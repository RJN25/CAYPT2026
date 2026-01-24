import argparse
import csv
import math
from collections import deque

import cv2
import numpy as np

# Optional but strongly recommended for node/antinode peak finding + skeletonization
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


def get_band_mask(gray, roi):
    """Segment dark rubber band from light background inside ROI."""
    x0, y0, x1, y1 = roi
    g = gray[y0:y1, x0:x1]

    # Normalize + blur to reduce noise
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)

    # Otsu threshold (band is darker -> invert)
    _, th = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    return th


def skeleton_centerline(mask):
    """Return skeleton (bool) and a per-row center x estimate."""
    # skimage expects bool mask
    skel = skeletonize(mask > 0)

    h, w = skel.shape
    xs = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        x_idx = np.where(skel[y])[0]
        if len(x_idx) >= 1:
            xs[y] = float(np.mean(x_idx))  # center of skeleton pixels in that row

    return skel, xs


def nan_interp(x):
    """Interpolate NaNs linearly."""
    n = len(x)
    idx = np.arange(n)
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    return np.interp(idx, idx[good], x[good]).astype(np.float32)


def smooth_1d(x, k=31):
    """Simple moving average smoothing; k should be odd."""
    k = max(3, int(k) | 1)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def draw_points(frame, pts, color, r=4):
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/mnt/data/triple_fast.mov")
    ap.add_argument("--out_video", default="annotated.mp4")
    ap.add_argument("--out_csv", default="metrics.csv")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("x0","y0","x1","y1"),
                    help="ROI rectangle around the band in full-frame coords (recommended).")
    ap.add_argument("--window", type=int, default=30, help="Frames for RMS amplitude window (~1s at 30fps).")
    ap.add_argument("--min_peak_dist", type=int, default=40, help="Min y-distance between nodes/antinodes peaks.")
    ap.add_argument("--debug_mask", action="store_true", help="Show mask window for tuning.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Default ROI: middle vertical strip (works for your setup if band is near center)
    if args.roi:
        roi = tuple(args.roi)
    else:
        roi = (W // 3, 0, 2 * W // 3, H)

    x0, y0, x1, y1 = roi
    roi_w, roi_h = x1 - x0, y1 - y0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (W, H))

    # Store last N centerlines to compute RMS amplitude along y
    buf = deque(maxlen=args.window)
    baseline = None

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow([
            "frame", "time_s",
            "mode_n", "avg_node_spacing_px", "amax_px",
            "roi_x0","roi_y0","roi_x1","roi_y1"
        ])

        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = get_band_mask(gray, roi)
            skel, xs = skeleton_centerline(mask)
            xs = nan_interp(xs)

            # If skeletonization failed (too few pixels), skip gracefully
            if not np.isfinite(xs).all():
                xs = nan_interp(xs)

            # Baseline: first stable centerline
            if baseline is None:
                baseline = xs.copy()

            disp = xs - baseline  # displacement in ROI coords
            buf.append(disp)

            # Build amplitude profile when buffer filled enough
            mode_n = ""
            avg_spacing = ""
            amax = ""

            nodes_pts = []
            antis_pts = []

            if len(buf) >= max(10, args.window // 2):
                X = np.stack(buf, axis=0)  # (T, Y)
                A = np.sqrt(np.mean(X**2, axis=0))  # RMS amplitude vs y
                A = smooth_1d(A, k=51)

                amax = float(np.max(A))

                # Antinodes: peaks of A(y)
                peaks, _ = find_peaks(A, distance=args.min_peak_dist, prominence=np.max(A) * 0.05)

                # Nodes: peaks of (-A) => minima of A
                mins, _ = find_peaks(-A, distance=args.min_peak_dist, prominence=np.max(A) * 0.02)

                # Mode number: count antinodes (rough but practical)
                mode_n = int(len(peaks))

                # Average node spacing: spacing between consecutive minima
                if len(mins) >= 2:
                    spacings = np.diff(mins)
                    avg_spacing = float(np.mean(spacings))

                # Convert peak y positions back to full-frame coords
                for y in mins:
                    x = xs[y]
                    nodes_pts.append((x0 + x, y0 + y))
                for y in peaks:
                    x = xs[y]
                    antis_pts.append((x0 + x, y0 + y))

                # Draw centerline polyline (sample every few pixels)
                pts = []
                for y in range(0, roi_h, 8):
                    pts.append((int(x0 + xs[y]), int(y0 + y)))
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), 2, lineType=cv2.LINE_AA)

                draw_points(frame, nodes_pts, (255, 0, 0), r=5)   # blue nodes
                draw_points(frame, antis_pts, (0, 0, 255), r=5)   # red antinodes

            # ROI rectangle
            cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

            # HUD
            t = frame_i / fps
            hud = [
                f"t={t:0.2f}s  fps={fps:0.1f}",
                f"mode n={mode_n}   avg node spacing(px)={avg_spacing}   Amax(px)={amax}",
                "Blue=nodes (min A(y))  Red=antinodes (max A(y))  Green=centerline",
            ]
            y_text = 25
            for line in hud:
                cv2.putText(frame, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                y_text += 24

            writer.write(frame)

            cw.writerow([frame_i, t, mode_n, avg_spacing, amax, x0, y0, x1, y1])

            if args.debug_mask:
                cv2.imshow("mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_i += 1

    cap.release()
    writer.release()
    if args.debug_mask:
        cv2.destroyAllWindows()

    print(f"Saved: {args.out_video}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
