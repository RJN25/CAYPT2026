import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


def band_mask(gray_roi: np.ndarray) -> np.ndarray:
    """Segment dark band on light background inside ROI."""
    g = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return th


def centerline_x_per_row(mask: np.ndarray) -> np.ndarray:
    """Skeletonize and return x(y) as mean skeleton x per row."""
    skel = skeletonize(mask > 0)
    h, w = skel.shape
    xs = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        x_idx = np.where(skel[y])[0]
        if len(x_idx):
            xs[y] = float(np.mean(x_idx))
    return xs


def nan_interp(x: np.ndarray) -> np.ndarray:
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    return np.interp(idx, idx[good], x[good]).astype(np.float32)


def smooth_1d(x: np.ndarray, k: int = 51) -> np.ndarray:
    k = max(3, int(k) | 1)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Realtime bounding boxes per standing-wave loop (mode count).")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--roi", type=int, nargs=4, metavar=("x0", "y0", "x1", "y1"),
                    help="ROI around the band (HIGHLY recommended).")
    ap.add_argument("--window_sec", type=float, default=1.0, help="Time window for RMS amplitude profile.")
    ap.add_argument("--min_peak_dist", type=int, default=50, help="Min vertical distance between antinodes (pixels).")
    ap.add_argument("--min_prom_frac", type=float, default=0.08,
                    help="Peak prominence as fraction of max amplitude (0.05-0.15 typical).")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")

    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    # Default ROI: middle strip (you should override with --roi)
    if args.roi:
        x0, y0, x1, y1 = args.roi
    else:
        x0, y0, x1, y1 = (W // 3, 0, 2 * W // 3, H)

    roi_h = y1 - y0

    # Tracking toggles
    tracking = True
    baseline = None

    # Buffer of displacements for RMS amplitude profile
    window_frames = max(10, int(args.window_sec * fps))
    disp_buf = deque(maxlen=window_frames)

    # FPS estimate
    last = time.time()
    fps_est = 0.0

    win = "Wave Boxes (q quit | t toggle tracking)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt)

        display = frame.copy()

        # Draw ROI
        cv2.rectangle(display, (x0, y0), (x1, y1), (200, 200, 200), 1)

        mode_n = 0

        if tracking:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_roi = gray[y0:y1, x0:x1]

            mask = band_mask(gray_roi)
            xs = centerline_x_per_row(mask)
            xs = nan_interp(xs)

            # If baseline not set, set it (string at rest-ish)
            if baseline is None:
                baseline = xs.copy()

            disp = xs - baseline
            disp_buf.append(disp)

            if len(disp_buf) >= max(8, window_frames // 2):
                X = np.stack(disp_buf, axis=0)          # (T, Y)
                A = np.sqrt(np.mean(X**2, axis=0))      # RMS amplitude along y
                A = smooth_1d(A, k=51)

                amax = float(np.max(A))
                prom = max(1e-6, amax * args.min_prom_frac)

                # Antinodes: peaks in A(y)
                peaks, _ = find_peaks(A, distance=args.min_peak_dist, prominence=prom)

                # Nodes: minima in A(y) -> peaks in -A(y)
                mins, _ = find_peaks(-A, distance=args.min_peak_dist, prominence=prom * 0.5)

                # We build boxes between adjacent nodes around each peak.
                # If node detection fails, fallback: just count peaks (antinodes) and box around them.
                mode_n = int(len(peaks))

                # Draw centerline (green)
                pts = []
                for yy in range(0, roi_h, 10):
                    if np.isfinite(xs[yy]):
                        pts.append((int(x0 + xs[yy]), int(y0 + yy)))
                for i in range(1, len(pts)):
                    cv2.line(display, pts[i - 1], pts[i], (0, 255, 0), 2, lineType=cv2.LINE_AA)

                # Build boxes
                boxes = []

                mins_sorted = np.sort(mins) if len(mins) else mins
                peaks_sorted = np.sort(peaks)

                if len(mins_sorted) >= 2:
                    # For each adjacent pair of nodes, see if an antinode lies between -> that's one "loop"
                    for i in range(len(mins_sorted) - 1):
                        y_top = int(mins_sorted[i])
                        y_bot = int(mins_sorted[i + 1])
                        mid_peaks = peaks_sorted[(peaks_sorted > y_top) & (peaks_sorted < y_bot)]
                        if len(mid_peaks) == 0:
                            continue

                        # Use that region as one loop box. Width based on max lateral excursion in that y-slab.
                        slab = disp[y_top:y_bot]
                        if len(slab) < 5:
                            continue

                        # Estimate how "wide" the motion gets in this slab
                        # Use RMS across time in slab: approximate by current amplitude A
                        slab_A = A[y_top:y_bot]
                        w_px = int(max(12, 6 * float(np.max(slab_A))))  # heuristic

                        # Center x from baseline in the middle of slab
                        y_mid = (y_top + y_bot) // 2
                        x_mid = float(xs[y_mid]) if np.isfinite(xs[y_mid]) else float(np.nanmean(xs[y_top:y_bot]))

                        if not np.isfinite(x_mid):
                            continue

                        x_left = int(x0 + x_mid - w_px)
                        x_right = int(x0 + x_mid + w_px)

                        boxes.append((x_left, y0 + y_top, x_right, y0 + y_bot))
                else:
                    # Fallback: box around each antinode peak y with a fixed height
                    for y_peak in peaks_sorted:
                        y_peak = int(y_peak)
                        h_box = 80
                        y_top = max(0, y_peak - h_box // 2)
                        y_bot = min(roi_h - 1, y_peak + h_box // 2)

                        slab_A = A[y_top:y_bot]
                        w_px = int(max(12, 6 * float(np.max(slab_A)))) if len(slab_A) else 30
                        x_mid = float(xs[y_peak]) if np.isfinite(xs[y_peak]) else float(np.nanmean(xs))

                        if not np.isfinite(x_mid):
                            continue

                        x_left = int(x0 + x_mid - w_px)
                        x_right = int(x0 + x_mid + w_px)
                        boxes.append((x_left, y0 + y_top, x_right, y0 + y_bot))

                # Draw boxes + label
                for j, (xl, yt, xr, yb) in enumerate(boxes, start=1):
                    xl = max(0, xl); xr = min(W - 1, xr)
                    yt = max(0, yt); yb = min(H - 1, yb)
                    cv2.rectangle(display, (xl, yt), (xr, yb), (0, 255, 255), 2)  # yellow
                    cv2.putText(display, f"{j}", (xl + 6, yt + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(display, f"{j}", (xl + 6, yt + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        # HUD
        hud = [
            f"TRACKING={'ON' if tracking else 'OFF'}   fps~{fps_est:.1f}",
            f"Mode estimate n = {mode_n}   (yellow boxes ~= loops/antinodes)",
            "Keys: t toggle tracking | q quit",
            "Tip: set a tight ROI around the band for stability."
        ]
        ytxt = 25
        for line in hud:
            cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            ytxt += 24

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            tracking = not tracking
            baseline = None
            disp_buf.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
