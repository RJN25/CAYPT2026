import argparse
import csv
import os
import time
from collections import deque

import cv2
import numpy as np

# Optional but recommended
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


def get_band_mask(gray_roi: np.ndarray) -> np.ndarray:
    """Segment dark rubber band from light background inside ROI."""
    g = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return th


def skeleton_centerline(mask: np.ndarray) -> np.ndarray:
    """Return per-row x centerline (float, with NaNs possible)."""
    skel = skeletonize(mask > 0)
    h, w = skel.shape
    xs = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        x_idx = np.where(skel[y])[0]
        if len(x_idx) >= 1:
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


def draw_polyline(frame, points, color=(0, 255, 0), thickness=2):
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, thickness, lineType=cv2.LINE_AA)


def draw_points(frame, pts, color, r=5):
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)


def make_writer(path, fps, size):
    """AVI MJPG is the most 'just works' on Windows."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w, h = size
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try a different --outdir (non-OneDrive) or install codecs.")
    return writer


def main():
    ap = argparse.ArgumentParser(description="Realtime rubber-band standing-wave tracker (OpenCV overlay + CSV).")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (0,1,2...). Try --list.")
    ap.add_argument("--list", action="store_true", help="List camera indices and exit.")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--outdir", type=str, default=r"C:\temp\caypt_vids", help="Where to save recordings/CSVs (avoid OneDrive).")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("x0","y0","x1","y1"),
                    help="ROI rectangle around the band. If omitted, uses middle vertical strip.")
    ap.add_argument("--window_sec", type=float, default=1.0, help="RMS window in seconds for nodes/antinodes.")
    ap.add_argument("--min_peak_dist", type=int, default=40, help="Min vertical distance between peaks in pixels.")
    args = ap.parse_args()

    def list_cameras(max_index=12):
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
            cap.release()
        return found

    if args.list:
        print("Available camera indices:", list_cameras())
        return

    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.cam}. Try: python realtime_band_tracker.py --list")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")

    H, W = frame.shape[:2]
    fps_guess = cap.get(cv2.CAP_PROP_FPS) or args.fps

    if args.roi:
        roi = tuple(args.roi)
    else:
        roi = (W // 3, 0, 2 * W // 3, H)

    x0, y0, x1, y1 = roi
    roi_h = y1 - y0

    # Tracking state
    tracking = False
    baseline = None

    # Buffer for RMS amplitude profile
    window_frames = max(10, int(args.window_sec * fps_guess))
    disp_buf = deque(maxlen=window_frames)

    # Recording state
    recording = False
    writer = None
    csv_file = None
    csv_writer = None
    rec_path = None
    csv_path = None

    # FPS estimate
    last = time.time()
    fps_est = 0.0

    window_name = "Realtime Band Tracker (t track | r rec | s snap | q quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Controls:")
    print("  t = toggle tracking ON/OFF")
    print("  r = start/stop recording + CSV logging")
    print("  s = snapshot")
    print("  q = quit")

    try:
        frame_i = 0
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

            # ROI visuals
            cv2.rectangle(display, (x0, y0), (x1, y1), (200, 200, 200), 1)

            mode_n = ""
            avg_node_spacing = ""
            amax = ""

            if tracking:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_roi = gray[y0:y1, x0:x1]

                mask = get_band_mask(gray_roi)
                xs = skeleton_centerline(mask)
                xs = nan_interp(xs)

                if baseline is None:
                    baseline = xs.copy()

                disp = xs - baseline
                disp_buf.append(disp)

                if len(disp_buf) >= max(8, window_frames // 2):
                    X = np.stack(disp_buf, axis=0)  # (T, Y)
                    A = np.sqrt(np.mean(X**2, axis=0))  # RMS amplitude profile
                    A = smooth_1d(A, k=51)

                    amax = float(np.max(A))

                    # Antinodes: peaks in A
                    peaks, _ = find_peaks(A, distance=args.min_peak_dist, prominence=max(1e-6, amax * 0.05))
                    # Nodes: minima of A -> peaks in -A
                    mins, _ = find_peaks(-A, distance=args.min_peak_dist, prominence=max(1e-6, amax * 0.02))

                    mode_n = int(len(peaks))

                    if len(mins) >= 2:
                        spacings = np.diff(mins)
                        avg_node_spacing = float(np.mean(spacings))

                    # Draw centerline polyline
                    pts = []
                    for yy in range(0, roi_h, 8):
                        x = xs[yy]
                        pts.append((int(x0 + x), int(y0 + yy)))
                    draw_polyline(display, pts, color=(0, 255, 0), thickness=2)

                    # Draw node / antinode dots
                    node_pts = [(x0 + xs[y], y0 + y) for y in mins if np.isfinite(xs[y])]
                    anti_pts = [(x0 + xs[y], y0 + y) for y in peaks if np.isfinite(xs[y])]
                    draw_points(display, node_pts, (255, 0, 0), r=5)   # blue
                    draw_points(display, anti_pts, (0, 0, 255), r=5)   # red

                # If recording, write metrics row every frame
                if recording and csv_writer is not None:
                    csv_writer.writerow([
                        frame_i,
                        now,
                        mode_n,
                        avg_node_spacing,
                        amax,
                        tracking,
                        roi
                    ])

            # HUD text
            hud = [
                f"cam={args.cam}  {W}x{H}  fps~{fps_est:.1f}",
                f"TRACKING={'ON' if tracking else 'OFF'}   REC={'ON' if recording else 'OFF'}",
                f"mode n={mode_n}   avg node spacing(px)={avg_node_spacing}   Amax(px)={amax}",
                "Keys: t track | r rec | s snap | q quit"
            ]
            ytxt = 25
            for line in hud:
                cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                ytxt += 24

            if recording and writer is not None:
                writer.write(display)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("t"):
                tracking = not tracking
                if tracking:
                    baseline = None
                    disp_buf.clear()
                print(f"[tracking] {'ON' if tracking else 'OFF'}")
            elif key == ord("s"):
                path = os.path.join(args.outdir, f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(path, display)
                print(f"[snapshot] {path}")
            elif key == ord("r"):
                if not recording:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(args.outdir, f"realtime_{ts}.avi")
                    csv_path = os.path.join(args.outdir, f"realtime_{ts}.csv")
                    writer = make_writer(rec_path, fps_guess if fps_guess > 1 else args.fps, (W, H))

                    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["frame", "unix_time", "mode_n", "avg_node_spacing_px", "amax_px", "tracking", "roi"])

                    recording = True
                    print(f"[REC ON] {rec_path}")
                    print(f"[CSV] {csv_path}")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    if csv_file:
                        csv_file.close()
                        csv_file = None
                        csv_writer = None
                    print("[REC OFF]")

            frame_i += 1

    finally:
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
