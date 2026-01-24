import os
import time
import csv
from collections import deque

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


class DragROI:
    def __init__(self):
        self.dragging = False
        self.active = False
        self.p0 = None
        self.p1 = None
        self.roi = None  # (x0,y0,x1,y1)
        self.armed = False

    def arm(self):
        self.armed = True
        self.dragging = False
        self.active = False
        self.p0 = None
        self.p1 = None
        self.roi = None

    def clear(self):
        self.dragging = False
        self.active = False
        self.p0 = None
        self.p1 = None
        self.roi = None
        self.armed = False

    def on_mouse(self, event, x, y, flags, param):
        if not self.armed:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.p0 = (x, y)
            self.p1 = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.p1 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.p1 = (x, y)
            x0 = min(self.p0[0], self.p1[0])
            y0 = min(self.p0[1], self.p1[1])
            x1 = max(self.p0[0], self.p1[0])
            y1 = max(self.p0[1], self.p1[1])

            # reject tiny boxes
            if (x1 - x0) > 20 and (y1 - y0) > 40:
                self.roi = (x0, y0, x1, y1)
                self.active = True

            self.armed = False


def band_mask(gray_roi: np.ndarray) -> np.ndarray:
    """Segment dark band on light background inside ROI."""
    g = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return th


def centerline_x_per_row(mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(mask > 0)
    h, _ = skel.shape
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


def make_writer_avi_mjpg(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w, h = size
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try saving to a non-OneDrive folder (e.g., C:\\temp\\caypt_vids).")
    return writer


def main():
    # --------- Settings you can edit quickly ----------
    CAM_INDEX = 0
    WIDTH, HEIGHT = 1920, 1080
    TARGET_FPS = 30
    OUTDIR = r"C:\temp\caypt_vids"  # avoid OneDrive for sanity
    WINDOW_SEC = 1.0               # RMS window length (s)
    MIN_PEAK_DIST = 50             # px along y between peaks
    MIN_PROM_FRAC = 0.08           # prominence as fraction of max amplitude
    SMOOTH_K = 51                  # smoothing length for A(y)
    # ---------------------------------------------------

    os.makedirs(OUTDIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, float(TARGET_FPS))

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")

    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

    roi_selector = DragROI()

    tracking = False
    baseline = None
    window_frames = max(10, int(WINDOW_SEC * fps))
    disp_buf = deque(maxlen=window_frames)

    # recording state
    recording = False
    writer = None
    csv_f = None
    csv_w = None
    rec_path = None
    csv_path = None

    # fps estimate
    last = time.time()
    fps_est = 0.0

    win = "Wave Boxes (b ROI | t track | r rec | s snap | c clear | q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, roi_selector.on_mouse)

    print("Controls:")
    print("  b = select ROI (click+drag)")
    print("  t = toggle tracking (ROI must be set)")
    print("  r = start/stop recording annotated video + CSV log")
    print("  s = snapshot")
    print("  c = clear ROI (stops tracking/recording)")
    print("  q = quit")

    frame_i = 0

    try:
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

            # draw ROI selection rectangle while dragging
            if roi_selector.dragging and roi_selector.p0 and roi_selector.p1:
                cv2.rectangle(display, roi_selector.p0, roi_selector.p1, (255, 255, 0), 2)

            mode_n = 0
            amax = None

            # active ROI tracking
            if roi_selector.active and roi_selector.roi:
                x0, y0, x1, y1 = roi_selector.roi
                cv2.rectangle(display, (x0, y0), (x1, y1), (200, 200, 200), 1)

                if tracking:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_roi = gray[y0:y1, x0:x1]

                    mask = band_mask(gray_roi)
                    xs = centerline_x_per_row(mask)
                    xs = nan_interp(xs)

                    if baseline is None:
                        baseline = xs.copy()

                    disp = xs - baseline
                    disp_buf.append(disp)

                    roi_h = y1 - y0

                    if len(disp_buf) >= max(8, window_frames // 2):
                        X = np.stack(disp_buf, axis=0)
                        A = np.sqrt(np.mean(X**2, axis=0))
                        A = smooth_1d(A, k=SMOOTH_K)

                        amax = float(np.max(A))
                        prom = max(1e-6, amax * MIN_PROM_FRAC)

                        peaks, _ = find_peaks(A, distance=MIN_PEAK_DIST, prominence=prom)
                        mins, _ = find_peaks(-A, distance=MIN_PEAK_DIST, prominence=prom * 0.5)

                        mode_n = int(len(peaks))

                        # draw centerline
                        pts = []
                        for yy in range(0, roi_h, 10):
                            if np.isfinite(xs[yy]):
                                pts.append((int(x0 + xs[yy]), int(y0 + yy)))
                        for i in range(1, len(pts)):
                            cv2.line(display, pts[i - 1], pts[i], (0, 255, 0), 2, lineType=cv2.LINE_AA)

                        # build boxes between adjacent nodes (mins)
                        boxes = []
                        mins_sorted = np.sort(mins)
                        peaks_sorted = np.sort(peaks)

                        if len(mins_sorted) >= 2:
                            for i in range(len(mins_sorted) - 1):
                                yt = int(mins_sorted[i])
                                yb = int(mins_sorted[i + 1])
                                mid_peaks = peaks_sorted[(peaks_sorted > yt) & (peaks_sorted < yb)]
                                if len(mid_peaks) == 0:
                                    continue

                                slab_A = A[yt:yb]
                                w_px = int(max(12, 6 * float(np.max(slab_A)))) if len(slab_A) else 30
                                y_mid = (yt + yb) // 2
                                x_mid = float(xs[y_mid]) if np.isfinite(xs[y_mid]) else float(np.nanmean(xs[yt:yb]))
                                if not np.isfinite(x_mid):
                                    continue

                                xl = int(x0 + x_mid - w_px)
                                xr = int(x0 + x_mid + w_px)
                                boxes.append((xl, y0 + yt, xr, y0 + yb))
                        else:
                            # fallback: box around each antinode if node detection fails
                            for y_peak in peaks_sorted:
                                y_peak = int(y_peak)
                                h_box = 80
                                yt = max(0, y_peak - h_box // 2)
                                yb = min(roi_h - 1, y_peak + h_box // 2)
                                slab_A = A[yt:yb]
                                w_px = int(max(12, 6 * float(np.max(slab_A)))) if len(slab_A) else 30
                                x_mid = float(xs[y_peak]) if np.isfinite(xs[y_peak]) else float(np.nanmean(xs))
                                if not np.isfinite(x_mid):
                                    continue
                                xl = int(x0 + x_mid - w_px)
                                xr = int(x0 + x_mid + w_px)
                                boxes.append((xl, y0 + yt, xr, y0 + yb))

                        # draw boxes + labels
                        for j, (xl, yt, xr, yb) in enumerate(boxes, start=1):
                            xl = max(0, xl); xr = min(W - 1, xr)
                            yt = max(0, yt); yb = min(H - 1, yb)
                            cv2.rectangle(display, (xl, yt), (xr, yb), (0, 255, 255), 2)
                            cv2.putText(display, f"{j}", (xl + 6, yt + 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                            cv2.putText(display, f"{j}", (xl + 6, yt + 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

                    # CSV logging (every frame while recording)
                    if recording and csv_w is not None:
                        csv_w.writerow([
                            frame_i,
                            now,
                            mode_n,
                            amax if amax is not None else "",
                            roi_selector.roi
                        ])

            # HUD
            hud = [
                f"fps~{fps_est:.1f}",
                f"ROI={'SET' if roi_selector.active else 'NOT SET'}  TRACK={'ON' if tracking else 'OFF'}  REC={'ON' if recording else 'OFF'}",
                f"mode n~{mode_n}  Amax(px)={amax if amax is not None else ''}",
                "Keys: b ROI | t track | r rec | s snap | c clear | q quit",
            ]
            ytxt = 25
            for line in hud:
                cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                ytxt += 24

            if roi_selector.armed:
                msg = "ROI SELECT MODE: click + drag a rectangle around the band, then release"
                cv2.putText(display, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

            # write annotated video if recording
            if recording and writer is not None:
                writer.write(display)

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("b"):
                roi_selector.arm()
                tracking = False
                baseline = None
                disp_buf.clear()

            elif key == ord("t"):
                if roi_selector.active:
                    tracking = not tracking
                    baseline = None
                    disp_buf.clear()

            elif key == ord("c"):
                # stop recording too
                if recording:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    if csv_f:
                        csv_f.close()
                        csv_f = None
                        csv_w = None
                    print("[REC OFF] (cleared ROI)")
                roi_selector.clear()
                tracking = False
                baseline = None
                disp_buf.clear()

            elif key == ord("s"):
                path = os.path.join(OUTDIR, f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(path, display)
                print(f"[SNAP] {path}")

            elif key == ord("r"):
                if not recording:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(OUTDIR, f"wave_boxes_{ts}.avi")
                    csv_path = os.path.join(OUTDIR, f"wave_boxes_{ts}.csv")

                    writer = make_writer_avi_mjpg(rec_path, fps if fps > 1 else TARGET_FPS, (W, H))
                    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
                    csv_w = csv.writer(csv_f)
                    csv_w.writerow(["frame", "unix_time", "mode_n", "amax_px", "roi"])

                    recording = True
                    print(f"[REC ON] {rec_path}")
                    print(f"[CSV]    {csv_path}")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    if csv_f:
                        csv_f.close()
                        csv_f = None
                        csv_w = None
                    print("[REC OFF]")

            frame_i += 1

    finally:
        if writer:
            writer.release()
        if csv_f:
            csv_f.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
