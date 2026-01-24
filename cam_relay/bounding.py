import time
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
        self.roi = None  # (x0,y0,x1,y1) with x0<x1,y0<y1
        self.armed = False

    def arm(self):
        self.armed = True
        self.dragging = False
        self.active = False
        self.p0 = None
        self.p1 = None
        self.roi = None

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
            # Reject tiny boxes
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
    th = cv2.morphologyEx(th, cv2.THRESH_BINARY, k, iterations=0)
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


def draw_points(frame, pts, color, r=5):
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)


def main():
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    # Set capture size (adjust if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")

    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    roi_selector = DragROI()

    tracking = False
    baseline = None

    window_sec = 1.0
    window_frames = max(10, int(window_sec * fps))
    disp_buf = deque(maxlen=window_frames)

    min_peak_dist = 50
    min_prom_frac = 0.08

    # FPS estimate
    last = time.time()
    fps_est = 0.0

    win = "Wave Boxes (b select ROI | t tracking | q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, roi_selector.on_mouse)

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

        # Draw current ROI selection drag box (while dragging)
        if roi_selector.dragging and roi_selector.p0 and roi_selector.p1:
            cv2.rectangle(display, roi_selector.p0, roi_selector.p1, (255, 255, 0), 2)

        # Draw active ROI
        if roi_selector.active and roi_selector.roi:
            x0, y0, x1, y1 = roi_selector.roi
            cv2.rectangle(display, (x0, y0), (x1, y1), (200, 200, 200), 1)

            mode_n = 0

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
                    A = smooth_1d(A, k=51)

                    amax = float(np.max(A))
                    prom = max(1e-6, amax * min_prom_frac)

                    peaks, _ = find_peaks(A, distance=min_peak_dist, prominence=prom)
                    mins, _ = find_peaks(-A, distance=min_peak_dist, prominence=prom * 0.5)

                    mode_n = int(len(peaks))

                    # Draw centerline (green)
                    pts = []
                    for yy in range(0, roi_h, 10):
                        if np.isfinite(xs[yy]):
                            pts.append((int(x0 + xs[yy]), int(y0 + yy)))
                    for i in range(1, len(pts)):
                        cv2.line(display, pts[i - 1], pts[i], (0, 255, 0), 2, lineType=cv2.LINE_AA)

                    # Build loop boxes between nodes
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
                        # Fallback: box around each antinode
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

                    # Draw boxes (yellow) + labels
                    for j, (xl, yt, xr, yb) in enumerate(boxes, start=1):
                        xl = max(0, xl); xr = min(W - 1, xr)
                        yt = max(0, yt); yb = min(H - 1, yb)
                        cv2.rectangle(display, (xl, yt), (xr, yb), (0, 255, 255), 2)
                        cv2.putText(display, f"{j}", (xl + 6, yt + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(display, f"{j}", (xl + 6, yt + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            # HUD for ROI active
            hud2 = f"ROI set. TRACKING={'ON' if tracking else 'OFF'}  Mode n~{mode_n}"
            cv2.putText(display, hud2, (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, hud2, (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Main HUD
        hud = [
            f"fps~{fps_est:.1f}",
            "Keys: b select ROI (click+drag) | t toggle tracking | q quit",
        ]
        ytxt = 25
        for line in hud:
            cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            ytxt += 24

        if roi_selector.armed:
            msg = "ROI SELECT MODE: click and drag a rectangle around the band"
            cv2.putText(display, msg, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, msg, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

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
            if roi_selector.active and roi_selector.roi:
                tracking = not tracking
                baseline = None
                disp_buf.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
