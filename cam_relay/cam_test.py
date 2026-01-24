import argparse
import csv
import os
import time
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np


@dataclass
class LineROI:
    p1: tuple[int, int] | None = None
    p2: tuple[int, int] | None = None
    active: bool = False


class ClickLineSelector:
    """
    Click two points on the frame to define a line ROI.
    Press 'l' to arm selection. Then click p1 and p2.
    """
    def __init__(self):
        self.roi = LineROI()
        self._armed = False

    def arm(self):
        self._armed = True
        self.roi = LineROI(active=True)

    def cancel(self):
        self._armed = False
        self.roi = LineROI(active=False)

    def on_mouse(self, event, x, y, flags, param):
        if not self._armed:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi.p1 is None:
                self.roi.p1 = (x, y)
            elif self.roi.p2 is None:
                self.roi.p2 = (x, y)
                self._armed = False  # done


def list_cameras(max_index: int = 10) -> list[int]:
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows-friendly
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
        cap.release()
    return available


def set_capture_props(cap: cv2.VideoCapture, width: int, height: int, fps: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))


def draw_hud(frame, text_lines, origin=(10, 25)):
    x, y = origin
    for line in text_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def sample_intensity_along_line(gray: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int = 200) -> np.ndarray:
    """
    Sample grayscale intensity along a line segment using linear interpolation.
    Returns an array shape (n,) of intensities.
    """
    x1, y1 = p1
    x2, y2 = p2
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)

    # Bilinear sampling
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1b = np.clip(x0 + 1, 0, gray.shape[1] - 1)
    y1b = np.clip(y0 + 1, 0, gray.shape[0] - 1)
    x0 = np.clip(x0, 0, gray.shape[1] - 1)
    y0 = np.clip(y0, 0, gray.shape[0] - 1)

    dx = xs - x0
    dy = ys - y0

    Ia = gray[y0, x0]
    Ib = gray[y0, x1b]
    Ic = gray[y1b, x0]
    Id = gray[y1b, x1b]

    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    return (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(np.float32)


def dominant_freq_hz(signal: np.ndarray, fs: float) -> float | None:
    """
    Estimate dominant frequency in Hz using FFT peak (excluding DC).
    Returns None if not enough data.
    """
    n = len(signal)
    if n < 32:
        return None

    # Window + detrend
    x = signal - np.mean(signal)
    win = np.hanning(n)
    xw = x * win

    # FFT
    spec = np.fft.rfft(xw)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if len(mag) < 3:
        return None

    mag[0] = 0.0  # remove DC
    peak_idx = int(np.argmax(mag))
    return float(freqs[peak_idx])


def make_writer(out_path: str, fps: float, frame_size: tuple[int, int]):
    """
    Try MP4 first; fallback to AVI.
    """
    w, h = frame_size
    ext = os.path.splitext(out_path)[1].lower()

    if ext in [".mp4", ".m4v"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        # fallback
        fallback = os.path.splitext(out_path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(fallback, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter (mp4/avi). Try installing a codec pack or use AVI.")
        return writer, fallback
    return writer, out_path


def main():
    ap = argparse.ArgumentParser(description="OBSBOT camera test relay + standing-wave quick probe (line ROI + FFT).")
    ap.add_argument("--list", action="store_true", help="List available camera indices and exit.")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0).")
    ap.add_argument("--width", type=int, default=1920, help="Capture width (try 3840 for 4K).")
    ap.add_argument("--height", type=int, default=1080, help="Capture height (try 2160 for 4K).")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS.")
    ap.add_argument("--outdir", type=str, default="captures", help="Directory for outputs.")
    ap.add_argument("--buffer-sec", type=float, default=4.0, help="Seconds of signal buffer for FFT.")
    ap.add_argument("--line-samples", type=int, default=200, help="Samples along the ROI line.")
    args = ap.parse_args()

    if args.list:
        cams = list_cameras(12)
        print("Available camera indices:", cams if cams else "(none found)")
        return

    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.cam}. Try --list and pick another index.")

    set_capture_props(cap, args.width, args.height, args.fps)

    # Read one frame to confirm actual size/FPS
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera opened but could not read frames.")
    h, w = frame.shape[:2]

    window = "WaveCam Test (q quit | r rec | s snap | l line ROI | c clear ROI)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    selector = ClickLineSelector()
    cv2.setMouseCallback(window, selector.on_mouse)

    # Recording
    recording = False
    writer = None
    out_video_path = None

    # ROI analysis buffer
    # We'll track a 1D signal: e.g., mean intensity along line (or variance).
    buffer_len = max(32, int(args.buffer_sec * args.fps))
    sig = deque(maxlen=buffer_len)
    t_sig = deque(maxlen=buffer_len)

    # CSV log for frequency estimates
    csv_path = os.path.join(args.outdir, f"freq_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["unix_time", "dominant_freq_hz", "signal_value", "roi_p1", "roi_p2"])

    # FPS measurement
    last_time = time.time()
    fps_est = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed; exiting.")
                break

            now = time.time()
            dt = now - last_time
            last_time = now
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if dt > 0 else fps_est

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ROI analysis if line is defined
            dom_hz = None
            if selector.roi.active and selector.roi.p1 and selector.roi.p2:
                p1, p2 = selector.roi.p1, selector.roi.p2
                cv2.line(display, p1, p2, (0, 255, 0), 2)

                line_vals = sample_intensity_along_line(gray, p1, p2, n=args.line_samples)

                # A simple scalar signal: spatial variance along line (sensitive to wave pattern changes)
                scalar = float(np.var(line_vals))

                sig.append(scalar)
                t_sig.append(now)

                # Estimate frequency once buffer has enough samples
                if len(sig) >= 64:
                    # Effective sampling rate based on timestamps
                    duration = t_sig[-1] - t_sig[0]
                    fs = (len(t_sig) - 1) / duration if duration > 0 else float(args.fps)
                    dom_hz = dominant_freq_hz(np.array(sig, dtype=np.float32), fs)

                csv_writer.writerow([now, dom_hz if dom_hz is not None else "", scalar, p1, p2])

            # HUD
            hud = [
                f"Cam {args.cam} | {w}x{h} | fps~{fps_est:.1f} (target {args.fps})",
                time.strftime("Local time: %Y-%m-%d %H:%M:%S"),
                "Keys: q quit | r rec | s snap | l set line ROI | c clear ROI",
            ]
            if recording:
                hud.append(f"REC: {os.path.basename(out_video_path) if out_video_path else '(writer?)'}")
            if dom_hz is not None:
                hud.append(f"ROI dominant freq: {dom_hz:.2f} Hz")
            elif selector.roi.active and selector.roi.p1 and selector.roi.p2:
                hud.append("ROI dominant freq: (warming up buffer...)")
            elif selector.roi.active:
                hud.append("ROI: click 2 points to set line")

            draw_hud(display, hud)

            # Write recording if enabled
            if recording and writer is not None:
                writer.write(frame)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_path = os.path.join(args.outdir, f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(snap_path, frame)
                print(f"[snapshot] {snap_path}")
            elif key == ord("r"):
                if not recording:
                    base = f"record_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    out_video_path = os.path.join(args.outdir, base)
                    writer, out_video_path = make_writer(out_video_path, float(args.fps), (w, h))
                    recording = True
                    print(f"[recording ON] {out_video_path}")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("[recording OFF]")
            elif key == ord("l"):
                selector.arm()
                print("[ROI] armed: click two points along the rubber band.")
                sig.clear()
                t_sig.clear()
            elif key == ord("c"):
                selector.cancel()
                print("[ROI] cleared.")
                sig.clear()
                t_sig.clear()

    finally:
        if writer:
            writer.release()
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[saved] frequency log: {csv_path}")


if __name__ == "__main__":
    main()
