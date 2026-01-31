"""
Slomo: slow down a video extremely for frame-by-frame analysis.

Reads an input video and writes a new video where each original frame
is repeated many times, so playback is very slow (e.g. 0.1x–0.05x speed).
Useful for analyzing fast motion (e.g. juggling, sports).

Run with no arguments to open a file picker and choose a .mov or .mp4.
Or: python slomo.py path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2

# Optional: file dialog when run with no args (tkinter is in the stdlib)
try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_FILEDIALOG = True
except ImportError:
    _HAS_FILEDIALOG = False


def get_video_info(cap: cv2.VideoCapture) -> tuple[float, int, int, int]:
    """Return (fps, width, height, frame_count)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, w, h, n


def slow_down_video(
    input_path: str | Path,
    output_path: str | Path,
    slowdown: float = 20.0,
    fourcc: str = "mp4v",
) -> None:
    """
    Create a slowed-down copy of the video.

    Each original frame is written `slowdown` times at the same output FPS,
    so effective playback speed = 1/slowdown (e.g. slowdown=20 → 0.05x speed).

    Args:
        input_path: Path to input video file.
        output_path: Path for output video file.
        slowdown: How many output frames per input frame (default 20 → very slow).
        fourcc: FourCC for output (default mp4v for .mp4).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    in_fps, w, h, total_frames = get_video_info(cap)
    # Output FPS = input FPS so we keep smooth motion; we just repeat each frame.
    out_fps = in_fps
    out_frame_count = int(total_frames * slowdown)

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(str(output_path), fourcc_code, out_fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    written = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for _ in range(int(slowdown)):
                out.write(frame)
                written += 1
                if written % 500 == 0:
                    print(f"  Written {written} / ~{out_frame_count} frames", end="\r")
        print(f"  Written {written} frames total. Done.")
    finally:
        cap.release()
        out.release()


VIDEO_TYPES = (
    ("Video files", "*.mp4 *.mov *.avi *.mkv"),
    ("MP4", "*.mp4"),
    ("QuickTime", "*.mov"),
    ("All files", "*.*"),
)


def pick_input_and_output(slowdown: float = 20.0) -> tuple[Path, Path] | None:
    """Open file picker for .mov/.mp4, then save-as for output. Returns (input_path, output_path) or None if cancelled."""
    if not _HAS_FILEDIALOG:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    input_path = filedialog.askopenfilename(
        title="Choose a video to analyze (.mov or .mp4)",
        filetypes=VIDEO_TYPES,
    )
    if not input_path:
        return None
    input_path = Path(input_path)
    default_name = f"{input_path.stem}_slomo{input_path.suffix or '.mp4'}"
    output_path = filedialog.asksaveasfilename(
        title="Save slowed video as",
        initialfile=default_name,
        initialdir=str(input_path.parent),
        filetypes=VIDEO_TYPES,
        defaultextension=input_path.suffix or ".mp4",
    )
    root.destroy()
    if not output_path:
        return None
    return input_path, Path(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Slow down a video extremely for frame-by-frame analysis."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input video file path (omit to open file picker)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output video path (default: input_slomo.<same ext>)",
    )
    parser.add_argument(
        "-s", "--slowdown",
        type=float,
        default=20.0,
        help="Output frames per input frame (default: 20 → ~0.05x speed)",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC for output (default: mp4v for .mp4)",
    )
    args = parser.parse_args()

    # No input path → open file picker
    if args.input is None:
        if not _HAS_FILEDIALOG:
            print("Usage: python slomo.py <path/to/video.mp4>", file=sys.stderr)
            print("Or install tkinter for a file picker.", file=sys.stderr)
            return 1
        picked = pick_input_and_output(slowdown=args.slowdown)
        if picked is None:
            print("Cancelled.")
            return 0
        args.input, args.output = picked
    elif args.output is None:
        stem = args.input.stem
        suffix = args.input.suffix or ".mp4"
        args.output = args.input.parent / f"{stem}_slomo{suffix}"

    try:
        slow_down_video(
            args.input,
            args.output,
            slowdown=args.slowdown,
            fourcc=args.fourcc,
        )
        print(f"Saved: {args.output}")
        return 0
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
