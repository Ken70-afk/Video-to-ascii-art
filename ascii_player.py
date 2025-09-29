#!/usr/bin/env python3
"""
ASCII Video Player
------------------
Render any video in your terminal as animated ASCII art.

Features
- Cross-platform (Windows, macOS, Linux)
- Auto uses video FPS (with optional cap/override)
- Aspect-ratio aware resizing for terminal characters
- Fit-to-terminal mode or fixed width
- Optional inversion and alternate character sets
- Vectorized NumPy mapping for speed
- Clear, typed, and documented code (portfolio friendly)

Dependencies: opencv-python, numpy
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from shutil import get_terminal_size
from typing import Tuple

import cv2
import numpy as np


# --------------------------- ASCII MAPPINGS ---------------------------

CHARSETS = {
    "classic": " .:-=+*#%@",
    "dense": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks": " ░▒▓█",
}


# --------------------------- CORE CONVERSION ---------------------------

def frame_to_ascii(
    frame_bgr: np.ndarray,
    out_width: int,
    char_aspect: float = 0.5,
    charset: str = "classic",
    invert: bool = False,
) -> str:
    """
    Convert a BGR frame (OpenCV) to an ASCII string.

    Args:
        frame_bgr: Input frame in BGR color space.
        out_width: Target character width for ASCII output.
        char_aspect: Approximate character width/height ratio (default 0.5).
        charset: One of CHARSETS keys.
        invert: If True, inverts brightness mapping.

    Returns:
        A string containing the ASCII art for the frame (with newlines).
    """
    chars = CHARSETS.get(charset, CHARSETS["classic"])
    if out_width < 1:
        out_width = 1

    # Compute output height based on aspect ratio:
    # chars are taller than wide; scale height down by char_aspect.
    h, w = frame_bgr.shape[:2]
    out_height = max(1, int((h * out_width / w) * char_aspect))

    # Resize for speed/quality
    resized = cv2.resize(frame_bgr, (out_width, out_height), interpolation=cv2.INTER_AREA)

    # Grayscale [0..255] -> normalize [0..1]
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if invert:
        gray = 1.0 - gray

    # Map to character indices using vectorized ops
    n = len(chars)
    idx = np.clip((gray * (n - 1)).astype(np.int32), 0, n - 1)

    # Build lines efficiently
    # Convert to array of bytes/str via lookup
    lut = np.frombuffer(chars.encode("utf-8"), dtype=np.uint8)
    # idx is 2D; map to bytes, then decode per line
    lines = []
    for row in idx:
        line_bytes = lut[row]
        lines.append(line_bytes.tobytes().decode("utf-8"))
    return "\n".join(lines)


# --------------------------- VIDEO LOOP ---------------------------

def play_video_ascii(
    video_path: Path,
    width: int | None,
    fit_terminal: bool,
    fps_override: float | None,
    fps_cap: float | None,
    clear_each_frame: bool,
    charset: str,
    invert: bool,
    char_aspect: float,
    loop: bool,
) -> None:
    """
    Stream a video file to the terminal as ASCII.
    """
    if not video_path.exists():
        print(f"Error: File not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    # Determine base FPS
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps <= 0 or np.isnan(vid_fps):
        vid_fps = 24.0  # reasonable default if metadata missing

    # Decide effective target FPS
    if fps_override and fps_override > 0:
        target_fps = fps_override
    else:
        target_fps = vid_fps

    if fps_cap and fps_cap > 0:
        target_fps = min(target_fps, fps_cap)

    frame_delay = 1.0 / max(1e-6, target_fps)

    # Pre-configure stdout to avoid encoding issues on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    # Compute width if fitting terminal
    if fit_terminal:
        cols, rows = get_terminal_size(fallback=(80, 24))
        # Leave one row for breathing room
        derived_width = max(10, cols)
    else:
        derived_width = width if width and width > 0 else 80

    # Playback loop
    last_print = 0.0
    start_time = time.perf_counter()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:  # if user asked for looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start
                    continue
            
                break


            ascii_frame = frame_to_ascii(
                frame,
                out_width=derived_width,
                char_aspect=char_aspect,
                charset=charset,
                invert=invert,
            )

            if clear_each_frame:
                os.system("cls" if os.name == "nt" else "clear")

            print(ascii_frame)

            frame_count += 1
            last_print = time.perf_counter()
            elapsed = last_print - start_time
            # Sleep to match target FPS
            # Note: Real-time exact sync is hard in terminals; this is good enough
            to_sleep = frame_delay - (time.perf_counter() - last_print)
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()


# --------------------------- CLI ---------------------------

def positive_float(v: str) -> float:
    try:
        x = float(v)
        if x <= 0:
            raise ValueError
        return x
    except ValueError as e:
        raise argparse.ArgumentTypeError("Value must be a positive number") from e


def positive_int(v: str) -> int:
    try:
        x = int(v)
        if x <= 0:
            raise ValueError
        return x
    except ValueError as e:
        raise argparse.ArgumentTypeError("Value must be a positive integer") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Play a video in the terminal using ASCII characters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", type=Path, help="Path to the video file")
    group_size = p.add_mutually_exclusive_group()
    group_size.add_argument("--width", type=positive_int, help="Output character width")
    group_size.add_argument("--fit-terminal", action="store_true", help="Fit width to current terminal columns")
    p.add_argument("--charset", choices=list(CHARSETS.keys()), default="classic", help="Character set for mapping")
    p.add_argument("--invert", action="store_true", help="Invert brightness mapping (dark ↔ bright)")
    p.add_argument("--char-aspect", type=positive_float, default=0.5, help="Character width/height ratio (≈0.5)")
    p.add_argument("--fps", type=positive_float, help="Override FPS (use instead of video FPS)")
    p.add_argument("--fps-cap", type=positive_float, help="Maximum FPS cap")
    p.add_argument("--no-clear", action="store_true", help="Do not clear terminal each frame")
    p.add_argument("--loop", action="store_true", help="Replay the video in a loop until interrupted")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    play_video_ascii(
        video_path=args.video,
        width=args.width,
        fit_terminal=bool(args.fit_terminal),
        fps_override=args.fps,
        fps_cap=args.fps_cap,
        clear_each_frame=not args.no_clear,
        charset=args.charset,
        invert=bool(args.invert),
        char_aspect=args.char_aspect,
        loop=args.loop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
