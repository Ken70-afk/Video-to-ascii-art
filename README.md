# ASCII Video Player

Render any video **in your terminal** as animated ASCII art — cross‑platform, fast, and easy to use.

https://github.com/your-username/ASCII-Video-Player (replace with your repo URL)

## Features
- Cross‑platform (Windows, macOS, Linux)
- Uses the video FPS automatically (with optional override or cap)
- Aspect‑ratio–aware scaling for readable ASCII
- Fit‑to‑terminal mode or fixed width
- Optional inversion and multiple character sets
- Vectorized NumPy mapping for speed
- Clean, typed, and documented Python code

## Requirements
- Python 3.9+ (tested with 3.13)
- Packages: `opencv-python`, `numpy`

Install:
```bash
python -m pip install -r requirements.txt
```

##  Usage

```bash
python ascii_player.py <video_path> [--width N | --fit-terminal] [options]
```

**Common examples**

```bash
# Fit width to your terminal; loop forever
python ascii_player.py "demo.mp4" --fit-terminal --loop

# Fixed character width (e.g., for consistent screenshots)
python ascii_player.py "demo.mp4" --width 100

# Dense charset for more detail; invert for light‑on‑dark schemes
python ascii_player.py "demo.mp4" --fit-terminal --charset dense --invert

# Cap FPS if your terminal can't keep up
python ascii_player.py "demo.mp4" --fit-terminal --fps-cap 24
```

### CLI Options

| Flag | Description |
|---|---|
| `video` | Path to the video file (required) |
| `--width <int>` | Output character width (mutually exclusive with `--fit-terminal`) |
| `--fit-terminal` | Fit width to terminal columns |
| `--charset [classic|dense|blocks]` | Character set for mapping |
| `--invert` | Invert brightness mapping (dark ↔ bright) |
| `--char-aspect <float>` | Character width/height ratio (default `0.5`) |
| `--fps <float>` | Override FPS (use instead of video FPS) |
| `--fps-cap <float>` | Maximum FPS cap |
| `--no-clear` | Do not clear the terminal each frame |
| `--loop` | Replay the video in a loop until interrupted |

> Tip: On Windows PowerShell, wrap paths with spaces in quotes.  
> Stop playback with **Ctrl+C**.

##  Quick test (sample video)
If you need a tiny test video, you can generate one with `ffmpeg`:

```bash
# 2-second color gradient test clip (requires ffmpeg)
ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=24 test.mp4
python ascii_player.py test.mp4 --fit-terminal
```

##  Troubleshooting
- **`ModuleNotFoundError: No module named 'cv2'`** → `python -m pip install opencv-python`
- **Nothing prints / exits immediately** → Ensure the path to the video is correct and that the format is supported by your OpenCV build. Try another small `.mp4`.
- **Too slow** → Lower `--width`, use `--fps-cap 24`, or try `--charset classic`.
- **Weird aspect ratio** → Adjust `--char-aspect` (typical terminals: `0.45`–`0.55`).

##  Project layout

```
.
├── ascii_player.py        # main script (put your file here)
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

##  License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---
