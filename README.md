# ChromaTrace - Real-Time Color + Shape Tracker

ChromaTrace is a real-time computer vision project that detects colored objects, classifies their shapes, and tracks them with motion trails. It is built to be safe for cameras (no device property writes) and runs on a standard webcam.

## Features
- Color segmentation in HSV (red, green, blue, yellow)
- Shape classification (triangle, square, rectangle, circle)
- Lightweight object tracking with stable IDs
- Motion trails and FPS overlay
- Optional video recording

## Quick Start

### 1) Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
python -m src.vision_lab
```

## For Non-Technical Users (Windows)
Download the latest `ChromaTrace.exe` from the GitHub Releases page, unzip it, and double-click to run.

## Build a Windows EXE (for release)
```powershell
.scripts\build_exe.ps1
```
The executable will be created at `dist\ChromaTrace.exe`.

## Controls
- `q` - Quit
- `r` - Toggle recording to `output.mp4`

## Configuration
Use command-line flags to adjust settings:
```bash
python -m src.vision_lab --camera 0 --width 960 --height 540 --min-area 1200 --record --output demo.mp4
```

## Notes
- This project does **not** set camera brightness/contrast/exposure properties.
- If your camera is unavailable on index 0, try `--camera 1` or `--camera 2`.

## Project Structure
```
ChromaTrace/
  src/
    colors.py
    tracker.py
    vision_lab.py
  README.md
  requirements.txt
```

## License
MIT
