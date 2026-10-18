
# Driver Drowsiness Detection

A real-time driver monitoring application built with Python, OpenCV and MediaPipe.

## Features

- Real-time webcam monitoring
- Eye Aspect Ratio (EAR) based eye-closure detection
- Head-pose and head-down detection
- PERCLOS-based drowsiness monitoring
- Audio alerts when drowsiness is detected
- Event logging and alert video recording

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pygame
- Pandas
- SciPy
- Matplotlib

## Requirements

- Python compatible with the listed dependencies
- Webcam
- Working audio output

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd driver-drowsiness-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

```bash
python run.py
```

Allow webcam access and audio output when prompted by your operating system.

## Notes

The application uses the MediaPipe Face Landmarker model included in `models/`.

The `log/` and `recordings/` directories are generated at runtime.

This project is intended for demonstration and educational purposes. It should not be relied upon as the sole safety system for driving.
