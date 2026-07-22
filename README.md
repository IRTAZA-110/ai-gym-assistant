# AI Gym Assistant

A real-time computer-vision fitness assistant that uses a webcam, MediaPipe Pose, and OpenCV to count repetitions and provide exercise feedback.

## Features

- Real-time pose landmark detection
- Rep and set tracking
- Configurable workout goals
- Exercise-specific form feedback
- Rest timer between sets
- Desktop interface built with CustomTkinter
- Support for bicep curls, shoulder presses, side raises, overhead claps, and jumping jacks

## Technology

- Python
- OpenCV
- MediaPipe Pose
- NumPy
- CustomTkinter
- Pillow

## Installation

```bash
git clone https://github.com/IRTAZA-110/ai-gym-assistant.git
cd ai-gym-assistant
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Connect a webcam, then run:

```bash
python main.py
```

## How it works

MediaPipe identifies body landmarks in each webcam frame. The application calculates joint angles and landmark distances, then applies exercise-specific state logic to detect completed repetitions and provide immediate feedback.

## Limitations

- Exercise thresholds may require adjustment for different camera angles and body proportions.
- The current implementation is designed for a single person in the frame.
- Good lighting and a clear full-body view improve tracking quality.

## Author

Irtaza Iftikhar Choudry

