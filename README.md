# Focus-Link

A robust, real-time Concentration Tracker built using MediaPipe, YOLOv8, and OpenCV. This system detects face, eyes, and head pose to estimate focus level, while simultaneously monitoring for environmental distractions, making it ideal for online-exam monitoring, driver fatigue detection, or attention-tracking demos.

## Features

* Real-time face & eye detection using MediaPipe Face Mesh
* Eye-closure timer (warns if eyes closed > 3 s)
* Auto-exit when no face detected > 10 s
* Concentration score bar (0-100 %)
* Detects noise and gives warning

⚠️ Setup & Installation

This project requires Python 3.11 for stable compatibility with the mediapipe library, Using newer versions (like 3.12+) may result in installation errors...

### To Quickstart, Tap Run Button

### otherwise, The Steps to RUN this code are as follows :

### 1. Install Python 3.11

Ensure you have Python 3.11 installed on your system.

### 2. Create requirements.txt

Before proceeding, create a file named requirements.txt in your project's root directory and paste the following content:

opencv-python>=4.8.0 
mediapipe==0.10.8 
numpy
ultralytics
sounddevice 

### 3. Create a Virtual Environment

Navigate to this project's directory in your terminal.
Run the command that matches your Python 3.11 installation:

```bash
# On Linux/macOS (if 'python3.11' is available):
python3.11 -m venv .venv

# On Windows (if you used the official installer):
py -3.11 -m venv .venv

# If Python 3.11 is your default 'python3' command:
python3 -m venv .venv
```

### 4. Activate the Environment

You must activate the environment before installing dependencies or running the script.
```bash
Operating System                                Command

(a.) Linux/macOSsource               .venv/bin/activate
(b.) Windows (Command Prompt)        .\.venv\Scripts\activate.bat
(c.) Windows (PowerShell)            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process (first)      \                                    .\.venv\Scripts\Activate.ps1                                    (Second)
```

### 5. Install Dependencies

With your virtual environment active, install the necessary packages:

```bash
pip install -r requirements.txt
```

### 6. Run the Tracker 

Execute the main script. The system will first prompt you for an audio calibration phase (please remain quiet!).

```bash
python FocusLink.py
```
To stop the tracker, press the q key while the video window is focused.