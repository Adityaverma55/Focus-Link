# Concentration-Tracker

A real-time Concentration Tracker built with MediaPipe and OpenCV. It detects face, eyes, and head pose to estimate focus level — useful for online-exam monitoring or attention-tracking demos.

## Features

* Real-time face & eye detection using MediaPipe Face Mesh
* Eye-closure timer (warns if eyes closed > 3 s)
* Auto-exit when no face detected > 10 s
* Concentration score bar (0-100 %)
* Detects noise and gives warning

## ⚠️ Setup & Installation

This project requires Python 3.11.

> **Note:** The mediapipe library does not yet support newer Python versions (like 3.12+). To run this project, you must use a virtual environment based on Python 3.11.

### Quickstart

#### 1. Install Python 3.11

Ensure you have Python 3.11 installed on your system.

#### 2. Create a Virtual Environment

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

#### 3. Activate the Environment

Run the command for your operating system:

```bash
# On Linux/macOS:
source .venv/bin/activate

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

#### 4. Install Dependencies

With your virtual environment active, install the packages:

```bash
pip install -r requirements.txt
```

#### 5. Run the Tracker

```bash
python ml.py
```

To stop, press `q`