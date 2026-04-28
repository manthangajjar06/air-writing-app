# Air Writing & Speech Recognition

A real-time computer vision web app that recognizes hand-drawn characters in the air using a webcam, powered by a trained EMNIST CNN model, MediaPipe hand tracking, and multi-language speech-to-text via Google Speech Recognition.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Cloud Deployment](#cloud-deployment)
- [Known Limitations (Cloud)](#known-limitations-cloud)
- [How It Works](#how-it-works)
- [Model Details](#model-details)

---

## Overview

This project started as a local desktop Flask application with real-time MJPEG video streaming, pinch-gesture drawing, mouse cursor control via `pyautogui`, and a speech recognition module. It has been adapted into a Streamlit web app for cloud deployment, with the desktop-only features clearly documented and disabled.

The core idea: use your index finger and thumb as a pen. Pinch to draw a character in the air in front of your webcam. The model predicts the character. Repeat to build words.

---

## Features

**Air Writing**
- Webcam-based hand detection using MediaPipe
- Pinch gesture (index + thumb) to draw on a virtual canvas
- Character prediction using a trained EMNIST CNN model
- Accumulated text output — predict letter by letter to form words

**Speech to Text**
- Upload a `.wav` audio file for transcription
- Powered by Google Speech Recognition
- Supports 6 language modes: English, Hindi, Gujarati, Hinglish, Gujarati-English mix, and auto-detect

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit |
| Hand Tracking | MediaPipe Hands |
| Computer Vision | OpenCV (headless) |
| Deep Learning | TensorFlow / Keras |
| Speech Recognition | SpeechRecognition (Google API) |
| Numerical Computing | NumPy |

---

## Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── air_writing_emnist.keras    # Trained EMNIST character recognition model
├── emnist_labels.npy           # Label index-to-character mapping
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A webcam
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/air-writing-app.git
cd air-writing-app

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Local tip:** If you're running locally and want full OpenCV support, replace `opencv-python-headless` with `opencv-python` in `requirements.txt`.

---

## Cloud Deployment

This app is configured for one-click deployment on [Streamlit Community Cloud](https://share.streamlit.io).

1. Push this repository to GitHub (all 6 files, nothing else)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** and select this repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

Streamlit will install dependencies from `requirements.txt` automatically. First build takes 2–3 minutes.

---

## Known Limitations (Cloud)

The original desktop version had several features that are fundamentally incompatible with any cloud server environment. These are disabled in this version and will not be re-added.

| Feature | Status | Reason |
|---|---|---|
| 🖱️ Cursor / mouse control | ❌ Removed | `pyautogui` controls a physical machine's mouse and screen. Cloud servers have neither. |
| 🎥 Live continuous video stream | ❌ Not possible | The original Flask app used MJPEG streaming over a persistent HTTP connection. Streamlit's execution model re-runs the script per interaction — there is no persistent frame loop. |
| ✍️ Real-time smooth stroke rendering | ⚠️ Degraded | Catmull-Rom spline smoothing and EMA filtering operated on 30fps continuous frames. Per-snapshot capture means strokes are drawn point-to-point between captures, not in real time. |
| 🔁 Auto-predict on pinch release | ⚠️ Changed | The original app detected the pinch-open transition across frames. Here, prediction is triggered manually via button click. |
| 👋 Two-hand click gestures | ❌ Removed | Depended entirely on `pyautogui.click()`. |

**If you need the full real-time experience**, run the app locally. All gesture tracking, smooth drawing, and cursor control work as intended on a local machine.

---

## How It Works

**Drawing Pipeline**

1. `st.camera_input()` captures a JPEG snapshot from the browser webcam
2. The frame is decoded with OpenCV and flipped horizontally (mirror view)
3. MediaPipe Hands detects landmarks for up to 1 hand
4. Normalized pinch distance between index tip (landmark 8) and thumb tip (landmark 4) is computed relative to hand size
5. If pinch distance < threshold (0.12), the index fingertip position is recorded and a stroke is drawn onto an in-memory canvas (NumPy array)
6. Canvas state is persisted across frames using `st.session_state`

**Prediction Pipeline**

1. The canvas is cropped to the bounding box of the drawn pixels
2. Padded to a square, resized to 28×28
3. Rotated 90° and mirrored to match EMNIST's orientation convention
4. Normalised to [0, 1] and passed to the Keras model
5. `argmax` of the softmax output is mapped to a character via `label_map`

**Speech Pipeline**

1. User uploads a `.wav` file
2. `speech_recognition.AudioFile` reads it into an `AudioData` object
3. `recognize_google()` is called for each language in the selected mode
4. The longest successful transcript is returned as the best result

---

## Model Details

- **Architecture:** Convolutional Neural Network trained on EMNIST
- **Input:** 28×28 grayscale image, single channel
- **Output:** 47 classes (EMNIST Balanced) or 62 classes (EMNIST ByClass), auto-detected at load time
- **File:** `air_writing_emnist.keras` (~12 MB)
- **Label mapping:** `emnist_labels.npy` — index-to-character dictionary loaded at startup

The model was trained on the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), which extends MNIST to handwritten letters and digits.
