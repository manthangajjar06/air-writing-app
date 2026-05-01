# ✍️ Air Writer AI — Touchless HCI System

**You don't touch the keyboard. The keyboard watches you.**

A real-time gesture-controlled writing, cursor automation, and speech recognition system powered by Computer Vision, Deep Learning, and NLP.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-0F9D58?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN_Inference-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_Server-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SpeechRecognition](https://img.shields.io/badge/Google_Speech-STT_API-4285F4?style=flat-square&logo=google&logoColor=white)](https://pypi.org/project/SpeechRecognition/)

Air Writer AI is a Flask application that captures webcam frames in real-time, extracts 21 hand landmarks via MediaPipe, classifies gestures through geometric heuristics, and maps them to three distinct interaction modes: air writing with CNN-based character recognition (EMNIST), OS-level cursor control via PyAutoGUI, and multilingual speech-to-text transcription via Google Speech API. Two hands are tracked simultaneously — the right hand controls writing and cursor movement, the left hand triggers click gestures. The entire vision + automation + ML pipeline runs inside a single frame loop at 30+ FPS.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Gesture Detection Engine](#gesture-detection-engine)
- [ML Prediction Pipeline](#ml-prediction-pipeline)
- [Cursor Control System](#cursor-control-system)
- [Speech Recognition Module](#speech-recognition-module)
- [Code Structure](#code-structure)
- [Limitations](#limitations)
- [Challenges Solved](#challenges-solved)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Architecture

```
User's Hand Moves in Front of Webcam
    → OpenCV captures frame, flips horizontally (mirror view)
    → MediaPipe extracts 21 hand landmarks per hand (up to 2 hands)
    → Geometric heuristics classify gesture state
    → Action dispatched based on mode:
        WRITE  → Pinch drawing on canvas → CNN prediction (EMNIST)
        CURSOR → Index finger mapped to screen coordinates → PyAutoGUI
        SPEECH → Browser MediaRecorder → WAV upload → Google Speech API
    → Flask serves MJPEG stream + REST endpoints to browser UI
```

```
┌──────────────────────────────────────────────────────────────────────┐
│  FLASK SERVER (app.py)                                               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  FRAME LOOP (generate_frames)                   30+ FPS       │  │
│  │                                                                │  │
│  │  cv2.VideoCapture ──► cv2.flip (mirror) ──► RGB convert       │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  MediaPipe Hands ──► 21 landmarks × 2 hands                   │  │
│  │       │                                                        │  │
│  │       ├─► Hand Classification (wrist x-position heuristic)    │  │
│  │       │       │                                                │  │
│  │       │       ├─► Right Hand (primary)                        │  │
│  │       │       │     ├─► EMA + Moving Avg smoothing            │  │
│  │       │       │     ├─► Pinch FSM (IDLE→DEBOUNCE→DRAW→END)   │  │
│  │       │       │     ├─► Finger-up detection (5 booleans)      │  │
│  │       │       │     └─► Mode dispatch:                        │  │
│  │       │       │           WRITE  → Canvas draw + CNN predict  │  │
│  │       │       │           CURSOR → PyAutoGUI.moveTo()         │  │
│  │       │       │           THUMB  → Mode switch (3s hold)      │  │
│  │       │       │                                                │  │
│  │       │       └─► Left Hand (secondary, CURSOR mode only)    │  │
│  │       │             ├─► Index only → Right click (0.5s hold) │  │
│  │       │             └─► Index+Middle → Left click (0.5s hold)│  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Canvas overlay ──► MJPEG encode ──► /video_feed endpoint     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  SPEECH PIPELINE                                               │  │
│  │                                                                │  │
│  │  Browser MediaRecorder ──► WAV blob ──► POST /transcribe      │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  ThreadPoolExecutor (up to 4 workers)                         │  │
│  │       │                                                        │  │
│  │       ├─► try_language(en-US)  ──┐                            │  │
│  │       ├─► try_language(hi-IN)  ──┤  Google Speech API         │  │
│  │       ├─► try_language(gu-IN)  ──┤  (parallel requests)       │  │
│  │       └─► try_language(en-IN)  ──┘                            │  │
│  │              │                                                 │  │
│  │              ▼                                                 │  │
│  │  Best result = max(transcript.length) ──► JSON response       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ROUTES:                                                             │
│    GET  /            → index.html (full UI)                          │
│    GET  /video_feed  → MJPEG stream (multipart/x-mixed-replace)     │
│    GET  /status      → { char, timestamp } (latest prediction)       │
│    GET  /mode        → { mode: "WRITE" | "CURSOR" }                  │
│    POST /transcribe  → { results[], best, best_lang }                │
└──────────────────────────────────────────────────────────────────────┘
```

The system is a monolithic Flask application that runs two pipelines inside a single process. The **vision pipeline** captures frames in a generator function (`generate_frames`) that yields MJPEG-encoded frames to the browser via a streaming HTTP response. The **speech pipeline** accepts audio uploads via a REST endpoint and dispatches transcription across multiple languages in parallel using `ThreadPoolExecutor`. State is shared between the frame loop and the API layer through thread-safe globals protected by a `Lock`.

---

## Features

| Category | Description |
|----------|-------------|
| **Air Writing** | Pinch-to-draw on a virtual canvas with Catmull-Rom spline interpolation for smooth strokes. CNN predicts characters from EMNIST (47 or 62 classes). |
| **Virtual Mouse** | Right hand index finger controls OS cursor via PyAutoGUI. Coordinate mapping from webcam active area to full screen resolution with EMA smoothing. |
| **Click Gestures** | Left hand triggers clicks in CURSOR mode. Index-only = right click. Index + middle = left click. 0.5-second hold with visual progress feedback. |
| **Mode Switching** | Thumb-only gesture held for 3 seconds toggles between WRITE and CURSOR modes. Circular progress indicator shown during hold. |
| **Dual-Hand Tracking** | Both hands tracked simultaneously. Hand classification uses wrist x-position heuristic (not MediaPipe's unreliable handedness label). |
| **Spline Smoothing** | Drawing uses a 2-stage smoothing pipeline: EMA filter → moving average. Catmull-Rom splines generate smooth curves through recent draw points. |
| **Speech-to-Text** | Multilingual transcription supporting English, Hindi, Gujarati, Hinglish (code-mixed), and auto-detect. Parallel API calls pick the best result. |
| **Real-Time Streaming** | MJPEG video feed served at 30+ FPS via Flask's streaming response. Canvas ink overlaid on camera feed with alpha blending. |
| **Prediction Polling** | Frontend polls `/status` endpoint to fetch the latest CNN prediction. Thread-safe access via `Lock`. |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Webcam (built-in or external)
- Microphone (for speech-to-text)
- Local machine with a physical display (required for PyAutoGUI)

### Installation

```bash
# Clone repository
git clone https://github.com/manthangajjar06/air-writing-app.git
cd air-writing-app

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Access at: [http://localhost:5000](http://localhost:5000)

### Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Frame capture, image processing, MJPEG encoding |
| `numpy` | Canvas manipulation, array operations, image transforms |
| `mediapipe` | 21-keypoint hand landmark detection |
| `tensorflow` | CNN model loading and inference (EMNIST) |
| `pyautogui` | OS-level cursor movement and click automation |
| `flask` | Web server, REST API, MJPEG streaming |
| `SpeechRecognition` | Google Speech API wrapper for audio transcription |

---

## Configuration

All tuning constants are in the `CONFIG` block at the top of `app.py` (lines 22–46):

```python
# Model & Camera
MODEL_PATH          = "air_writing_emnist.keras"   # Trained CNN model path
LABEL_NPY           = "emnist_labels.npy"          # Label mappings
CAMERA_INDEX        = 0                            # Webcam device index
LINE_THICKNESS      = 12                           # Stroke width on canvas
SMOOTH_WINDOW       = 7                            # Moving-average window size
MIN_PIXELS_TO_PREDICT = 50                         # Min ink pixels before prediction

# Pinch Gesture Thresholds (normalized to hand size)
PINCH_START_NORM    = 0.14    # Pinch begins when distance < this
PINCH_END_NORM      = 0.20    # Pinch ends when distance > this
START_HOLD_SEC      = 0.08    # Debounce: pinch must hold 80ms to activate
END_HOLD_SEC        = 0.15    # Debounce: release must hold 150ms to deactivate

# Gesture Hold Durations
GESTURE_HOLD_TIME   = 3.0     # Seconds to hold thumb for mode switch
PREDICT_HOLD_TIME   = 1.0     # Seconds to hold index for prediction
CLICK_HOLD_TIME     = 0.5     # Seconds to hold left-hand gesture for click

# Smoothing (EMA)
EMA_POINTER_ALPHA   = 0.22    # Pointer smoothing (lower = smoother, laggier)
EMA_PINCH_ALPHA     = 0.35    # Pinch distance smoothing

# Cursor Mapping
FRAME_REDUCTION     = 100     # Active area margin (virtual mousepad inset)
SMOOTHENING         = 5       # Cursor movement smoothing factor
```

**Hysteresis design:** `PINCH_START_NORM` (0.14) and `PINCH_END_NORM` (0.20) create a dead zone that prevents flickering when the pinch distance hovers near a single threshold. Combined with temporal debouncing (`START_HOLD_SEC`, `END_HOLD_SEC`), this produces a 4-state finite state machine that eliminates false activations.

---

## Usage

1. Open [http://localhost:5000](http://localhost:5000) in your browser.
2. The webcam feed starts automatically with the WRITE mode active.
3. Use gestures to interact (see gesture map below).
4. For speech-to-text, select a language and click the record button in the UI.

### Gesture Control Map

| Gesture | Hand | Mode | Action | Hold Time |
|---------|------|------|--------|-----------|
| Pinch (Index + Thumb) | Right | WRITE | Draw on canvas | Immediate |
| Index Finger Only | Right | WRITE | Trigger CNN prediction | 1.0s |
| Two Fingers (Index + Middle) | Right | WRITE | Clear canvas | 3.0s |
| Thumb Only (all others down) | Right | Both | Toggle WRITE ↔ CURSOR mode | 3.0s |
| Point (Index Finger) | Right | CURSOR | Move OS cursor | Immediate |
| Index Finger Only | Left | CURSOR | Right click | 0.5s |
| Index + Middle | Left | CURSOR | Left click | 0.5s |

---

## Gesture Detection Engine

This is the core interaction layer that makes touchless control reliable.

### Problem

Raw hand tracking data from MediaPipe is noisy. Pinch distance fluctuates frame-to-frame, causing false draw activations. MediaPipe's built-in handedness labels (`Left`/`Right`) are unreliable on horizontally flipped frames. Naive threshold-based gesture detection produces constant flickering.

### Solution: 4-State Pinch FSM with Hysteresis

The pinch lifecycle is modeled as a finite state machine with 4 states:

```
IDLE ──► DEBOUNCE_START ──► DRAWING ──► DEBOUNCE_END ──► IDLE
  ▲           │                              │
  └───────────┘ (pinch released too early)   │
  ▲                                          │
  └──────────────────────────────────────────┘ (release confirmed)
```

| State | Entry Condition | Exit Condition | Purpose |
|-------|----------------|----------------|---------|
| `PINCH_IDLE` | Default / release confirmed | `norm_pin ≤ 0.14` | Waiting for pinch |
| `PINCH_DEBOUNCE_START` | Distance drops below start threshold | Hold for 80ms OR distance bounces back | Filters accidental touches |
| `PINCH_DRAWING` | Debounce timer satisfied | `norm_pin > 0.20` | Active drawing state |
| `PINCH_DEBOUNCE_END` | Distance exceeds end threshold | Hold for 150ms OR distance drops back | Prevents premature stroke termination |

**Normalized distance:** Pinch distance is divided by hand size (`wrist → middle-finger-base`) to make thresholds invariant to the user's distance from the camera.

**EMA smoothing:** Both the pointer position and pinch distance pass through exponential moving average filters before any threshold logic, removing high-frequency noise at the signal level.

### Hand Classification (Dual-Hand)

MediaPipe's `handedness` label is unreliable on mirrored frames. Instead, the system uses a **wrist x-position heuristic**:

- **1 hand detected** → always treated as right (primary) hand
- **2 hands detected** → the hand with the higher wrist x-coordinate in the mirrored frame is classified as the right hand

This is deterministic and frame-rate independent — no ML-based handedness required.

### Finger-Up Detection

Each finger's state (up/down) is determined by comparing landmark y-coordinates:

```python
# Right hand (mirrored frame)
thumb  = lm[4].x < lm[3].x        # Thumb extends sideways (x-axis)
index  = lm[8].y < lm[6].y        # Tip above PIP joint
middle = lm[12].y < lm[10].y
ring   = lm[16].y < lm[14].y
pinky  = lm[20].y < lm[18].y

# Left hand — thumb direction reversed
thumb  = lm[4].x > lm[3].x
```

These 5 booleans are combined to detect compound gestures (thumb-only, index-only, two-fingers, etc.).

---

## ML Prediction Pipeline

### Model

- **Architecture:** CNN trained on EMNIST dataset
- **Input:** 28×28 grayscale image, single channel
- **Output:** 47 classes (EMNIST Balanced) or 62 classes (EMNIST ByClass)
- **Format:** `.keras` (TensorFlow/Keras SavedModel)

The model auto-detects its class count at load time and selects the appropriate label map:

```python
out_classes = model.output_shape[-1]
if out_classes == 62:
    label_map = EMNIST_BYCLASS_CHARS   # 0-9, A-Z, a-z (62 classes)
else:
    label_map = EMNIST_BALANCED_CHARS   # 0-9, A-Z, subset of a-z (47 classes)
```

### Canvas-to-Prediction Flow

```
Canvas (full frame resolution, binary)
    → Crop to bounding box of non-zero pixels (np.nonzero)
    → Pad to square with zero-padding
    → Resize to 28×28 (cv2.INTER_AREA for anti-aliasing)
    → Rotate 90° × 3 (np.rot90) + horizontal flip (np.fliplr)
    → Normalize to [0, 1] float32
    → Reshape to (1, 28, 28, 1)
    → model.predict() → argmax → label_map lookup
    → Canvas cleared after prediction
```

**Why rotate + flip?** EMNIST stores images transposed relative to standard orientation. The `rot90(3) + fliplr` combination maps the user's natural writing orientation to EMNIST's expected input format.

### Drawing Quality: Catmull-Rom Splines

Raw draw points from hand tracking are spaced irregularly and produce jagged strokes. The system uses **Catmull-Rom spline interpolation** when 4+ recent points are available:

```python
def catmull_rom_spline(p0, p1, p2, p3, steps=8):
    # Generates smooth curve through p1→p2 using p0, p3 as control points
    # 8 interpolated points per segment
```

When fewer than 4 points exist, it falls back to **linear interpolation** with adaptive step count based on distance. Additionally, a filled circle is drawn at every cursor position to eliminate gaps in fast strokes.

---

## Cursor Control System

### Coordinate Mapping

The webcam frame is divided into an **active area** (inset by `FRAME_REDUCTION = 100px` on each side). Only finger positions within this rectangle are mapped to screen coordinates:

```python
# Map camera active area → full screen resolution
x_screen = np.interp(x_cam, (100, frame_w - 100), (0, SCREEN_W))
y_screen = np.interp(y_cam, (100, frame_h - 100), (0, SCREEN_H))
```

This creates a "virtual mousepad" effect — the user doesn't need to reach the extreme edges of the camera frame to reach screen corners.

### Smoothing

Cursor movement uses a simple **exponential smoothing** formula:

```python
clocX = plocX + (x_screen - plocX) / SMOOTHENING   # SMOOTHENING = 5
clocY = plocY + (y_screen - plocY) / SMOOTHENING
pyautogui.moveTo(clocX, clocY)
```

Higher `SMOOTHENING` values produce smoother but laggier cursor movement. The value of 5 balances responsiveness with stability.

### Click Gestures (Left Hand)

In CURSOR mode, the left hand controls clicks via hold-to-confirm gestures:

| Left Hand Gesture | Click Type | Hold Time |
|-------------------|------------|-----------|
| Index finger only | Right click | 0.5s |
| Index + Middle fingers | Left click | 0.5s |

Each gesture shows a **circular progress indicator** on the left hand's index fingertip. The click fires exactly once per hold cycle (`left_gesture_fired` flag prevents repeated triggers). A green flash confirms the click visually.

---

## Speech Recognition Module

### Architecture

The speech pipeline is a REST endpoint (`POST /transcribe`) that accepts WAV audio uploads from the browser's `MediaRecorder` API and dispatches them to Google Speech API.

### Multilingual Support

| Language Mode | API Calls | Languages |
|---------------|-----------|-----------|
| `en-US` | 1 | English (US) |
| `hi-IN` | 1 | Hindi |
| `gu-IN` | 1 | Gujarati |
| `hinglish` | 3 (parallel) | Hindi + English (India) + English (US) |
| `guj-eng` | 3 (parallel) | Gujarati + English (India) + English (US) |
| `auto` | 4 (parallel) | All supported languages |

For multi-language modes, a `ThreadPoolExecutor` with 4 workers sends parallel requests. The **longest successful transcript** is selected as the best result — longer transcripts typically indicate the correct language match.

### Standalone Module

The `speech-to-text/` directory contains a standalone version of the speech module with its own Flask app and UI template. This was developed independently and later integrated into the main `app.py`.

---

## Code Structure

```
app.py (~600 lines)
│
├─ CONFIG ─────────────── Model paths, camera index, gesture thresholds,
│                          smoothing constants, timing parameters
│
├─ GLOBALS ────────────── Canvas (numpy array), prediction state (dict + Lock),
│                          mode ("WRITE"/"CURSOR"), screen dimensions
│
├─ RESOURCE LOADING ───── TensorFlow model + EMNIST label map (auto-detect 47/62),
│                          MediaPipe Hands (2 hands, 0.7 confidence)
│
├─ Section 1: Math & Interpolation
│   ├─ euclid()                    Euclidean distance between two points
│   ├─ catmull_rom_spline()        Smooth curve through 4 control points (8 steps)
│   ├─ interpolate_points()        Linear interpolation fallback
│   ├─ normalized_pinch_distance() Pinch distance / hand size (camera-invariant)
│   ├─ check_fingers_up()          5-boolean finger state (right hand, mirrored)
│   └─ check_fingers_up_left()     5-boolean finger state (left hand, mirrored)
│
├─ Section 2: Image Processing
│   └─ prepare_for_emnist()        Crop → pad → resize → rotate → normalize → reshape
│
├─ Section 3: Frame Loop (generate_frames)
│   ├─ Hand classification         Wrist x-position heuristic (1 or 2 hands)
│   ├─ Right hand processing       EMA + moving avg → pinch FSM → mode dispatch
│   │   ├─ Mode switch             Thumb-only 3s hold → toggle WRITE/CURSOR
│   │   ├─ WRITE mode              Pinch FSM → spline draw → gesture detect
│   │   │   ├─ Drawing             Catmull-Rom / linear interpolation on canvas
│   │   │   ├─ Predict gesture     Index-only 1s hold → CNN inference → clear canvas
│   │   │   └─ Clear gesture       Two-fingers 3s hold → canvas reset
│   │   └─ CURSOR mode             Coordinate mapping → EMA smoothing → moveTo()
│   ├─ Left hand processing        Click gestures (CURSOR mode only)
│   │   ├─ Index only              Right click (0.5s hold)
│   │   └─ Index + Middle          Left click (0.5s hold)
│   └─ Frame encoding              Canvas overlay → MJPEG yield
│
├─ Section 4: Speech Recognition
│   └─ try_language()              Single-language transcription via Google Speech API
│
└─ Section 5: Flask Routes
    ├─ GET  /                      Serve index.html
    ├─ GET  /video_feed            MJPEG streaming response
    ├─ GET  /status                Latest prediction JSON (thread-safe)
    ├─ GET  /mode                  Current mode JSON
    └─ POST /transcribe            Audio upload → parallel transcription → best result
```

---

## Project Structure

```
air-writing-app/
│
├── app.py                          # Monolithic server: Flask + CV + ML + Mouse + Speech
│
├── air_writing_emnist.keras        # Trained CNN model (EMNIST Balanced or ByClass)
├── emnist_labels.npy               # NumPy label array (legacy, auto-detected from model)
├── predictions_log.csv             # Historical prediction log
│
├── templates/
│   └── index.html                  # Main UI: video feed, gesture guide, speech recorder
│
├── speech-to-text/                 # Standalone speech module (developed independently)
│   ├── app.py                      # Isolated Flask app for speech-to-text
│   └── templates/
│       └── index.html              # Standalone speech UI
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Limitations

| Constraint | Details |
|------------|---------|
| **Local only** | Cannot be deployed on cloud platforms. PyAutoGUI requires a physical display and OS-level access. Cloud environments are headless (no GUI). |
| **Single user** | The frame loop runs in the Flask process. Only one client can receive the MJPEG stream at a time. |
| **Webcam required** | No file upload or pre-recorded video mode. Requires a live camera feed. |
| **Right-hand bias** | Single-hand mode always assumes right hand. Left-handed users must use two hands for full functionality. |
| **EMNIST only** | Character recognition limited to single characters (A-Z, a-z, 0-9). No word-level or sentence-level recognition. |
| **Speech API** | Requires internet access for Google Speech API. No offline transcription. |
| **Browser** | Tested on Chrome. Speech recording uses `MediaRecorder` API which may behave differently across browsers. |

---

## Challenges Solved

| Challenge | Solution |
|-----------|----------|
| **Pinch flickering** | 4-state FSM with hysteresis thresholds (0.14 start / 0.20 end) + temporal debouncing (80ms / 150ms) |
| **Jittery cursor** | 2-stage smoothing: EMA filter (α=0.22) → 7-point moving average. Separate EMA on pinch distance (α=0.35) |
| **Jagged strokes** | Catmull-Rom spline interpolation through 4 recent points. Linear interpolation fallback. Gap-filling circles at every cursor position |
| **Hand misclassification** | Wrist x-position heuristic instead of MediaPipe's unreliable handedness label |
| **EMNIST orientation** | `rot90(3) + fliplr` transforms natural writing orientation to EMNIST's transposed format |
| **Coordinate mapping** | Virtual mousepad (100px inset) mapped to full screen resolution via `np.interp`. Prevents edge-reaching |
| **Accidental clicks** | 0.5-second hold-to-confirm with single-fire flag. Visual progress circle provides feedback |
| **Code-mixed speech** | Parallel transcription across multiple language codes. Longest transcript wins |

---

## Future Improvements

- Multi-character word recognition (sequence models, CTC decoding)
- Custom gesture mapping (user-defined gesture → action bindings)
- Mobile / AR version (WebRTC-based, no PyAutoGUI dependency)
- Edge deployment optimization (TFLite, ONNX Runtime)
- Left-hand primary mode for left-handed users
- Multi-client streaming (WebSocket-based frame distribution)

---

## Disclaimer

This project is built for **educational and research purposes only**. It is a technical demonstration of real-time computer vision, deep learning inference, OS-level automation, and speech recognition integration. The author assumes no responsibility for how it is used.

---

<div align="center">

*From raw landmarks to smooth splines — every frame is a gesture waiting to be understood.*

</div>
