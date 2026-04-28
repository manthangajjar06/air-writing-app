# 🚀 AI-Powered Air Writing, Virtual Mouse & Speech Recognition

<p align="center"> <b>Touchless Human-Computer Interaction using Computer Vision, ML & NLP</b> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/> <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-ff6f00?style=for-the-badge"/> <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge"/> </p>

---

## 📌 Overview

This project presents a **next-generation Human-Computer Interaction (HCI)** system that eliminates the need for physical input devices.

Using a webcam and microphone, users can:

* ✍️ Write in the air and get AI predictions
* 🖱 Control the system cursor via gestures
* 🎤 Convert speech into text in real-time

The system integrates **Computer Vision + Deep Learning + OS Automation + NLP** into a single interactive pipeline.

---

## 🎯 Key Highlights

* Real-time hand tracking at **30+ FPS**
* CNN-based handwritten character recognition (EMNIST)
* Gesture-driven OS-level mouse control
* Seamless speech-to-text integration
* Fully touchless interaction system

---

## ⚙️ System Architecture

### 🔁 Vision Pipeline

* Webcam input via OpenCV
* Hand landmark detection using MediaPipe (21 keypoints)
* Gesture classification using geometric heuristics
* Action mapping:

  * Drawing → Canvas update
  * Prediction → CNN inference
  * Mouse Mode → OS control

---

### 🔊 Audio Pipeline

* Microphone input capture
* Speech segmentation (pause detection)
* Google Speech API processing
* Real-time UI rendering

---

## 🔄 System Workflow

The system runs two pipelines simultaneously inside a single loop:

### 1. Vision-Control Loop (Real-Time)

1. Frame is captured using OpenCV
2. MediaPipe extracts hand landmarks (21 points)
3. Distances between fingers are calculated
4. Gesture state is determined
5. Based on state:

   * **Draw Mode** → Points added to canvas
   * **Predict Mode** → Canvas processed by ML model
   * **Mouse Mode** → Cursor control using PyAutoGUI

---

### 2. Audio Processing Flow

1. Microphone captures audio
2. SpeechRecognition detects pause
3. Audio sent to Google API
4. Text returned and displayed on UI

---

## 🧠 Core Features

### ✍️ Air Writing & Recognition

* Draw characters using pinch gesture
* Automatic canvas tracking with NumPy
* AI prediction triggered via gesture hold

---

### 🖱 Virtual Mouse

* Cursor movement via hand tracking
* Gesture-based clicks:

  * Single pinch → Left click
  * Double pinch → Right click

---

### 🎤 Speech-to-Text

* Continuous voice capture
* Real-time transcription
* Integrated with web interface

---

## ✋ Gesture Control Map

| Gesture               | Action            |
| --------------------- | ----------------- |
| Pinch (Index + Thumb) | Draw              |
| Index Finger Hold     | Predict Character |
| Two Fingers Up        | Clear Canvas      |
| Thumb Up              | Enable Mouse Mode |
| 1 Pinch               | Left Click        |
| 2 Pinches             | Right Click       |

---

## 🛠 Tech Stack

| Layer           | Technology         |
| --------------- | ------------------ |
| Backend         | Flask              |
| Computer Vision | OpenCV             |
| Hand Tracking   | MediaPipe          |
| ML Model        | TensorFlow / Keras |
| Automation      | PyAutoGUI          |
| Speech          | SpeechRecognition  |
| Data Processing | NumPy              |
| Frontend        | HTML / CSS / JS    |

---

## 📂 Project Structure

```bash
├── app.py  
├── speech-to-text/  
│   └── app.py  
├── models/  
│   ├── air_writing_emnist.keras  
│   └── emnist_labels.npy  
├── static/  
├── templates/  
├── requirements.txt  
└── README.md
```

---

## 🔍 Code Explanation

### 🧩 Main Application (`app.py`)

* Acts as the **central controller**
* Handles:

  * Webcam streaming
  * Hand tracking
  * Gesture detection
  * ML prediction
  * Mouse automation

---

### ✋ Gesture Detection Logic

* Uses landmark indices:

  * Thumb tip → 4
  * Index tip → 8
* Calculates Euclidean distance
* Applies threshold logic to detect gestures

---

### 🧠 ML Prediction Flow

1. Capture drawn canvas
2. Convert to grayscale
3. Resize to 28×28
4. Normalize pixel values
5. Pass into CNN (EMNIST model)
6. Return predicted character

---

### 🖱 Mouse Control Logic

* Map webcam coordinates → screen resolution
* Apply smoothing to reduce jitter
* Use PyAutoGUI:

  * `moveTo()`
  * `click()`

---

### 🎤 Speech Module (`speech-to-text/`)

* Captures microphone input
* Sends audio to Google Speech API
* Returns text asynchronously
* Injects output into frontend

---

## 🚀 Installation & Setup

### 🔧 Prerequisites

* Python 3.8+
* Webcam + Microphone
* Local machine with GUI

---

### ⚡ Setup

```bash
# Clone repository  
git clone https://github.com/YOUR_USERNAME/Air-Writing-Gesture-Mouse.git  
cd Air-Writing-Gesture-Mouse  

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

Access locally:

[http://localhost:5000](http://localhost:5000)

---

## ⚠️ Deployment Limitation

This project **cannot be deployed on cloud platforms**.

### Reason:

* Uses **PyAutoGUI** for system-level interaction
* Requires:

  * Physical display
  * OS-level access
* Cloud environments are **headless (no GUI)**

✔ Works locally
❌ Not compatible with standard hosting

---

## 🧩 Challenges Solved

* Real-time gesture detection without latency
* Mapping 3D landmarks to 2D screen coordinates
* Stable drawing using noisy hand tracking data
* Integrating CV + ML + OS automation in one loop

---

## 📈 Future Improvements

* Multi-character word recognition
* Gesture customization
* Mobile/AR version
* Edge deployment optimization

---

This project is open-source and available under the **MIT License**.
