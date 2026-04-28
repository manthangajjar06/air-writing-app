# ✍️ Air Writing & Speech Recognition

A web app that uses **MediaPipe hand tracking** + a trained **EMNIST CNN model** to recognize characters drawn in the air via webcam, plus **multi-language speech-to-text**.

## 🚀 Live Demo
> Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## ⚠️ Cloud Limitations

| ❌ Feature | Reason Disabled |
|---|---|
| 🖱️ Cursor / Mouse Control | `pyautogui` requires a physical screen |
| 🎥 Live video stream | Replaced with per-frame camera snapshots |
| ✍️ Real-time smooth drawing | Continuous frame loop not available on Streamlit |
| 🔁 Auto-predict on gesture | Gesture state not persistent across frames |

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies
├── air_writing_emnist.keras   # Trained character recognition model
├── emnist_labels.npy          # Label mapping
└── README.md
```
