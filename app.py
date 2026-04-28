import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
import math
import os
import io
import urllib.request
import speech_recognition as sr

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Air Writing & Speech Recognition",
    page_icon="✍️",
    layout="wide",
)

st.error("""
### ⚠️ Features Disabled on Cloud — Read Before Using

This is a **cloud-hosted** version of the Air Writing app. The following features from the original desktop app **do not work** here and have been removed:

| ❌ Disabled Feature | Why it doesn't work on a server |
|---|---|
| 🖱️ **Cursor / Mouse Control** | `pyautogui` controls the physical mouse of a machine. Servers have no physical screen or mouse. |
| 🎥 **Live continuous video stream** | Flask's MJPEG `/video_feed` route requires a persistent connection. Streamlit captures one snapshot at a time instead. |
| ✍️ **Real-time smooth pinch-drawing** | Catmull-Rom spline smoothing and EMA filtering worked on a 30fps video loop — not possible with frame-by-frame camera snapshots. |
| 🔁 **Auto-predict on pinch release** | The original app watched for gesture *transitions* across frames in real-time. Streamlit re-runs the script per frame, so gesture state is not continuous. |
| 👋 **Two-hand click gestures** | Depended on `pyautogui.click()` — disabled for same reason as mouse control. |

✅ **What still works:** Drawing via webcam frame capture · Character prediction with the trained EMNIST model · Full speech-to-text with multi-language support.
""")

st.title("✍️ Air Writing & 🎤 Speech Recognition")

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
MODEL_PATH      = "air_writing_emnist.keras"
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
PINCH_THRESH    = 0.12
LINE_THICKNESS  = 12
MIN_PIXELS      = 50

# Hand skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

EMNIST_BYCLASS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
]
EMNIST_BALANCED = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# ─────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────
@st.cache_resource
def load_emnist_model():
    if not os.path.exists(MODEL_PATH):
        return None, {}
    mdl  = tf.keras.models.load_model(MODEL_PATH)
    n    = mdl.output_shape[-1]
    lmap = {i: EMNIST_BYCLASS[i] for i in range(62)} if n == 62 \
           else {i: EMNIST_BALANCED[i] for i in range(47)}
    return mdl, lmap

@st.cache_resource
def load_hand_detector():
    """Download the hand-landmarker task model and create a Tasks-API detector."""
    if not os.path.exists(HAND_MODEL_PATH):
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

model, label_map = load_emnist_model()
hand_detector    = load_hand_detector()

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
for key, default in [("canvas", None), ("predicted_text", ""), ("prev_pt", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
def pinch_dist(landmarks):
    ix, th, wr, mid = landmarks[8], landmarks[4], landmarks[0], landmarks[9]
    d = math.hypot(ix.x - th.x, ix.y - th.y)
    s = math.hypot(wr.x - mid.x, wr.y - mid.y)
    return d / (s + 1e-6)

def draw_hand(frame, landmarks, h, w):
    """Draw hand skeleton using OpenCV (no mp.solutions dependency)."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 120), 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 180, 90), 1)

def prepare_for_emnist(canvas_img):
    ys, xs = np.nonzero(canvas_img)
    if len(xs) == 0:
        return None
    crop = canvas_img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w  = crop.shape
    s     = max(h, w)
    ph, pw = (s - h) // 2, (s - w) // 2
    square  = np.pad(crop, ((ph, s-h-ph), (pw, s-w-pw)), 'constant')
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    resized = np.rot90(resized, 3)
    resized = np.fliplr(resized)
    return (resized.astype("float32") / 255.0).reshape(1, 28, 28, 1)

def transcribe_audio(audio_bytes, lang_mode):
    lang_groups = {
        "en-US":    ["en-US"],
        "hi-IN":    ["hi-IN"],
        "gu-IN":    ["gu-IN"],
        "hinglish": ["hi-IN", "en-IN", "en-US"],
        "guj-eng":  ["gu-IN", "en-IN", "en-US"],
        "auto":     ["en-US", "hi-IN", "gu-IN", "en-IN"],
    }
    languages  = lang_groups.get(lang_mode, ["en-US"])
    recognizer = sr.Recognizer()
    wav_io     = io.BytesIO(audio_bytes)
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
    results = []
    for lang in languages:
        try:
            text = recognizer.recognize_google(audio_data, language=lang)
            results.append({"lang": lang, "transcript": text})
        except Exception:
            results.append({"lang": lang, "transcript": None})
    return results

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["✍️ Air Writing", "🎤 Speech to Text"])

# ══════════════════════════════════════════
#  TAB 1 — AIR WRITING
# ══════════════════════════════════════════
with tab1:
    st.subheader("Draw a character using your hand")
    st.info(
        "📷 Click **Take Photo** to capture a frame. "
        "**Pinch** (bring index finger and thumb together) to draw. "
        "Open your hand to stop. Hit **Predict** when done drawing a character."
    )

    col_cam, col_canvas = st.columns([3, 2])

    with col_cam:
        camera_image = st.camera_input("Point your camera at your hand and click Take Photo")

        if camera_image:
            nparr  = np.frombuffer(camera_image.getvalue(), np.uint8)
            frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]

            if st.session_state.canvas is None or st.session_state.canvas.shape != (h, w):
                st.session_state.canvas = np.zeros((h, w), dtype=np.uint8)

            # ── MediaPipe Tasks API detection ──
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = hand_detector.detect(mp_image)

            pinching = False

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]   # list of NormalizedLandmark

                draw_hand(frame, lm, h, w)

                norm_p = pinch_dist(lm)
                ix_x   = int(lm[8].x * w)
                ix_y   = int(lm[8].y * h)
                cursor = (ix_x, ix_y)

                if norm_p < PINCH_THRESH:
                    pinching = True
                    cv2.circle(frame, cursor, 10, (0, 255, 0), -1)
                    cv2.circle(frame, cursor, 15, (255, 255, 255), 2)
                    if st.session_state.prev_pt is not None:
                        cv2.line(st.session_state.canvas,
                                 st.session_state.prev_pt, cursor,
                                 255, LINE_THICKNESS, cv2.LINE_AA)
                    else:
                        cv2.circle(st.session_state.canvas, cursor, LINE_THICKNESS // 2, 255, -1)
                    st.session_state.prev_pt = cursor
                    cv2.putText(frame, "DRAWING", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    st.session_state.prev_pt = None
                    cv2.putText(frame, "OPEN HAND — not drawing", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.putText(frame, f"Pinch: {norm_p:.2f}  (< {PINCH_THRESH} to draw)",
                            (10, h - 15), cv2.FONT_HERSHEY_PLAIN, 1.1,
                            (0, 255, 0) if pinching else (180, 180, 180), 1)
            else:
                st.session_state.prev_pt = None
                cv2.putText(frame, "No hand detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Overlay canvas ink (spring-green tint)
            mask = st.session_state.canvas > 0
            if np.any(mask):
                ink = np.array([0, 255, 127], dtype=np.uint8)
                frame[mask] = cv2.addWeighted(
                    frame[mask], 0.3,
                    np.full_like(frame[mask], ink), 0.7, 0
                )

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption="Frame with hand tracking overlay",
                     use_container_width=True)

    with col_canvas:
        st.subheader("Drawing Canvas")
        canvas_disp = st.session_state.canvas if st.session_state.canvas is not None \
                      else np.zeros((240, 320), dtype=np.uint8)
        st.image(canvas_disp, caption="What the model sees",
                 use_container_width=True, clamp=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Predict Character", use_container_width=True):
                if model is None:
                    st.error("EMNIST model file not found.")
                elif st.session_state.canvas is None or \
                        np.count_nonzero(st.session_state.canvas) < MIN_PIXELS:
                    st.warning("Draw something first — canvas looks empty.")
                else:
                    inp = prepare_for_emnist(st.session_state.canvas)
                    if inp is not None:
                        preds = model.predict(inp, verbose=0)
                        idx   = int(np.argmax(preds))
                        char  = label_map.get(idx, "?")
                        conf  = float(np.max(preds)) * 100
                        st.success(f"Predicted: **`{char}`** ({conf:.1f}% confidence)")
                        st.session_state.predicted_text += char
                        st.session_state.canvas  = np.zeros_like(st.session_state.canvas)
                        st.session_state.prev_pt = None

        with c2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                if st.session_state.canvas is not None:
                    st.session_state.canvas  = np.zeros_like(st.session_state.canvas)
                    st.session_state.prev_pt = None

        st.markdown("---")
        st.markdown("### 📝 Accumulated Text")
        st.markdown(f"## `{st.session_state.predicted_text or '—'}`")
        if st.button("🧹 Clear Text"):
            st.session_state.predicted_text = ""

# ══════════════════════════════════════════
#  TAB 2 — SPEECH TO TEXT
# ══════════════════════════════════════════
with tab2:
    st.subheader("Speech to Text")
    st.info("Upload a **.wav** audio file. Multi-language support: English, Hindi, Gujarati, and mixed modes.")

    lang_options = {
        "English (en-US)":              "en-US",
        "Hindi (hi-IN)":                "hi-IN",
        "Gujarati (gu-IN)":             "gu-IN",
        "Hinglish (Hindi + English)":   "hinglish",
        "Gujarati + English":           "guj-eng",
        "Auto-detect (all languages)":  "auto",
    }
    lang_label = st.selectbox("Language / Mode", list(lang_options.keys()))
    lang_mode  = lang_options[lang_label]

    audio_file = st.file_uploader("Upload WAV audio file", type=["wav"])

    if audio_file:
        st.audio(audio_file, format="audio/wav")
        audio_bytes = audio_file.read()

        if st.button("🎙️ Transcribe", use_container_width=True):
            if len(audio_bytes) < 100:
                st.error("Audio file is too short or empty.")
            else:
                with st.spinner("Sending to Google Speech Recognition…"):
                    try:
                        results    = transcribe_audio(audio_bytes, lang_mode)
                        successful = [r for r in results if r["transcript"]]
                        if successful:
                            best = max(successful, key=lambda r: len(r["transcript"]))
                            st.success(f"**Best result ({best['lang']}):**  {best['transcript']}")
                            if len(results) > 1:
                                with st.expander("All language results"):
                                    for r in results:
                                        icon = "✅" if r["transcript"] else "❌"
                                        st.write(f"{icon} **{r['lang']}**: "
                                                 f"{r['transcript'] or '(could not understand)'}")
                        else:
                            st.error("Could not understand audio in any selected language. "
                                     "Try a cleaner recording or switch language mode.")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
    else:
        st.caption("No file uploaded yet.")
