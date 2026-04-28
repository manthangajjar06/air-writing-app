import os
import time
import json
import math
from threading import Lock
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
import speech_recognition as sr
import io
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIG =================
MODEL_PATH = "air_writing_emnist.keras"
LABEL_NPY = "emnist_labels.npy"
CAMERA_INDEX = 0
LINE_THICKNESS = 12
SMOOTH_WINDOW = 7
MIN_PIXELS_TO_PREDICT = 50

# Pinch & Gesture Tuning
PINCH_START_NORM = 0.14
PINCH_END_NORM   = 0.20
START_HOLD_SEC   = 0.08
END_HOLD_SEC     = 0.15
GESTURE_HOLD_TIME = 3.0
PREDICT_HOLD_TIME = 1.0
AUTO_PREDICT_COOLDOWN = 2.0
MIN_MOVE_PIXELS = 5
CLICK_HOLD_TIME = 0.5       # Seconds to hold left-hand gesture before click fires

# Smoothing (EMA)
EMA_POINTER_ALPHA = 0.22   # Lower = smoother but laggier (0.15-0.35)
EMA_PINCH_ALPHA   = 0.35   # Smooths pinch distance to prevent flicker

# Cursor / Mouse Tuning (The "Feasible" Logic)
FRAME_REDUCTION = 100       # "Active Area" Margin (Virtual Mousepad size)
SMOOTHENING = 5             # Higher = Smoother cursor, but more lag (5-7 is good)

# ================= FLASK APP =================
app = Flask(__name__)

# Global Variables
canvas = None
latest_prediction = {"char": "", "timestamp": 0}
model = None
label_map = None
current_mode = "WRITE"  # Modes: "WRITE" or "CURSOR"
state_lock = Lock()

# Get Screen Size
try:
    SCREEN_W, SCREEN_H = pyautogui.size()
except:
    SCREEN_W, SCREEN_H = 1920, 1080

# ================= LOAD RESOURCES =================
EMNIST_BALANCED_CHARS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]
EMNIST_BYCLASS_CHARS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
]

if os.path.exists(MODEL_PATH):
    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    out_classes = model.output_shape[-1]
    if out_classes == 62:
        label_map = {i: EMNIST_BYCLASS_CHARS[i] for i in range(62)}
    else:
        label_map = {i: EMNIST_BALANCED_CHARS[i] for i in range(47)}
else:
    print("WARNING: Model not found.")
    label_map = {}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Visual Styles
mp_style_small_lm = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
mp_style_small_con = mp_draw.DrawingSpec(color=(0, 200, 0), thickness=1)

# ================= HELPER FUNCTIONS =================
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def catmull_rom_spline(p0, p1, p2, p3, steps=8):
    """Generate smooth curve points using Catmull-Rom spline through p1→p2.
    p0 and p3 are control points for curvature continuity."""
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * ((2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
        points.append((int(x), int(y)))
    return points

def interpolate_points(p1, p2, steps=4):
    """Fallback linear interpolation between two points."""
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        x = int(p1[0] + (p2[0] - p1[0]) * t)
        y = int(p1[1] + (p2[1] - p1[1]) * t)
        points.append((x, y))
    return points

def normalized_pinch_distance(landmarks):
    ix, th, wr, mid = landmarks[8], landmarks[4], landmarks[0], landmarks[9]
    pinch_d = math.hypot(ix.x - th.x, ix.y - th.y)
    hand_size = math.hypot(wr.x - mid.x, wr.y - mid.y)
    return pinch_d / (hand_size + 1e-6)

def check_fingers_up(lm):
    # Thumb: extends sideways, use x-axis (works for right hand in mirrored view)
    thumb = lm[4].x < lm[3].x
    return [thumb, lm[8].y < lm[6].y, lm[12].y < lm[10].y, lm[16].y < lm[14].y, lm[20].y < lm[18].y]

def check_fingers_up_left(lm):
    """Check fingers for the LEFT hand in a flipped (mirrored) frame.
    Thumb direction is reversed compared to right hand."""
    thumb = lm[4].x > lm[3].x  # Reversed for left hand in mirrored view
    return [thumb, lm[8].y < lm[6].y, lm[12].y < lm[10].y, lm[16].y < lm[14].y, lm[20].y < lm[18].y]

def prepare_for_emnist(canvas_img):
    ys, xs = np.nonzero(canvas_img)
    if len(xs) == 0: return None
    crop = canvas_img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    s = max(h, w)
    pad_h, pad_w = (s-h)//2, (s-w)//2
    square = np.pad(crop, ((pad_h, s-h-pad_h), (pad_w, s-w-pad_w)), 'constant')
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    resized = np.rot90(resized, 3)
    resized = np.fliplr(resized)
    return (resized.astype("float32") / 255.0).reshape(1, 28, 28, 1)

# ================= VIDEO GENERATOR =================
def generate_frames():
    global canvas, latest_prediction, current_mode
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    pts = deque(maxlen=SMOOTH_WINDOW)  # Moving-average deque for secondary smoothing
    
    # Constants
    PINCH_IDLE, PINCH_DEBOUNCE_START, PINCH_DRAWING, PINCH_DEBOUNCE_END = 0, 1, 2, 3
    
    # State Variables
    pinch_state = PINCH_IDLE
    pinch_state_changed_at = 0.0
    last_draw_point = None
    gesture_start_time = 0.0
    last_prediction_time = 0.0
    active_gesture = None
    mode_switch_cooldown = 0.0
    
    # EMA Smoothing State
    ema_x, ema_y = 0.0, 0.0
    ema_pinch = 0.5
    ema_initialized = False
    draw_history = deque(maxlen=8)  # Recent draw points for Catmull-Rom spline
    
    # Cursor Mode — Mouse Smoothing State
    plocX, plocY = 0, 0  # Previous Location (for smoothing)
    clocX, clocY = 0, 0  # Current Location
    
    # Cursor Mode — Left Hand Click Gesture State
    left_gesture = None          # "RIGHT_CLICK" or "LEFT_CLICK" or None
    left_gesture_start = 0.0     # When the current gesture started
    left_gesture_fired = False   # Single-fire flag (prevents repeated clicks)
    
    # Init Canvas
    _, frame = cap.read()
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    
    pyautogui.FAILSAFE = False

    while True:
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        status_color = (0, 255, 0) if current_mode == "WRITE" else (255, 0, 255)
        status_text = f"MODE: {current_mode}"
        if result.multi_hand_landmarks:
            # ---- Classify Hands ----
            # We avoid MediaPipe's handedness label (unreliable on flipped frames).
            # Instead: 1 hand = always primary (right), 2 hands = wrist x-position.
            # In mirrored view, the right hand's wrist has a HIGHER x value.
            right_hand_lm = None
            left_hand_lm = None

            # Draw landmarks for all detected hands
            for hlm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

            if len(result.multi_hand_landmarks) == 1:
                # Single hand — always treat as right (primary) hand
                right_hand_lm = result.multi_hand_landmarks[0]
            else:
                # Two hands — the one with higher wrist x = right hand (mirrored view)
                h0 = result.multi_hand_landmarks[0]
                h1 = result.multi_hand_landmarks[1]
                if h0.landmark[0].x > h1.landmark[0].x:
                    right_hand_lm, left_hand_lm = h0, h1
                else:
                    right_hand_lm, left_hand_lm = h1, h0

            # ---- Process Right Hand (Primary: cursor, writing, mode switch) ----
            hand = right_hand_lm
            if hand:
                lm = hand.landmark
                ix = lm[8]  # Index finger tip
                raw_x, raw_y = ix.x * frame_w, ix.y * frame_h

                # Stage 1: EMA Smoothed Pointer (removes high-frequency noise)
                if not ema_initialized:
                    ema_x, ema_y = raw_x, raw_y
                    ema_initialized = True
                else:
                    ema_x = EMA_POINTER_ALPHA * raw_x + (1 - EMA_POINTER_ALPHA) * ema_x
                    ema_y = EMA_POINTER_ALPHA * raw_y + (1 - EMA_POINTER_ALPHA) * ema_y

                # Stage 2: Moving-average on top of EMA (hybrid smoothing)
                pts.append((int(ema_x), int(ema_y)))
                avg_x = int(sum(p[0] for p in pts) / len(pts))
                avg_y = int(sum(p[1] for p in pts) / len(pts))
                cursor_pt = (avg_x, avg_y)

                # EMA Smoothed Pinch Distance (prevents flicker)
                raw_pinch = normalized_pinch_distance(lm)
                ema_pinch = EMA_PINCH_ALPHA * raw_pinch + (1 - EMA_PINCH_ALPHA) * ema_pinch
                norm_pin = ema_pinch

                fingers = check_fingers_up(lm)
                now = time.time()
                is_pinching = False

                # --- 1. MODE SWITCHING (Thumb Only, Right Hand Only) ---
                thumb_only = fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
                if thumb_only:
                    if active_gesture != "THUMB":
                        gesture_start_time = now
                        active_gesture = "THUMB"
                    duration = now - gesture_start_time
                    progress = min(duration / GESTURE_HOLD_TIME, 1.0)

                    cv2.ellipse(frame, cursor_pt, (30, 30), 0, 0, 360 * progress, (255, 255, 0), 4)
                    cv2.putText(frame, "SWITCHING...", (cursor_pt[0]-40, cursor_pt[1]-40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

                    if duration >= GESTURE_HOLD_TIME and now > mode_switch_cooldown:
                        current_mode = "CURSOR" if current_mode == "WRITE" else "WRITE"
                        gesture_start_time = 0
                        active_gesture = None
                        mode_switch_cooldown = now + 1.0
                else:
                    if active_gesture == "THUMB":
                        gesture_start_time = 0
                        active_gesture = None

                # --- 2. WRITE MODE LOGIC (unchanged) ---
                if current_mode == "WRITE":
                    if pinch_state == PINCH_IDLE:
                        if norm_pin <= PINCH_START_NORM:
                            pinch_state = PINCH_DEBOUNCE_START
                            pinch_state_changed_at = now
                    elif pinch_state == PINCH_DEBOUNCE_START:
                        if norm_pin > PINCH_START_NORM: pinch_state = PINCH_IDLE
                        elif now - pinch_state_changed_at >= START_HOLD_SEC:
                            pinch_state = PINCH_DRAWING
                            pinch_state_changed_at = now
                            last_draw_point = cursor_pt
                    elif pinch_state == PINCH_DRAWING:
                        is_pinching = True
                        # Visual feedback: filled dot + ring at cursor
                        cv2.circle(frame, cursor_pt, 8, (0, 255, 0), -1)
                        cv2.circle(frame, cursor_pt, 12, (255, 255, 255), 2)
                        # Draw a filled circle on canvas at cursor to fill gaps
                        cv2.circle(canvas, cursor_pt, LINE_THICKNESS // 2, 255, -1)
                        if norm_pin > PINCH_END_NORM:
                            pinch_state = PINCH_DEBOUNCE_END
                            pinch_state_changed_at = now
                            draw_history.clear()
                        else:
                            if last_draw_point is None:
                                last_draw_point = cursor_pt
                                draw_history.clear()
                            draw_history.append(cursor_pt)
                            dist = euclid(last_draw_point, cursor_pt)
                            if dist >= MIN_MOVE_PIXELS:
                                # Use Catmull-Rom spline when we have enough history
                                if len(draw_history) >= 4:
                                    p0 = draw_history[-4]
                                    p1 = draw_history[-3]
                                    p2 = draw_history[-2]
                                    p3 = draw_history[-1]
                                    spline_pts = catmull_rom_spline(p0, p1, p2, p3, steps=8)
                                    all_pts = [last_draw_point] + spline_pts
                                    for i in range(len(all_pts) - 1):
                                        cv2.line(canvas, all_pts[i], all_pts[i+1], 255, LINE_THICKNESS, cv2.LINE_AA)
                                else:
                                    # Fallback: linear interpolation with gap fill
                                    steps = max(2, int(dist / 4))
                                    interp = interpolate_points(last_draw_point, cursor_pt, steps)
                                    all_pts = [last_draw_point] + interp
                                    for i in range(len(all_pts) - 1):
                                        cv2.line(canvas, all_pts[i], all_pts[i+1], 255, LINE_THICKNESS, cv2.LINE_AA)
                                last_draw_point = cursor_pt
                    elif pinch_state == PINCH_DEBOUNCE_END:
                        if norm_pin <= PINCH_END_NORM: pinch_state = PINCH_DRAWING
                        elif now - pinch_state_changed_at >= END_HOLD_SEC:
                            pinch_state = PINCH_IDLE
                            last_draw_point = None

                    # Gestures (Index=Predict, Two=Clear)
                    if not is_pinching and not thumb_only:
                        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                            if active_gesture != "PREDICT":
                                gesture_start_time = now
                                active_gesture = "PREDICT"
                            duration = now - gesture_start_time
                            progress = min(duration / PREDICT_HOLD_TIME, 1.0)
                            cv2.ellipse(frame, cursor_pt, (20, 20), 0, 0, 360 * progress, (255, 100, 0), 4)
                            if duration >= PREDICT_HOLD_TIME:
                                if np.count_nonzero(canvas) > MIN_PIXELS_TO_PREDICT and model:
                                    inp = prepare_for_emnist(canvas)
                                    if inp is not None:
                                        preds = model.predict(inp, verbose=0)
                                        idx = np.argmax(preds)
                                        char = label_map.get(idx, "?")
                                        with state_lock:
                                            latest_prediction = {"char": char, "timestamp": time.time()}
                                        canvas[:] = 0
                                gesture_start_time = 0
                                active_gesture = None
                        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                            if active_gesture != "CLEAR":
                                gesture_start_time = now
                                active_gesture = "CLEAR"
                            duration = now - gesture_start_time
                            progress = min(duration / GESTURE_HOLD_TIME, 1.0)
                            cv2.ellipse(frame, cursor_pt, (20, 20), 0, 0, 360 * progress, (0, 0, 255), 4)
                            if duration >= GESTURE_HOLD_TIME:
                                canvas[:] = 0
                                gesture_start_time = 0
                                active_gesture = None
                        else:
                            gesture_start_time = 0
                            active_gesture = None

                # --- 3. CURSOR MODE — Right Hand Moves Cursor ---
                elif current_mode == "CURSOR":
                    # Draw Active Area Box (Virtual Mousepad)
                    cv2.rectangle(frame, (FRAME_REDUCTION, FRAME_REDUCTION), (frame_w - FRAME_REDUCTION, frame_h - FRAME_REDUCTION), (255, 0, 255), 2)

                    # Get raw camera coordinates for index finger
                    x1, y1 = ix.x * frame_w, ix.y * frame_h

                    # Map coordinates: Camera Active Area -> Full Screen
                    x3 = np.interp(x1, (FRAME_REDUCTION, frame_w - FRAME_REDUCTION), (0, SCREEN_W))
                    y3 = np.interp(y1, (FRAME_REDUCTION, frame_h - FRAME_REDUCTION), (0, SCREEN_H))

                    # Apply Smoothing (Exponential Moving Average)
                    clocX = plocX + (x3 - plocX) / SMOOTHENING
                    clocY = plocY + (y3 - plocY) / SMOOTHENING

                    # Move mouse cursor
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY

            else:
                # Right hand not visible — reset right-hand state
                pts.clear()
                ema_initialized = False
                if active_gesture == "THUMB":
                    gesture_start_time = 0
                    active_gesture = None
                if pinch_state != PINCH_IDLE:
                    pinch_state = PINCH_IDLE
                    last_draw_point = None

            # ---- Process Left Hand (Click Gestures, Cursor Mode Only) ----
            if left_hand_lm and current_mode == "CURSOR":
                left_lm = left_hand_lm.landmark
                left_fingers = check_fingers_up_left(left_lm)
                now = time.time()

                # Detect gesture: ☝️ Index only = Right Click, ✌️ Index+Middle = Left Click
                index_only = left_fingers[1] and not left_fingers[2] and not left_fingers[3] and not left_fingers[4]
                two_fingers = left_fingers[1] and left_fingers[2] and not left_fingers[3] and not left_fingers[4]

                if index_only:
                    current_left_gesture = "RIGHT_CLICK"
                elif two_fingers:
                    current_left_gesture = "LEFT_CLICK"
                else:
                    current_left_gesture = None

                if current_left_gesture:
                    # Start or continue tracking the gesture hold time
                    if current_left_gesture != left_gesture:
                        left_gesture = current_left_gesture
                        left_gesture_start = now
                        left_gesture_fired = False

                    duration = now - left_gesture_start
                    progress = min(duration / CLICK_HOLD_TIME, 1.0)

                    # Visual feedback: progress circle on left hand index finger
                    left_ix = left_lm[8]
                    left_pt = (int(left_ix.x * frame_w), int(left_ix.y * frame_h))

                    if current_left_gesture == "RIGHT_CLICK":
                        click_color = (0, 0, 255)    # Red in BGR
                        click_label = "RIGHT CLICK"
                    else:
                        click_color = (255, 100, 0)  # Blue in BGR
                        click_label = "LEFT CLICK"

                    cv2.ellipse(frame, left_pt, (25, 25), 0, 0, int(360 * progress), click_color, 3)
                    cv2.putText(frame, click_label, (left_pt[0] - 50, left_pt[1] - 35),
                                cv2.FONT_HERSHEY_PLAIN, 1, click_color, 2)

                    # Fire click ONCE after hold time reaches threshold
                    if not left_gesture_fired and duration >= CLICK_HOLD_TIME:
                        if current_left_gesture == "RIGHT_CLICK":
                            pyautogui.click(button='right')
                            print("Right Click (Left Hand Index)")
                        else:
                            pyautogui.click()
                            print("Left Click (Left Hand Two Fingers)")
                        left_gesture_fired = True
                        # Visual confirmation: green flash
                        cv2.circle(frame, left_pt, 30, (0, 255, 0), -1)
                else:
                    # No valid gesture on left hand — reset
                    left_gesture = None
                    left_gesture_fired = False
            else:
                # Left hand not visible or not in cursor mode — reset
                left_gesture = None
                left_gesture_fired = False

        else:
            # No hands detected at all — reset everything
            pts.clear()
            ema_initialized = False
            gesture_start_time = 0
            active_gesture = None
            if pinch_state != PINCH_IDLE:
                pinch_state = PINCH_IDLE
                last_draw_point = None
            left_gesture = None
            left_gesture_fired = False

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (0,0,0), -1)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Overlay canvas ink on full camera frame
        mask = canvas > 0
        if np.any(mask):
            ink_color = np.array([0, 255, 127], dtype=np.uint8)  # Spring green
            frame[mask] = cv2.addWeighted(frame[mask], 0.3, np.full_like(frame[mask], ink_color), 0.7, 0)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ================= SPEECH RECOGNITION =================
def try_language(audio_bytes, lang_code):
    """Try transcribing with a specific language."""
    try:
        recognizer = sr.Recognizer()
        wav_io = io.BytesIO(audio_bytes)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=lang_code)
        return {"lang": lang_code, "transcript": text, "error": None}
    except sr.UnknownValueError:
        return {"lang": lang_code, "transcript": None, "error": "Could not understand"}
    except sr.RequestError as e:
        return {"lang": lang_code, "transcript": None, "error": str(e)}
    except Exception as e:
        return {"lang": lang_code, "transcript": None, "error": str(e)}

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with state_lock:
        return jsonify(latest_prediction)

@app.route('/mode')
def get_mode():
    return jsonify({"mode": current_mode})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    print("[TRANSCRIBE] Request received")
    if 'audio' not in request.files:
        print("[TRANSCRIBE] Error: No audio file in request")
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    print(f"[TRANSCRIBE] Audio size: {len(audio_bytes)} bytes")
    if len(audio_bytes) < 100:
        print("[TRANSCRIBE] Error: Audio too short")
        return jsonify({"error": "Audio too short"}), 400
    lang_mode = request.form.get('lang_mode', 'en-US')
    print(f"[TRANSCRIBE] Language mode: {lang_mode}")
    lang_groups = {
        "en-US":    ["en-US"],
        "hi-IN":    ["hi-IN"],
        "gu-IN":    ["gu-IN"],
        "hinglish": ["hi-IN", "en-IN", "en-US"],
        "guj-eng":  ["gu-IN", "en-IN", "en-US"],
        "auto":     ["en-US", "hi-IN", "gu-IN", "en-IN"],
    }
    languages = lang_groups.get(lang_mode, ["en-US"])
    try:
        if len(languages) == 1:
            print(f"[TRANSCRIBE] Single language: {languages[0]}")
            result = try_language(audio_bytes, languages[0])
            print(f"[TRANSCRIBE] Result: {result}")
            if result["transcript"]:
                return jsonify({"results": [result], "best": result["transcript"]})
            else:
                return jsonify({"results": [result], "best": "(could not understand audio)"})
        else:
            print(f"[TRANSCRIBE] Multi-language: {languages}")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(try_language, audio_bytes, lang) for lang in languages]
                results = []
                for i, fut in enumerate(futures):
                    try:
                        r = fut.result(timeout=30)
                        results.append(r)
                        print(f"[TRANSCRIBE] Worker {languages[i]}: {r}")
                    except Exception as e:
                        print(f"[TRANSCRIBE] Worker {languages[i]} timed out: {e}")
                        results.append({"lang": languages[i], "transcript": None, "error": f"Timeout ({str(e)})"})
            successful = [r for r in results if r["transcript"]]
            print(f"[TRANSCRIBE] Successful: {len(successful)}/{len(results)}")
            if successful:
                best = max(successful, key=lambda r: len(r["transcript"]))
                return jsonify({"results": results, "best": best["transcript"], "best_lang": best["lang"]})
            else:
                return jsonify({"results": results, "best": "(could not understand audio in any language)"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[TRANSCRIBE] Exception: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)