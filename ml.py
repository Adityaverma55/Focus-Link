# 4ml_fixed_timers.py
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque

# ---------------- CONFIG ----------------
CALIB_SECONDS = 5.0
EAR_BLINK_THRESHOLD = 0.18
BLINK_CONSEC_FRAMES = 2
FACE_DET_CONF = 0.45
EYE_VARIANCE_THRESHOLD = 200.0   # occlusion check: variance too low => likely covered
EYE_MEAN_DARK = 45.0             # occlusion check: mean too dark => covered
GAZE_X_DELTA = 0.07              # threshold around calibrated center for LEFT/RIGHT
GAZE_Y_DELTA = 0.06              # threshold around calibrated center for UP/DOWN
SCORE_SMOOTH = 6
NOISE_SENSITIVITY = 2.0          # how much above baseline RMS counts as noise
AUDIO_CALIB_SECONDS = 1.0
AUDIO_SR = 22050
AUDIO_BLOCKSIZE = 1024
EYES_CLOSED_SECONDS = 3.0        # <-- display eyes closed if closed for this long
NO_FACE_SECONDS = 10.0           # <-- exit if no face detected this many seconds
# ----------------------------------------

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

# landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = 468
RIGHT_IRIS = 473

# audio globals (thread-safe)
_audio_rms = 0.0
_audio_lock = threading.Lock()
_audio_stream = None
_audio_baseline = 1e-6

def audio_callback(indata, frames, time_info, status):
    global _audio_rms
    mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata[:,0]
    rms = float(np.sqrt(np.mean(np.square(mono))))
    with _audio_lock:
        _audio_rms = rms

def start_audio():
    global _audio_stream
    try:
        _audio_stream = sd.InputStream(callback=audio_callback,
                                       blocksize=AUDIO_BLOCKSIZE,
                                       samplerate=AUDIO_SR,
                                       channels=1)
        _audio_stream.start()
        return True
    except Exception as e:
        print("Audio stream start failed:", e)
        return False

def calibrate_audio_baseline():
    global _audio_baseline
    try:
        print(f"Calibrating microphone for {AUDIO_CALIB_SECONDS:.1f}s — please be quiet...")
        rec = sd.rec(int(AUDIO_CALIB_SECONDS * AUDIO_SR), samplerate=AUDIO_SR, channels=1, dtype='float64')
        sd.wait()
        mono = rec[:,0]
        _audio_baseline = max(1e-6, float(np.sqrt(np.mean(np.square(mono)))))
        print(f"Audio baseline RMS = {_audio_baseline:.6f}")
    except Exception as e:
        print("Audio calibration failed:", e)
        _audio_baseline = 1e-6

# ---------- Helpers ----------
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    try:
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if C <= 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0

def eye_region_stats(gray, landmarks, eye_indices, w, h, pad=6):
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    x1 = max(min(xs) - pad, 0); x2 = min(max(xs) + pad, w-1)
    y1 = max(min(ys) - pad, 0); y2 = min(max(ys) + pad, h-1)
    if x2 <= x1 or y2 <= y1:
        return None
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(np.mean(region)), float(np.var(region))

def get_iris_avg(landmarks):
    try:
        return (landmarks[LEFT_IRIS].x + landmarks[RIGHT_IRIS].x) / 2.0, \
               (landmarks[LEFT_IRIS].y + landmarks[RIGHT_IRIS].y) / 2.0
    except Exception:
        return None, None

def compute_concentration(gaze_ok, head_ok, blink_recent, occluded, noise_flag):
    # weights (tunable): gaze 40%, head 30%, blink penalty 20%, noise 10%
    gaze_score = 1.0 if gaze_ok else 0.0
    head_score = 1.0 if head_ok else 0.0
    blink_pen = 0.0 if not blink_recent else 0.5
    noise_pen = 1.0 if noise_flag else 0.0
    base = 0.4*gaze_score + 0.3*head_score + 0.2*(1.0 - blink_pen) + 0.1*(1.0 - noise_pen)
    if occluded:
        base *= 0.2
    return int(np.clip(base * 100.0, 0, 100))

# ---------- Start audio thread and calibrate ----------
audio_ok = start_audio()
if audio_ok:
    calibrate_audio_baseline()
else:
    print("Warning: audio disabled; noise detection will be off.")

# ---------- Camera and calibration ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera. Close other apps or check device.")
    raise SystemExit

cv2.namedWindow("Concentration Tracker", cv2.WINDOW_NORMAL)
print("Camera opened. Starting gaze calibration — look straight at the camera now.")

calib_x = []
calib_y = []
calib_start = time.time()
while time.time() - calib_start < CALIB_SECONDS:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = face_detection.process(rgb)
    mesh = face_mesh.process(rgb)
    if mesh.multi_face_landmarks and det.detections:
        landmarks = mesh.multi_face_landmarks[0].landmark
        avgx, avgy = get_iris_avg(landmarks)
        if avgx is not None:
            calib_x.append(avgx); calib_y.append(avgy)
    cv2.putText(frame, "Calibrating gaze (keep eyes on camera)...", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

if not calib_x:
    print("Calibration failed: no face/iris detected. Retry with better lighting/position.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

baseline_x = float(np.mean(calib_x))
baseline_y = float(np.mean(calib_y))
print(f"Calibration complete. Baseline iris center = ({baseline_x:.3f}, {baseline_y:.3f})")

# ---------- Main variables ----------
score_buf = deque(maxlen=SCORE_SMOOTH)
frame_no = 0
blink_frames = 0
last_blink_time = 0.0
BLINK_MIN_SEP = 0.35  # seconds between blink events

# --- Timers we add (initialized here so they're in scope) ---
eyes_closed_start = None
no_face_start = None

print("Tracker running. Press 'q' in the window to quit.")

# ---------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed; exiting main loop.")
        break

    frame_no += 1
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection confidence and mesh
    det = face_detection.process(rgb)
    mesh = face_mesh.process(rgb)
    face_conf = 0.0
    if det.detections:
        face_conf = max([d.score[0] for d in det.detections])

    occluded = False
    blink_event = False
    gaze_dir = "UNKNOWN"
    concentration = 0

    if mesh.multi_face_landmarks and face_conf >= FACE_DET_CONF:
        # Reset no-face timer since we have a face now
        no_face_start = None

        lm = mesh.multi_face_landmarks[0].landmark

        # draw lightweight mesh
        mp_drawing.draw_landmarks(frame, mesh.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,150,255), thickness=1))

        # EAR blink detection (temporal)
        left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear > 0 and avg_ear < EAR_BLINK_THRESHOLD:
            blink_frames += 1
        else:
            if blink_frames >= BLINK_CONSEC_FRAMES:
                now = time.time()
                if now - last_blink_time > BLINK_MIN_SEP:
                    blink_event = True
                    last_blink_time = now
            blink_frames = 0

        # --- NEW: Eyes closed continuous timer (3 seconds) ---
        if avg_ear > 0 and avg_ear < EAR_BLINK_THRESHOLD:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            # else: keep the start time
            # we do not change detection logic, just show overlay when time passes threshold
            if time.time() - eyes_closed_start >= EYES_CLOSED_SECONDS:
                cv2.putText(frame, "EYES CLOSED > 3s", (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            eyes_closed_start = None

        # Occlusion: check pixel stats inside eye regions
        left_stats = eye_region_stats(gray, lm, LEFT_EYE, w, h)
        right_stats = eye_region_stats(gray, lm, RIGHT_EYE, w, h)
        if left_stats is None or right_stats is None:
            occluded = True
        else:
            lmean, lvar = left_stats; rmean, rvar = right_stats
            if (lvar < EYE_VARIANCE_THRESHOLD or lmean < EYE_MEAN_DARK) and \
               (rvar < EYE_VARIANCE_THRESHOLD or rmean < EYE_MEAN_DARK):
                occluded = True

        # gaze using iris avg and calibrated baseline
        avgx, avgy = get_iris_avg(lm)
        if avgx is None:
            gaze_dir = "UNKNOWN"
        else:
            dx = avgx - baseline_x
            dy = avgy - baseline_y
            # assign direction
            if abs(dx) <= GAZE_X_DELTA and abs(dy) <= GAZE_Y_DELTA:
                gaze_dir = "CENTER"
            elif abs(dx) > abs(dy):
                gaze_dir = "LEFT" if dx < 0 else "RIGHT"
            else:
                gaze_dir = "UP" if dy < 0 else "DOWN"

        # head center heuristic (nose)
        try:
            nose = lm[1]
            nose_x = nose.x; nose_y = nose.y
            head_ok = (abs(nose_x - 0.5) < 0.22 and abs(nose_y - 0.5) < 0.18)
        except Exception:
            head_ok = False

        # audio noise check (normalized)
        with _audio_lock:
            current_rms = _audio_rms
        noise_flag = False
        if _audio_baseline > 0 and current_rms > _audio_baseline * NOISE_SENSITIVITY:
            noise_flag = True

        # compute concentration
        gaze_ok = (gaze_dir == "CENTER")
        concentration = compute_concentration(gaze_ok, head_ok, blink_event, occluded, noise_flag)
    else:
        # no reliable face
        concentration = 0
        occluded = True
        gaze_dir = "NO_FACE"
        noise_flag = False

        # --- NEW: Start/advance no-face timer; exit after threshold ---
        if no_face_start is None:
            no_face_start = time.time()
        else:
            elapsed_no_face = time.time() - no_face_start
            # optional: draw countdown on frame so user knows
            cv2.putText(frame, f"No face: {int(elapsed_no_face)}s/{int(NO_FACE_SECONDS)}s", (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed_no_face >= NO_FACE_SECONDS:
                print(f"No face detected for {NO_FACE_SECONDS} seconds. Exiting...")
                break

    # Reset no_face_start if face is present (handled above at detection start)
    # (nothing else to do here)

    score_buf.append(concentration)
    smooth_score = int(np.mean(score_buf)) if len(score_buf) > 0 else concentration

    # status label priority
    if not mesh.multi_face_landmarks or face_conf < FACE_DET_CONF:
        status = "NO FACE"
    elif occluded:
        status = "OCCLUDED"
    else:
        # noisy has precedence if noise present
        with _audio_lock:
            curr = _audio_rms
        noisy = (_audio_baseline > 0 and curr > _audio_baseline * NOISE_SENSITIVITY)
        if noisy:
            status = "NOISY"
        elif blink_event:
            status = "BLINK"
        elif smooth_score < 55:
            status = "DISTRACTED"
        else:
            status = "CONCENTRATED"

    # Draw UI
    bar_x, bar_y, bar_w, bar_h = 18, 22, 320, 30
    fill = int((smooth_score/100.0) * bar_w)
    color = (0,200,0) if smooth_score >= 60 else (0,140,255) if smooth_score >= 40 else (0,80,200)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (30,30,30), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (200,200,200), 2)
    cv2.putText(frame, f"{smooth_score}%", (bar_x+bar_w+10, bar_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Status: {status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Gaze: {gaze_dir}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if blink_event:
        cv2.putText(frame, "BLINK", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    if noise_flag:
        cv2.putText(frame, "NOISE", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(frame, f"Frame: {frame_no}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Concentration Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
try:
    if _audio_stream is not None:
        _audio_stream.stop()
        _audio_stream.close()
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")
