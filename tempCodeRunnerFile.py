import cv2
import mediapipe as mp
import numpy as np
import math
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import sounddevice as sd
import sys
import threading
import time

# --- CONFIGURATION ---

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
# Initialize Face Mesh for up to 3 faces
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# YOLOv8 setup
yolo_model = YOLO('yolov8n.pt')
# List of "distraction" objects YOLO can detect
DISTRACTION_CLASSES = ['cell phone', 'book', 'laptop', 'mouse', 'remote', 'keyboard','pencil']

# --- !!! THIS IS THE PART YOU MUST TUNE !!! ---
# Concentration Logic Parameters
EAR_THRESHOLD = 0.18  # If score is 0%, change this to 0.16
HEAD_POSE_THRESHOLD = 30  # If score is 0%, change this to 40
ROLLING_AVERAGE_FRAMES = 90  # Number of frames for live concentration score

# --- THIS IS THE "ADD-ON" FOR NOISE ---
NOISE_SENSITIVITY = 2.0  # If "Loud Noise" is always on, change this to 3.0 or 4.0
AUDIO_CALIB_SECONDS = 3.0  # How long to listen for silence
AUDIO_SR = 22050
AUDIO_BLOCKSIZE = 1024
# --- END "ADD-ON" ---

# Threshold for counting a 'person' detection (Fix for 'hand as person' bug)
PERSON_CONFIDENCE_THRESHOLD = 0.6 # (60% confidence)


# --- DATA STORAGE ---
# For Pillar 1: Concentration Tracking
person_concentration_history = {
    0: deque(maxlen=ROLLING_AVERAGE_FRAMES),
    1: deque(maxlen=ROLLING_AVERAGE_FRAMES),
    2: deque(maxlen=ROLLING_AVERAGE_FRAMES)
}
# For Final Log
person_total_concentration_frames = {0: 0, 1: 0, 2: 0}
person_frame_count = {0: 0, 1: 0, 2: 0} # <-- FIXED BUG: Tracks frames per person
total_frames = 0

# For Pillar 2: Distraction Logging
distraction_log = []
last_seen_distractions = set()

# --- "ADD-ON" FOR NOISE (Thread-safe variables) ---
_audio_rms = 0.0
_audio_lock = threading.Lock()
_audio_stream = None
_audio_baseline = 1e-6 # Default silent level, will be calibrated


# --- HELPER FUNCTIONS (PILLAR 1) ---

def get_ear(landmarks, eye_indices):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    # Vertical landmarks
    v1 = landmarks[eye_indices[1]]
    v2 = landmarks[eye_indices[2]]
    v3 = landmarks[eye_indices[3]]
    # Horizontal landmarks
    h1 = landmarks[eye_indices[0]]
    h2 = landmarks[eye_indices[4]]
    
    # Euclidean distance
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)*2 + (p1.y - p2.y)*2)

    # Calculate vertical and horizontal distances
    vertical_dist1 = dist(v1, v3)
    vertical_dist2 = dist(v2, landmarks[eye_indices[5]]) # Using index 5 for the last point
    horizontal_dist = dist(h1, h2)
    
    if horizontal_dist == 0:
        return 0.0

    # EAR formula
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def get_head_pose(face_landmarks, frame_shape):
    """Estimates head pose (yaw, pitch, roll) from face landmarks."""
    h, w = frame_shape
    
    # 3D model points of a generic face
    face_3d_model_points = np.array([
        [0.0, 0.0, 0.0],      # Nose tip
        [0.0, -330.0, -65.0], # Chin
        [-225.0, 170.0, -135.0], # Left eye left corner
        [225.0, 170.0, -135.0],  # Right eye right corner
        [-150.0, -150.0, -125.0], # Left Mouth corner
        [150.0, -150.0, -125.0]  # Right mouth corner
    ])
    
    # Key 2D image points from MediaPipe
    face_2d_image_points = np.array([
        [face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h],      # Nose tip
        [face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h],  # Chin
        [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h],  # Left eye left corner
        [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h],    # Right eye right corner
        [face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h],  # Left Mouth corner
        [face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h]    # Right mouth corner
    ], dtype=np.double)

    # Camera matrix (approximatio)
    focal_length = w
    cam_center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, cam_center[0]],
        [0, focal_length, cam_center[1]],
        [0, 0, 1]
    ], dtype=np.double)
    
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.double)
    
    try:
        # Solve for pose
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            face_3d_model_points, face_2d_image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Combine rotation matrix and translation vector
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        
        # Decompose to get Euler angles
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        
        return -euler_angles[1], euler_angles[0], euler_angles[2]
    except Exception:
        return 0.0, 0.0, 0.0 # Return 0s if solvePnP fails


# Eye landmark indices from MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


# --- "ADD-ON" FOR NOISE (Helper Functions) ---
def audio_callback(indata, frames, time_info, status):
    """
    This callback runs in a separate thread.
    It safely updates the global RMS value.
    """
    global _audio_rms
    if status:
        print(status, file=sys.stderr)
    
    # Use float32 for RMS calculation
    rms = np.sqrt(np.mean(indata**2))
    
    with _audio_lock:
        _audio_rms = float(rms)

def calibrate_audio_baseline():
    """
    Listens for a few seconds to find the average "silent"
    noise level of the room.
    """
    global _audio_baseline
    global cap # Use the global cap object
    try:
        print(f"\nCalibrating microphone for {AUDIO_CALIB_SECONDS:.1f}s — PLEASE BE QUIET... (Press 'q' in window to skip)")
        
        # Temporarily start a recording stream
        rec_data = []
        def rec_callback(indata, frames, time_info, status):
            rec_data.append(indata.copy())
            
        # Need to open camera before calibration to check for 'q'
        if not cap.isOpened():
            cap.open(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        rec_stream = sd.InputStream(
            callback=rec_callback, 
            samplerate=AUDIO_SR, 
            blocksize=AUDIO_BLOCKSIZE, 
            channels=1, 
            dtype='float32'
        )
        
        rec_stream.start()
        
        # Wait for calibration time, but allow OpenCV window to be responsive
        start_time = time.time()
        while time.time() - start_time < AUDIO_CALIB_SECONDS:
            ret, frame = cap.read() # Read frame to show a window
            if ret:
                # Show a message on the calibration frame
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "CALIBRATING... PLEASE BE QUIET", (50, 360), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Concentration and Distraction Analyzer', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'): # Allow skipping
                print("Audio calibration skipped.")
                break
        
        rec_stream.stop()
        rec_stream.close()

        if not rec_data:
            print("Audio calibration failed: No data recorded.")
            _audio_baseline = 1e-6 # Use default
            return

        # Calculate RMS of the entire recording
        full_recording = np.concatenate(rec_data, axis=0)
        mono = full_recording[:,0]
        _audio_baseline = max(1e-6, float(np.sqrt(np.mean(np.square(mono)))))
        print(f"Audio baseline RMS = {_audio_baseline:.6f}")
        
    except Exception as e:
        print(f"Audio calibration failed: {e}")
        _audio_baseline = 1e-6 # Use default

def start_audio_stream():
    """
    Starts the continuous, non-blocking audio stream for real-time analysis.
    """
    global _audio_stream
    try:
        _audio_stream = sd.InputStream(
            callback=audio_callback,
            samplerate=AUDIO_SR,
            blocksize=AUDIO_BLOCKSIZE,
            channels=1,
            dtype='float32'
        )
        _audio_stream.start()
        print("Real-time audio stream started.")
        return True
    except Exception as e:
        print(f"Warning: Could not start audio stream. {e}")
        print("Noise detection will be disabled.")
        return False
# --- END "ADD-ON" ---


# --- MAIN PROGRAM ---
print("Starting Concentration Tracker... Press 'q' to quit.")
cap = cv2.VideoCapture(0)
# This check is now inside calibrate_audio_baseline, but we leave it
# here as a fallback.
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- "ADD-ON" FOR NOISE (Calibration) ---
# We add this block to calibrate the audio
calibrate_audio_baseline()

# And this block to start the audio stream
if not start_audio_stream():
    _audio_stream = None # Ensure it's None if it fails
# --- END "ADD-ON" ---


# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    total_frames += 1
    
    # Flip the frame horizontally for a "mirror" view
    frame = cv2.flip(frame, 1)
    
    # Create copies for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yolo_frame = frame.copy() 
    
    # --- PILLAR 2: ENVIRONMENT DETECTOR (YOLOv8) ---
    yolo_results = yolo_model(yolo_frame, verbose=False)
    
    current_distractions_in_frame = set()
    num_persons_detected = 0 # High-confidence person count

    for box in yolo_results[0].boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        # Check for defined distraction classes
        if class_name in DISTRACTION_CLASSES:
            current_distractions_in_frame.add(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Count persons
        elif class_name == 'person':
            # We still draw the box, so you can see what it's thinking
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Person ({confidence:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # BUT: Only count it as a distraction if confidence is high
            if confidence > PERSON_CONFIDENCE_THRESHOLD:
                num_persons_detected += 1
    
    # --- PILLAR 1: CONCENTRATION TRACKER (MediaPipe) ---
    rgb_frame.flags.writeable = False # Optimize
    mp_results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True

    live_scores = {0: "N/A", 1: "N/A", 2: "N/A"}
    num_faces_detected = 0

    # --- "ADD-ON" FOR NOISE (Get current status) ---
    # We add this check here
    with _audio_lock:
        current_rms = _audio_rms
    # Check if current noise is significantly louder than baseline
    is_noisy = (current_rms > _audio_baseline * NOISE_SENSITIVITY)
    # --- END "ADD-ON" ---

    if mp_results.multi_face_landmarks:
        num_faces_detected = len(mp_results.multi_face_landmarks)
        
        for person_id, face_landmarks in enumerate(mp_results.multi_face_landmarks):
            # Draw the face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            is_concentrating = False
            try:
                # 1. Check Head Pose
                yaw, pitch, roll = get_head_pose(face_landmarks, frame.shape[:2])
                
                # 2. Check Eye Blinks
                left_ear = get_ear(face_landmarks.landmark, LEFT_EYE_INDICES)
                right_ear = get_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0
                
                # --- Original Concentration Logic ---
                head_forward = (abs(yaw) < HEAD_POSE_THRESHOLD) and (abs(pitch) < HEAD_POSE_THRESHOLD)
                eyes_open = (ear > EAR_THRESHOLD)
                
                # --- THIS IS THE "ADD-ON" ---
                # You are only concentrating if eyes are open, head is forward, AND it's not noisy
                if head_forward and eyes_open and not is_noisy:
                    is_concentrating = True
                # --- END "ADD-ON" ---

                # Draw head pose info for debugging
                cv2.putText(frame, f"P{person_id+1} Yaw: {yaw:.0f}", (10, 90 + person_id*60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"P{person_id+1} Pitch: {pitch:.0f}", (10, 110 + person_id*60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"P{person_id+1} EAR: {ear:.2f}", (10, 130 + person_id*60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            except Exception as e:
                # print(f"Error processing face {person_id}: {e}")
                pass # solvePnP can sometimes fail

            # Update concentration history
            if is_concentrating:
                person_concentration_history[person_id].append(1)
                person_total_concentration_frames[person_id] += 1
            else:
                person_concentration_history[person_id].append(0)
            
            # --- FIXED BUG: Track frame count for this person ---
            person_frame_count[person_id] += 1 

            # Calculate live score
            history = person_concentration_history[person_id]
            if len(history) > 0:
                live_score_perc = (sum(history) / len(history)) * 100
                live_scores[person_id] = f"{live_score_perc:.0f}%"
            else:
                live_scores[person_id] = "..."
                
    # --- "THE MERGE": Correlate and Log ---

    # Check for "other" people as distractions
    if num_persons_detected > num_faces_detected:
        current_distractions_in_frame.add("other person")
    
    # --- "ADD-ON" FOR NOISE (Logging) ---
    if is_noisy:
        current_distractions_in_frame.add("Loud Noise")

    # Log newly detected distractions
    newly_detected_distractions = current_distractions_in_frame - last_seen_distractions
    for obj_name in newly_detected_distractions:
        event_time = datetime.now()
        distraction_log.append((event_time, obj_name))
        print(f"LOGGED Event: {event_time.strftime('%H:%M:%S')} - Detected {obj_name}")
    
    last_seen_distractions = current_distractions_in_frame


    # --- DISPLAY FINAL FRAME ---
    
    # Display Pillar 1 Scores
    cv2.putText(frame, f"Person 1: {live_scores[0]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Person 2: {live_scores[1]}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Person 3: {live_scores[2]}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    # Display Pillar 3 "Merge" Alert
    distraction_alert_text = ""
    # --- "ADD-ON" FOR NOISE (Display) ---
    if current_distractions_in_frame: # This now includes "Loud Noise"
        distraction_alert_text = "DISTRACTION: " + ", ".join(current_distractions_in_frame)
        
    if distraction_alert_text:
        cv2.putText(frame, distraction_alert_text, (frame.shape[1] - 700, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Correlate low score with distraction
    for i in range(3):
        score_val = live_scores[i]
        if score_val != "N/A" and score_val != "..." and distraction_alert_text:
            if float(score_val[:-1]) < 50.0: # If score < 50% and distraction is present
                cv2.putText(frame, f"Person {i+1} Distracted!", (10, 180 + (i*30)), # Moved text down
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Concentration and Distraction Analyzer', frame)

    # This is the correct location, inside the while loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- SHUTDOWN AND LOGGING ---
# This block is now correctly OUTSIDE the loop
cap.release()

# --- "ADD-ON" FOR NOISE (Shutdown) ---
if _audio_stream:
    _audio_stream.stop()
    _audio_stream.close()
    print("\nAudio stream stopped.")

cv2.destroyAllWindows()

print("\n" + "="*50)
print("             PROGRAM ENDED - FINAL LOG")
print("="*50)

# 1. Final Concentration Percentage Log
print("\n--- FINAL CONCENTRATION LOG ---")
if total_frames > 0:
    for i in range(3):
        # --- FIXED BUG: This now calculates % correctly per person ---
        if person_frame_count[i] > 0:
            perc = (person_total_concentration_frames[i] / person_frame_count[i]) * 100
            print(f"Person {i+1} Overall Concentration: {perc:.2f}%")
        else:
            print(f"Person {i+1} was not detected.")
    # --- FIXED BUG: This line is now correctly indented ---
    print(f"\nTotal frames processed: {total_frames}")
else:
    print("No frames were processed.")

# 2. Distraction Object Log
print("\n--- DISTRACTION OBJECT LOG ---")
if not distraction_log:
    print("No distraction objects were detected during the session.")
else:
    print(f"Total distraction events logged: {len(distraction_log)}")
    # This log will now show "Loud Noise" events
    for event_time, obj_name in distraction_log:
        print(f"[{event_time.strftime('%Y-%m-%d %H:%M:%S')}] Detected: {obj_name}")

print("\n" + "="*50)