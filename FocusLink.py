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

# --- CONFIGURATION (Optimization Parameters) ---
# Set this to a lower value (e.g., 320 or 480) for low-end PCs
# Processing will happen on a downscaled frame, but display will be original size
PROCESS_RESOLUTION_WIDTH = 640 
MAX_FPS = 25 # Limit the main loop FPS to save CPU/GPU resources
FRAME_SKIP_YOLO = 3 # Run YOLO on every Nth frame (YOLO is the most expensive part)

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
# Initialize Face Mesh for up to 3 faces (max_num_faces=3 is fine)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# YOLOv8 setup - using the fast 'n' model is good
yolo_model = YOLO('yolov8n.pt')
# List of "distraction" objects YOLO can detect
DISTRACTION_CLASSES = ['cell phone', 'book', 'laptop', 'mouse', 'remote', 'keyboard','pencil']

# --- !!! THIS IS THE PART YOU MUST TUNE !!! ---
# Concentration Logic Parameters
EAR_THRESHOLD = 0.18
HEAD_POSE_THRESHOLD = 30
ROLLING_AVERAGE_FRAMES = 90

# --- THIS IS THE "ADD-ON" FOR NOISE ---
NOISE_SENSITIVITY = 2.0
AUDIO_CALIB_SECONDS = 3.0
AUDIO_SR = 22050
AUDIO_BLOCKSIZE = 1024

# Threshold for counting a 'person' detection (Fix for 'hand as person' bug)
PERSON_CONFIDENCE_THRESHOLD = 0.6 # (60% confidence)


# --- DATA STORAGE ---
person_concentration_history = {
    0: deque(maxlen=ROLLING_AVERAGE_FRAMES),
    1: deque(maxlen=ROLLING_AVERAGE_FRAMES),
    2: deque(maxlen=ROLLING_AVERAGE_FRAMES)
}
person_total_concentration_frames = {0: 0, 1: 0, 2: 0}
person_frame_count = {0: 0, 1: 0, 2: 0}
total_frames = 0
yolo_frame_counter = 0 # Counter for frame skipping

# For Pillar 2: Distraction Logging
distraction_log = []
last_seen_distractions = set()
# Cache YOLO results from the last run to use for skipped frames
last_yolo_boxes = []

# --- "ADD-ON" FOR NOISE (Thread-safe variables) ---
_audio_rms = 0.0
_audio_lock = threading.Lock()
_audio_stream = None
_audio_baseline = 1e-6 


# --- HELPER FUNCTIONS (PILLAR 1) ---

def get_ear(landmarks, eye_indices):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    
    # Extract coordinates directly from normalized MediaPipe landmarks
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # Vertical landmarks - using indices 1, 2, 3 for vertical
    # NOTE: The original code used 1, 2, 3, but the 6-point model (P1..P6) typically uses
    # P2 and P6, and P3 and P5 for vertical. Let's stick to the original logic's indices
    v1 = landmarks[eye_indices[1]]
    v2 = landmarks[eye_indices[2]]
    v3 = landmarks[eye_indices[3]]
    # Horizontal landmarks
    h1 = landmarks[eye_indices[0]]
    h2 = landmarks[eye_indices[4]]
    
    # Calculate vertical and horizontal distances (using index 5 for the last point as per original logic)
    vertical_dist1 = dist(v1, v3)
    vertical_dist2 = dist(v2, landmarks[eye_indices[5]]) 
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
        [0.0, 0.0, 0.0],      # Nose tip (1)
        [0.0, -330.0, -65.0], # Chin (152)
        [-225.0, 170.0, -135.0], # Left eye left corner (263)
        [225.0, 170.0, -135.0],  # Right eye right corner (33)
        [-150.0, -150.0, -125.0], # Left Mouth corner (287)
        [150.0, -150.0, -125.0]  # Right mouth corner (57)
    ])
    
    # Key 2D image points from MediaPipe - Use .x * w and .y * h for pixel coords
    face_2d_image_points = np.array([
        [face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h],     
        [face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h], 
        [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h], 
        [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h],   
        [face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h], 
        [face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h]   
    ], dtype=np.double)

    # Camera matrix (approximatio) - using w as focal length is a common approximation
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
        # Solve for pose - cv2.SOLVEPNP_ITERATIVE is a good default
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            face_3d_model_points, face_2d_image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0
            
        # Convert rotation vector to rotation matrix and then to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        
        # Euler angles: yaw, pitch, roll
        return -euler_angles[1][0], euler_angles[0][0], euler_angles[2][0]
    except Exception:
        # Silently fail, it happens when key points are obscured
        return 0.0, 0.0, 0.0 


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
    # Ensure indata is not empty or all zeros before calculation
    if indata.size > 0:
        rms = np.sqrt(np.mean(indata**2))
        with _audio_lock:
            _audio_rms = float(rms)

def calibrate_audio_baseline():
    """
    Listens for a few seconds to find the average "silent"
    noise level of the room.
    """
    global _audio_baseline
    global cap 
    try:
        print(f"\nCalibrating microphone for {AUDIO_CALIB_SECONDS:.1f}s — PLEASE BE QUIET... (Press 'q' in window to skip)")
        
        # Need to open camera *before* calibration to check for 'q'
        if not cap.isOpened():
            cap.open(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        rec_data = []
        def rec_callback(indata, frames, time_info, status):
            rec_data.append(indata.copy())
            
        rec_stream = sd.InputStream(
            callback=rec_callback, 
            samplerate=AUDIO_SR, 
            blocksize=AUDIO_BLOCKSIZE, 
            channels=1, 
            dtype='float32'
        )
        
        rec_stream.start()
        
        start_time = time.time()
        while time.time() - start_time < AUDIO_CALIB_SECONDS:
            ret, frame = cap.read()
            if ret:
                # Show a message on the calibration frame
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "CALIBRATING... PLEASE BE QUIET", (50, 360), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Concentration and Distraction Analyzer', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Audio calibration skipped.")
                break
        
        rec_stream.stop()
        rec_stream.close()

        if not rec_data:
            print("Audio calibration failed: No data recorded.")
            _audio_baseline = 1e-6 
            return

        # Calculate RMS of the entire recording
        full_recording = np.concatenate(rec_data, axis=0)
        # Handle stereo/mono if necessary, assuming one channel 'channels=1' was requested
        mono = full_recording.flatten() 
        _audio_baseline = max(1e-6, float(np.sqrt(np.mean(np.square(mono)))))
        print(f"Audio baseline RMS = {_audio_baseline:.6f}")
        
    except Exception as e:
        print(f"Audio calibration failed: {e}")
        _audio_baseline = 1e-6 

def start_audio_stream():
    """
    Starts the continuous, non-blocking audio stream for real-time analysis.
    """
    global _audio_stream
    try:
        # Check if default device exists before starting
        sd.query_devices() 
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

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    sys.exit()

# Try to set a standard high-res for better quality display, 
# but we will downscale for processing later.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Frame Rate Control Setup ---
# Calculate the delay needed to not exceed MAX_FPS
if MAX_FPS > 0:
    target_frame_time_ms = 1000 / MAX_FPS
else:
    target_frame_time_ms = 0
last_frame_time = time.time()

# --- Optimization: Set OpenCV Preferred Backend (Optional but helpful) ---
# Use DSHOW on Windows for better camera initialization, or AUTO.
# cv2.setPreference(cv2.CAP_DSHOW) # Only uncomment on Windows if you have issues

# --- "ADD-ON" FOR NOISE (Calibration) ---
calibrate_audio_baseline()
if not start_audio_stream():
    _audio_stream = None 

# --- MAIN LOOP ---
while cap.isOpened():
    frame_start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    total_frames += 1
    
    # Flip the frame horizontally for a "mirror" view
    frame = cv2.flip(frame, 1)
    
    # Get original dimensions for upscaling bounding boxes later
    H, W = frame.shape[:2]
    
    # --- Optimization: Downscale for Processing ---
    if W > PROCESS_RESOLUTION_WIDTH:
        scale_factor = PROCESS_RESOLUTION_WIDTH / W
        proc_frame = cv2.resize(frame, (PROCESS_RESOLUTION_WIDTH, int(H * scale_factor)))
        p_H, p_W = proc_frame.shape[:2]
    else:
        scale_factor = 1.0
        proc_frame = frame.copy()
        p_H, p_W = H, W
    
    rgb_proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)


    # --- PILLAR 2: ENVIRONMENT DETECTOR (YOLOv8) ---
    # Optimization: Only run YOLO every N frames
    yolo_frame_counter += 1
    if yolo_frame_counter % FRAME_SKIP_YOLO == 0:
        yolo_results = yolo_model(proc_frame, verbose=False)
        last_yolo_boxes = yolo_results[0].boxes
    
    # Use the latest YOLO results (either fresh or cached)
    current_distractions_in_frame = set()
    num_persons_detected = 0 

    for box in last_yolo_boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        
        # YOLO bounding boxes are based on proc_frame, so upscale them for the display frame
        x1_p, y1_p, x2_p, y2_p = map(int, box.xyxy[0])
        
        # Upscale coordinates back to original frame size (W, H)
        x1 = int(x1_p / scale_factor)
        y1 = int(y1_p / scale_factor)
        x2 = int(x2_p / scale_factor)
        y2 = int(y2_p / scale_factor)
        
        confidence = float(box.conf[0])
        
        # Check for defined distraction classes
        if class_name in DISTRACTION_CLASSES:
            current_distractions_in_frame.add(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Count persons
        elif class_name == 'person':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Person ({confidence:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if confidence > PERSON_CONFIDENCE_THRESHOLD:
                num_persons_detected += 1
    
    # --- PILLAR 1: CONCENTRATION TRACKER (MediaPipe) ---
    # MediaPipe runs on the downscaled frame for performance
    rgb_proc_frame.flags.writeable = False 
    mp_results = face_mesh.process(rgb_proc_frame)
    rgb_proc_frame.flags.writeable = True

    live_scores = {0: "N/A", 1: "N/A", 2: "N/A"}
    num_faces_detected = 0

    # --- "ADD-ON" FOR NOISE (Get current status) ---
    with _audio_lock:
        current_rms = _audio_rms
    is_noisy = (current_rms > _audio_baseline * NOISE_SENSITIVITY)

    if mp_results.multi_face_landmarks:
        num_faces_detected = len(mp_results.multi_face_landmarks)
        
        for person_id, face_landmarks in enumerate(mp_results.multi_face_landmarks):
            
            # Upscale landmarks for drawing and head pose calculation (if scale_factor != 1.0)
            # The landmarks are normalized (0 to 1), so we use the original (W, H) to get pixel values for drawing
            face_landmarks_upscaled = face_landmarks
            
            # Draw the face mesh on the original frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_upscaled, # MediaPipe handles the drawing using normalized coords
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            is_concentrating = False
            try:
                # Get head pose. Note: Head pose is calculated using the *downscaled* pixel coords
                # inside the function, but since normalized landmarks are used, it's fine.
                # The crucial part is using frame.shape[:2] which is the size the landmarks were normalized to
                # (which is p_W, p_H from the proc_frame), so we pass that size.
                yaw, pitch, roll = get_head_pose(face_landmarks, (p_H, p_W))
                
                # Check Eye Blinks
                left_ear = get_ear(face_landmarks.landmark, LEFT_EYE_INDICES)
                right_ear = get_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0
                
                # Concentration Logic
                head_forward = (abs(yaw) < HEAD_POSE_THRESHOLD) and (abs(pitch) < HEAD_POSE_THRESHOLD)
                eyes_open = (ear > EAR_THRESHOLD)
                
                if head_forward and eyes_open and not is_noisy:
                    is_concentrating = True

                # Draw head pose info for debugging on the original frame
                cv2.putText(frame, f"P{person_id+1} Yaw: {yaw:.0f}", (10, 90 + person_id*60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"P{person_id+1} Pitch: {pitch:.0f}", (10, 110 + person_id*60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"P{person_id+1} EAR: {ear:.2f}", (10, 130 + person_id*60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            except Exception:
                pass # solvePnP can sometimes fail

            # Update concentration history
            if is_concentrating:
                person_concentration_history[person_id].append(1)
                person_total_concentration_frames[person_id] += 1
            else:
                person_concentration_history[person_id].append(0)
            
            person_frame_count[person_id] += 1 

            # Calculate live score
            history = person_concentration_history[person_id]
            if len(history) > 0:
                live_score_perc = (sum(history) / len(history)) * 100
                live_scores[person_id] = f"{live_score_perc:.0f}%"
            else:
                live_scores[person_id] = "..."
                
    # --- "THE MERGE": Correlate and Log ---

    if num_persons_detected > num_faces_detected:
        current_distractions_in_frame.add("other person")
    
    if is_noisy:
        current_distractions_in_frame.add("Loud Noise")

    # Log *newly* detected distractions
    newly_detected_distractions = current_distractions_in_frame - last_seen_distractions
    for obj_name in newly_detected_distractions:
        event_time = datetime.now()
        distraction_log.append((event_time, obj_name))
        print(f"LOGGED Event: {event_time.strftime('%H:%M:%S')} - Detected {obj_name}")
    
    last_seen_distractions = current_distractions_in_frame


    # --- DISPLAY FINAL FRAME ---
    
    # Display Pillar 1 Scores
    cv2.putText(frame, f"P1 Score: {live_scores[0]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"P2 Score: {live_scores[1]}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"P3 Score: {live_scores[2]}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    # Display Pillar 3 "Merge" Alert
    distraction_alert_text = ""
    if current_distractions_in_frame:
        # Use only the top 3 distractions for display if there are many
        display_distractions = list(current_distractions_in_frame)[:3] 
        distraction_alert_text = "DISTRACTION: " + ", ".join(display_distractions)
        
    if distraction_alert_text:
        # Position the text on the top right
        text_size = cv2.getTextSize(distraction_alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        cv2.putText(frame, distraction_alert_text, (text_x, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Correlate low score with distraction
    for i in range(3):
        score_val = live_scores[i]
        if score_val != "N/A" and score_val != "..." and distraction_alert_text:
            try:
                if float(score_val[:-1]) < 50.0:
                    cv2.putText(frame, f"Person {i+1} Distracted!", (10, H - 30 - (i*30)), # Moved text to bottom-left
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except ValueError:
                pass # Skip if score is "..."

    cv2.imshow('Concentration and Distraction Analyzer', frame)

    # --- Optimization: Frame Rate Control ---
    # Calculate time taken for this frame
    frame_end_time = time.time()
    time_spent_ms = (frame_end_time - frame_start_time) * 1000
    
    # Calculate required delay
    delay_ms = max(1, int(target_frame_time_ms - time_spent_ms)) 
    
    # Use the calculated delay in waitKey
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break

# --- SHUTDOWN AND LOGGING ---
cap.release()

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
        if person_frame_count[i] > 0:
            perc = (person_total_concentration_frames[i] / person_frame_count[i]) * 100
            print(f"Person {i+1} Overall Concentration: {perc:.2f}%")
        else:
            print(f"Person {i+1} was not detected.")
    print(f"\nTotal frames processed: {total_frames}")
else:
    print("No frames were processed.")

# 2. Distraction Object Log
print("\n--- DISTRACTION OBJECT LOG ---")
if not distraction_log:
    print("No distraction objects were detected during the session.")
else:
    print(f"Total distraction events logged: {len(distraction_log)}")
    for event_time, obj_name in distraction_log:
        print(f"[{event_time.strftime('%Y-%m-%d %H:%M:%S')}] Detected: {obj_name}")

print("\n" + "="*50)