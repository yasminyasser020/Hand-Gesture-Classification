import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import joblib
import numpy as np


class_names = {
    0: "call", 1: "dislike", 2: "fist", 3: "four", 4: "like",
    5: "mute", 6: "ok", 7: "one", 8: "palm", 9: "peace",
    10: "peace_inverted", 11: "rock", 12: "stop", 13: "stop_inverted",
    14: "three", 15: "three2", 16: "two_up", 17: "two_up_inverted" }

model = joblib.load('svm_model.pkl')

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),  (9,13), (13,14), (14,15),
    (15,16),  (13,17), (0,17), (17,18), (18,19), (19,20) ]


cap = cv2.VideoCapture("demo_video.mp4")

# capture one "test frame" to find the real dimensions
success, test_frame = cap.read()
if not success:
    print("Error: Could not access camera.")
    exit()


h, w, _ = test_frame.shape

# Define VideoWriter using the exact detected dimensions
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (w, h))

cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)

print(f"System Ready! Resolution: {w}x{h}")

# ==========================================
#  MAIN PROCESSING LOOP
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # --- DRAWING (SKELETON & DOTS) ---
        for conn in HAND_CONNECTIONS:
            start = hand_landmarks[conn[0]]
            end = hand_landmarks[conn[1]]
        
            cv2.line(frame, (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), (0, 255, 0), 2)
        
        for lm in hand_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

        # ---- FEATURE EXTRACTION ----
        ## Recenter (x,y) landmarks using wrist point as origin (0,0) (Landmark 0 x1,y1)
        ## Scaling the landmarks by dividing them by the middle finger tip position
        wrist = hand_landmarks[0]
        mid_tip = hand_landmarks[12]
        
        landmarks_list = []
        for lm in hand_landmarks:
            
            rel_x = (lm.x - wrist.x) / (mid_tip.x)
            rel_y = (lm.y - wrist.y) / (mid_tip.y)
            rel_z = lm.z - wrist.z
            landmarks_list.extend([rel_x, rel_y, rel_z])
        
        # --- PREDICTION ---
        input_data = np.array(landmarks_list).reshape(1, -1)
        prediction_id = model.predict(input_data)[0]
        label = class_names.get(prediction_id, f"ID: {prediction_id}")

        # ----TEXT DISPLAY----
        cv2.rectangle(frame, (0,0), (420, 60), (0,0,0), -1)
        cv2.putText(frame, f"Gesture: {label}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  
    out.write(frame)
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Process finished. Video saved as 'output.mp4'")