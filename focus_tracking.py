import cv2
import mediapipe as mp
import time
import requests

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start webcam capture
cap = cv2.VideoCapture(0)

start_time = time.time()
focus_lost = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        start_time = time.time()  # Reset focus timer
        focus_lost = False
    else:
        if time.time() - start_time > 30:  # 30 sec threshold
            focus_lost = True

    # Send focus status to backend API
    requests.post("http://127.0.0.1:8000/focus/", json={"focus_lost": focus_lost})

    # Display status on screen
    cv2.putText(frame, f"Focus Lost: {focus_lost}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if focus_lost else (0, 255, 0), 2)
    
    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
