import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_path = "V1.mp4" 
  
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    
    "output_with_pose.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

knee_angles = []

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        
        landmarks = results.pose_landmarks.landmark

        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        

        h = (int(hip.x * width), int(hip.y * height))
        k = (int(knee.x * width), int(knee.y * height))
        a = (int(ankle.x * width), int(ankle.y * height))
        

        angle = calculate_angle(h, k, a)
        knee_angles.append(angle)

        cv2.putText(
            frame,
            f"Knee Angle: {int(angle)} deg",
            (k[0] - 40, k[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    out.write(frame)

    cv2.imshow("Pose Analysis", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()

if knee_angles:
    print(f"Frames analyzed: {len(knee_angles)}")
    print(f"Min knee angle: {min(knee_angles):.2f} deg")
    print(f"Max knee angle: {max(knee_angles):.2f} deg")
    print(f"Average knee angle: {np.mean(knee_angles):.2f} deg")