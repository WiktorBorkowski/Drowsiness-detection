import cv2
import math
import mediapipe as mp

MODEL_PATH = "face_landmarker.task"

EAR_THRESHOLD = 0.15

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)
landmarker = FaceLandmarker.create_from_options(options)

def eye_aspect_ratio(landmarks, eye_indices):
    def coord(i):
        return landmarks[i].x, landmarks[i].y
    p1, p2, p3, p4, p5, p6 = [coord(i) for i in eye_indices]
    vertical1 = math.dist(p2, p6)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def eye_points_pixels(landmarks, eye_indices, frame):
    h, w = frame.shape[:2]
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6)

    result = landmarker.detect(mp_image)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE)

        left_pts = eye_points_pixels(landmarks, LEFT_EYE, frame)
        right_pts = eye_points_pixels(landmarks, RIGHT_EYE, frame)

        eyes_status = "Open" if left_EAR >= EAR_THRESHOLD or right_EAR >= EAR_THRESHOLD else "Closed"
        cv2.putText(frame, f"Eyes: {eyes_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for pts in [left_pts, right_pts]:
            for p in pts:
                cv2.circle(frame, p, 2, (0, 0, 255), -1)
            cv2.line(frame, pts[1], pts[5], (255, 0, 0), 1)
            cv2.line(frame, pts[2], pts[4], (255, 0, 0), 1)
            cv2.line(frame, pts[0], pts[3], (0, 255, 0), 1)

    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Full Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
