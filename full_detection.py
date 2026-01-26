import cv2
import math
import mediapipe as mp # Run activate_this.py to activate venv.
# If installing manually install numpy v1.26.1 after mediapipe

MODEL_PATH = "face_landmarker.task"

EAR_THRESHOLD = 0.15
MAR_THRESHOLD = 0.6

# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)
landmarker = FaceLandmarker.create_from_options(options)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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

def mouth_aspect_ratio(landmarks):
    # New points for mouth opening
    top = (landmarks[0].x, landmarks[0].y)       # Upper lip center
    bottom = (landmarks[17].x, landmarks[17].y) # Lower lip center
    left = (landmarks[61].x, landmarks[61].y)   # Left corner
    right = (landmarks[291].x, landmarks[291].y) # Right corner

    vertical = math.dist(top, bottom)
    horizontal = math.dist(left, right)
    return vertical / horizontal

def mouth_points_pixels(landmarks, frame):
    h, w = frame.shape[:2]
    points = [0, 17, 61, 291]  # Same points as MAR
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in points]

# Face detection (optional, just for drawing)
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

    # Draw face rectangles (optional)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # Eyes
        left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
        eyes_status = "Open" if left_EAR >= EAR_THRESHOLD or right_EAR >= EAR_THRESHOLD else "Closed"
        cv2.putText(frame, f"Eyes: {eyes_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw eyes
        for pts in [LEFT_EYE, RIGHT_EYE]:
            pts_pixels = eye_points_pixels(landmarks, pts, frame)
            for p in pts_pixels:
                cv2.circle(frame, p, 2, (0, 0, 255), -1)
            cv2.line(frame, pts_pixels[1], pts_pixels[5], (255, 0, 0), 1)
            cv2.line(frame, pts_pixels[2], pts_pixels[4], (255, 0, 0), 1)
            cv2.line(frame, pts_pixels[0], pts_pixels[3], (0, 255, 0), 1)

        # Mouth
        mar = mouth_aspect_ratio(landmarks)
        mouth_status = "Open" if mar > MAR_THRESHOLD else "Closed"
        cv2.putText(frame, f"Mouth: {mouth_status}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw mouth points
        mouth_pts = mouth_points_pixels(landmarks, frame)
        for p in mouth_pts:
            cv2.circle(frame, p, 3, (0, 255, 255), -1)
        cv2.line(frame, mouth_pts[0], mouth_pts[1], (255, 255, 0), 1) # top-bottom
        cv2.line(frame, mouth_pts[2], mouth_pts[3], (255, 255, 0), 1) # left-right

    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Full Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
