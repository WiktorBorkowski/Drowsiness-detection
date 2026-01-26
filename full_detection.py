import cv2
import math
import time
import mediapipe as mp

# ---------------- Configuration ----------------
MODEL_PATH = "face_landmarker.task"

EAR_THRESHOLD = 0.2 # 0.3 for poor lighting
MAR_THRESHOLD = 0.5
YAWN_DURATION = 1.2  # seconds
gi
# Eye & mouth landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------- Mediapipe Setup ----------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)
landmarker = FaceLandmarker.create_from_options(options)

# ---------------- Helper Functions ----------------
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
    top = (landmarks[0].x, landmarks[0].y)
    bottom = (landmarks[17].x, landmarks[17].y)
    left = (landmarks[61].x, landmarks[61].y)
    right = (landmarks[291].x, landmarks[291].y)
    vertical = math.dist(top, bottom)
    horizontal = math.dist(left, right)
    return vertical / horizontal

def mouth_points_pixels(landmarks, frame):
    h, w = frame.shape[:2]
    points = [0, 17, 61, 291]
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in points]

# ---------------- Video Capture ----------------
cap = cv2.VideoCapture(0)

# Blink & Yawn Tracking
blink_counter = 0
eye_closed = False

yawn_counter = 0
mouth_open_start = None

prev_time = time.time()
fps = 0

# ---------------- Main Loop ----------------
while True:
    # FPS calculation
    curr_time = time.time()
    dt = curr_time - prev_time
    fps = 1 / dt if dt > 0 else 0
    prev_time = curr_time

    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- Grayscale Display ----------------
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # so we can draw colored text/lines

    # ---------------- Mediapipe Detection ----------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # ----- Eyes -----
        left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
        eyes_status = "Open" if left_EAR >= EAR_THRESHOLD or right_EAR >= EAR_THRESHOLD else "Closed"
        cv2.putText(display_frame, f"Eyes: {eyes_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if eyes_status == "Closed" and not eye_closed:
            eye_closed = True
        elif eyes_status == "Open" and eye_closed:
            blink_counter += 1
            eye_closed = False

        # Draw eyes landmarks
        for pts in [LEFT_EYE, RIGHT_EYE]:
            pts_pixels = eye_points_pixels(landmarks, pts, display_frame)
            for p in pts_pixels:
                cv2.circle(display_frame, p, 2, (0, 0, 255), -1)
            cv2.line(display_frame, pts_pixels[1], pts_pixels[5], (255, 0, 0), 1)
            cv2.line(display_frame, pts_pixels[2], pts_pixels[4], (255, 0, 0), 1)
            cv2.line(display_frame, pts_pixels[0], pts_pixels[3], (0, 255, 0), 1)

        # ----- Mouth -----
        mar = mouth_aspect_ratio(landmarks)
        mouth_status = "Open" if mar > MAR_THRESHOLD else "Closed"
        cv2.putText(display_frame, f"Mouth: {mouth_status}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        current_time = time.time()
        if mouth_status == "Open":
            if mouth_open_start is None:
                mouth_open_start = current_time
            elif current_time - mouth_open_start >= YAWN_DURATION:
                yawn_counter += 1
                mouth_open_start = None
        else:
            mouth_open_start = None

        # Draw mouth landmarks
        mouth_pts = mouth_points_pixels(landmarks, display_frame)
        for p in mouth_pts:
            cv2.circle(display_frame, p, 3, (0, 255, 255), -1)
        cv2.line(display_frame, mouth_pts[0], mouth_pts[1], (255, 255, 0), 1)
        cv2.line(display_frame, mouth_pts[2], mouth_pts[3], (255, 255, 0), 1)

    # ---------------- Info Overlay ----------------
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display_frame, f"Blinks: {blink_counter}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Yawns: {yawn_counter}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(display_frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---------------- Display ----------------
    cv2.imshow("Full Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()
