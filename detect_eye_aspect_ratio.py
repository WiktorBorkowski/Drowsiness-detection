import cv2
import math
import mediapipe as mp

MODEL_PATH = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create FaceLandmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)
landmarker = FaceLandmarker.create_from_options(options)

# calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_indices):
    def coord(i):
        return landmarks[i].x, landmarks[i].y

    p1, p2, p3, p4, p5, p6 = [coord(i) for i in eye_indices]
    vertical1 = math.dist(p2, p6)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Eye landmark index sets
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


image = cv2.imread("images/person_eyes_open.png")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # MUST BE COLOUR!!

# 3. Wrap as mp.Image for the Tasks API
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


result = landmarker.detect(mp_image)

if result.face_landmarks:
    landmarks = result.face_landmarks[0]

    left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE)

    print(f"Left EAR: {left_EAR:.2f}, Right EAR: {right_EAR:.2f}")

    EAR_THRESHOLD = 0.25
    if left_EAR < EAR_THRESHOLD and right_EAR < EAR_THRESHOLD:
        print("Eyes are closed")
    else:
        print("Eyes are open")
else:
    print("No face detected")
