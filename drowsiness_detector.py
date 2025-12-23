import cv2
import numpy as np
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('Face_and_eye_yolo/V8/yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

print(model.names)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Run YOLOv8 detection
    results = model(frame_resized)  # returns a list of Results objects

    # Render bounding boxes and labels on the frame
    annotated_frame = results[0].plot()  # YOLOv8 way


    # Display the result
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
