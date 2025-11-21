import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#img = 'https://free-images.com/lg/7c5c/people_looking_talking_on.jpg'
cap = cv2.VideoCapture(0) #0=webcam, enter filename for videos
while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame, size=320)  #size = 320 speeds up but messes with program?
    cv2.imshow('Object detection', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#results = model(img)
#results.print()

#results.show()


