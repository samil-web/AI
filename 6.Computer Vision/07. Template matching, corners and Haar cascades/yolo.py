import cv2
import matplotlib.pyplot as plt
import numpy as np
from helpers import imshow
%matplotlib inline


background_subtractor = cv2.createBackgroundSubtractorMOG2(history = 10,varThreshold = 5,detectShadow = True)

capture = cv2.VideoCapture(0)

while True:
    ret,frame = capture.read()

    if not ret:
        break

    foreground_mask = background_subtractor.apply(frame)

    k = cv2.waitKey(30)

    if k == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)