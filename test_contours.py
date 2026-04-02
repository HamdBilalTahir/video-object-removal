import cv2
import numpy as np
mask = np.ones((100, 100), dtype=np.float64)
_, thresh = cv2.threshold(mask, 127, 255, 0)
cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
