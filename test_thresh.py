import cv2
import numpy as np
mask = np.ones((100, 100, 1), dtype=np.float64)
cv2.threshold(mask, 127, 255, 0)
