import cv2
import numpy as np
msk = np.ones((100, 100), dtype=np.int64)
kernel = np.ones((3, 3), dtype=np.uint8)
cv2.dilate(msk, kernel, iterations=1)
