import cv2
import numpy as np

msk = np.ones((100, 100), dtype=np.float64)
kernel = np.ones((3, 3), dtype=np.uint8)
try:
    cv2.dilate(msk, kernel, iterations=1)
    print("cv2.dilate works with CV_64F")
except Exception as e:
    print(f"Error: {e}")
