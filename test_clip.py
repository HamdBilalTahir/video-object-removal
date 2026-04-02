import cv2
import numpy as np
result = np.ones((100, 100, 3), dtype=np.float64) * 100
result2 = np.clip(result, 0, 255).astype(np.uint8)
print(result2.dtype)
cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
