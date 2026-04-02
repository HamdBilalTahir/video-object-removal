import torch
import numpy as np
from iopaint.model import LaMa
from iopaint.schema import InpaintRequest, HDStrategy

lama = LaMa(device=torch.device("cpu"))
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:,:,0] = 255 
msk = np.zeros((100, 100), dtype=np.uint8)
req = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL, hd_strategy_crop_trigger_size=800, hd_strategy_crop_margin=32, hd_strategy_resize_limit=1280)
res = lama(img, msk, req)
print("Result shape:", res.shape)
print("Result color (should be 255,0,0 if RGB matching):", res[0,0])