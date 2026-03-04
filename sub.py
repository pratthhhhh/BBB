import cv2
import numpy as np
import matplotlib.pyplot as plt

ch01_path = ""
ch03_path = ""
out_path  = ""

# Read as grayscale
ch01 = cv2.imread(ch01_path, cv2.IMREAD_GRAYSCALE)
ch03 = cv2.imread(ch03_path, cv2.IMREAD_GRAYSCALE)
if ch01 is None: raise FileNotFoundError(ch01_path)
if ch03 is None: raise FileNotFoundError(ch03_path)

# Ensure same size
if ch01.shape != ch03.shape:
    ch03 = cv2.resize(ch03, (ch01.shape[1], ch01.shape[0]), interpolation=cv2.INTER_NEAREST)

# Subtract: ch03 - ch01 (clip negatives)
diff = ch03.astype(np.int16) - ch01.astype(np.int16)
diff = np.clip(diff, 0, 255).astype(np.uint8)

# Make fluorescent red output (only red channel)
out_bgr = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
out_bgr[..., 2] = diff

# Save
cv2.imwrite(out_path, out_bgr)
print("Saved:", out_path)