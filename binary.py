import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("")
if img_bgr is None:
    raise FileNotFoundError("No input")

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Tuning
lower = np.array([35,  40,  40], dtype=np.uint8)
upper = np.array([90, 255, 255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower, upper)  

out_bgr = np.zeros_like(img_bgr)
out_bgr[mask > 0] = (0, 255, 0) #BGR

# Plot (convert BGR->RGB for matplotlib)
out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,6))
plt.imshow(out_rgb)
plt.axis("off")
plt.title("Binary")
plt.show()
