import cv2
import numpy as np
import matplotlib.pyplot as plt

img_rgb = out_rgb.copy()

mask = (img_rgb[..., 1] > 0).astype(np.uint8)  # green channel > 0
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

min_area = 20  

boxed = img_rgb.copy()
for lab in range(1, num_labels):  # 0 = background
    x = stats[lab, cv2.CC_STAT_LEFT]
    y = stats[lab, cv2.CC_STAT_TOP]
    w = stats[lab, cv2.CC_STAT_WIDTH]
    h = stats[lab, cv2.CC_STAT_HEIGHT]
    area = stats[lab, cv2.CC_STAT_AREA]

    if area < min_area:
        continue

    # rectangle in RGB (red box)
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.putText(boxed, f"{lab}", (x, max(0, y-5)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

plt.figure(figsize=(7,7))
plt.imshow(boxed)
plt.axis("off")
plt.title(f"Bounding boxes (area >= {min_area})")
plt.show()

# print("Total segments (excluding background):", num_labels - 1)
# print("Segments shown (area >= min_area):", int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= min_area)))
