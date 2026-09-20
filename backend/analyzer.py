import numpy as np
import cv2


def detect_anomalies(image, block_size=32):
    """
    Detect anomalies using local brightness deviation.
    Only BRIGHT outliers are flagged.
    """
    img = image.astype(np.float32)
    h, w = img.shape
    anomaly_map = np.zeros_like(img)

    global_mean = img.mean()
    global_std = img.std() + 1e-6

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            local_mean = block.mean()

            # Only flag blocks BRIGHTER than average
            diff = local_mean - global_mean
            if diff > 0:
                score = diff / global_std
            else:
                score = 0

            anomaly_map[i:i+block_size, j:j+block_size] = score

    # Smooth
    anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)
    anomaly_map = cv2.normalize(anomaly_map, None, 0, 1, cv2.NORM_MINMAX)
    return anomaly_map


def find_anomaly_regions(anomaly_map, threshold=0.6, min_area=2500):
    """
    Find bounding boxes around suspicious spots.
    Ignores border noise and small regions.
    """
    h, w = anomaly_map.shape

    # Ignore the border
    border = 60
    masked = anomaly_map.copy()
    masked[:border, :] = 0
    masked[-border:, :] = 0
    masked[:, :border] = 0
    masked[:, -border:] = 0

    # Only strong values
    binary = (masked > threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    regions = [cv2.boundingRect(c) for c in contours
               if cv2.contourArea(c) >= min_area]

    # Filter out giant boxes (over 15% of image)
    img_area = h * w
    regions = [r for r in regions
               if (r[2] * r[3]) < img_area * 0.15]

    return regions