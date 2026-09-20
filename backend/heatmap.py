import cv2
import numpy as np


def generate_heatmap(original_gray, anomaly_map, alpha=0.55):
    """
    Generate a smoother heatmap with reduced blockiness.
    Uses strong Gaussian blur to smooth the anomaly map.
    """
    # Resize anomaly map to match original resolution
    anomaly_map = cv2.resize(
        anomaly_map,
        (original_gray.shape[1], original_gray.shape[0])
    )

    # Strong Gaussian blur to smooth the blocky artifacts
    smooth_map = cv2.GaussianBlur(anomaly_map.astype(np.float32),
                                   (31, 31), 0)

    # Normalize back to 0-1
    smooth_map = cv2.normalize(smooth_map, None, 0, 1, cv2.NORM_MINMAX)

    # Convert original grayscale to BGR
    original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

    # Apply JET colormap
    heat_raw = cv2.applyColorMap(
        (smooth_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # Blend heatmap with original image
    mask = (smooth_map > 0.55).astype(np.float32)[..., None]
    overlay = (original_rgb * (1 - alpha * mask) +
               heat_raw * (alpha * mask)).astype(np.uint8)

    return heat_raw, overlay


def draw_boxes(image, regions, color=(255, 140, 0), thickness=2):
    """
    Draw colored rectangles around each region.
    Orange color, thickness 2, with white labels.
    """
    out = image.copy()
    for idx, (x, y, w, h) in enumerate(regions, 1):
        # Draw rectangle
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

        # Draw label background
        label = f"R{idx}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            out,
            (x, max(y - th - 8, 0)),
            (x + tw + 10, max(y, th + 8)),
            color,
            -1
        )

        # Draw label text (white on orange)
        cv2.putText(
            out,
            label,
            (x + 5, max(y - 4, th + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    return out