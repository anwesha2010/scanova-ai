import cv2
import numpy as np

def generate_heatmap(original_gray, anomaly_map, alpha=0.55):
    """
    Turn anomaly_map into a colored heatmap blended on the original.
    Returns (raw_heatmap, blended_overlay).
    """
    original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

    heat_raw = cv2.applyColorMap(
        (anomaly_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # Only show color where anomaly is strong
    mask = (anomaly_map > 0.65).astype(np.float32)[..., None]
    overlay = (original_rgb * (1 - alpha * mask) +
               heat_raw * (alpha * mask)).astype(np.uint8)

    return heat_raw, overlay


def draw_boxes(image, regions):
    """Draw red rectangles around each region."""
    out = image.copy()
    for idx, (x, y, w, h) in enumerate(regions, 1):
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(out, f"R{idx}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out