"""
Explainability Layer
====================
Provides honest, non-fabricated explanations of what the AI did.
Since our models are autoencoders (not classifiers with Grad-CAM),
we explain reconstruction-error based detection honestly.
"""

import numpy as np
import cv2


def find_top_hotspots(anomaly_map, k=3):
    """Find top-k hotspot coordinates in the anomaly map."""
    if anomaly_map.size == 0:
        return []

    h, w = anomaly_map.shape
    small = cv2.resize(anomaly_map.astype(np.float32), (32, 32),
                       interpolation=cv2.INTER_AREA)

    hotspots = []
    work = small.copy()

    for _ in range(k):
        idx = np.unravel_index(np.argmax(work), work.shape)
        y_s, x_s = idx
        y = int(y_s / 32 * h)
        x = int(x_s / 32 * w)
        score = float(small[y_s, x_s])

        if score < 0.3:
            break

        hotspots.append({"x": x, "y": y, "score": score})

        y0 = max(0, y_s - 4)
        y1 = min(32, y_s + 4)
        x0 = max(0, x_s - 4)
        x1 = min(32, x_s + 4)
        work[y0:y1, x0:x1] = 0

    return hotspots


def explain_reconstruction_error(anomaly_map, regions, modality="chest_xray"):
    """
    Explain the anomaly map in terms of reconstruction error.
    This is HONEST: our model reconstructs images and flags regions
    it couldn't reconstruct well.
    """
    if anomaly_map is None or anomaly_map.size == 0:
        return {
            "headline": "No analysis available.",
            "detail": "The model did not produce a valid anomaly map.",
            "top_hotspots": [],
        }

    n_regions = len(regions) if regions else 0

    if n_regions == 0:
        headline = "The model did not find regions of concern."
        detail = (
            "The autoencoder reconstructed this image with low error, "
            "which suggests it resembles the normal training data."
        )
    elif n_regions <= 2:
        headline = f"The model found {n_regions} region(s) of possible interest."
        detail = (
            "These are areas the model could not reconstruct as accurately "
            "as the rest of the image. This MAY indicate a structural "
            "difference — or may be due to image quality."
        )
    else:
        headline = f"The model flagged {n_regions} region(s) for review."
        detail = (
            "Multiple regions showed higher reconstruction error. This can "
            "indicate structural differences OR simply differences in the "
            "image compared to the training data (age, equipment, exposure)."
        )

    modality_notes = {
        "chest_xray": (
            "Common sources of reconstruction error in chest X-rays include "
            "rib shadows, cardiac silhouette, and image exposure."
        ),
        "brain_mri": (
            "Common sources of reconstruction error in brain MRI include "
            "ventricle size, scan slice position, and contrast differences."
        ),
        "chest_ct": (
            "Common sources of reconstruction error in chest CT include "
            "breathing motion, slice thickness, and contrast phase."
        ),
        "bone_scan": (
            "Common sources of reconstruction error in bone scans include "
            "positioning angle and exposure differences."
        ),
    }

    top_hotspots = find_top_hotspots(anomaly_map, k=3)

    return {
        "headline": headline,
        "detail": detail,
        "modality_note": modality_notes.get(modality, ""),
        "n_regions": n_regions,
        "top_hotspots": top_hotspots,
        "method": "Autoencoder reconstruction error",
        "honest_note": (
            "This is NOT attention-based. Our model does not use Grad-CAM. "
            "This is honest reconstruction-error analysis."
        ),
    }