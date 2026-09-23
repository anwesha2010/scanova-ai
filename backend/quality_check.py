"""
Pre-Scan Image Quality Check
=============================
Validates the uploaded image before AI analysis.
Checks practical things only — does NOT pretend to do
medical-grade quality assessment.
"""

import numpy as np
import cv2


def check_image_quality(image):
    """
    Run practical quality checks on a grayscale image.

    Args:
        image: 2D numpy array (grayscale)

    Returns:
        dict with:
            - checks: list of (label, passed, detail)
            - passed: count
            - total: count
            - overall: "good" | "fair" | "poor"
            - message: human-readable summary
    """
    checks = []
    h, w = image.shape

    # ---- Check 1: Image loaded ----
    checks.append((
        "Image detected",
        image is not None and image.size > 0,
        f"{w} × {h} px"
    ))

    # ---- Check 2: Resolution ----
    min_dim = min(h, w)
    resolution_ok = min_dim >= 224
    checks.append((
        "Resolution check",
        resolution_ok,
        "Good" if resolution_ok else "Low — at least 224×224 recommended"
    ))

    # ---- Check 3: Contrast ----
    p5, p95 = np.percentile(image, [5, 95])
    contrast = float(p95 - p5)
    contrast_ok = contrast > 40
    checks.append((
        "Contrast check",
        contrast_ok,
        f"Dynamic range: {contrast:.0f}/255"
    ))

    # ---- Check 4: Brightness ----
    mean_brightness = float(image.mean())
    brightness_ok = 20 < mean_brightness < 235
    checks.append((
        "Brightness check",
        brightness_ok,
        f"Mean: {mean_brightness:.0f}/255"
    ))

    # ---- Check 5: Sharpness (Laplacian variance) ----
    laplacian_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    sharpness_ok = laplacian_var > 50
    checks.append((
        "Sharpness check",
        sharpness_ok,
        f"Focus score: {laplacian_var:.1f}"
    ))

    # ---- Check 6: Orientation / aspect ratio ----
    aspect = w / max(h, 1)
    aspect_ok = 0.3 < aspect < 3.0
    checks.append((
        "Orientation check",
        aspect_ok,
        f"Aspect ratio: {aspect:.2f}"
    ))

    # ---- Overall assessment ----
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)

    if passed >= total - 1:
        overall = "good"
        message = "✅ Image looks suitable for AI analysis."
    elif passed >= total - 2:
        overall = "fair"
        message = "⚠️ Image quality is acceptable, but results may be less reliable."
    else:
        overall = "poor"
        message = ("❌ This image may not be suitable for reliable AI "
                   "analysis. Please upload a clearer image.")

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "overall": overall,
        "message": message,
    }