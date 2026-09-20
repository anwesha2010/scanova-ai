def generate_report(anomaly_map, regions, threshold=0.6):
    """Create a clean text report."""

    if regions:
        # Average anomaly strength INSIDE the detected regions
        total_pixel_score = 0
        total_pixels = 0
        for (x, y, w, h) in regions:
            region_scores = anomaly_map[y:y+h, x:x+w]
            total_pixel_score += (region_scores > threshold).sum()
            total_pixels += w * h
        # Percent of the box that is "anomalous"
        risk_pct = (total_pixel_score / max(total_pixels, 1)) * 100
    else:
        risk_pct = 0.0

    if len(regions) == 0:
        status = "✅ No significant unusual regions detected"
        level = "safe"
    elif len(regions) <= 2:
        status = f"⚠️ {len(regions)} minor pattern(s) detected"
        level = "warning"
    else:
        status = f"🔴 {len(regions)} region(s) require review"
        level = "danger"

    return {
        "status": status,
        "level": level,
        "regions_found": len(regions),
        "affected_area_percent": round(risk_pct, 2),
        "disclaimer": ("Assistive tool only. Final diagnosis must be "
                       "made by a qualified radiologist.")
    }