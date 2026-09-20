def generate_report(anomaly_map, regions, threshold=0.6):
    """Create a clean text report."""
    
    # Calculate affected area from REGIONS only (not raw pixels)
    if regions:
        region_area = sum(w * h for (_, _, w, h) in regions)
        total_area = anomaly_map.shape[0] * anomaly_map.shape[1]
        risk_pct = (region_area / total_area) * 100
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