def generate_report(anomaly_map, regions, threshold=0.6):
    """Create a simple text report."""
    high_risk = (anomaly_map > threshold).sum()
    total = anomaly_map.size
    risk_pct = (high_risk / total) * 100

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