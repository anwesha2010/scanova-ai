def generate_report(anomaly_map, regions, threshold=0.6):
    """Create a clean text report."""
    high_risk = (anomaly_map > threshold).sum()
    total = anomaly_map.size
    risk_pct = (high_risk / total) * 100

    if len(regions) == 0:
        status = "✅ No significant unusual regions detected"
        level = "safe"
    elif len(regions) <= 3:
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
                       "made by a qualified radiologist."),
    }


def patient_explanation(report):
    """Return a plain-language explanation of the result."""
    level = report.get("level", "safe")
    regions = report.get("regions_found", 0)

    if level == "safe":
        return (
            "The AI did not find any regions that stood out as unusual. "
            "This is a reassuring result — but it does NOT rule out every "
            "possible issue. Only a qualified professional can confirm "
            "that the image is normal."
        )
    elif level == "warning":
        return (
            f"The AI noticed {regions} area(s) that look slightly "
            "different from typical images. This does NOT mean something "
            "is wrong. It could be normal anatomy, image quality, or a "
            "real finding. A qualified healthcare professional should "
            "review the image."
        )
    else:
        return (
            f"The AI flagged {regions} area(s) that it could not "
            "interpret as typical. This is a signal for professional "
            "review — NOT a diagnosis. Many flagged regions turn out to "
            "be normal. Please have a qualified healthcare professional "
            "examine the image."
        )


def clinical_explanation(report):
    """Return a technical description for clinical mode."""
    level = report.get("level", "safe")
    regions = report.get("regions_found", 0)
    affected = report.get("affected_area_percent", 0)

    return (
        f"Model output: {level.upper()} | "
        f"Regions flagged: {regions} | "
        f"Affected area: {affected:.2f}% | "
        f"Method: autoencoder reconstruction error | "
        f"Post-processing: threshold + morphological filtering"
    )