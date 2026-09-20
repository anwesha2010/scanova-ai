import streamlit as st
from datetime import datetime


def init_analytics():
    """Initialize analytics state."""
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []


def record_scan(filename, regions, status, level, duration):
    """Record one analysis in the session history."""
    init_analytics()
    st.session_state.scan_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "filename": filename,
        "regions": regions,
        "status": status,
        "level": level,
        "duration": round(duration, 2),
    })


def get_stats():
    """Compute summary statistics."""
    init_analytics()
    history = st.session_state.scan_history

    if not history:
        return {
            "total": 0,
            "avg_time": 0.0,
            "safe": 0,
            "warning": 0,
            "danger": 0,
            "history": [],
        }

    total = len(history)
    avg_time = sum(h["duration"] for h in history) / total
    safe = sum(1 for h in history if h["level"] == "safe")
    warning = sum(1 for h in history if h["level"] == "warning")
    danger = sum(1 for h in history if h["level"] == "danger")

    return {
        "total": total,
        "avg_time": round(avg_time, 2),
        "safe": safe,
        "warning": warning,
        "danger": danger,
        "history": history,
    }