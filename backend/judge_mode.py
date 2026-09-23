"""
Judge Mode — Guided Demo Flow
=============================
State machine for the 90-second InnoEx demo.
"""

import streamlit as st


JUDGE_STEPS = [
    {
        "id": "welcome",
        "title": "Welcome to SCANOVA",
        "subtitle": "Step 1 of 8 · Welcome",
        "message": (
            "SCANOVA is an AI-powered medical imaging assistant. "
            "We'll walk through the app in about 90 seconds. "
            "Click Next to begin."
        ),
    },
    {
        "id": "upload",
        "title": "Step 1 — Upload Image",
        "subtitle": "Step 2 of 8 · Image Upload",
        "message": (
            "Scroll down and click '🎯 Try Sample' to load a demonstration "
            "chest X-ray. You can also upload your own PNG, JPG, or DICOM file."
        ),
    },
    {
        "id": "quality",
        "title": "Step 2 — Image Quality Check",
        "subtitle": "Step 3 of 8 · Pre-Scan Quality",
        "message": (
            "SCANOVA validates the image before analysis. The pre-scan "
            "panel checks resolution, contrast, brightness, sharpness, "
            "and orientation. This is responsible AI — we don't analyze "
            "bad inputs."
        ),
    },
    {
        "id": "analyze",
        "title": "Step 3 — AI Analysis",
        "subtitle": "Step 4 of 8 · AI Analysis",
        "message": (
            "Click '🔍 Analyze Image' to run both AI engines. "
            "SCANOVA has a statistical engine AND a deep learning engine. "
            "You can compare them side-by-side."
        ),
    },
    {
        "id": "result",
        "title": "Step 4 — AI Result",
        "subtitle": "Step 5 of 8 · Model Output",
        "message": (
            "Here's the result. The heatmap shows regions the model could "
            "not reconstruct as accurately as the rest of the image. "
            "This is called reconstruction error — it's not a diagnosis."
        ),
    },
    {
        "id": "explain",
        "title": "Step 5 — Explainable AI",
        "subtitle": "Step 6 of 8 · Explainability",
        "message": (
            "SCANOVA explains WHY it flagged each region. "
            "No black box. Judges and doctors can see the reasoning."
        ),
    },
    {
        "id": "patient",
        "title": "Step 6 — Patient View",
        "subtitle": "Step 7 of 8 · Patient Mode",
        "message": (
            "In the sidebar, toggle 'Patient View'. The SAME result is now "
            "explained in plain language anyone can understand. "
            "This is one of our key differentiators."
        ),
    },
    {
        "id": "closing",
        "title": "Thank You",
        "subtitle": "Step 8 of 8 · Closing",
        "message": (
            "SCANOVA does not replace the doctor. It explores how AI can "
            "help us understand medical images and communicate AI-generated "
            "findings more clearly. Thank you for watching."
        ),
    },
]


def init_judge_mode():
    if "judge_mode" not in st.session_state:
        st.session_state.judge_mode = False
    if "judge_step" not in st.session_state:
        st.session_state.judge_step = 0


def judge_next():
    st.session_state.judge_step = min(
        st.session_state.judge_step + 1,
        len(JUDGE_STEPS) - 1,
    )


def judge_prev():
    st.session_state.judge_step = max(st.session_state.judge_step - 1, 0)


def judge_reset():
    st.session_state.judge_step = 0


def judge_exit():
    st.session_state.judge_mode = False
    judge_reset()


def get_current_step():
    idx = st.session_state.get("judge_step", 0)
    idx = min(max(idx, 0), len(JUDGE_STEPS) - 1)
    return JUDGE_STEPS[idx]


def render_judge_banner():
    """Render the judge mode banner at top of the page."""
    if not st.session_state.get("judge_mode", False):
        return

    step = get_current_step()
    total = len(JUDGE_STEPS)
    current = st.session_state.judge_step + 1

    # Progress bar
    progress_pct = int((current / total) * 100)

    banner_html = (
        '<div class="judge-banner">'
        '<div class="judge-banner-top">'
        f'<div class="judge-banner-step">{step["subtitle"]}</div>'
        '<div class="judge-banner-progress-wrap">'
        '<div class="judge-banner-progress-bar">'
        f'<div class="judge-banner-progress-fill" style="width: {progress_pct}%;"></div>'
        '</div>'
        f'<span class="judge-banner-progress-text">{current} / {total}</span>'
        '</div>'
        '</div>'
        f'<div class="judge-banner-title">{step["title"]}</div>'
        f'<div class="judge-banner-message">{step["message"]}</div>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


def render_judge_controls():
    """Render Prev / Next / Exit controls."""
    if not st.session_state.get("judge_mode", False):
        return

    total = len(JUDGE_STEPS)
    current = st.session_state.judge_step

    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

    with col1:
        if st.button("← Prev", disabled=current == 0, key="judge_prev_btn"):
            judge_prev()
            st.rerun()

    with col2:
        if current < total - 1:
            if st.button("Next →", type="primary", key="judge_next_btn"):
                judge_next()
                st.rerun()
        else:
            if st.button("Finish", type="primary", key="judge_finish_btn"):
                judge_exit()
                st.rerun()

    with col3:
        if st.button("✕ Exit", key="judge_exit_btn"):
            judge_exit()
            st.rerun()

    st.markdown("---")