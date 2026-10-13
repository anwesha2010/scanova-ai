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
        "subtitle": "Step 1 of 9 · Welcome",
        "message": (
            "SCANOVA is an AI-powered medical imaging assistant. "
            "We'll walk through the app in about 90 seconds. "
            "Click Next to begin."
        ),
    },
    {
        "id": "upload",
        "title": "Step 1 — Upload Image",
        "subtitle": "Step 2 of 9 · Image Upload",
        "message": (
            "Scroll down and click '🎯 Try Sample' to load a demonstration "
            "chest X-ray. You can also upload your own PNG, JPG, or DICOM file."
        ),
    },
    {
        "id": "quality",
        "title": "Step 2 — Image Quality Check",
        "subtitle": "Step 3 of 9 · Pre-Scan Quality",
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
        "subtitle": "Step 4 of 9 · AI Analysis",
        "message": (
            "Click '🔍 Analyze Image' to run both AI engines. "
            "SCANOVA has a statistical engine AND a deep learning engine. "
            "You can compare them side-by-side."
        ),
    },
    {
        "id": "result",
        "title": "Step 4 — AI Result",
        "subtitle": "Step 5 of 9 · Model Output",
        "message": (
            "Here's the result. The heatmap shows regions the model could "
            "not reconstruct as accurately as the rest of the image. "
            "This is called reconstruction error — it's not a diagnosis."
        ),
    },
    {
        "id": "explain",
        "title": "Step 5 — Explainable AI",
        "subtitle": "Step 6 of 9 · Explainability",
        "message": (
            "SCANOVA explains WHY it flagged each region. "
            "No black box. Judges and doctors can see the reasoning."
        ),
    },
    {
        "id": "patient",
        "title": "Step 6 — Patient View",
        "subtitle": "Step 7 of 9 · Patient Mode",
        "message": (
            "In the sidebar, toggle 'Patient View'. The SAME result is now "
            "explained in plain language anyone can understand. "
            "This is one of our key differentiators."
        ),
    },
    {
        "id": "closing",
        "title": "Thank You",
        "subtitle": "Step 8 of 9 · Closing",
        "message": (
            "SCANOVA does not replace the doctor. It explores how AI can "
            "help us understand medical images and communicate AI-generated "
            "findings more clearly. Thank you for watching."
        ),
    },
    {
        "id": "summary",
        "title": "Scorecard Summary",
        "subtitle": "Step 9 of 9 · Summary",
        "message": (
            "Everything we showed you, in one card. "
            "Screenshot this or scan the QR code to revisit the demo."
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


def render_judge_summary_card():
    """Render the final scorecard — printable, screenshot-friendly."""
    if not st.session_state.get("judge_mode", False):
        return

    step = get_current_step()
    if step.get("id") != "summary":
        return

    st.markdown("---")

    card_html = """
<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 16px; padding: 2rem 2.25rem; border-left: 6px solid #0ea5e9; max-width: 860px; margin: 1rem auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.2em; color: #0369a1; text-transform: uppercase; margin-bottom: 0.5rem;">🏆 InnoEx 2026 · SCANOVA Scorecard</div>
<div style="font-size: 1.75rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; margin-bottom: 1.25rem;">SCANOVA 2.0 — The 90-Second Summary</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.5rem;">
<div>
<div style="font-size: 0.7rem; font-weight: 700; color: #0369a1; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;">Problem</div>
<div style="color: #334155; line-height: 1.5;">Radiologists read hundreds of scans a day. AI should assist them — not replace them.</div>
</div>
<div>
<div style="font-size: 0.7rem; font-weight: 700; color: #0369a1; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;">Solution</div>
<div style="color: #334155; line-height: 1.5;">An explainable AI that flags unusual regions and shows exactly where the model looked.</div>
</div>
</div>

<div style="margin-bottom: 1.5rem;">
<div style="font-size: 0.7rem; font-weight: 700; color: #0369a1; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem;">Measured Results (50-image test set)</div>
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;">
<div style="background: white; border-radius: 10px; padding: 0.85rem; text-align: center; border: 1px solid #bae6fd;"><div style="font-size: 1.5rem; font-weight: 800; color: #0ea5e9;">74.0%</div><div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">Accuracy</div></div>
<div style="background: white; border-radius: 10px; padding: 0.85rem; text-align: center; border: 1px solid #bae6fd;"><div style="font-size: 1.5rem; font-weight: 800; color: #0ea5e9;">100.0%</div><div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">Recall</div></div>
<div style="background: white; border-radius: 10px; padding: 0.85rem; text-align: center; border: 1px solid #bae6fd;"><div style="font-size: 1.5rem; font-weight: 800; color: #0ea5e9;">65.8%</div><div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">Precision</div></div>
<div style="background: white; border-radius: 10px; padding: 0.85rem; text-align: center; border: 1px solid #bae6fd;"><div style="font-size: 1.5rem; font-weight: 800; color: #0ea5e9;">79.4%</div><div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">F1 Score</div></div>
</div>
<div style="font-size: 0.85rem; color: #475569; margin-top: 0.6rem;"><strong>100% recall by design.</strong> In medical screening, missing a sick patient is worse than a false alarm.</div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.25rem;">
<div>
<div style="font-size: 0.7rem; font-weight: 700; color: #16a34a; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;">✓ What's Built</div>
<ul style="color: #334155; line-height: 1.7; padding-left: 1.1rem; margin: 0; font-size: 0.9rem;">
<li>4 specialized ONNX models</li>
<li>Real Grad-CAM explainability</li>
<li>DICOM + PNG + JPG support</li>
<li>Patient-friendly language view</li>
<li>PDF report with QR code</li>
<li>Clinician feedback loop</li>
</ul>
</div>
<div>
<div style="font-size: 0.7rem; font-weight: 700; color: #dc2626; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;">⚠ Honest Limitations</div>
<ul style="color: #334155; line-height: 1.7; padding-left: 1.1rem; margin: 0; font-size: 0.9rem;">
<li>Not clinically validated</li>
<li>Pediatric chest X-ray data only</li>
<li>74% accuracy — not clinical-grade</li>
<li>Grad-CAM local only</li>
</ul>
</div>
</div>

<div style="border-top: 1px solid #bae6fd; padding-top: 1rem; font-size: 0.85rem; color: #64748b; text-align: center;">Live demo: <a href="https://scanova-ai.streamlit.app" style="color: #0369a1; font-weight: 600;">scanova-ai.streamlit.app</a> · Built by Anwesha Rout for InnoEx 2026</div>
</div>
"""

    st.html(card_html)