import streamlit as st
from PIL import Image
import numpy as np
import io
import time

from backend.preprocess import preprocess
from backend.analyzer import detect_anomalies, find_anomaly_regions
from backend.heatmap import generate_heatmap, draw_boxes
from backend.report import generate_report
from utils.analytics import init_analytics, record_scan, get_stats


# ---------- Page Setup ----------
st.set_page_config(
    page_title="SCANOVA AI — Medical Image Analysis",
    page_icon="🩻",
    layout="wide"
)

init_analytics()


# ---------- Load Custom CSS ----------
try:
    with open("branding/styles.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass


# ---------- HERO ----------
st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">AI-POWERED MEDICAL IMAGING</div>
        <h1 class="hero-title">SCANOVA AI</h1>
        <p class="hero-tagline">
            Upload X-ray, MRI, or CT scans. Our AI identifies potentially 
            unusual regions and highlights them with medical-grade heatmaps — 
            faster than manual review.
        </p>
        <p class="hero-team">
            Built by <strong>Anwesha Rout</strong> • 
            <strong>SCANOVA Team</strong> • v2.0
        </p>
    </div>
""", unsafe_allow_html=True)


# ---------- Disclaimer ----------
st.warning(
    "⚠️ **Assistive tool only.** This does not replace professional medical "
    "diagnosis. Always consult a qualified radiologist for clinical decisions."
)

st.divider()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.caption("Tune how sensitive the AI should be.")

    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.3, max_value=0.9, value=0.85, step=0.05,
        help="Higher = more detections (may include false positives)"
    )
    block_size = st.select_slider(
        "Analysis Block Size",
        options=[16, 32, 64], value=32,
        help="Smaller = more detailed, slower"
    )
    show_boxes = st.checkbox("Show bounding boxes", value=True)

    st.divider()
    st.markdown("### 📊 Session Stats")
    try:
        stats = get_stats()
        st.caption(f"Scans this session: **{stats['total']}**")
        if stats["total"] > 0:
            st.caption(f"Avg analysis time: **{stats['avg_time']}s**")
            st.caption(f"✅ {stats['safe']}  ⚠️ {stats['warning']}  🔴 {stats['danger']}")
    except Exception:
        pass

    st.divider()
    st.caption("Python • OpenCV • Streamlit")


# ---------- Upload Section Header ----------
st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">📤</span>
        <div>
            <h2 class="section-header-text">Upload Medical Image</h2>
            <p class="section-header-desc">Supported formats: PNG, JPG, JPEG</p>
        </div>
    </div>
""", unsafe_allow_html=True)


# ---------- Upload ----------
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

col1, col2 = st.columns([3, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded

with col2:
    st.write("")
    sample_btn = st.button(
        "🎯 Try Sample",
        use_container_width=True,
        help="Load a sample chest X-ray"
    )

if sample_btn:
    try:
        with open("assets/sample_xray.png", "rb") as f:
            sample_bytes = f.read()
        sample_io = io.BytesIO(sample_bytes)
        sample_io.name = "sample_xray.png"
        st.session_state.uploaded_file = sample_io
        st.rerun()
    except FileNotFoundError:
        st.error("Sample image not found. Please upload your own.")

uploaded_file = st.session_state.uploaded_file

analyze_btn = st.button(
    "🔍 Analyze Image",
    type="primary",
    disabled=uploaded_file is None
)


# ---------- Process ----------
if uploaded_file is not None:
    st.success(f"✅ Loaded: **{uploaded_file.name}**")

    if analyze_btn:
        file_bytes = uploaded_file.read()
        start_time = time.time()

        with st.spinner("Analyzing image..."):
            original, enhanced, normalized = preprocess(file_bytes)
            anomaly_map = detect_anomalies(enhanced, block_size=block_size)
            regions = find_anomaly_regions(anomaly_map, threshold=sensitivity)

            raw_heat, overlay = generate_heatmap(original, anomaly_map)
            if show_boxes:
                overlay = draw_boxes(overlay, regions)

            report = generate_report(anomaly_map, regions, threshold=sensitivity)

        analysis_duration = time.time() - start_time

        st.session_state.original = original
        st.session_state.processed = enhanced
        st.session_state.heatmap = raw_heat
        st.session_state.overlay = overlay
        st.session_state.regions = regions
        st.session_state.report = report

        record_scan(
            filename=uploaded_file.name,
            regions=report["regions_found"],
            status=report["status"],
            level=report["level"],
            duration=analysis_duration
        )

    # ---------- Display Results ----------
    if "original" in st.session_state and st.session_state.original is not None:
        st.divider()

        st.markdown("""
            <div class="section-header">
                <span class="section-header-icon">🖼️</span>
                <div>
                    <h2 class="section-header-text">Analysis Results</h2>
                    <p class="section-header-desc">Four views of the AI analysis pipeline</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        mobile_view = st.toggle("📱 Mobile View (stack images)", value=False)

        if mobile_view:
            st.image(st.session_state.original,
                     caption="① Original (resized)", use_container_width=True)
            st.image(st.session_state.processed,
                     caption="② Preprocessed (CLAHE)", use_container_width=True)
            st.image(st.session_state.heatmap,
                     caption="③ Anomaly Heatmap", use_container_width=True)
            st.image(st.session_state.overlay,
                     caption="④ Detection Overlay", use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(st.session_state.original,
                         caption="① Original (resized)", use_container_width=True)
                st.image(st.session_state.heatmap,
                         caption="③ Anomaly Heatmap", use_container_width=True)
            with c2:
                st.image(st.session_state.processed,
                         caption="② Preprocessed (CLAHE)", use_container_width=True)
                st.image(st.session_state.overlay,
                         caption="④ Detection Overlay", use_container_width=True)

        # ---------- Report ----------
        st.divider()

        st.markdown("""
            <div class="section-header">
                <span class="section-header-icon">📋</span>
                <div>
                    <h2 class="section-header-text">Analysis Report</h2>
                    <p class="section-header-desc">Summary of detected patterns</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        r = st.session_state.report

        if mobile_view:
            c1, c2 = st.columns(2)
            c1.metric("Regions Found", r["regions_found"])
            c2.metric("Status", r["level"].upper())
            st.metric("Affected Area", f"{r['affected_area_percent']}%")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Regions Found", r["regions_found"])
            c2.metric("Affected Area", f"{r['affected_area_percent']}%")
            c3.metric("Status", r["level"].upper())

        if r["level"] == "safe":
            st.success(r["status"])
        elif r["level"] == "warning":
            st.warning(r["status"])
        else:
            st.error(r["status"])

        st.info(r["disclaimer"])

        # ---------- Downloads ----------
        col_dl1, col_dl2 = st.columns(2)

        buf = io.BytesIO()
        Image.fromarray(st.session_state.overlay).save(buf, format="PNG")
        with col_dl1:
            st.download_button(
                "⬇️ Download PNG",
                buf.getvalue(),
                "scanova_result.png",
                "image/png",
                use_container_width=True
            )

        try:
            from backend.pdf_report import generate_pdf_report
            pdf_bytes = generate_pdf_report(
                st.session_state.report,
                st.session_state.overlay
            )
            with col_dl2:
                st.download_button(
                    "📄 Download PDF Report",
                    pdf_bytes,
                    "scanova_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            with col_dl2:
                st.warning(f"PDF unavailable: {e}")


# ---------- Footer ----------
st.markdown("""
    <div class="app-footer">
        <div class="footer-line">
            <strong>SCANOVA AI</strong> — Medical Image Analysis Assistant
        </div>
        <div class="footer-line">
            Designed & built by the SCANOVA Team
        </div>
        <div>
            <span class="footer-badge">Python</span>
            <span class="footer-badge">OpenCV</span>
            <span class="footer-badge">Streamlit</span>
            <span class="footer-badge">v2.0</span>
        </div>
        <div class="footer-line" style="margin-top: 1rem; font-size: 0.75rem;">
            ⚠️ Not for clinical use. Always consult a qualified radiologist.
        </div>
    </div>
""", unsafe_allow_html=True)