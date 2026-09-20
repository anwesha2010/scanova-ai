import streamlit as st
from PIL import Image
import numpy as np
import io

from backend.preprocess import preprocess
from backend.analyzer import detect_anomalies, find_anomaly_regions
from backend.heatmap import generate_heatmap, draw_boxes
from backend.report import generate_report

# ---------- Setup ----------
st.set_page_config(
    page_title="SCANOVA AI",
    page_icon="🩻",
    layout="wide"
)

# ---------- Header ----------
st.title("🩻 SCANOVA AI")
st.caption("AI-Powered Medical Image Analysis Assistant")

st.warning(
    "⚠️ **Assistive tool only.** This does not replace professional "
    "medical diagnosis. Always consult a qualified radiologist."
)

st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")

    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.3, max_value=0.9, value=0.75, step=0.05
    )
    block_size = st.select_slider(
        "Analysis Block Size",
        options=[16, 32, 64], value=32
    )
    show_boxes = st.checkbox("Show bounding boxes", value=True)

    st.divider()
    st.caption("Built with Python • OpenCV • Streamlit")

# ---------- Upload ----------
st.markdown("### 📤 Upload Medical Image")

uploaded_file = st.file_uploader(
    "Supported: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"]
)

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

        with st.spinner("Analyzing image..."):
            # 1. Preprocess
            original, enhanced, normalized = preprocess(file_bytes)

            # 2. Detect anomalies
            anomaly_map = detect_anomalies(enhanced, block_size=block_size)
            regions = find_anomaly_regions(anomaly_map, threshold=sensitivity)

            # 3. Generate heatmap
            raw_heat, overlay = generate_heatmap(original, anomaly_map)
            if show_boxes:
                overlay = draw_boxes(overlay, regions)

            # 4. Report
            report = generate_report(anomaly_map, regions,
                                     threshold=sensitivity)

        # Save to session state
        st.session_state.original = original
        st.session_state.processed = enhanced
        st.session_state.heatmap = raw_heat
        st.session_state.overlay = overlay
        st.session_state.regions = regions
        st.session_state.report = report

    # ---------- Display ----------
    if "original" in st.session_state and st.session_state.original is not None:
        st.divider()
        st.markdown("### 🖼️ Analysis Results")

        col1, col2 = st.columns(2)
        with col1:
                        st.image(st.session_state.original,
                     caption="① Original (resized)",
                     use_container_width=True)
            st.image(st.session_state.heatmap,
                     caption="③ Anomaly Heatmap",
                     use_container_width=True)
                with col1:
            st.image(st.session_state.original,
                     caption="① Original (resized)",
                     use_container_width=True)
            st.image(st.session_state.heatmap,
                     caption="③ Anomaly Heatmap",
                     use_container_width=True)
        st.divider()
        st.markdown("### 📋 Analysis Report")
        r = st.session_state.report

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

        # Download button
        buf = io.BytesIO()
        Image.fromarray(st.session_state.overlay).save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Result",
            buf.getvalue(),
            "scanova_result.png",
            "image/png"
        )