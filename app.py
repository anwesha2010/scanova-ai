import streamlit as st
from PIL import Image
import numpy as np
import io

from backend.preprocess import preprocess
from backend.analyzer import detect_anomalies, find_anomaly_regions
from backend.heatmap import generate_heatmap, draw_boxes
from backend.report import generate_report


# ---------- Page Setup ----------
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
        min_value=0.3, max_value=0.9, value=0.85, step=0.05
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

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

col1, col2 = st.columns([3, 1])

with col1:
    uploaded = st.file_uploader(
        "Supported: PNG, JPG, JPEG",
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

        with st.spinner("Analyzing image..."):
            original, enhanced, normalized = preprocess(file_bytes)
            anomaly_map = detect_anomalies(enhanced, block_size=block_size)
            regions = find_anomaly_regions(anomaly_map, threshold=sensitivity)

            raw_heat, overlay = generate_heatmap(original, anomaly_map)
            if show_boxes:
                overlay = draw_boxes(overlay, regions)

            report = generate_report(anomaly_map, regions, threshold=sensitivity)

        st.session_state.original = original
        st.session_state.processed = enhanced
        st.session_state.heatmap = raw_heat
        st.session_state.overlay = overlay
        st.session_state.regions = regions
        st.session_state.report = report

    # ---------- Display Results ----------
    if "original" in st.session_state and st.session_state.original is not None:
        st.divider()
        st.markdown("### 🖼️ Analysis Results")

        mobile_view = st.toggle("📱 Mobile View (stack images)", value=False)

        if mobile_view:
            st.image(
                st.session_state.original,
                caption="① Original (resized)",
                use_container_width=True
            )
            st.image(
                st.session_state.processed,
                caption="② Preprocessed (CLAHE)",
                use_container_width=True
            )
            st.image(
                st.session_state.heatmap,
                caption="③ Anomaly Heatmap",
                use_container_width=True
            )
            st.image(
                st.session_state.overlay,
                caption="④ Detection Overlay",
                use_container_width=True
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    st.session_state.original,
                    caption="① Original (resized)",
                    use_container_width=True
                )
                st.image(
                    st.session_state.heatmap,
                    caption="③ Anomaly Heatmap",
                    use_container_width=True
                )
            with c2:
                st.image(
                    st.session_state.processed,
                    caption="② Preprocessed (CLAHE)",
                    use_container_width=True
                )
                st.image(
                    st.session_state.overlay,
                    caption="④ Detection Overlay",
                    use_container_width=True
                )

        # ---------- Report ----------
        st.divider()
        st.markdown("### 📋 Analysis Report")
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
        buf = io.BytesIO()
        Image.fromarray(st.session_state.overlay).save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Result (PNG)",
            buf.getvalue(),
            "scanova_result.png",
            "image/png"
        )

        try:
            from backend.pdf_report import generate_pdf_report
            pdf_bytes = generate_pdf_report(
                st.session_state.report,
                st.session_state.overlay
            )
            st.download_button(
                "📄 Download PDF Report",
                pdf_bytes,
                "scanova_report.pdf",
                "application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF generation unavailable: {e}")