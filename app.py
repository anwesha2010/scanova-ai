import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import base64

from backend.preprocess import preprocess
from backend.analyzer import detect_anomalies, find_anomaly_regions
from backend.heatmap import generate_heatmap, draw_boxes
from backend.report import generate_report
from utils.analytics import init_analytics, record_scan, get_stats

# Deep Learning analyzer (ONNX-based)
try:
    from backend.dl_analyzer import detect_anomalies_dl
    DL_AVAILABLE = True
except Exception:
    DL_AVAILABLE = False
    def detect_anomalies_dl(*args, **kwargs):
        return None

try:
    from streamlit_extras.let_it_rain import rain
    CONFETTI_AVAILABLE = True
except Exception:
    CONFETTI_AVAILABLE = False


# ---------- Page Setup ----------
st.set_page_config(
    page_title="SCANOVA AI — Medical Image Analysis",
    page_icon="branding/favicon.png",
    layout="wide"
)

init_analytics()

# Initialize session state for settings
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "⚡ Statistical (fast)"
if "modality_choice" not in st.session_state:
    st.session_state.modality_choice = "🌐 Auto-detect"
if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = 0.65
if "block_size" not in st.session_state:
    st.session_state.block_size = 32
if "show_boxes" not in st.session_state:
    st.session_state.show_boxes = True

if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "stat_results" not in st.session_state:
    st.session_state.stat_results = None
if "dl_results" not in st.session_state:
    st.session_state.dl_results = None
if "original" not in st.session_state:
    st.session_state.original = None


# ---------- Load Custom CSS ----------
try:
    with open("branding/styles.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass


# ---------- Top Announcement Banner ----------
st.markdown(
    '<div class="top-banner">'
    '<div class="top-banner-content">'
    '<span class="top-banner-pill">NEW</span>'
    '<span class="top-banner-text">'
    '<span class="gradient-text">Revolutionizing Medical Imaging</span> '
    'with <strong>AI-Powered X-ray Analysis</strong>'
    '</span>'
    '<span class="top-banner-arrow">→</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# HERO SECTION
# ==========================================================
hero_stats = get_stats()

try:
    with open("branding/hero_image.png", "rb") as f:
        hero_img_b64 = base64.b64encode(f.read()).decode()
    hero_img_tag = (
        f'<div class="hero-image-wrap">'
        f'<img src="data:image/png;base64,{hero_img_b64}" alt="SCANOVA AI hero" />'
        f'<div class="hero-image-badge">Live scan preview</div>'
        f'</div>'
    )
except FileNotFoundError:
    hero_img_tag = ""

st.markdown(f"""
<div class="hero-split">
<div class="hero-left">
<div class="hero-eyebrow"><span class="hero-eyebrow-dot"></span>BUILT BY STUDENTS · DEPLOYED TO PRODUCTION</div>
<h1 class="hero-headline">AI that sees <em>what<br>tired eyes miss.</em></h1>
<p class="hero-sub">SCANOVA AI analyzes X-rays, MRIs, and CT scans in seconds — flagging potentially unusual regions with medical-grade heatmaps. <strong>Not a replacement for radiologists. A second pair of eyes.</strong></p>
<div class="hero-ctas"><a href="#upload" class="cta-primary">Analyze a scan →</a><a href="/About" target="_self" class="cta-secondary">Read our story</a></div>
<div class="trust-strip"><span class="trust-item"><strong>Python</strong></span><span class="trust-item"><strong>OpenCV</strong></span><span class="trust-item"><strong>ONNX</strong></span><span class="trust-item">DICOM-ready</span><span class="trust-item">MIT Licensed</span><span class="trust-item">v3.0 · InnoEx 2026</span></div>
</div>
<div class="hero-right">{hero_img_tag}</div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# MODALITY GRID
# ==========================================================
st.markdown(
    '<div class="modality-section">'
    '<div class="modality-eyebrow">SUPPORTED MODALITIES</div>'
    '<h2 class="modality-title">One tool. <em>Four scan types.</em></h2>'
    '<p class="modality-desc">SCANOVA ships with four specialized neural networks — one per modality — plus an auto-router that picks the right one.</p>'
    '</div>',
    unsafe_allow_html=True
)

mod_col1, mod_col2, mod_col3, mod_col4 = st.columns(4)

cards = [
    ("c1", "01 / 04", "🩻", "Chest X-rays",
     "Production-ready. Trained on 500 frontal chest X-rays. Detects density variations in lung fields."),
    ("c2", "02 / 04", "🧠", "Brain MRI",
     "Trained on 500 normal brain MRIs. Flags asymmetric regions and abnormal signal patterns."),
    ("c3", "03 / 04", "🫁", "Chest CT",
     "Trained on 100 chest CT slices. Highlights structural irregularities in lung tissue."),
    ("c4", "04 / 04", "🦴", "Bone Scan",
     "Trained on 500 bone X-rays. Surfaces density discontinuities in long bones and joints."),
]

for col, (cls, num, icon, title, desc) in zip(
    [mod_col1, mod_col2, mod_col3, mod_col4], cards
):
    with col:
        st.markdown(
            f'<div class="modality-card {cls}">'
            f'<span class="modality-card-number">{num}</span>'
            f'<span class="modality-card-icon">{icon}</span>'
            f'<h3 class="modality-card-title">{title}</h3>'
            f'<p class="modality-card-desc">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True
        )


# ==========================================================
# SETTINGS — SIDEBAR (desktop) + MOBILE EXPANDER (mobile)
# ============================================================

# ---------- Sidebar (desktop) ----------
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.caption("Tune how sensitive the AI should be.")

    if DL_AVAILABLE:
        analysis_mode = st.radio(
            "AI Engine",
            options=[
                "⚡ Statistical (fast)",
                "🧠 Deep Learning (accurate)",
                "⚖️ Compare Both"
            ],
            index=0,
            help="Compare mode runs both engines side-by-side.",
            key="sidebar_engine"
        )
        st.session_state.analysis_mode = analysis_mode

        if "Deep Learning" in analysis_mode or "Compare" in analysis_mode:
            modality_choice = st.selectbox(
                "Modality",
                options=[
                    "🌐 Auto-detect",
                    "🩻 Chest X-ray",
                    "🧠 Brain MRI",
                    "🫁 Chest CT",
                    "🦴 Bone Scan"
                ],
                index=0,
                help="Auto-detect uses the trained modality classifier.",
                key="sidebar_modality"
            )
        else:
            modality_choice = st.session_state.modality_choice
        st.session_state.modality_choice = modality_choice
    else:
        analysis_mode = st.session_state.analysis_mode
        modality_choice = st.session_state.modality_choice

    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.3, max_value=0.9, value=0.65, step=0.05,
        help="Higher = more detections (may include false positives)",
        key="sidebar_sensitivity"
    )
    st.session_state.sensitivity = sensitivity

    block_size = st.select_slider(
        "Analysis Block Size",
        options=[16, 32, 64], value=32,
        help="Smaller = more detailed, slower",
        key="sidebar_block"
    )
    st.session_state.block_size = block_size

    show_boxes = st.checkbox("Show bounding boxes", value=True,
                              key="sidebar_boxes")
    st.session_state.show_boxes = show_boxes

    st.divider()
    st.markdown("### 📊 Session Stats")
    try:
        sidebar_stats = get_stats()
        st.caption(f"Scans this session: **{sidebar_stats['total']}**")
        if sidebar_stats["total"] > 0:
            st.caption(f"Avg analysis time: **{sidebar_stats['avg_time']}s**")
            st.caption(
                f"✅ {sidebar_stats['safe']}  "
                f"⚠️ {sidebar_stats['warning']}  "
                f"🔴 {sidebar_stats['danger']}"
            )
    except Exception:
        pass

    st.divider()
    st.caption("Python • OpenCV • ONNX • Streamlit")


# ---------- Mobile-only settings expander ----------
st.markdown('<div class="mobile-settings">', unsafe_allow_html=True)

with st.expander("⚙️ Analysis Settings (tap to open)", expanded=False):
    st.caption("Same settings as the sidebar — for mobile use.")

    if DL_AVAILABLE:
        analysis_mode_m = st.radio(
            "AI Engine",
            options=[
                "⚡ Statistical (fast)",
                "🧠 Deep Learning (accurate)",
                "⚖️ Compare Both"
            ],
            index=0,
            key="mobile_engine"
        )
        # Sync to session state
        st.session_state.analysis_mode = analysis_mode_m
        analysis_mode = analysis_mode_m

        if "Deep Learning" in analysis_mode_m or "Compare" in analysis_mode_m:
            modality_choice_m = st.selectbox(
                "Modality",
                options=[
                    "🌐 Auto-detect",
                    "🩻 Chest X-ray",
                    "🧠 Brain MRI",
                    "🫁 Chest CT",
                    "🦴 Bone Scan"
                ],
                index=0,
                key="mobile_modality"
            )
            st.session_state.modality_choice = modality_choice_m
            modality_choice = modality_choice_m

    sensitivity_m = st.slider(
        "Detection Sensitivity",
        min_value=0.3, max_value=0.9, value=0.65, step=0.05,
        key="mobile_sensitivity"
    )
    st.session_state.sensitivity = sensitivity_m
    sensitivity = sensitivity_m

    block_size_m = st.select_slider(
        "Analysis Block Size",
        options=[16, 32, 64], value=32,
        key="mobile_block"
    )
    st.session_state.block_size = block_size_m
    block_size = block_size_m

    show_boxes_m = st.checkbox("Show bounding boxes", value=True,
                                key="mobile_boxes")
    st.session_state.show_boxes = show_boxes_m
    show_boxes = show_boxes_m

st.markdown('</div>', unsafe_allow_html=True)


# ---------- Disclaimer + Live Stats Row ----------
col_left, col_right = st.columns([3, 1])

with col_left:
    st.warning(
        "⚠️ **Assistive tool only.** This does not replace professional medical "
        "diagnosis. Always consult a qualified radiologist for clinical decisions."
    )

with col_right:
    st.markdown(f"""
<div class="live-card">
<div class="live-card-header"><span class="live-card-title">Live Session</span><span class="live-pulse">Active</span></div>
<div class="live-stat-row"><span class="live-stat-label">Scans this session</span><span class="live-stat-value accent">{hero_stats['total']}</span></div>
<div class="live-stat-row"><span class="live-stat-label">Avg analysis time</span><span class="live-stat-value">{hero_stats['avg_time']}s</span></div>
<div class="live-stat-row"><span class="live-stat-label">Safe / Warn / Danger</span><span class="live-stat-value" style="font-size: 1rem;">{hero_stats['safe']} / {hero_stats['warning']} / {hero_stats['danger']}</span></div>
</div>
""", unsafe_allow_html=True)


# ---------- Upload Section ----------
st.markdown(
    '<a id="upload"></a>'
    '<div class="section-header">'
    '<span class="section-header-icon">📤</span>'
    '<div><h2 class="section-header-text">Upload Medical Image</h2>'
    '<p class="section-header-desc">Supported formats: PNG · JPG · DICOM (.dcm)</p></div>'
    '</div>',
    unsafe_allow_html=True
)


# ---------- Upload ----------
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

col1, col2 = st.columns([3, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        label_visibility="collapsed"
    )
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded

with col2:
    st.write("")
    sample_btn = st.button(
        "🎯 Try Sample",
        use_container_width=True,
        help="Load a sample chest X-ray (auto-analyzes)"
    )

if sample_btn:
    try:
        with open("assets/sample_xray.png", "rb") as f:
            sample_bytes = f.read()
        sample_io = io.BytesIO(sample_bytes)
        sample_io.name = "sample_xray.png"
        st.session_state.uploaded_file = sample_io
        st.session_state.auto_analyze = True
        st.rerun()
    except FileNotFoundError:
        st.error("Sample image not found. Please upload your own.")

uploaded_file = st.session_state.uploaded_file

analyze_btn = st.button(
    "🔍 Analyze Image",
    type="primary",
    disabled=uploaded_file is None
)

if st.session_state.get("auto_analyze", False) and uploaded_file is not None:
    analyze_btn = True
    st.session_state.auto_analyze = False


# ---------- Process ----------
if uploaded_file is not None:
    st.success(f"✅ Loaded: **{uploaded_file.name}**")

    if analyze_btn:
        file_bytes = uploaded_file.read()
        start_time = time.time()

        with st.spinner(f"Analyzing with {analysis_mode}..."):
            original, enhanced, normalized = preprocess(file_bytes)

            modality_map = {
                "🌐 Auto-detect": "auto",
                "🩻 Chest X-ray": "chest_xray",
                "🧠 Brain MRI":  "brain_mri",
                "🫁 Chest CT":   "chest_ct",
                "🦴 Bone Scan":  "bone_scan",
            }
            selected_modality = modality_map.get(modality_choice, "auto")

            # ---- Statistical ----
            stat_start = time.time()
            anomaly_map_stat = detect_anomalies(enhanced, block_size=block_size)
            regions_stat = find_anomaly_regions(anomaly_map_stat, threshold=sensitivity)
            heat_stat, overlay_stat = generate_heatmap(original, anomaly_map_stat)
            if show_boxes:
                overlay_stat = draw_boxes(overlay_stat, regions_stat)
            report_stat = generate_report(anomaly_map_stat, regions_stat, threshold=sensitivity)
            stat_time = time.time() - stat_start

            # ---- Deep Learning ----
            anomaly_map_dl = None
            regions_dl = []
            heat_dl = None
            overlay_dl = None
            report_dl = None
            dl_time = 0

            if DL_AVAILABLE and ("Deep Learning" in analysis_mode or "Compare" in analysis_mode):
                dl_start = time.time()
                anomaly_map_dl = detect_anomalies_dl(enhanced, modality=selected_modality)
                if anomaly_map_dl is not None:
                    regions_dl = find_anomaly_regions(anomaly_map_dl, threshold=sensitivity)
                    heat_dl, overlay_dl = generate_heatmap(original, anomaly_map_dl)
                    if show_boxes:
                        overlay_dl = draw_boxes(overlay_dl, regions_dl)
                    report_dl = generate_report(anomaly_map_dl, regions_dl, threshold=sensitivity)
                dl_time = time.time() - dl_start

            # ---- Primary result ----
            if "Compare" in analysis_mode and anomaly_map_dl is not None:
                anomaly_map, regions, raw_heat, overlay, report = (
                    anomaly_map_dl, regions_dl, heat_dl, overlay_dl, report_dl
                )
            elif "Deep Learning" in analysis_mode and anomaly_map_dl is not None:
                anomaly_map, regions, raw_heat, overlay, report = (
                    anomaly_map_dl, regions_dl, heat_dl, overlay_dl, report_dl
                )
            else:
                anomaly_map, regions, raw_heat, overlay, report = (
                    anomaly_map_stat, regions_stat, heat_stat, overlay_stat, report_stat
                )

        analysis_duration = time.time() - start_time

        st.session_state.original = original
        st.session_state.processed = enhanced
        st.session_state.heatmap = raw_heat
        st.session_state.overlay = overlay
        st.session_state.regions = regions
        st.session_state.report = report
        st.session_state.analysis_mode = analysis_mode

        st.session_state.compare_mode = "Compare" in analysis_mode
        st.session_state.stat_results = {
            "regions": regions_stat,
            "heat": heat_stat,
            "overlay": overlay_stat,
            "report": report_stat,
            "time": stat_time,
        }
        st.session_state.dl_results = {
            "regions": regions_dl,
            "heat": heat_dl,
            "overlay": overlay_dl,
            "report": report_dl,
            "time": dl_time,
        } if anomaly_map_dl is not None else None

        engine_label = (
            "DL" if "Deep Learning" in analysis_mode
            else ("Compare" if "Compare" in analysis_mode else "Statistical")
        )
        modality_label = selected_modality if selected_modality != "auto" else "auto-detect"

        record_scan(
            filename=uploaded_file.name,
            regions=report["regions_found"],
            status=report["status"],
            level=report["level"],
            duration=analysis_duration,
            modality=modality_label,
            engine=engine_label
        )

    # ---------- Display Results ----------
    if st.session_state.get("original") is not None:

        # ========== COMPARE MODE ==========
        if st.session_state.get("compare_mode", False) and st.session_state.get("dl_results") is not None:
            st.divider()
            st.markdown(
                '<div class="section-header">'
                '<span class="section-header-icon">⚖️</span>'
                '<div><h2 class="section-header-text">Side-by-Side Comparison</h2>'
                '<p class="section-header-desc">Same image, two AI engines</p></div>'
                '</div>',
                unsafe_allow_html=True
            )

            stat = st.session_state.stat_results
            dl = st.session_state.dl_results

            stack_compare = st.toggle("📱 Stack vertically (mobile view)", value=False)

            if stack_compare:
                st.markdown("### ⚡ Statistical Engine")
                st.image(stat["overlay"], use_container_width=True,
                         caption="Detection Overlay")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Regions", len(stat["regions"]))
                c2.metric("Time", f"{stat['time']:.2f}s")
                c3.metric("Affected", f"{stat['report']['affected_area_percent']}%")
                c4.metric("Status", stat["report"]["level"].upper())

                st.divider()

                st.markdown("### 🧠 Deep Learning Engine")
                st.image(dl["overlay"], use_container_width=True,
                         caption="Detection Overlay")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Regions", len(dl["regions"]))
                c2.metric("Time", f"{dl['time']:.2f}s")
                c3.metric("Affected", f"{dl['report']['affected_area_percent']}%")
                c4.metric("Status", dl["report"]["level"].upper())
            else:
                col_stat, col_dl = st.columns(2)

                with col_stat:
                    st.markdown("### ⚡ Statistical Engine")
                    st.image(stat["overlay"], use_container_width=True,
                             caption="Detection Overlay")
                    c1, c2 = st.columns(2)
                    c1.metric("Regions", len(stat["regions"]))
                    c2.metric("Time", f"{stat['time']:.2f}s")
                    c3, c4 = st.columns(2)
                    c3.metric("Affected", f"{stat['report']['affected_area_percent']}%")
                    c4.metric("Status", stat["report"]["level"].upper())

                with col_dl:
                    st.markdown("### 🧠 Deep Learning Engine")
                    st.image(dl["overlay"], use_container_width=True,
                             caption="Detection Overlay")
                    c1, c2 = st.columns(2)
                    c1.metric("Regions", len(dl["regions"]))
                    c2.metric("Time", f"{dl['time']:.2f}s")
                    c3, c4 = st.columns(2)
                    c3.metric("Affected", f"{dl['report']['affected_area_percent']}%")
                    c4.metric("Status", dl["report"]["level"].upper())

            st.info(
                "💡 **Statistical** is fast and rule-based. "
                "**Deep Learning** uses a trained neural network. "
                "Compare the heatmaps and regions to see the difference."
            )

        # ========== SINGLE ENGINE RESULTS ==========
        else:
            st.divider()

            st.markdown(
                '<div class="section-header">'
                '<span class="section-header-icon">🖼️</span>'
                '<div><h2 class="section-header-text">Analysis Results</h2>'
                f'<p class="section-header-desc">Engine: {st.session_state.get("analysis_mode", "Statistical")}</p></div>'
                '</div>',
                unsafe_allow_html=True
            )

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

        st.markdown(
            '<div class="section-header">'
            '<span class="section-header-icon">📋</span>'
            '<div><h2 class="section-header-text">Analysis Report</h2>'
            '<p class="section-header-desc">Summary of detected patterns</p></div>'
            '</div>',
            unsafe_allow_html=True
        )

        r = st.session_state.report

        c1, c2, c3 = st.columns(3)
        c1.metric("Regions Found", r["regions_found"])
        c2.metric("Affected Area", f"{r['affected_area_percent']}%")
        c3.metric("Status", r["level"].upper())

        if r["level"] == "safe":
            st.success(r["status"])
            if CONFETTI_AVAILABLE:
                rain(emoji="✨", font_size=26, falling_speed=6, animation_length=1.2)
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


# ==========================================================
# FAQ SECTION
# ==========================================================
st.divider()

st.markdown(
    '<div class="section-header">'
    '<span class="section-header-icon">❓</span>'
    '<div><h2 class="section-header-text">Frequently Asked Questions</h2>'
    '<p class="section-header-desc">Quick answers. Full FAQ on the dedicated page.</p></div>'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("🩺 Is SCANOVA AI a medical device?"):
    st.markdown(
        "**No.** SCANOVA AI is an assistive research and educational tool. "
        "It is not FDA, MCI, or CE-cleared and must not be used for clinical "
        "decision-making. Always consult a qualified radiologist."
    )

with st.expander("🧠 How does the AI actually work?"):
    st.markdown(
        "SCANOVA runs on **four specialized neural networks** — one per modality "
        "(chest X-ray, brain MRI, chest CT, bone scan). A modality classifier "
        "auto-routes your upload to the right model. The model reconstructs "
        "the image; regions it can't reconstruct well are flagged as anomalies."
    )

with st.expander("⚖️ What is Compare mode?"):
    st.markdown(
        "Compare mode runs both the **Statistical engine** and the "
        "**Deep Learning engine** on the same image, side-by-side. "
        "It's the easiest way to see the difference between classical and "
        "modern AI approaches."
    )

with st.expander("🏥 Does it support DICOM files?"):
    st.markdown(
        "**Yes.** SCANOVA reads **DICOM** (`.dcm`) files — the standard format "
        "used by hospital radiology systems. We handle MONOCHROME1/MONOCHROME2, "
        "Hounsfield units, window/level, and multi-frame images automatically."
    )

with st.expander("🔒 Is my data stored?"):
    st.markdown(
        "**No.** Images are processed in-memory only. Nothing is saved, "
        "logged, or shared. Close the tab and it's gone."
    )

with st.expander("💸 Is it free?"):
    st.markdown(
        "**Yes.** MIT-licensed, free forever. No signup, no limits. "
        "Source code on [GitHub](https://github.com/anwesha2010/scanova-ai)."
    )

st.markdown(
    "📖 **See all 11 questions →** "
    "[Open the full FAQ page](/FAQ)"
)


# ==========================================================
# FOUNDER QUOTE
# ==========================================================
st.markdown(
    '<div class="founder-quote">'
    '"We built SCANOVA because we watched radiologists drown in '
    'hundreds of scans a day. AI shouldn\'t replace them — it should '
    'give them back the time to think."'
    '<span class="attr">— <strong>Anwesha Rout</strong>, Founder · SCANOVA AI</span>'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
    '<div class="app-footer">'
    '<div class="footer-line"><strong>SCANOVA AI</strong> — Medical Image Analysis Assistant</div>'
    '<div class="footer-line">Designed & built by the SCANOVA Team</div>'
    '<div>'
    '<span class="footer-badge">Python</span>'
    '<span class="footer-badge">OpenCV</span>'
    '<span class="footer-badge">ONNX</span>'
    '<span class="footer-badge">DICOM</span>'
    '<span class="footer-badge">Streamlit</span>'
    '<span class="footer-badge">v3.0</span>'
    '</div>'
    '<div class="footer-line" style="margin-top: 1rem; font-size: 0.75rem;">⚠️ Not for clinical use. Always consult a qualified radiologist.</div>'
    '</div>',
    unsafe_allow_html=True
)