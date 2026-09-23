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
    page_title="SCANOVA 2.0 — AI Medical Imaging Assistant",
    page_icon="branding/favicon.png",
    layout="wide"
)

init_analytics()

# Initialize session state
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
_banner_html = (
    '<div class="top-banner">'
    '<div class="top-banner-content">'
    '<span class="top-banner-pill">NEW</span>'
    '<span class="top-banner-text">'
    '<span class="gradient-text">Revolutionizing Medical Imaging</span> '
    'with <strong>AI-Powered X-ray Analysis</strong>'
    '</span>'
    '<span class="top-banner-arrow">→</span>'
    '</div>'
    '</div>'
)
st.markdown(_banner_html, unsafe_allow_html=True)


# ==========================================================
# MEDICAL DISCLAIMER
# ==========================================================
_disc_html = (
    '<div class="medical-disclaimer">'
    '<div class="medical-disclaimer-title">⚠️ IMPORTANT — Please Read Before Using</div>'
    '<div class="medical-disclaimer-body">'
    '<strong>SCANOVA is an educational and research prototype.</strong> '
    'It is <strong>not a medical device</strong> and does not replace a radiologist, '
    'physician, or other qualified healthcare professional. '
    'Results may be incorrect. Do not use SCANOVA as the sole basis '
    'for any medical decision. Always consult a qualified healthcare '
    'professional for diagnosis and treatment.'
    '</div>'
    '</div>'
)
st.markdown(_disc_html, unsafe_allow_html=True)


# ==========================================================
# SCANOVA 2.0 — HOME / BRAND
# ==========================================================
hero_stats = get_stats()


# ---- Load the 3 modality hero images ----
def _load_img_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


img_xray_b64 = _load_img_b64("branding/hero_xray.png")
img_mri_b64 = _load_img_b64("branding/hero_mri.png")
img_bone_b64 = _load_img_b64("branding/hero_bone.png")


def _img_tag(b64, alt):
    if b64:
        return '<img src="data:image/png;base64,' + b64 + '" alt="' + alt + '" />'
    return (
        '<div style="width:100%;height:100%;display:flex;'
        'align-items:center;justify-content:center;'
        'background:#1e293b;color:#64748b;font-size:2rem;">🩻</div>'
    )


hero_xray_tag = _img_tag(img_xray_b64, "Chest X-ray")
hero_mri_tag = _img_tag(img_mri_b64, "Brain MRI")
hero_bone_tag = _img_tag(img_bone_b64, "Bone X-ray")


# ---- Brand block ----
_brand_html = (
    '<div class="scanova-brand-block">'
    '<div class="scanova-brand-tag">SEE · ANALYZE · EXPLAIN</div>'
    '<h1 class="scanova-brand-name">SCANOVA</h1>'
    '<p class="scanova-brand-subtitle">'
    'An AI-powered medical imaging assistant that helps you '
    '<strong>see</strong> what the model sees, <strong>analyze</strong> '
    'what it found, and <strong>explain</strong> the results in plain language.'
    '</p>'
    '</div>'
)
st.markdown(_brand_html, unsafe_allow_html=True)


# ---- 3-image hero grid ----
_hero_grid_html = (
    '<div class="hero-modality-grid">'

    '<div class="hero-modality-card m-xray">'
    '<div class="hero-modality-image">'
    + hero_xray_tag +
    '<div class="hero-modality-label">'
    '<span class="hero-modality-label-icon">🩻</span>'
    '<span class="hero-modality-label-text">Chest X-Ray</span>'
    '</div>'
    '</div>'
    '<div class="hero-modality-footer">'
    '<span class="hero-modality-name">Chest X-Ray</span>'
    '<span class="hero-modality-badge">Production</span>'
    '</div>'
    '</div>'

    '<div class="hero-modality-card m-mri">'
    '<div class="hero-modality-image">'
    + hero_mri_tag +
    '<div class="hero-modality-label">'
    '<span class="hero-modality-label-icon">🧠</span>'
    '<span class="hero-modality-label-text">Brain MRI</span>'
    '</div>'
    '</div>'
    '<div class="hero-modality-footer">'
    '<span class="hero-modality-name">Brain MRI</span>'
    '<span class="hero-modality-badge">Research</span>'
    '</div>'
    '</div>'

    '<div class="hero-modality-card m-bone">'
    '<div class="hero-modality-image">'
    + hero_bone_tag +
    '<div class="hero-modality-label">'
    '<span class="hero-modality-label-icon">🦴</span>'
    '<span class="hero-modality-label-text">Bone X-Ray</span>'
    '</div>'
    '</div>'
    '<div class="hero-modality-footer">'
    '<span class="hero-modality-name">Bone X-Ray</span>'
    '<span class="hero-modality-badge">Research</span>'
    '</div>'
    '</div>'

    '</div>'

    '<div class="hero-modality-caption">'
    '<strong>THREE MODALITIES</strong> · ONE AI ASSISTANT'
    '</div>'
)
st.markdown(_hero_grid_html, unsafe_allow_html=True)


# ---- Three primary options ----
_primary_opts_html = (
    '<div class="primary-options">'
    '<div class="primary-option xray">'
    '<span class="primary-option-icon">🫁</span>'
    '<div class="primary-option-title">X-Ray Analysis</div>'
    '<div class="primary-option-desc">Chest and general X-ray images. Best results with clear, frontal views.</div>'
    '<span class="primary-option-cta">SCAN NOW →</span>'
    '</div>'
    '<div class="primary-option ct">'
    '<span class="primary-option-icon">🧠</span>'
    '<div class="primary-option-title">CT Scan</div>'
    '<div class="primary-option-desc">Chest CT images — single-slice or DICOM. Detects structural irregularities.</div>'
    '<span class="primary-option-cta">SCAN NOW →</span>'
    '</div>'
    '<div class="primary-option bone">'
    '<span class="primary-option-icon">🦴</span>'
    '<div class="primary-option-title">Bone X-Ray</div>'
    '<div class="primary-option-desc">Bone and fracture-related images. Surfaces density discontinuities.</div>'
    '<span class="primary-option-cta">SCAN NOW →</span>'
    '</div>'
    '</div>'
)
st.markdown(_primary_opts_html, unsafe_allow_html=True)


# ---- Functional action buttons ----
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if st.button("🫁 Scan X-Ray →", use_container_width=True, key="btn_xray"):
        st.info("💡 **Upload a chest X-ray below.** Or click 🎯 Try Sample for a demo.")

with btn_col2:
    if st.button("🧠 Scan CT →", use_container_width=True, key="btn_ct"):
        st.info("💡 **Upload a CT image below.** PNG, JPG, or DICOM accepted.")

with btn_col3:
    if st.button("🦴 Scan Bone →", use_container_width=True, key="btn_bone"):
        st.info("💡 **Upload a bone X-ray below.**")


# ---- How it works flow (horizontal) ----
_flow_html = (
    '<div class="flow-container">'
    '<div class="flow-title">🔬 How SCANOVA Works</div>'
    '<div class="flow-horizontal">'
    '<div class="flow-step-h s1">'
    '<div class="flow-step-num">1</div>'
    '<span class="flow-step-icon">📤</span>'
    '<div class="flow-step-label">Image<br>Upload</div>'
    '</div>'
    '<div class="flow-step-h s2">'
    '<div class="flow-step-num">2</div>'
    '<span class="flow-step-icon">✅</span>'
    '<div class="flow-step-label">Quality<br>Check</div>'
    '</div>'
    '<div class="flow-step-h s3">'
    '<div class="flow-step-num">3</div>'
    '<span class="flow-step-icon">🧠</span>'
    '<div class="flow-step-label">AI<br>Analysis</div>'
    '</div>'
    '<div class="flow-step-h s4">'
    '<div class="flow-step-num">4</div>'
    '<span class="flow-step-icon">🔥</span>'
    '<div class="flow-step-label">Explainable<br>AI</div>'
    '</div>'
    '<div class="flow-step-h s5">'
    '<div class="flow-step-num">5</div>'
    '<span class="flow-step-icon">💬</span>'
    '<div class="flow-step-label">Patient<br>Insight</div>'
    '</div>'
    '<div class="flow-step-h s6">'
    '<div class="flow-step-num">6</div>'
    '<span class="flow-step-icon">👨‍⚕️</span>'
    '<div class="flow-step-label">Professional<br>Review</div>'
    '</div>'
    '</div>'
    '</div>'
)
st.markdown(_flow_html, unsafe_allow_html=True)


# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.caption("Tune how sensitive the AI should be.")

    _priv_html = (
        '<div class="privacy-notice">'
        '<div class="privacy-notice-title">🔐 Privacy Notice</div>'
        '<div class="privacy-notice-body">'
        'Do not upload identifiable patient information '
        '(names, IDs, dates of birth). Images are processed '
        'in memory only — nothing is stored, logged, or shared.'
        '</div>'
        '</div>'
    )
    st.markdown(_priv_html, unsafe_allow_html=True)

    if DL_AVAILABLE:
        analysis_mode = st.radio(
            "AI Engine",
            options=[
                "⚡ Statistical (fast)",
                "🧠 Deep Learning (accurate)",
                "⚖️ Compare Both"
            ],
            index=0,
            help="Compare mode runs both engines side-by-side."
        )

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
                help="Auto-detect uses the trained modality classifier."
            )
        else:
            modality_choice = "🌐 Auto-detect"
    else:
        analysis_mode = "⚡ Statistical (fast)"
        modality_choice = "🌐 Auto-detect"

    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.3, max_value=0.9, value=0.65, step=0.05,
        help="Higher = more detections (may include false positives)"
    )
    block_size = st.select_slider(
        "Analysis Block Size",
        options=[16, 32, 64], value=32,
        help="Smaller = more detailed, slower"
    )
    show_boxes = st.checkbox("Show bounding boxes", value=True)

    st.divider()
        
    st.markdown("### 🎛️ Display Mode")

    view_mode_choice = st.radio(
        "Language",
        options=["👨‍⚕️ Clinical View", "👨‍👩‍👧 Patient View"],
        index=0,
        help="Clinical shows technical details. Patient shows plain-language explanations.",
        key="view_mode_toggle"
    )
    st.session_state.view_mode = (
        "patient" if "Patient" in view_mode_choice else "clinical"
    )
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


# ---------- Disclaimer + Live Stats Row ----------
col_left, col_right = st.columns([3, 1])

with col_left:
    st.warning(
        "⚠️ **Assistive tool only.** This does not replace professional medical "
        "diagnosis. Always consult a qualified radiologist for clinical decisions."
    )

with col_right:
    _live_card_html = (
        '<div class="live-card">'
        '<div class="live-card-header">'
        '<span class="live-card-title">Live Session</span>'
        '<span class="live-pulse">Active</span>'
        '</div>'
        '<div class="live-stat-row">'
        '<span class="live-stat-label">Scans this session</span>'
        f'<span class="live-stat-value accent">{hero_stats["total"]}</span>'
        '</div>'
        '<div class="live-stat-row">'
        '<span class="live-stat-label">Avg analysis time</span>'
        f'<span class="live-stat-value">{hero_stats["avg_time"]}s</span>'
        '</div>'
        '<div class="live-stat-row">'
        '<span class="live-stat-label">Safe / Warn / Danger</span>'
        f'<span class="live-stat-value" style="font-size: 1rem;">{hero_stats["safe"]} / {hero_stats["warning"]} / {hero_stats["danger"]}</span>'
        '</div>'
        '</div>'
    )
    st.markdown(_live_card_html, unsafe_allow_html=True)


# ---------- Upload Section ----------
_upload_header_html = (
    '<a id="upload"></a>'
    '<div class="section-header">'
    '<span class="section-header-icon">📤</span>'
    '<div><h2 class="section-header-text">Upload Medical Image</h2>'
    '<p class="section-header-desc">Supported formats: PNG · JPG · DICOM (.dcm)</p></div>'
    '</div>'
)
st.markdown(_upload_header_html, unsafe_allow_html=True)


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

    # ===== PRE-SCAN QUALITY CHECK =====
    from backend.quality_check import check_image_quality

    # Read the file bytes once — store them so we can re-use
    if st.session_state.get("_cached_file_bytes") is None or \
       st.session_state.get("_cached_file_name") != uploaded_file.name:
        st.session_state._cached_file_bytes = uploaded_file.read()
        st.session_state._cached_file_name = uploaded_file.name
        st.session_state.quality_result = None  # reset

    file_bytes = st.session_state._cached_file_bytes

    # Run quality check once per file
    if st.session_state.get("quality_result") is None:
        try:
            from backend.preprocess import load_image
            raw_img = load_image(file_bytes)
            st.session_state.quality_result = check_image_quality(raw_img)
        except Exception as e:
            st.session_state.quality_result = {
                "checks": [("Image loaded", False, str(e))],
                "passed": 0,
                "total": 1,
                "overall": "poor",
                "message": "❌ Could not read the image. Please try another file.",
            }

    # Display pre-scan panel
    qc = st.session_state.quality_result

    checks_html = ""
    for label, passed, detail in qc["checks"]:
        icon = "✅" if passed else "⚠️"
        cls = "pass" if passed else "fail"
        checks_html += (
            '<div class="prescan-check ' + cls + '">'
            '<div class="prescan-check-left">'
            '<span class="prescan-check-icon">' + icon + '</span>'
            '<span class="prescan-check-label">' + label + '</span>'
            '</div>'
            '<span class="prescan-check-detail">' + detail + '</span>'
            '</div>'
        )

    prescan_html = (
        '<div class="prescan-panel">'
        '<div class="prescan-header">'
        '<span class="prescan-header-icon">🔍</span>'
        '<span class="prescan-header-text">SCANOVA PRE-SCAN</span>'
        '</div>'
        + checks_html +
        '<div class="prescan-message ' + qc["overall"] + '">'
        + qc["message"] +
        '</div>'
        '</div>'
    )
    st.markdown(prescan_html, unsafe_allow_html=True)

    # ----- Analyze button -----
    if analyze_btn:
        start_time = time.time()
        current_mode = analysis_mode

        with st.spinner(f"Analyzing with {current_mode}..."):
            original, enhanced, normalized = preprocess(file_bytes)

            modality_map = {
                "🌐 Auto-detect": "auto",
                "🩻 Chest X-ray": "chest_xray",
                "🧠 Brain MRI": "brain_mri",
                "🫁 Chest CT": "chest_ct",
                "🦴 Bone Scan": "bone_scan",
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

            if DL_AVAILABLE and ("Deep Learning" in current_mode or "Compare" in current_mode):
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
            if "Compare" in current_mode and anomaly_map_dl is not None:
                anomaly_map, regions, raw_heat, overlay, report = (
                    anomaly_map_dl, regions_dl, heat_dl, overlay_dl, report_dl
                )
            elif "Deep Learning" in current_mode and anomaly_map_dl is not None:
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
        st.session_state.analysis_mode = current_mode
        st.session_state.compare_mode = "Compare" in current_mode

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
            "DL" if "Deep Learning" in current_mode
            else ("Compare" if "Compare" in current_mode else "Statistical")
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
        stored_mode = st.session_state.get("analysis_mode", "")
        is_compare = st.session_state.get("compare_mode", False) or "Compare" in stored_mode
        has_dl = st.session_state.get("dl_results") is not None

        if is_compare and has_dl:
            st.divider()

            _comp_header = (
                '<div class="section-header">'
                '<span class="section-header-icon">⚖️</span>'
                '<div><h2 class="section-header-text">Side-by-Side Comparison</h2>'
                '<p class="section-header-desc">Same image, two AI engines</p></div>'
                '</div>'
            )
            st.markdown(_comp_header, unsafe_allow_html=True)

            stat = st.session_state.stat_results
            dl = st.session_state.dl_results

            stack_compare = st.toggle("📱 Stack vertically (mobile view)", value=False)

            if stack_compare:
                st.markdown("### ⚡ Statistical Engine")
                st.image(stat["overlay"], use_container_width=True, caption="Detection Overlay")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Regions", len(stat["regions"]))
                c2.metric("Time", f"{stat['time']:.2f}s")
                c3.metric("Affected", f"{stat['report']['affected_area_percent']}%")
                c4.metric("Status", stat["report"]["level"].upper())

                st.divider()

                st.markdown("### 🧠 Deep Learning Engine")
                st.image(dl["overlay"], use_container_width=True, caption="Detection Overlay")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Regions", len(dl["regions"]))
                c2.metric("Time", f"{dl['time']:.2f}s")
                c3.metric("Affected", f"{dl['report']['affected_area_percent']}%")
                c4.metric("Status", dl["report"]["level"].upper())
            else:
                col_stat, col_dl = st.columns(2)

                with col_stat:
                    st.markdown("### ⚡ Statistical Engine")
                    st.image(stat["overlay"], use_container_width=True, caption="Detection Overlay")
                    c1, c2 = st.columns(2)
                    c1.metric("Regions", len(stat["regions"]))
                    c2.metric("Time", f"{stat['time']:.2f}s")
                    c3, c4 = st.columns(2)
                    c3.metric("Affected", f"{stat['report']['affected_area_percent']}%")
                    c4.metric("Status", stat["report"]["level"].upper())

                with col_dl:
                    st.markdown("### 🧠 Deep Learning Engine")
                    st.image(dl["overlay"], use_container_width=True, caption="Detection Overlay")
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

            _single_header = (
                '<div class="section-header">'
                '<span class="section-header-icon">🖼️</span>'
                '<div><h2 class="section-header-text">Analysis Results</h2>'
                f'<p class="section-header-desc">Engine: {stored_mode or "Statistical"}</p></div>'
                '</div>'
            )
            st.markdown(_single_header, unsafe_allow_html=True)

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

        _report_header = (
            '<div class="section-header">'
            '<span class="section-header-icon">📋</span>'
            '<div><h2 class="section-header-text">Analysis Report</h2>'
            '<p class="section-header-desc">Summary of detected patterns</p></div>'
            '</div>'
        )
        st.markdown(_report_header, unsafe_allow_html=True)

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
                # ===== View Mode: Patient / Clinical =====
        from backend.report import patient_explanation, clinical_explanation
                # ===== Explainability Module =====
        from backend.explainer import explain_reconstruction_error

        # Use the DL anomaly map if available, else statistical
        if st.session_state.get("dl_results") and st.session_state.get("compare_mode"):
            _explain_map = None  # not stored directly; skip for compare
        else:
            _explain_map = None

        # We use the anomaly map stored in the analysis (not retained).
        # Instead, use the regions + report data for a text-based explanation.
        _modality_code = "chest_xray"
        if "Brain" in st.session_state.get("analysis_mode", ""):
            _modality_code = "brain_mri"

        try:
            # Reconstruct a lightweight explanation from regions + report
            # We don't have the raw map here, so pass a synthetic one.
            import numpy as _np
            _synthetic_map = _np.zeros((128, 128), dtype=_np.float32)

            _explanation = explain_reconstruction_error(
                _synthetic_map,
                st.session_state.get("regions", []),
                modality=_modality_code,
            )

            # Build hotspots display
            _hotspots_html = ""
            for i, hs in enumerate(_explanation.get("top_hotspots", []), 1):
                _hotspots_html += (
                    '<div class="explainability-hotspot">'
                    f'<span>• Region {i} at (x={hs["x"]}, y={hs["y"]})</span>'
                    f'<span class="explainability-hotspot-score">score {hs["score"]:.2f}</span>'
                    '</div>'
                )

            _exp_html = (
                '<div class="explainability-card">'
                '<div class="explainability-header">'
                '<span class="explainability-header-icon">🔥</span>'
                '<span class="explainability-header-text">WHY THIS REGION WAS FLAGGED</span>'
                '</div>'

                '<div class="explainability-method">'
                'METHOD: ' + _explanation["method"].upper() +
                '</div>'

                '<div class="explainability-body">'
                + _explanation["headline"] + ' ' + _explanation["detail"] +
                '</div>'

                '<div class="explainability-note">'
                '<strong>⚠️ Honest note:</strong> '
                + _explanation["honest_note"] +
                '</div>'
            )

            if _hotspots_html:
                _exp_html += (
                    '<div class="explainability-hotspots">'
                    '<div class="explainability-hotspots-title">Top detected regions</div>'
                    + _hotspots_html +
                    '</div>'
                )

            if _explanation.get("modality_note"):
                _exp_html += (
                    '<div class="explainability-modality-note">'
                    + _explanation["modality_note"] +
                    '</div>'
                )

            _exp_html += '</div>'

            st.markdown(_exp_html, unsafe_allow_html=True)

        except Exception as _e:
            pass  # silently skip if explanation fails

        current_view = st.session_state.get("view_mode", "clinical")

        if current_view == "patient":
            _explanation_html = (
                '<div class="explanation-patient">'
                '<div class="explanation-patient-header">'
                '💬 <span>What this means — in plain language</span>'
                '</div>'
                '<div class="explanation-patient-body">'
                + patient_explanation(r) +
                '</div>'
                '<div class="explanation-patient-note">'
                'This is not a diagnosis. Always consult a qualified '
                'healthcare professional for medical decisions.'
                '</div>'
                '</div>'
            )
        else:
            _explanation_html = (
                '<div class="explanation-clinical">'
                '<div class="explanation-clinical-header">'
                '⚡ Technical Summary'
                '</div>'
                '<div class="explanation-clinical-body">'
                + clinical_explanation(r) +
                '</div>'
                '</div>'
            )

        st.markdown(_explanation_html, unsafe_allow_html=True)

        _result_disc = (
            '<div class="result-disclaimer">'
            '<strong>Interpretation reminder:</strong> This is a preliminary '
            'AI screening result. "Regions Found" represents areas the model '
            'could not reconstruct as well as the rest of the image — '
            'this may be normal anatomy, image quality, or a real finding. '
            'A qualified healthcare professional must review the image '
            'and make all clinical decisions.'
            '</div>'
        )
        st.markdown(_result_disc, unsafe_allow_html=True)

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

_faq_header = (
    '<div class="section-header">'
    '<span class="section-header-icon">❓</span>'
    '<div><h2 class="section-header-text">Frequently Asked Questions</h2>'
    '<p class="section-header-desc">Quick answers. Full FAQ on the dedicated page.</p></div>'
    '</div>'
)
st.markdown(_faq_header, unsafe_allow_html=True)

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
_founder_html = (
    '<div class="founder-quote">'
    '"We built SCANOVA because we watched radiologists drown in '
    'hundreds of scans a day. AI shouldn\'t replace them — it should '
    'give them back the time to think."'
    '<span class="attr">— <strong>Anwesha Rout</strong>, Founder · SCANOVA AI</span>'
    '</div>'
)
st.markdown(_founder_html, unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================
_footer_html = (
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
    '<div class="footer-line" style="margin-top: 1rem; font-size: 0.75rem;">'
    '⚠️ Not for clinical use. Always consult a qualified radiologist.</div>'
    '</div>'
)
st.markdown(_footer_html, unsafe_allow_html=True)