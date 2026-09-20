import streamlit as st

st.set_page_config(
    page_title="About — SCANOVA AI",
    page_icon="✨",
    layout="wide"
)

try:
    with open("branding/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ---------- HERO ----------
st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">ABOUT THE PROJECT</div>
        <h1 class="hero-title">Made by students,<br>for the future of medicine.</h1>
        <p class="hero-tagline">
            We're a small team of Gen-Z builders who got tired of waiting 
            for AI to reach the clinic. So we built SCANOVA.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------- MISSION ----------
st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">🎯</span>
        <div>
            <h2 class="section-header-text">Our Mission</h2>
            <p class="section-header-desc">Why we built this</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
Medical imaging saves lives — but analyzing thousands of scans is slow, 
tiring, and easy to miss small details in. We believe **AI shouldn't replace 
radiologists, it should give them superpowers.**

SCANOVA AI is our proof-of-concept: an assistive tool that scans medical 
images, flags potentially unusual regions, and visualizes them with 
medical-grade heatmaps — **in seconds, not hours.**
""")

st.divider()

# ---------- HOW IT WORKS ----------
st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">⚙️</span>
        <div>
            <h2 class="section-header-text">How It Works</h2>
            <p class="section-header-desc">The pipeline, end to end</p>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 1️⃣ Upload & Preprocess")
    st.caption(
        "Your X-ray, MRI, or CT scan is resized, denoised, and "
        "contrast-enhanced using CLAHE (adaptive histogram equalization)."
    )

    st.markdown("#### 2️⃣ AI Analysis")
    st.caption(
        "The image is scanned in blocks. Regions that deviate strongly "
        "from surrounding tissue are flagged as potential anomalies."
    )

with c2:
    st.markdown("#### 3️⃣ Heatmap Generation")
    st.caption(
        "Flagged regions are colored with a medical colormap and overlaid "
        "on the original image, so you can see *exactly* where to look."
    )

    st.markdown("#### 4️⃣ Report & Export")
    st.caption(
        "Get a summary report, download the annotated image, or export "
        "a formatted PDF for documentation."
    )

st.divider()

# ---------- TECH STACK ----------
st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">🛠️</span>
        <div>
            <h2 class="section-header-text">Built With</h2>
            <p class="section-header-desc">The stack behind SCANOVA</p>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frontend", "Streamlit")
c2.metric("Vision", "OpenCV")
c3.metric("Numerics", "NumPy")
c4.metric("Reports", "ReportLab")

st.divider()

# ---------- TEAM ----------
st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">👥</span>
        <div>
            <h2 class="section-header-text">The Team</h2>
            <p class="section-header-desc">Who built this</p>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 🎨")
with col2:
    st.markdown("#### Anwesha Rout")
    st.caption(
        "**Creator & Developer.** Built the end-to-end pipeline — "
        "from image preprocessing to deployment. Passionate about "
        "making AI practical for real-world problems."
    )

st.markdown("---")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 🤝")
with col2:
    st.markdown("#### SCANOVA Team")
    st.caption(
        "**Mentorship & Infrastructure.** Guided architecture decisions, "
        "deployment strategy, and the design system that powers this app."
    )

st.divider()

# ---------- DISCLAIMER ----------
st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">⚠️</span>
        <div>
            <h2 class="section-header-text">Important Disclaimer</h2>
        </div>
    </div>
""", unsafe_allow_html=True)

st.warning(
    "**SCANOVA AI is an assistive tool, not a diagnostic device.** "
    "It is designed for educational and research purposes. All results "
    "must be reviewed by a qualified radiologist. Never rely solely on "
    "this tool for clinical decisions."
)

# ---------- LINKS ----------
st.divider()
st.markdown("""
    <div class="app-footer">
        <div class="footer-line">
            <strong>SCANOVA AI</strong> — Version 2.0
        </div>
        <div class="footer-line">
            Open source under the MIT License
        </div>
        <div>
            <span class="footer-badge">GitHub</span>
            <span class="footer-badge">Streamlit Cloud</span>
        </div>
    </div>
""", unsafe_allow_html=True)