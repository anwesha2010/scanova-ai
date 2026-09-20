import streamlit as st

st.set_page_config(
    page_title="FAQ — SCANOVA AI",
    page_icon="❓",
    layout="wide"
)

# Load CSS
try:
    with open("branding/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# FAQ-specific styles
st.markdown("""
<style>
    .faq-hero {
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #e4e4e7;
        margin-bottom: 2rem;
    }
    .faq-eyebrow {
        display: inline-block;
        background: #09090b;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        padding: 6px 14px;
        border-radius: 100px;
        text-transform: uppercase;
        margin-bottom: 1.25rem;
    }
    .faq-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.5rem, 5vw, 3.25rem);
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #09090b;
        line-height: 1.05;
        margin-bottom: 0.75rem;
    }
    .faq-title em {
        color: #0ea5e9;
        font-style: italic;
    }
    .faq-subtitle {
        font-size: 1.1rem;
        color: #71717a;
        max-width: 640px;
        line-height: 1.6;
    }
    /* Style expanders */
    [data-testid="stExpander"] {
        border: 1px solid #e4e4e7 !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        background: #ffffff !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #0ea5e9 !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.08) !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #09090b !important;
        padding: 6px 0 !important;
    }
    [data-testid="stExpander"] p {
        color: #3f3f46 !important;
        line-height: 1.7 !important;
        font-size: 0.98rem !important;
    }
    .faq-cta {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border-radius: 14px;
        border-left: 4px solid #0ea5e9;
        text-align: center;
    }
    .faq-cta h3 {
        font-size: 1.4rem;
        font-weight: 800;
        color: #09090b;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .faq-cta p {
        color: #52525b;
        font-size: 1rem;
        margin: 0;
    }
    .faq-cta a {
        color: #0ea5e9 !important;
        font-weight: 700;
        text-decoration: none !important;
    }
    .faq-cta a:hover {
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
    <div class="faq-hero">
        <div class="faq-eyebrow">FAQ</div>
        <h1 class="faq-title">Questions we get <em>all the time.</em></h1>
        <p class="faq-subtitle">
            Straight answers. No marketing fluff. If something still isn't clear,
            <a href="https://github.com/anwesha2010/scanova-ai" target="_blank"
               style="color:#0ea5e9; font-weight:600;">open an issue on GitHub</a>.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------- THE QUESTIONS ----------
faqs = [
    (
        "🩺 Is SCANOVA AI a medical device?",
        "**No.** SCANOVA AI is an assistive research and educational tool. "
        "It is **not** FDA, MCI, CE, or any other regulatory body-cleared "
        "and **must not** be used for actual clinical decision-making. "
        "Always consult a qualified radiologist."
    ),
    (
        "🖼️ What image types can I upload?",
        "Currently supported: **X-rays, MRIs, and CT scans** in `.png`, `.jpg`, "
        "or `.jpeg` format. We do not yet support DICOM (`.dcm`) files, but "
        "this is on the roadmap. Best results come from **clear, frontal** "
        "chest X-rays where anatomy is well-centered."
    ),
    (
        "🧠 How does the AI actually work?",
        "Two stages:\n\n"
        "1. **Preprocessing** — the image is resized, denoised, and "
        "contrast-enhanced using CLAHE (adaptive histogram equalization).\n\n"
        "2. **Anomaly detection** — the image is scanned in blocks. "
        "Regions that deviate strongly from surrounding tissue are flagged. "
        "An upcoming version will use a trained PyTorch autoencoder for "
        "even higher accuracy."
    ),
    (
        "🎯 Why does it sometimes flag normal tissue?",
        "Every anomaly-detection system has **false positives**. Our tool "
        "surfaces *potentially* unusual regions — it does not diagnose. "
        "A radiologist reviews every flag and decides what matters. "
        "**The AI doesn't replace judgement — it just saves time.**"
    ),
    (
        "🔒 Is my data stored or shared?",
        "**No.** Images are processed in-memory only during your session. "
        "Nothing is written to a database, saved to a disk, or sent to a "
        "third party. When you close your browser, everything is gone. "
        "For a production release, we'd add this transparency to our "
        "privacy policy."
    ),
    (
        "💸 Is it really free?",
        "**Yes.** SCANOVA AI is **MIT-licensed** and free forever. "
        "No signup, no credit card, no usage limits. The full source code "
        "is on "
        "[GitHub](https://github.com/anwesha2010/scanova-ai) — fork it, "
        "learn from it, improve it."
    ),
    (
        "⚠️ Can I use this for my patients?",
        "**No. Absolutely not.** SCANOVA AI is for **education, research, "
        "and demonstration only**. It has not been validated against any "
        "clinical dataset, has not been reviewed by a medical board, and "
        "should never be used to make real diagnostic decisions. If you "
        "have a medical concern, see a doctor."
    ),
    (
        "👥 Who built this?",
        "SCANOVA AI was designed and built by a **student team led by "
        "Anwesha Rout**, with mentorship on architecture, deployment, and "
        "design. We are not affiliated with any hospital, clinic, or "
        "medical institution. This is an independent project built for "
        "the InnoEx 2026 competition."
    ),
    (
        "🚀 What's next for SCANOVA?",
        "Three items on our roadmap:\n\n"
        "- **🧠 Deep learning** — PyTorch autoencoder for higher accuracy\n"
        "- **📄 DICOM support** — read directly from hospital formats\n"
        "- **🌍 Multi-language** — Arabic + English for regional clinics\n\n"
        "Have a feature idea? Open an issue on GitHub."
    ),
]

for question, answer in faqs:
    with st.expander(question):
        st.markdown(answer)

# ---------- CTA ----------
st.markdown("""
    <div class="faq-cta">
        <h3>Still curious?</h3>
        <p>
            Read <a href="/About" target="_self">our story</a> or
            try the <a href="/" target="_self">live demo</a> yourself.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div class="app-footer">
        <div class="footer-line">
            <strong>SCANOVA AI</strong> — Medical Image Analysis Assistant
        </div>
        <div>
            <span class="footer-badge">MIT Licensed</span>
            <span class="footer-badge">Made in Kuwait</span>
            <span class="footer-badge">v3.0</span>
        </div>
    </div>
""", unsafe_allow_html=True)