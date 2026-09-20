import streamlit as st
import base64

st.set_page_config(
    page_title="About - SCANOVA AI",
    page_icon="ℹ️",
    layout="wide"
)

# Load custom CSS
try:
    with open("branding/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Header with logo
try:
    with open("branding/logo.svg", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <div class="brand-header">
            <img src="data:image/svg+xml;base64,{logo_b64}" class="brand-logo" />
            <h1 class="brand-title">About SCANOVA AI</h1>
        </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.title("ℹ️ About SCANOVA AI")

st.markdown("""
### 🎯 Purpose
SCANOVA AI is an **assistive tool** that helps medical professionals 
quickly identify potentially unusual regions in X-ray, MRI, and CT scans.

### 🔄 Workflow
`Upload → Preprocess → AI Analysis → Heatmap → Report`

### 🧠 How It Works
1. **Preprocess** — Resize, denoise, and enhance contrast
2. **Analyze** — Scan image in 32×32 blocks, flag deviations
3. **Heatmap** — Color-code suspicious regions
4. **Report** — Generate summary with metrics

### 🛠️ Technology Stack
| Component | Tool |
|-----------|------|
| Frontend | Streamlit |
| Image Processing | OpenCV, PIL |
| Numerical | NumPy |
| Deployment | Streamlit Cloud |

### ⚠️ Important Disclaimer
This tool **does not replace** professional medical diagnosis.
All results must be reviewed by a qualified radiologist.
""")

st.divider()

st.markdown("""
### 📊 Version
**v1.0** — Initial release

### 🔗 Links
- [GitHub Repository](https://github.com/anwesha2010/scanova-ai)
- [Streamlit Documentation](https://docs.streamlit.io)
""")