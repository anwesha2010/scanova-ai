import streamlit as st

st.set_page_config(
    page_title="Our Story — SCANOVA AI",
    page_icon="✍️",
    layout="wide"
)

# Load CSS
try:
    with open("branding/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Custom handwritten styles
st.markdown("""
<style>
    .story-hero {
        padding: 3rem 0 2rem 0;
        position: relative;
    }
    .story-eyebrow {
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
    .story-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.5rem, 5vw, 3.5rem);
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #09090b;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .story-title em {
        color: #0ea5e9;
        font-style: italic;
    }
    .story-subtitle {
        font-size: 1.1rem;
        color: #71717a;
        margin-bottom: 2rem;
        max-width: 640px;
    }
    .handwritten {
        font-family: 'Caveat', cursive;
        font-size: 1.85rem;
        line-height: 1.45;
        color: #18181b;
        background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
        padding: 2rem 2.25rem;
        border-radius: 4px 12px 12px 4px;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);
        position: relative;
        transform: rotate(-0.4deg);
    }
    .handwritten::before {
        content: '📌';
        position: absolute;
        top: -14px;
        left: 24px;
        font-size: 1.5rem;
    }
    .story-body {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        line-height: 1.75;
        color: #3f3f46;
        max-width: 720px;
        margin-bottom: 1.5rem;
    }
    .story-body strong {
        color: #09090b;
        font-weight: 700;
    }
    .story-sign {
        font-family: 'Caveat', cursive;
        font-size: 2rem;
        color: #0ea5e9;
        margin-top: 2rem;
        transform: rotate(-1deg);
    }
    .story-sign small {
        display: block;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #71717a;
        letter-spacing: 0.05em;
        margin-top: 4px;
        transform: rotate(0);
    }
    .value-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 2.5rem 0;
    }
    .value-card {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    .value-card:hover {
        border-color: #0ea5e9;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.08);
    }
    .value-icon {
        font-size: 1.75rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    .value-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #09090b;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    .value-desc {
        font-size: 0.9rem;
        color: #71717a;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)

# Load Caveat handwritten font
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# ---------- HERO ----------
st.markdown("""
    <div class="story-hero">
        <div class="story-eyebrow">OUR STORY</div>
        <h1 class="story-title">We built this at 2am,<br><em>because someone had to.</em></h1>
        <p class="story-subtitle">Short version. The long version needs coffee.</p>
    </div>
""", unsafe_allow_html=True)

# ---------- HANDWRITTEN NOTE ----------
st.markdown("""
    <div class="handwritten">
        We're students. We're not doctors. But we've sat in hospital waiting rooms,
        watched radiologists scan image after image, and wondered — <em>why isn't AI helping here yet?</em>
    </div>
""", unsafe_allow_html=True)

# ---------- BODY ----------
st.markdown("""
    <p class="story-body">
        So we built SCANOVA AI. A small thing. Something that looks at a chest X-ray
        and says, <strong>"hey, look here"</strong> — without pretending to be smarter than the person
        reading it.
    </p>
    <p class="story-body">
        It's not perfect. It's not clinical. But it's <strong>honest</strong>. And it's the
        first step in a much longer journey.
    </p>
    <p class="story-body">
        If this tool helps even one radiologist catch one thing they'd have missed —
        <strong>we did our job.</strong>
    </p>
""", unsafe_allow_html=True)

# ---------- SIGNATURE ----------
st.markdown("""
    <div class="story-sign">
        — Anwesha
        <small>FOUNDER · SCANOVA AI · 2026</small>
    </div>
""", unsafe_allow_html=True)

# ---------- VALUES ----------
st.markdown("""
    <div style="margin-top: 3rem; margin-bottom: 1rem;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                    color: #0ea5e9; letter-spacing: 0.14em; font-weight: 700;
                    text-transform: uppercase; margin-bottom: 8px;">
            WHAT WE BELIEVE
        </div>
        <h2 style="font-size: 1.75rem; font-weight: 800; color: #09090b;
                   letter-spacing: -0.03em; margin: 0;">
            Three things we refuse to compromise on.
        </h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="value-grid">
        <div class="value-card">
            <span class="value-icon">🩺</span>
            <div class="value-title">Assist, never replace</div>
            <div class="value-desc">
                AI is a second pair of eyes. The final call always belongs
                to a trained radiologist.
            </div>
        </div>
        <div class="value-card">
            <span class="value-icon">🔍</span>
            <div class="value-title">Show, don't hide</div>
            <div class="value-desc">
                Every flag comes with a visible heatmap. No black boxes,
                no "trust me" — just where and why.
            </div>
        </div>
        <div class="value-card">
            <span class="value-icon">🌱</span>
            <div class="value-title">Open by default</div>
            <div class="value-desc">
                MIT licensed. Free forever. Built to be forked, learned from,
                and improved by anyone.
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------- CTA ----------
st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
        <div style="font-size: 1.1rem; color: #3f3f46; padding-top: 8px;">
            Want to see it in action? Try a sample scan — no signup, no data collection.
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(
        '<a href="/" target="_self" class="cta-primary">← Back to app</a>',
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ---------- FOOTER ----------
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