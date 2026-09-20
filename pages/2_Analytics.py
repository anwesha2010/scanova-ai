import streamlit as st
from utils.analytics import get_stats, init_analytics

st.set_page_config(
    page_title="Analytics - SCANOVA AI",
    page_icon="📊",
    layout="wide"
)

# ---------- Load Custom CSS ----------
try:
    with open("branding/styles.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ---------- Header ----------
st.markdown(
    '<h1 style="color: #0f172a; font-weight: 800; '
    'letter-spacing: -0.02em;">📊 Analytics</h1>',
    unsafe_allow_html=True
)
st.caption("Session statistics for SCANOVA AI")
st.divider()

# ---------- Stats ----------
init_analytics()
stats = get_stats()

if stats["total"] == 0:
    st.info("📭 No scans yet this session. Upload and analyze an image to see statistics.")
else:
    # Top row metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Scans", stats["total"])
    c2.metric("Avg Time", f"{stats['avg_time']}s")
    c3.metric("Session Status",
              "ACTIVE" if stats["total"] > 0 else "IDLE")

    st.divider()

    # Status breakdown
    st.markdown("### 📈 Status Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ SAFE", stats["safe"])
    c2.metric("⚠️ WARNING", stats["warning"])
    c3.metric("🔴 DANGER", stats["danger"])

    st.divider()

    # Recent scans table
    st.markdown("### 🕒 Recent Scans")
    history = stats["history"][::-1]  # newest first

    for scan in history[:10]:
        emoji = {"safe": "✅", "warning": "⚠️", "danger": "🔴"}.get(
            scan["level"], "ℹ️"
        )
        with st.container():
            cols = st.columns([3, 1, 1, 1])
            cols[0].write(f"📄 **{scan['filename']}**")
            cols[1].write(f"🕐 {scan['timestamp']}")
            cols[2].write(f"⏱️ {scan['duration']}s")
            cols[3].write(f"{emoji} {scan['level'].upper()}")

    st.divider()

    # Clear button
    if st.button("🗑️ Clear Session History"):
        st.session_state.scan_history = []
        st.rerun()

st.divider()
st.caption("Session-only data. Refreshing the browser clears analytics.")