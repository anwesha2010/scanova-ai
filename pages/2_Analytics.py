import streamlit as st
from utils.analytics import get_stats, init_analytics

st.set_page_config(
    page_title="Analytics — SCANOVA AI",
    page_icon="📊",
    layout="wide"
)

# Load CSS
try:
    with open("branding/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Custom styles for history
st.markdown("""
<style>
    .history-item {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    .history-item:hover {
        border-color: #0ea5e9;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.08);
        transform: translateX(4px);
    }
    .history-filename {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #09090b;
        font-size: 1rem;
        margin: 0;
    }
    .history-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #71717a;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .history-tags {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .history-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 100px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .tag-safe { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .tag-warning { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .tag-danger { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .tag-neutral { background: #f4f4f5; color: #3f3f46; border: 1px solid #e4e4e7; }
    .tag-engine { background: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<h1 style="color: #09090b; font-weight: 800; letter-spacing: -0.02em;">📊 Analytics</h1>',
    unsafe_allow_html=True
)
st.caption("Session statistics and scan history for SCANOVA AI")
st.divider()

init_analytics()
stats = get_stats()

if stats["total"] == 0:
    st.info("📭 No scans yet this session. Upload and analyze an image to see statistics.")
else:
    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", stats["total"])
    c2.metric("Avg Time", f"{stats['avg_time']}s")
    c3.metric("Safe / Warning", f"{stats['safe']} / {stats['warning']}")
    c4.metric("Danger", stats["danger"])

    st.divider()

    # Status breakdown
    st.markdown("### 📈 Status Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ SAFE", stats["safe"])
    c2.metric("⚠️ WARNING", stats["warning"])
    c3.metric("🔴 DANGER", stats["danger"])

    st.divider()

    # Scan history
    st.markdown("### 📚 Scan History")
    st.caption(f"All {stats['total']} scans from this session (most recent first)")

    history = stats["history"][::-1]  # newest first

    for scan in history:
        # Emoji by status
        level_emoji = {"safe": "✅", "warning": "⚠️", "danger": "🔴"}.get(scan["level"], "ℹ️")
        level_class = f"tag-{scan['level']}" if scan["level"] in ["safe", "warning", "danger"] else "tag-neutral"

        # Modality display
        modality_display = scan.get("modality", "unknown").replace("_", " ").title()
        engine_display = scan.get("engine", "statistical")

        st.markdown(f"""
        <div class="history-item">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <p class="history-filename">📄 {scan['filename']}</p>
                    <p class="history-meta">🕐 {scan['timestamp']} &nbsp;·&nbsp; ⏱️ {scan['duration']}s</p>
                </div>
            </div>
            <div class="history-tags">
                <span class="history-tag tag-neutral">🧠 {modality_display}</span>
                <span class="history-tag tag-engine">⚙️ {engine_display}</span>
                <span class="history-tag {level_class}">{level_emoji} {scan['level'].upper()}</span>
                <span class="history-tag tag-neutral">{scan['regions']} region(s)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Clear button
    if st.button("🗑️ Clear Session History", use_container_width=False):
        st.session_state.scan_history = []
        st.rerun()

st.divider()
st.caption("Session-only data. Refreshing the browser clears analytics.")