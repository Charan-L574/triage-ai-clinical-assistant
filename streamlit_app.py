"""
streamlit_app.py — TriageAI multi-page Streamlit frontend.
"""
import streamlit as st

st.set_page_config(
    page_title="TriageAI — Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.esi-card {
    border-radius: 12px; padding: 20px; text-align: center;
    color: white; font-weight: 700; margin-bottom: 16px;
}
.esi-number { font-size: 3.5rem; line-height: 1; }
.esi-label { font-size: 1.2rem; margin-top: 4px; }
.escalate-banner {
    background: #E24B4A; color: white; border-radius: 8px;
    padding: 12px 16px; font-weight: 700; font-size: 1.1rem;
    text-align: center; margin: 8px 0;
}
.confidence-bar-wrap { margin: 8px 0; }
.metric-card {
    background: #1e1e2e; border-radius: 10px; padding: 16px;
    border: 1px solid #333;
}
.tag {
    display: inline-block; border-radius: 4px; padding: 2px 8px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
}
.tag-symptom { background:#4d1f1f; color:#ff8080; }
.tag-body    { background:#1a2d4d; color:#7eb8ff; }
.tag-cond    { background:#2d1a4d; color:#c07bff; }
.tag-med     { background:#1a3d2d; color:#6ee7a0; }
.rec-card {
    background:#1a1a2e; border-left: 4px solid #7eb8ff;
    border-radius: 8px; padding: 14px; margin: 8px 0;
}
.timeline-dot {
    width:14px; height:14px; border-radius:50%;
    display:inline-block; margin-right:8px;
}
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Streamlit Navigation ────────────────────────────────────────────────────────
pg = st.navigation([
    st.Page("frontend/page_triage.py", title="Triage Assessment", icon="🏥"),
    st.Page("frontend/page_history.py", title="Case History", icon="📋"),
    st.Page("frontend/page_status.py", title="System Status", icon="⚙️"),
    st.Page("frontend/page_about.py", title="About & Docs", icon="ℹ️"),
])
pg.run()
