"""
pages/page_status.py — Page 6: System Status with model health and charts.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import streamlit as st
import plotly.graph_objects as go
from frontend.ui_helpers import api_get, ESI_COLORS

st.title("⚙️ System Status")
st.caption("Live model availability and aggregate system performance metrics.")

# ── Top metrics ────────────────────────────────────────────────────────────────
stats = api_get("/api/stats")
health = api_get("/api/health")

if stats:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cases", stats.get("total_cases", 0))
    m2.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.0%}")
    all_rt = stats.get("avg_response_time_ms", {})
    avg_rt_all = round(sum(all_rt.values()) / len(all_rt)) if all_rt else 0
    m3.metric("Avg Response Time", f"{avg_rt_all} ms")
    if health:
        uptime_sec = health.get("uptime_seconds", 0)
        h, m = divmod(uptime_sec // 60, 60)
        m4.metric("System Uptime", f"{h}h {m}m")

st.divider()

# ── Model status cards ─────────────────────────────────────────────────────────
st.markdown("### 🤖 Model Status")

if st.button("🔄 Refresh Model Availability"):
    st.rerun()

model_status_data = api_get("/api/models/status")

if model_status_data:
    models = model_status_data.get("models", [])
    cols = st.columns(3)
    for i, m in enumerate(models):
        with cols[i % 3]:
            enabled = m.get("enabled", False)
            dot = "🟢" if enabled else "🔴"
            success_rate = m.get("success_rate")
            avg_rt = m.get("avg_response_time_ms")
            last_used = str(m.get("last_used", "Never"))[:19]
            total_calls = m.get("total_calls", 0)

            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:10px;padding:16px;
                        border:1px solid #333;margin-bottom:8px;">
                <div style="font-size:1.1rem;font-weight:700;margin-bottom:8px">
                    {dot} {m.get('name', '?')}
                </div>
                <div style="color:#aaa;font-size:0.85rem">
                    Model ID: <code>{m.get('model_id','?')}</code><br>
                    Calls: {total_calls}<br>
                    Success Rate: {f"{success_rate:.0%}" if success_rate is not None else "N/A"}<br>
                    Avg RT: {f"{avg_rt}ms" if avg_rt else "N/A"}<br>
                    Last Used: {last_used}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Performance Charts")

if stats:
    model_dist = stats.get("model_distribution", {})
    esi_dist = stats.get("esi_distribution", {})
    avg_rt_by_model = stats.get("avg_response_time_ms", {})

    ch1, ch2 = st.columns(2)

    with ch1:
        if model_dist:
            fig = go.Figure(go.Pie(
                labels=list(model_dist.keys()),
                values=list(model_dist.values()),
                hole=0.45,
                marker_colors=["#7eb8ff", "#c07bff", "#6ee7a0"],
            ))
            fig.update_layout(
                title="Model Usage Distribution", height=320,
                paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with ch2:
        if avg_rt_by_model:
            fig = go.Figure(go.Bar(
                x=list(avg_rt_by_model.keys()),
                y=list(avg_rt_by_model.values()),
                marker_color=["#7eb8ff", "#c07bff", "#6ee7a0"],
            ))
            fig.update_layout(
                title="Avg Response Time per Model (ms)", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", margin=dict(t=40, b=0, l=0, r=0),
                yaxis_title="ms",
            )
            st.plotly_chart(fig, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        if esi_dist:
            esi_keys = sorted(esi_dist.keys())
            fig = go.Figure(go.Bar(
                x=[f"ESI {k}" for k in esi_keys],
                y=[esi_dist[k] for k in esi_keys],
                marker_color=[ESI_COLORS.get(int(k), "#888") for k in esi_keys],
            ))
            fig.update_layout(
                title="ESI Level Distribution", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", margin=dict(t=40, b=0, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with ch4:
        if health:
            st.markdown("#### 🔌 System Health")
            hk1, hk2 = st.columns(2)
            hk1.metric("Database", "✅ Connected" if health.get("database_connected") else "❌ Error")
            hk2.metric("API Version", health.get("app_version", "?"))
            st.caption(f"Primary model: `{health.get('primary_model','?')}`")
            rag_ready = health.get("rag_index_ready", False)
            st.metric("RAG Index", "✅ Ready" if rag_ready else "⏳ Building/Not ready")

if not stats:
    st.info("No data available. Run some triage cases first, or check that the API is running.")
