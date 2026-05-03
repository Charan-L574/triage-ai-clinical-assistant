"""
pages/page_history.py — Page 5: Case History with filtering, pagination, detail view.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import streamlit as st
import pandas as pd
import httpx
from frontend.ui_helpers import api_get, ESI_COLORS, render_result_column
import config as _cfg

API = _cfg.STREAMLIT_API_BASE_URL

st.title("📋 Case History")
st.caption("Browse, filter, and re-view all past triage assessments.")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    esi_filter = st.multiselect("ESI Level", [1, 2, 3, 4, 5])
    model_filter = st.selectbox("Model Used", ["All", "MedGemma-27B", "Groq-Llama-3.3", "Gemini-2.5-Flash", "emergency_fallback"])
    date_from = st.date_input("From Date", value=None)
    date_to = st.date_input("To Date", value=None)

# ── Build query params ─────────────────────────────────────────────────────────
params = {"page": 1, "page_size": 50}
if esi_filter:
    params["min_esi"] = min(esi_filter)
    params["max_esi"] = max(esi_filter)
if model_filter != "All":
    params["model_used"] = model_filter
if date_from:
    params["date_from"] = str(date_from)
if date_to:
    params["date_to"] = str(date_to)

data = api_get("/api/cases", params=params)

if not data:
    st.info("No cases found or API unavailable.")
    st.stop()

cases = data.get("cases", [])
total = data.get("total", 0)

# ── Export ─────────────────────────────────────────────────────────────────────
if cases:
    exp_rows = [{
        "Session ID": c["session_id"],
        "Timestamp": c["timestamp"],
        "Chief Complaint": c["chief_complaint"][:80],
        "ESI Level": c["triage_level"],
        "ESI Label": c["esi_label"],
        "Escalate": c["escalate_immediately"],
        "Confidence": c["confidence_score"],
        "Model": c["model_used"],
        "Completeness": c["data_completeness_score"],
    } for c in cases]
    csv = pd.DataFrame(exp_rows).to_csv(index=False)
    st.download_button("⬇️ Export CSV", csv, "triage_history.csv", "text/csv")

st.caption(f"Showing {len(cases)} of {total} cases")

# ── Table ──────────────────────────────────────────────────────────────────────
if not cases:
    st.info("No cases match the current filters.")
    st.stop()

# Column Headers
with st.container():
    hcol1, hcol2, hcol3, hcol4, hcol5, hcol6 = st.columns([2, 4, 1, 1, 2, 1])
    hcol1.markdown("**Timestamp**")
    hcol2.markdown("**Chief Complaint**")
    hcol3.markdown("**Level**")
    hcol4.markdown("**Escalate**")
    hcol5.markdown("**Model & Conf**")
    hcol6.markdown("**Action**")
st.divider()

for c in cases:
    lvl = c.get("triage_level", "?")
    color = ESI_COLORS.get(lvl, "#888") if isinstance(lvl, int) else "#888"
    ts = c.get("timestamp", "")[:19]
    complaint = c.get("chief_complaint", "")[:60]
    escalate = "🚨 YES" if c.get("escalate_immediately") else "No"
    conf = f"{c.get('confidence_score', 0):.0%}"
    model = c.get("model_used", "?")

    row_key = f"case_{c['session_id']}"

    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns([2, 4, 1, 1, 2, 1])
        col1.caption(ts)
        col2.markdown(complaint)
        col3.markdown(
            f'<span style="background:{color};color:white;padding:2px 6px;'
            f'border-radius:4px;font-weight:700;">ESI {lvl}</span>',
            unsafe_allow_html=True,
        )
        col4.markdown(escalate)
        col5.caption(f"{model} | {conf}")

        if col6.button("👁️", key=f"view_{c['session_id']}"):
            if st.session_state.get("expanded_case") == c["session_id"]:
                st.session_state.pop("expanded_case", None)
            else:
                st.session_state["expanded_case"] = c["session_id"]

        # Expanded detail
        if st.session_state.get("expanded_case") == c["session_id"]:
            with st.container():
                full_result_json = c.get("full_result_json")
                if full_result_json:
                    try:
                        full_result = json.loads(full_result_json)
                        render_result_column(full_result)
                    except Exception:
                        st.json(c)

                # Delete button
                if st.button(f"🗑️ Delete Case", key=f"del_{c['session_id']}"):
                    try:
                        r = httpx.delete(f"{API}/api/cases/{c['session_id']}", timeout=10)
                        if r.status_code == 200:
                            st.success("Case deleted.")
                            st.session_state.pop("expanded_case", None)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

        st.divider()
