"""
pages/ui_helpers.py — Shared UI rendering helpers for all TriageAI pages.
"""
from __future__ import annotations
import json
import streamlit as st
import httpx
import config as _cfg

API = _cfg.STREAMLIT_API_BASE_URL
TIMEOUT = 60


def api_post(endpoint: str, data: dict, timeout: int = TIMEOUT) -> dict | None:
    try:
        r = httpx.post(f"{API}{endpoint}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("❌ Cannot reach the TriageAI API. Is the backend running? (`uvicorn main:app --port 8000`)")
    except httpx.TimeoutException:
        st.error("⏱️ Request timed out. The pipeline is still processing — try again.")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_get(endpoint: str, params: dict = None, timeout: int = 15) -> dict | None:
    try:
        r = httpx.get(f"{API}{endpoint}", params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("❌ Cannot reach the TriageAI API.")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


ESI_COLORS = {1: "#E24B4A", 2: "#EF9F27", 3: "#FAC775", 4: "#378ADD", 5: "#639922"}
ESI_LABELS = {1: "Immediate", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}


def render_esi_card(triage_level: int, esi_label: str, escalate: bool):
    color = ESI_COLORS.get(triage_level, "#888")
    st.markdown(f"""
    <div class="esi-card" style="background:{color};">
        <div class="esi-number">ESI {triage_level}</div>
        <div class="esi-label">{esi_label}</div>
        {"<div style='margin-top:8px;font-size:1rem;background:rgba(0,0,0,0.3);border-radius:6px;padding:4px;'>🚨 ESCALATE IMMEDIATELY</div>" if escalate else ""}
    </div>""", unsafe_allow_html=True)


def render_confidence_bar(score: float, label: str):
    pct = int(score * 100)
    color = "#4caf50" if pct >= 75 else "#ff9800" if pct >= 50 else "#f44336"
    st.markdown(f"""
    <div class="confidence-bar-wrap">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.85rem;color:#aaa;">Confidence</span>
            <span style="font-weight:600;color:{color};">{pct}% — {label}</span>
        </div>
        <div style="background:#333;border-radius:4px;height:8px;">
            <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_completeness_bar(score: float):
    pct = int(score * 100)
    color = "#4caf50" if pct >= 70 else "#ff9800" if pct >= 40 else "#f44336"
    st.markdown(f"""
    <div class="confidence-bar-wrap">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.85rem;color:#aaa;">Input Completeness</span>
            <span style="font-weight:600;color:{color};">{pct}%</span>
        </div>
        <div style="background:#333;border-radius:4px;height:8px;">
            <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_entity_tags(entities: dict):
    tags_html = ""
    cat_map = {
        "symptoms": ("tag-symptom", "🔴 Symptoms"),
        "body_parts": ("tag-body", "🔵 Body Parts"),
        "conditions": ("tag-cond", "🟣 Conditions"),
        "medications": ("tag-med", "🟢 Medications"),
    }
    for cat, (cls, label) in cat_map.items():
        items = entities.get(cat, [])
        if items:
            tags_html += f"<div style='margin:4px 0'><span style='color:#aaa;font-size:0.8rem'>{label}:</span> "
            tags_html += "".join(f'<span class="tag {cls}">{t}</span>' for t in items[:8])
            tags_html += "</div>"
    if tags_html:
        st.markdown(tags_html, unsafe_allow_html=True)


def render_differentials(diffs: list):
    if not diffs:
        return
    def sort_key(dx):
        rop_score = 0 if dx.get("rule_out_priority") == "RULE OUT FIRST" else 1
        lk = dx.get("likelihood", "Low")
        lk_score = {"High": 0, "Moderate": 1, "Low": 2}.get(lk, 3)
        return (rop_score, lk_score)

    sorted_diffs = sorted(diffs, key=sort_key)
    
    likelihood_colors = {"High": "#f44336", "Moderate": "#ff9800", "Low": "#4caf50"}
    for dx in sorted_diffs:
        lk = dx.get("likelihood", "Low")
        color = likelihood_colors.get(lk, "#888")
        rop = dx.get("rule_out_priority", "Standard")
        badge = "🔴 **RULE OUT FIRST**" if rop == "RULE OUT FIRST" else ""
        icd = f" `{dx['icd_code']}`" if dx.get("icd_code") else ""
        with st.container():
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{dx.get('condition_name','?')}**{icd} {badge}")
            col2.markdown(f"<span style='color:{color};font-weight:700'>{lk}</span>", unsafe_allow_html=True)
            if dx.get("supporting_findings"):
                st.caption("✅ " + " · ".join(dx["supporting_findings"][:3]))
            if dx.get("against_findings"):
                st.caption("❌ " + " · ".join(dx["against_findings"][:2]))
            st.divider()


def render_result_column(result: dict):
    """Full result rendering for the right column of Page 1."""
    if not result:
        return

    level = result.get("triage_level", 2)
    label = result.get("esi_label", "")
    escalate = result.get("escalate_immediately", False)

    render_esi_card(level, label, escalate)
    render_confidence_bar(result.get("confidence_score", 0), result.get("confidence_label", ""))

    if escalate:
        st.error("🚨 **ESCALATE IMMEDIATELY** — Call emergency services now.")

    # Red flags
    for flag in result.get("red_flags_detected", []):
        st.warning(f"⛔ {flag}")

    # Missing vitals
    mv = result.get("missing_vitals", [])
    if mv:
        st.info(f"ℹ️ **Missing vitals:** {', '.join(mv)} — increased uncertainty")

    # Ambiguity
    for af in result.get("ambiguity_flags", []):
        st.warning(f"⚠️ {af}")

    # Timeline analysis
    tla = result.get("symptom_timeline_analysis")
    if tla:
        st.markdown(f"> 📈 **Timeline insight:** {tla}")

    # Recommendation
    rec = result.get("recommendations", {})
    if rec.get("action"):
        st.markdown("### 📋 Clinical Interventions")
        st.markdown(f"""
        <div class="rec-card" style="background:#f8f9fa; color:#1E1E1E; padding:15px; border-radius:8px; border-left:4px solid #378add;">
        <b>Target Action:</b> {rec.get('action','')}<br>
        <b>Urgency:</b> {rec.get('urgency','')}<br>
        <b>Timeframe:</b> {rec.get('timeframe','')}<hr style="margin:10px 0; border-color:#ddd;">
        <b>Nursing Interventions:</b> {rec.get('nursing_interventions','')}<br>
        <b>Diagnostic Considerations:</b> {rec.get('diagnostic_considerations','')}
        </div>""", unsafe_allow_html=True)

    # Differentials
    diffs = result.get("differential_diagnoses", [])
    if diffs:
        st.markdown("### 🔬 Differential Diagnoses")
        render_differentials(diffs)

    # RAG Diseases
    rag_diseases = result.get("rag_extracted_diseases", [])
    if rag_diseases:
        with st.expander("📚 Vector Search Results (RAG)", expanded=False):
            st.caption("Raw diseases extracted from the 12,000+ ontology FAISS database based on symptom embeddings.")
            for i, d in enumerate(rag_diseases, 1):
                name = d.get('name', 'Unknown')
                dist = d.get('distance', 0.0)
                icds = d.get('icd_codes', [])
                icd_str = f" `{icds[0]}`" if icds else ""
                
                # Match the differential column layout
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"**{i}. {name.title()}**{icd_str}")
                
                # Color code distance: closer to 0 is better
                color = "#4caf50" if dist < 1.05 else "#ff9800" if dist < 1.15 else "#888888"
                col2.markdown(f"<span style='color:{color};font-weight:700'>L2: {dist:.4f}</span>", unsafe_allow_html=True)
                
                if len(icds) > 1:
                    st.caption("Other ICDs: " + " · ".join(icds[1:4]))
                st.divider()

    # Reasoning
    steps = result.get("reasoning_steps", [])
    if steps:
        with st.expander("🧠 Clinical Reasoning Steps"):
            for i, s in enumerate(steps, 1):
                st.markdown(f"{i}. {s}")

    # Follow-up questions
    fqs = result.get("follow_up_questions", [])
    if fqs:
        with st.expander("❓ Follow-up Questions"):
            for fq in fqs:
                prio = fq.get("priority", "medium").upper()
                st.markdown(f"**[{prio}]** {fq.get('question','')}")
                st.caption(fq.get("clinical_rationale", ""))

    # Missing vital impact
    mvi = result.get("missing_vital_impact", {})
    if mvi:
        with st.expander("📊 Missing Vital Impact"):
            for vital, impact in mvi.items():
                st.markdown(f"**{vital}:** {impact}")

    # Model info
    st.caption(f"**Primary Assessment Model:** `{result.get('model_used','?')}`")

    # Safety disclaimers
    discs = result.get("safety_disclaimers", [])
    if discs:
        with st.expander("⚠️ Safety Disclaimers", expanded=True):
            for d in discs:
                st.info(d)

    # Download report
    _offer_download(result)


def _offer_download(result: dict):
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from utils.report_generator import ReportGenerator
        from models.patient import PatientInput
        from models.triage_output import TriageResult as TR

        # Create a mock PatientInput just for the report since we don't have the original easily
        p = PatientInput(chief_complaint=result.get("raw_llm_output", "N/A")[:100] + "...")
        
        # We need to construct TR. Since result might have dicts for differentials instead of objects, 
        # TR(**result) should work since Pydantic auto-converts dicts to models.
        report = ReportGenerator.generate_text_report(TR(**result), p)
        
        st.download_button(
            "📄 Download Clinical Report",
            data=report,
            file_name=f"triage_report_{result.get('session_id','unknown')[:8]}.txt",
            mime="text/plain",
        )
    except Exception as e:
        import traceback
        st.error(f"Failed to generate report: {e}")
        st.caption(f"Traceback: {traceback.format_exc()}")


def vital_status_js_indicator(value, field: str, age: int = None) -> str:
    """Return color and message for a vital sign value (pure Python logic)."""
    pediatric = age is not None and age < 18
    normal_ranges = {
        "heart_rate": (60, 100) if not pediatric else (70, 130),
        "spo2_percent": (95, 100),
        "blood_pressure_systolic": (90, 140),
        "temperature_celsius": (36.1, 37.2),
        "respiratory_rate": (12, 20) if not pediatric else (20, 40),
        "glucose_mmol": (3.9, 7.8),
    }
    critical_ranges = {
        "heart_rate": (40, 150),
        "spo2_percent": (90, 100),
        "blood_pressure_systolic": (80, 200),
        "temperature_celsius": (35.0, 40.0),
        "respiratory_rate": (8, 30),
        "glucose_mmol": (3.0, 25.0),
    }
    if value is None:
        return "⬜", ""
    r = normal_ranges.get(field)
    cr = critical_ranges.get(field)
    if cr and (value < cr[0] or value > cr[1]):
        return "🔴", f"⚠️ **Critical value** — outside safe range"
    if r and (value < r[0] or value > r[1]):
        return "🟡", f"Borderline value (normal: {r[0]}–{r[1]})"
    return "🟢", ""


def render_hitl_panel(thread_id: str, questions: list, reason: str):
    """Render the Human-in-the-Loop clarification panel when LangGraph interrupts."""
    st.error("⏸️ **Pipeline Interrupted — Clinical Clarification Required**")
    if reason:
        st.warning(f"**Trigger Reason:** {reason}")
    
    st.markdown("The AI requires additional context before generating a final ESI level. Please answer the following to resume the assessment:")
    
    answers = {}
    with st.form("hitl_form", border=True):
        for idx, q in enumerate(questions):
            qid = q.get("id", str(idx))
            st.markdown(f"#### {idx+1}. {q.get('question')}")
            if q.get('reason'):
                st.caption(f"*{q.get('reason')}*")
            answers[qid] = st.text_input("Answer", key=f"hitl_ans_{qid}", label_visibility="collapsed")
            if idx < len(questions) - 1:
                st.divider()
                
        submit = st.form_submit_button("Submit Answers & Resume Triage", type="primary")
        
        if submit:
            # Validate they didn't submit totally empty
            if not any(answers.values()):
                st.error("Please answer at least one question to proceed.")
                st.stop()
                
            payload = {
                "thread_id": thread_id,
                "answers": answers
            }
            with st.spinner("Resuming AI pipeline and analyzing new data..."):
                resp = api_post("/api/triage/resume", payload, timeout=150)
                if resp:
                    if resp.get("status") == "complete":
                        # Save result and clear HITL state
                        st.session_state["triage_result"] = resp["result"]
                        st.session_state["hitl_thread_id"] = None
                        st.rerun()
                    elif resp.get("status") == "pending_clarification":
                        st.session_state["hitl_questions"] = resp["questions"]
                        st.session_state["hitl_reason"] = resp.get("reason", "")
                        st.rerun()
