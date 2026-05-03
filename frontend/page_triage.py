"""
pages/page_triage.py — Page 1: Triage Assessment (3 input modes + results).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from frontend.ui_helpers import api_post, api_get, render_result_column, vital_status_js_indicator, render_hitl_panel

st.title("🏥 Triage Assessment")
st.caption("Enter patient information in any of the three modes below.")

# ── Shared age state for pediatric vital indicators ────────────────────────────
if "triage_age" not in st.session_state:
    st.session_state["triage_age"] = None
if "hitl_thread_id" not in st.session_state:
    st.session_state["hitl_thread_id"] = None
if "hitl_questions" not in st.session_state:
    st.session_state["hitl_questions"] = []
if "hitl_reason" not in st.session_state:
    st.session_state["hitl_reason"] = ""

left, right = st.columns([55, 45])

with left:
    tab1, tab2, tab3 = st.tabs(["📝 Form Input", "💬 Free-Text", "📅 Timeline Builder"])

    # ════════════════════════════════════════════════════════════════════════
    # MODE 1 — Form Input
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        with st.expander("Chief Complaint", expanded=True):
            complaint = st.text_area("Describe the main complaint *", height=100,
                                      placeholder="e.g. Severe chest pain radiating to left arm, started 30 minutes ago",
                                      key="form_complaint")
            c1, c2 = st.columns(2)
            duration = c1.text_input("Duration", placeholder="e.g. 2 hours", key="form_dur")
            onset = c2.selectbox("Onset", ["Unknown", "Sudden", "Gradual"], key="form_onset")
            progression = st.selectbox("Progression", ["Unknown", "Worsening", "Stable", "Improving"], key="form_prog")

            if complaint and len(complaint.strip()) < 15:
                st.warning("⚠️ Chief complaint is very short. Consider adding more detail, or use the **Clarify** button.")

            if st.button("🔍 Clarify Before Triage", key="form_clarify"):
                with st.spinner("Getting clarifying questions..."):
                    resp = api_post("/api/clarify", {"chief_complaint": complaint or "N/A"})
                    if resp:
                        qs = resp.get("questions", [])
                        if qs:
                            st.session_state["clarify_qs"] = qs
                            st.success("Answer these questions to improve accuracy:")
                        else:
                            st.info("No additional clarification needed.")

            if "clarify_qs" in st.session_state:
                clarify_answers = []
                for i, q in enumerate(st.session_state["clarify_qs"], 1):
                    ans = st.text_input(f"Q{i}: {q.get('question','')}", key=f"cq_{i}")
                    clarify_answers.append(ans)
                st.session_state["clarify_answers"] = clarify_answers

        with st.expander("Vital Signs", expanded=True):
            age_v = st.session_state.get("triage_age")
            v1, v2 = st.columns(2)
            hr = v1.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=0, key="f_hr") or None
            spo2 = v2.number_input("SpO2 (%)", min_value=0, max_value=100, value=0, key="f_spo2") or None
            sbp = v1.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=0, key="f_sbp") or None
            temp = v2.number_input("Temperature (°C)", min_value=0.0, max_value=45.0, value=0.0, step=0.1, key="f_temp") or None
            rr = v1.number_input("Respiratory Rate", min_value=0, max_value=60, value=0, key="f_rr") or None
            gluc = v2.number_input("Blood Glucose (mmol/L)", min_value=0.0, max_value=60.0, value=0.0, step=0.1, key="f_gluc") or None

            # Live vital indicators
            for field, val, label in [
                ("heart_rate", hr, "HR"), ("spo2_percent", spo2, "SpO2"),
                ("blood_pressure_systolic", sbp, "BP"), ("temperature_celsius", temp, "Temp"),
                ("respiratory_rate", rr, "RR"), ("glucose_mmol", gluc, "Glucose"),
            ]:
                if val and val > 0:
                    dot, msg = vital_status_js_indicator(val, field, age_v)
                    if msg:
                        st.markdown(f"{dot} **{label}:** {msg}")

        with st.expander("Patient Details"):
            pc1, pc2, pc3 = st.columns(3)
            age = pc1.number_input("Age", min_value=0, max_value=120, value=0, key="f_age") or None
            if age:
                st.session_state["triage_age"] = age
            sex = pc2.selectbox("Sex", ["Not specified", "Male", "Female", "Other"], key="f_sex")
            weight = pc3.number_input("Weight (kg)", min_value=0.0, value=0.0, key="f_weight") or None

        with st.expander("Medical History"):
            comorbid_raw = st.text_input("Comorbidities (comma-separated)", key="f_comorbid")
            meds_raw = st.text_input("Current Medications (comma-separated)", key="f_meds")
            allergy_raw = st.text_input("Allergies (comma-separated)", key="f_allergy")
            surg_raw = st.text_area("Surgical History", height=60, key="f_surg")
            fam_raw = st.text_area("Family History", height=60, key="f_fam")
            hc1, hc2 = st.columns(2)
            smoking = hc1.selectbox("Smoking Status", ["Not specified", "Never", "Ex-smoker", "Current"], key="f_smoke")
            alcohol = hc2.selectbox("Alcohol Use", ["Not specified", "None", "Occasional", "Regular", "Heavy"], key="f_alc")

        pain = st.slider("Pain Scale", 0, 10, 0, key="f_pain",
                         help="0 = No pain, 10 = Worst possible pain")
        st.caption(f"Selected pain level: **{pain}/10**")

        submitted_form = st.button("🚀 Run Triage Assessment", type="primary", key="form_submit",
                                    use_container_width=True, disabled=not complaint)

        if submitted_form and complaint:
            clarify_text = ""
            if "clarify_answers" in st.session_state:
                answers = [a for a in st.session_state["clarify_answers"] if a.strip()]
                if answers:
                    clarify_text = " Additional info: " + ". ".join(answers)

            payload = {
                "chief_complaint": complaint + clarify_text,
                "symptom_duration": duration or None,
                "symptom_onset": onset.lower() if onset != "Unknown" else None,
                "symptom_progression": progression.lower() if progression != "Unknown" else None,
                "pain_scale": pain if pain > 0 else None,
                "heart_rate": int(hr) if hr else None,
                "spo2_percent": int(spo2) if spo2 else None,
                "blood_pressure_systolic": int(sbp) if sbp else None,
                "temperature_celsius": float(temp) if temp else None,
                "respiratory_rate": int(rr) if rr else None,
                "glucose_mmol": float(gluc) if gluc else None,
                "age": int(age) if age else None,
                "sex": sex.lower() if sex != "Not specified" else None,
                "weight_kg": float(weight) if weight else None,
                "comorbidities": [x.strip() for x in comorbid_raw.split(",") if x.strip()],
                "current_medications": [x.strip() for x in meds_raw.split(",") if x.strip()],
                "allergies": [x.strip() for x in allergy_raw.split(",") if x.strip()],
                "surgical_history": [surg_raw.strip()] if surg_raw.strip() else [],
                "family_history": [fam_raw.strip()] if fam_raw.strip() else [],
                "smoking_status": smoking if smoking != "Not specified" else None,
                "alcohol_use": alcohol if alcohol != "Not specified" else None,
                "input_mode": "form",
            }
            with st.spinner("Running AI pipeline..."):
                resp = api_post("/api/triage/start", payload, timeout=150)
                if resp:
                    if resp.get("status") == "pending_clarification":
                        st.session_state["hitl_thread_id"] = resp["thread_id"]
                        st.session_state["hitl_questions"] = resp["questions"]
                        st.session_state["hitl_reason"] = resp.get("reason", "")
                        st.rerun()
                    elif resp.get("status") == "complete":
                        st.session_state["triage_result"] = resp["result"]
                        st.session_state["hitl_thread_id"] = None

    # ════════════════════════════════════════════════════════════════════════
    # MODE 2 — Free-text
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        ft_text = st.text_area(
            "Describe the patient's condition in your own words",
            height=200,
            placeholder=(
                "e.g. I'm a 52-year-old man with diabetes and high blood pressure. "
                "Since this morning I've had a crushing pain in my chest that goes into my left arm. "
                "I'm sweating a lot and feel nauseous. The pain is about 8 out of 10. "
                "I take metformin and lisinopril. No allergies."
            ),
            key="ft_text",
        )

        # Live entity preview
        if ft_text and len(ft_text) > 20:
            from frontend.ui_helpers import render_entity_tags
            # Client-side keyword scan (no API)
            kw_map = {
                "symptoms": ["pain","fever","nausea","vomiting","dizziness","cough","breathless",
                             "swelling","headache","rash","fatigue","weakness","chest pain"],
                "body_parts": ["chest","arm","leg","head","neck","abdomen","back","shoulder","heart","lung"],
                "conditions": ["diabetes","hypertension","asthma","copd","stroke","angina","allergy"],
                "medications": ["metformin","insulin","lisinopril","aspirin","ibuprofen","warfarin"],
            }
            preview_ents = {}
            txt_lower = ft_text.lower()
            for cat, terms in kw_map.items():
                found = [t for t in terms if t in txt_lower]
                if found:
                    preview_ents[cat] = found
            if preview_ents:
                st.markdown("**🔎 Detected terms:**")
                render_entity_tags(preview_ents)
            else:
                st.warning("No medical terms detected — try being more specific.")

        st.markdown("**Optional vital signs:**")
        fv1, fv2, fv3 = st.columns(3)
        ft_hr = fv1.number_input("Heart Rate", 0, 300, 0, key="ft_hr") or None
        ft_spo2 = fv2.number_input("SpO2 %", 0, 100, 0, key="ft_spo2") or None
        ft_sbp = fv3.number_input("Systolic BP", 0, 300, 0, key="ft_sbp") or None

        submitted_ft = st.button("🚀 Extract and Run Triage", type="primary", key="ft_submit",
                                  use_container_width=True, disabled=not ft_text)
        if submitted_ft and ft_text:
            payload = {
                "chief_complaint": ft_text,
                "heart_rate": int(ft_hr) if ft_hr else None,
                "spo2_percent": int(ft_spo2) if ft_spo2 else None,
                "blood_pressure_systolic": int(ft_sbp) if ft_sbp else None,
                "input_mode": "freetext",
            }
            with st.spinner("Running AI pipeline..."):
                resp = api_post("/api/triage/start", payload, timeout=150)
                if resp:
                    if resp.get("status") == "pending_clarification":
                        st.session_state["hitl_thread_id"] = resp["thread_id"]
                        st.session_state["hitl_questions"] = resp["questions"]
                        st.session_state["hitl_reason"] = resp.get("reason", "")
                        st.rerun()
                    elif resp.get("status") == "complete":
                        st.session_state["triage_result"] = resp["result"]
                        st.session_state["hitl_thread_id"] = None

    # ════════════════════════════════════════════════════════════════════════
    # MODE 3 — Timeline Builder
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.caption("Add symptoms in chronological order (oldest first).")

        if "timeline_events" not in st.session_state:
            st.session_state["timeline_events"] = []

        # Add event
        tl1, tl2 = st.columns([1, 2])
        tl_when = tl1.text_input("When?", placeholder="e.g. 3 days ago", key="tl_when")
        tl_what = tl2.text_input("What happened?", placeholder="e.g. Mild headache started", key="tl_what")

        tlb1, tlb2 = st.columns(2)
        if tlb1.button("➕ Add to Timeline", key="tl_add"):
            if tl_when and tl_what:
                st.session_state["timeline_events"].append(
                    {"timestamp_description": tl_when, "description": tl_what}
                )
                st.rerun()
        if tlb2.button("↩️ Remove Last", key="tl_remove") and st.session_state["timeline_events"]:
            st.session_state["timeline_events"].pop()
            st.rerun()

        # Render timeline
        events = st.session_state["timeline_events"]
        if events:
            st.markdown("**📅 Your Timeline:**")
            n = len(events)
            for i, ev in enumerate(events):
                worsening = any(w in ev["description"].lower()
                                for w in ["worse","severe","unbearable","can't breathe","cannot breathe"])
                dot_color = "#f44336" if i == n-1 and worsening else ("#ff9800" if i >= n//2 else "#888")
                st.markdown(
                    f'<span class="timeline-dot" style="background:{dot_color}"></span>'
                    f'**{ev["timestamp_description"]}** — {ev["description"]}',
                    unsafe_allow_html=True,
                )

            # Trajectory preview
            if n >= 2:
                last = events[-1]["description"].lower()
                escalating = any(w in last for w in ["worse","severe","unbearable","critical","can't breathe"])
                pattern = "⚠️ **Escalating pattern detected**" if escalating else "📊 Gradual progression"
                st.info(f"{pattern} — {n} events over the reported period")

        # Vitals for timeline mode
        with st.expander("Vital Signs & Patient Details"):
            tv1, tv2 = st.columns(2)
            tl_hr = tv1.number_input("Heart Rate", 0, 300, 0, key="tl_hr") or None
            tl_spo2 = tv2.number_input("SpO2 %", 0, 100, 0, key="tl_spo2") or None
            tl_sbp = tv1.number_input("Systolic BP", 0, 300, 0, key="tl_sbp") or None
            tl_temp = tv2.number_input("Temperature °C", 0.0, 45.0, 0.0, key="tl_temp") or None
            tl_age = tv1.number_input("Age", 0, 120, 0, key="tl_age") or None
            tl_sex = tv2.selectbox("Sex", ["Not specified","Male","Female","Other"], key="tl_sex")

        submitted_tl = st.button("🚀 Run Triage with Timeline", type="primary", key="tl_submit",
                                  use_container_width=True, disabled=len(events) == 0)
        if submitted_tl and events:
            chief = " → ".join(e["description"] for e in events)
            payload = {
                "chief_complaint": chief,
                "symptom_timeline": events,
                "heart_rate": int(tl_hr) if tl_hr else None,
                "spo2_percent": int(tl_spo2) if tl_spo2 else None,
                "blood_pressure_systolic": int(tl_sbp) if tl_sbp else None,
                "temperature_celsius": float(tl_temp) if tl_temp else None,
                "age": int(tl_age) if tl_age else None,
                "sex": tl_sex.lower() if tl_sex != "Not specified" else None,
                "input_mode": "timeline",
            }
            with st.spinner("Running AI pipeline with timeline..."):
                resp = api_post("/api/triage/start", payload, timeout=150)
                if resp:
                    if resp.get("status") == "pending_clarification":
                        st.session_state["hitl_thread_id"] = resp["thread_id"]
                        st.session_state["hitl_questions"] = resp["questions"]
                        st.session_state["hitl_reason"] = resp.get("reason", "")
                        st.rerun()
                    elif resp.get("status") == "complete":
                        st.session_state["triage_result"] = resp["result"]
                        st.session_state["hitl_thread_id"] = None

# ── Right column: results ──────────────────────────────────────────────────────
with right:
    # HITL Clarification Panel takes priority
    if st.session_state.get("hitl_thread_id"):
        render_hitl_panel(
            thread_id=st.session_state["hitl_thread_id"],
            questions=st.session_state["hitl_questions"],
            reason=st.session_state["hitl_reason"],
        )
    elif "triage_result" not in st.session_state:
        st.info("ℹ️ **ESI Scale Reference**")
        for lvl, (label, color, ex) in {
            1: ("Immediate", "#E24B4A", "Cardiac arrest, respiratory failure"),
            2: ("Emergent", "#EF9F27", "Chest pain, stroke symptoms"),
            3: ("Urgent", "#FAC775", "Abdominal pain, high fever"),
            4: ("Less Urgent", "#378ADD", "UTI, minor sprain"),
            5: ("Non-Urgent", "#639922", "Cold symptoms, medication refill"),
        }.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                f'<div style="background:{color};color:white;font-weight:700;padding:2px 10px;border-radius:4px;min-width:40px;text-align:center;">ESI {lvl}</div>'
                f'<div><b>{label}</b> — {ex}</div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Submit a patient case on the left to see the AI assessment here.")
    else:
        render_result_column(st.session_state["triage_result"])
        if st.button("🔄 Clear Results", key="clear_res"):
            del st.session_state["triage_result"]
            st.rerun()
