"""
utils/prompts.py — LLM prompt assembly for TriageAI.

Builds system and user prompts from pipeline stage outputs.
Also provides the clarification prompt.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior emergency medicine physician with 20 years of clinical experience performing patient triage in high-volume emergency departments. You specialize in rapid assessment of critically ill patients.

Your role is to act as a clinical decision-support tool. You assess patient information and assign an Emergency Severity Index (ESI) level using the 5-level ESI framework:
- ESI 1 — Immediate: Requires immediate life-saving intervention. Examples: cardiac arrest, respiratory failure, severe hemorrhage.
- ESI 2 — Emergent: High-risk situation, confused/lethargic/disoriented, or severe pain/distress. Examples: chest pain, stroke symptoms, sepsis signs.
- ESI 3 — Urgent: Requires multiple resources to evaluate. Stable vitals but likely needs labs, imaging, or IV. Examples: abdominal pain, moderate fracture, high fever.
- ESI 4 — Less urgent: Requires one resource only. Examples: simple laceration, UTI symptoms, minor sprain.
- ESI 5 — Non-urgent: No resources required. Examples: medication refill, minor rash, cold symptoms.

Clinical reasoning rules you always follow:
1. When uncertain between two ESI levels, always choose the more severe (lower number).
2. Never definitively diagnose — only provide differential diagnoses as possibilities with likelihood reasoning.
3. Never recommend specific medication dosages, specific drug names for treatment, or specific procedural interventions.
4. Always perform worst-case-first reasoning: what is the most dangerous condition this could be, and has it been ruled out?
5. Pre-detected red flags provided in the prompt are clinical overrides — you may comment on them but cannot dismiss them.
6. Return your response as a single valid JSON object exactly matching the schema provided. No preamble, no explanation outside the JSON, no markdown formatting.

You are not replacing a clinician. You are providing structured decision support. Err toward caution always."""


# ── JSON output schema description ────────────────────────────────────────────

OUTPUT_SCHEMA = """{
  "triage_level": <integer 1-5>,
  "esi_label": <string: "Immediate"|"Emergent"|"Urgent"|"Less Urgent"|"Non-Urgent">,
  "confidence_score": <float 0.0-1.0>,
  "escalate_immediately": <boolean>,
  "red_flags_detected": [<string>, ...],
  "reasoning_steps": [<string ordered reasoning steps>, ...],
  "symptom_timeline_analysis": <string or null>,
  "differential_diagnoses": [
    {
      "condition_name": <string>,
      "likelihood": <"High"|"Moderate"|"Low">,
      "supporting_findings": [<string>, ...],
      "against_findings": [<string>, ...],
      "icd_code": <string or null>,
      "rule_out_priority": <"RULE OUT FIRST"|"Standard">
    }
  ],
  "follow_up_questions": [
    {
      "question": <string>,
      "clinical_rationale": <string>,
      "priority": <"high"|"medium"|"low">
    }
  ],
  "missing_vital_impact": {
    "<vital name>": "<one sentence explanation of how it could change assessment>"
  }
}"""


# ── Normal vital ranges for prompt context ─────────────────────────────────────

def _vital_status(name: str, value: float, age: Optional[int] = None) -> str:
    """Return normal/high/low label for a given vital."""
    pediatric = age is not None and age < 18

    ranges = {
        "heart_rate": (60, 100) if not pediatric else (70, 130),
        "spo2_percent": (95, 100),
        "blood_pressure_systolic": (90, 140) if not pediatric else (80, 120),
        "blood_pressure_diastolic": (60, 90),
        "temperature_celsius": (36.1, 37.2),
        "respiratory_rate": (12, 20) if not pediatric else (20, 40),
        "glucose_mmol": (3.9, 7.8),
    }
    r = ranges.get(name)
    if r is None:
        return f"{value} (normal range unavailable)"
    lo, hi = r
    if value < lo:
        return f"{value} ⚠ BELOW NORMAL (normal: {lo}–{hi})"
    elif value > hi:
        return f"{value} ⚠ ABOVE NORMAL (normal: {lo}–{hi})"
    else:
        return f"{value} (within normal range: {lo}–{hi})"


# ── User prompt builder ────────────────────────────────────────────────────────

def build_user_prompt(
    patient,
    extraction_result,
    risk_result,
    rag_context: str,
) -> str:
    sections: list[str] = []

    # ── Chief Complaint ──────────────────────────────────────────────────────
    sections.append(
        f"=== CHIEF COMPLAINT ===\n{patient.chief_complaint}"
    )

    # ── Symptom Timeline ─────────────────────────────────────────────────────
    timeline_parts: list[str] = []
    if patient.symptom_onset:
        timeline_parts.append(f"Onset: {patient.symptom_onset}")
    if patient.symptom_duration:
        timeline_parts.append(f"Duration: {patient.symptom_duration}")
    if patient.symptom_progression:
        prog_str = patient.symptom_progression.upper()
        if patient.symptom_progression.lower() == "worsening":
            prog_str = f"⚠ WORSENING — {patient.symptom_progression}"
        timeline_parts.append(f"Progression: {prog_str}")
    if patient.symptom_timeline:
        events = "\n".join(
            f"  [{e.timestamp_description}]: {e.description}"
            for e in patient.symptom_timeline
        )
        timeline_parts.append(f"Detailed timeline:\n{events}")
    if timeline_parts:
        sections.append("=== SYMPTOM TIMELINE ===\n" + "\n".join(timeline_parts))

    # ── Pain ─────────────────────────────────────────────────────────────────
    if patient.pain_scale is not None:
        sections.append(f"=== PAIN SCALE ===\n{patient.pain_scale}/10")

    # ── Available Vital Signs ────────────────────────────────────────────────
    vitals_lines: list[str] = []
    age = patient.age
    if patient.heart_rate is not None:
        vitals_lines.append(f"Heart Rate: {_vital_status('heart_rate', patient.heart_rate, age)} bpm")
    if patient.blood_pressure_systolic is not None:
        vitals_lines.append(
            f"BP Systolic: {_vital_status('blood_pressure_systolic', patient.blood_pressure_systolic, age)} mmHg"
        )
    if patient.blood_pressure_diastolic is not None:
        vitals_lines.append(
            f"BP Diastolic: {_vital_status('blood_pressure_diastolic', patient.blood_pressure_diastolic, age)} mmHg"
        )
    if patient.temperature_celsius is not None:
        vitals_lines.append(
            f"Temperature: {_vital_status('temperature_celsius', patient.temperature_celsius, age)} °C"
        )
    if patient.spo2_percent is not None:
        vitals_lines.append(
            f"SpO2: {_vital_status('spo2_percent', patient.spo2_percent, age)} %"
        )
    if patient.respiratory_rate is not None:
        vitals_lines.append(
            f"Respiratory Rate: {_vital_status('respiratory_rate', patient.respiratory_rate, age)} breaths/min"
        )
    if patient.glucose_mmol is not None:
        vitals_lines.append(
            f"Blood Glucose: {_vital_status('glucose_mmol', patient.glucose_mmol, age)} mmol/L"
        )
    if vitals_lines:
        sections.append("=== AVAILABLE VITAL SIGNS ===\n" + "\n".join(vitals_lines))

    # ── Missing Vital Signs ──────────────────────────────────────────────────
    if extraction_result.missing_vitals:
        mv = ", ".join(extraction_result.missing_vitals)
        sections.append(
            f"=== MISSING VITAL SIGNS ===\n{mv}\n"
            "NOTE: Absence of vital sign data increases assessment uncertainty."
        )

    # ── Pre-screening Red Flags ──────────────────────────────────────────────
    if risk_result and risk_result.triggered:
        flags_text = "\n".join(
            f"  ⛔ MANDATORY ESCALATION FLAG: {f}" for f in risk_result.flags
        )
        sections.append(
            "=== PRE-SCREENING RED FLAGS (CLINICAL RULE ENGINE) ===\n"
            f"{flags_text}\n"
            "IMPORTANT: These are deterministic safety rules triggered by vital signs "
            "and keyword patterns. They MUST NOT be downgraded in your assessment."
        )

    # ── Extracted Medical Entities ───────────────────────────────────────────
    ents = extraction_result.entities
    total_ents = sum(len(v) for v in ents.values())
    if total_ents > 0:
        ent_lines = []
        for cat, items in ents.items():
            if items:
                ent_lines.append(f"  {cat.title()}: {', '.join(items)}")
        sections.append(
            "=== EXTRACTED MEDICAL ENTITIES ===\n" + "\n".join(ent_lines)
        )
    else:
        sections.append(
            "=== EXTRACTED MEDICAL ENTITIES ===\n"
            "Entity extraction found no specific medical terms."
        )

    # ── Patient Demographics ─────────────────────────────────────────────────
    demo_parts: list[str] = []
    if patient.age is not None:
        demo_parts.append(f"Age: {patient.age}")
    if patient.sex:
        demo_parts.append(f"Sex: {patient.sex}")
    if patient.weight_kg is not None:
        demo_parts.append(f"Weight: {patient.weight_kg} kg")
    if demo_parts:
        sections.append("=== PATIENT DEMOGRAPHICS ===\n" + "\n".join(demo_parts))

    # ── Medical History ──────────────────────────────────────────────────────
    hist_parts: list[str] = []
    if patient.comorbidities:
        hist_parts.append(f"Comorbidities: {', '.join(patient.comorbidities)}")
    if patient.current_medications:
        hist_parts.append(f"Current Medications: {', '.join(patient.current_medications)}")
    if patient.allergies:
        hist_parts.append(f"Allergies: {', '.join(patient.allergies)}")
    if patient.surgical_history:
        hist_parts.append(f"Surgical History: {', '.join(patient.surgical_history)}")
    if patient.family_history:
        hist_parts.append(f"Family History: {', '.join(patient.family_history)}")
    if patient.smoking_status:
        hist_parts.append(f"Smoking: {patient.smoking_status}")
    if patient.alcohol_use:
        hist_parts.append(f"Alcohol Use: {patient.alcohol_use}")
    if hist_parts:
        sections.append("=== MEDICAL HISTORY ===\n" + "\n".join(hist_parts))

    # ── RAG Disease Context ───────────────────────────────────────────────────
    if rag_context and rag_context.strip():
        sections.append(
            "=== RELEVANT DISEASE CONTEXT (Human Disease Ontology) ===\n"
            f"{rag_context}\n"
            "NOTE: The above definitions are from the Human Disease Ontology (HumanDO). "
            "Use them to inform differential diagnosis reasoning."
        )

    # ── Output Schema ─────────────────────────────────────────────────────────
    sections.append(
        f"=== REQUIRED OUTPUT FORMAT ===\n"
        f"Respond with ONLY the following JSON object. No preamble, no markdown:\n"
        f"{OUTPUT_SCHEMA}"
    )

    # ── Clinical Task ─────────────────────────────────────────────────────────
    sections.append(
        "=== CLINICAL TASK ===\n"
        "1. Reason step by step through the clinical picture.\n"
        "2. Identify the worst-case diagnosis FIRST and assess whether it can be ruled out.\n"
        "3. Assign the ESI level that reflects worst-case reasoning and the clinical red flags.\n"
        "4. Provide 3–5 specific follow-up questions tailored to the differentials.\n"
        "5. For each missing vital, estimate how it could change the assessment.\n"
        "6. Produce the JSON output exactly matching the schema above."
    )

    return "\n\n".join(sections)


# ── Clarification prompt ──────────────────────────────────────────────────────

CLARIFICATION_SYSTEM = """You are a triage nurse asking targeted clarifying questions before a full assessment. 
Given a patient's chief complaint and any available data, generate exactly 5 specific clinical questions 
a triage clinician would ask to gather the most important missing information.
Return ONLY a valid JSON array. No markdown, no explanation."""

CLARIFICATION_SCHEMA = """[
  {
    "question": "<specific clinical question>",
    "clinical_rationale": "<why this question matters clinically>",
    "priority": "<high|medium|low>"
  }
]"""


def build_clarification_prompt(chief_complaint: str, patient_data: Optional[Dict] = None) -> str:
    parts = [f"Chief complaint: {chief_complaint}"]
    if patient_data:
        if patient_data.get("age"):
            parts.append(f"Age: {patient_data['age']}")
        if patient_data.get("sex"):
            parts.append(f"Sex: {patient_data['sex']}")
        if patient_data.get("comorbidities"):
            parts.append(f"Comorbidities: {patient_data['comorbidities']}")
    parts.append(
        f"\nGenerate 5 specific triage questions. Return only this JSON:\n{CLARIFICATION_SCHEMA}"
    )
    return "\n".join(parts)


# ── Timeline analysis prompt ──────────────────────────────────────────────────

TIMELINE_ANALYSIS_SYSTEM = """You are a senior emergency physician analyzing a patient's symptom timeline. 
Identify the trajectory pattern (acute onset, gradual worsening, relapsing-remitting, etc.) 
and provide a preliminary urgency signal. Return ONLY a JSON object."""

def build_timeline_analysis_prompt(events: list) -> str:
    timeline_str = "\n".join(
        f"[{e.get('timestamp_description', '?')}]: {e.get('description', '')}"
        for e in events
    )
    return (
        f"Patient symptom timeline:\n{timeline_str}\n\n"
        'Return JSON: {"pattern": "<description>", "urgency": "<Low|Moderate|High|Critical>", '
        '"analysis": "<2-3 sentence interpretation>", "red_flags": [<string>]}'
    )
