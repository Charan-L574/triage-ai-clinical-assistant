"""
utils/report_generator.py — Exportable clinical handoff report generator.
"""

from __future__ import annotations

from datetime import datetime, timezone

import config
from models.patient import PatientInput
from models.triage_output import TriageResult


class ReportGenerator:
    """Generates a formatted plain-text clinical handoff report."""

    @staticmethod
    def generate_text_report(result: TriageResult, patient: PatientInput) -> str:
        lines: list[str] = []

        def section(title: str) -> None:
            lines.append("")
            lines.append("=" * 60)
            lines.append(f"  {title}")
            lines.append("=" * 60)

        def row(label: str, value) -> None:
            lines.append(f"  {label:<30} {value}")

        # ── Header ─────────────────────────────────────────────────────────────
        lines.append("=" * 60)
        lines.append("   TriageAI Clinical Triage Report")
        lines.append("   *** FOR DECISION-SUPPORT USE ONLY — NOT FOR CLINICAL USE ***")
        lines.append("=" * 60)
        row("Session ID:", result.session_id)
        row("Report Generated:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        row("Assessment Timestamp:", result.timestamp)

        # ── Patient Summary ────────────────────────────────────────────────────
        section("PATIENT SUMMARY")
        row("Chief Complaint:", patient.chief_complaint[:200])
        if patient.age:
            row("Age:", patient.age)
        if patient.sex:
            row("Sex:", patient.sex)
        if patient.symptom_duration:
            row("Symptom Duration:", patient.symptom_duration)
        if patient.symptom_onset:
            row("Onset:", patient.symptom_onset)
        if patient.symptom_progression:
            row("Progression:", patient.symptom_progression)
        if patient.comorbidities:
            row("Comorbidities:", ", ".join(patient.comorbidities))
        if patient.current_medications:
            row("Medications:", ", ".join(patient.current_medications))
        if patient.allergies:
            row("Allergies:", ", ".join(patient.allergies))

        # ── Vital Signs ────────────────────────────────────────────────────────
        section("VITAL SIGNS")
        vitals_provided = False
        if patient.heart_rate is not None:
            row("Heart Rate:", f"{patient.heart_rate} bpm")
            vitals_provided = True
        if patient.blood_pressure_systolic is not None:
            row("Blood Pressure:", f"{patient.blood_pressure_systolic}/{patient.blood_pressure_diastolic} mmHg")
            vitals_provided = True
        if patient.temperature_celsius is not None:
            row("Temperature:", f"{patient.temperature_celsius} °C")
            vitals_provided = True
        if patient.spo2_percent is not None:
            row("SpO2:", f"{patient.spo2_percent}%")
            vitals_provided = True
        if patient.respiratory_rate is not None:
            row("Respiratory Rate:", f"{patient.respiratory_rate} breaths/min")
            vitals_provided = True
        if patient.glucose_mmol is not None:
            row("Blood Glucose:", f"{patient.glucose_mmol} mmol/L")
            vitals_provided = True
        if not vitals_provided:
            lines.append("  No vital signs provided.")
        if result.missing_vitals:
            lines.append(f"\n  MISSING: {', '.join(result.missing_vitals)}")

        # ── Triage Assessment ──────────────────────────────────────────────────
        section("TRIAGE ASSESSMENT")
        row("ESI Level:", f"ESI {result.triage_level} — {result.esi_label}")
        row("Confidence:", f"{result.confidence_score:.0%} ({result.confidence_label})")
        row("Escalate Immediately:", "YES ⚠" if result.escalate_immediately else "No")
        row("Data Completeness:", f"{result.data_completeness_score:.0%}")

        # ── Red Flags ──────────────────────────────────────────────────────────
        if result.red_flags_detected:
            section("DETECTED RED FLAGS")
            for flag in result.red_flags_detected:
                lines.append(f"  ⛔ {flag}")

        # ── Differential Diagnoses ─────────────────────────────────────────────
        if result.differential_diagnoses:
            section("DIFFERENTIAL DIAGNOSES")
            for i, dx in enumerate(result.differential_diagnoses, 1):
                priority = " [RULE OUT FIRST]" if dx.rule_out_priority == "RULE OUT FIRST" else ""
                lines.append(f"  {i}. {dx.condition_name} — {dx.likelihood}{priority}")
                if dx.icd_code:
                    lines.append(f"     ICD: {dx.icd_code}")
                if dx.supporting_findings:
                    lines.append(f"     Supporting: {'; '.join(dx.supporting_findings)}")
                if dx.against_findings:
                    lines.append(f"     Against: {'; '.join(dx.against_findings)}")

        # ── Clinical Reasoning ─────────────────────────────────────────────────
        if result.reasoning_steps:
            section("CLINICAL REASONING")
            for i, step in enumerate(result.reasoning_steps, 1):
                lines.append(f"  {i}. {step}")

        # ── Recommendation ─────────────────────────────────────────────────────
        section("RECOMMENDATION")
        rec = result.recommendations
        row("Action:", rec.action)
        row("Urgency:", rec.urgency)
        row("Timeframe:", rec.timeframe)
        row("Location/Nursing:", rec.nursing_interventions)
        row("Diagnostics:", rec.diagnostic_considerations)

        # ── Follow-up Questions ────────────────────────────────────────────────
        if result.follow_up_questions:
            section("FOLLOW-UP QUESTIONS")
            for i, fq in enumerate(result.follow_up_questions, 1):
                lines.append(f"  {i}. [{fq.priority.upper()}] {fq.question}")
                lines.append(f"     Rationale: {fq.clinical_rationale}")

        # ── Model Information ──────────────────────────────────────────────────
        section("MODEL INFORMATION")
        row("Model Used:", result.model_used)
        row("Response Time:", f"{result.model_response_time_ms} ms")
        row("RAG Context Used:", str(result.rag_context_used))
        row("Rule Engine Triggered:", str(result.rule_engine_triggered))
        row("Safety Override Applied:", str(result.safety_override_applied))

        # ── Safety Disclaimers ─────────────────────────────────────────────────
        section("SAFETY DISCLAIMERS")
        for disc in result.safety_disclaimers:
            lines.append(f"  • {disc}")

        # ── Footer ─────────────────────────────────────────────────────────────
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  TriageAI v{config.APP_VERSION} — Research Prototype")
        lines.append(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("=" * 60)

        return "\n".join(lines)
