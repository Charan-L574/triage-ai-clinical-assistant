"""
pipeline/safety_validator.py — Stage 7 (final gate): Safety validation and override.

Cannot be disabled. Applies risk engine overrides, confidence-based escalation,
dosage stripping, and mandatory disclaimers.
"""

from __future__ import annotations

import re
import time
from typing import List

import config
from models.patient import PatientInput
from models.triage_output import TriageResult
from pipeline.risk_engine import RiskResult
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Mandatory disclaimers (always appended) ────────────────────────────────────

MANDATORY_DISCLAIMERS = [
    "TriageAI is an AI decision-support prototype. It has not been clinically validated "
    "and must not replace professional medical judgment.",
    "If you believe you are experiencing a medical emergency, call your local emergency number immediately.",
    "This assessment is based solely on the information provided and may be incomplete.",
]

# ── Dosage pattern stripping ───────────────────────────────────────────────────

DOSAGE_PATTERNS = [
    re.compile(r"take\s+\d+\s*(?:mg|mcg|g|ml|units?)", re.IGNORECASE),
    re.compile(r"administer\s+\d+\s*(?:mg|mcg|g|ml|units?)", re.IGNORECASE),
    re.compile(r"give\s+\d+\s*(?:mg|mcg|g|ml|units?)", re.IGNORECASE),
    re.compile(r"dose\s+of\s+\d+\s*(?:mg|mcg|g|ml|units?)", re.IGNORECASE),
    re.compile(r"\d+\s*(?:mg|mcg|g|ml|units?)\s+(?:of\s+)?\w+", re.IGNORECASE),
]

DOSAGE_REPLACEMENT = "follow healthcare provider instructions for dosing"


class SafetyValidator:
    """
    Stage 7 — the final mandatory gate.
    Applies risk engine overrides, confidence escalation, dosage stripping,
    and appends all required disclaimers.
    """

    def validate(
        self,
        result: TriageResult,
        patient: PatientInput,
        risk_result: RiskResult,
    ) -> TriageResult:
        t0 = time.monotonic()

        # ── Risk engine ESI override ─────────────────────────────────────────
        if (
            risk_result.triggered
            and risk_result.forced_esi_level is not None
            and result.triage_level > risk_result.forced_esi_level
        ):
            old_level = result.triage_level
            result.triage_level = risk_result.forced_esi_level
            result.safety_override_applied = True
            msg = (
                f"SAFETY OVERRIDE: AI model assessed ESI {old_level}, but the clinical "
                f"rule engine detected critical red flags and has escalated to ESI "
                f"{risk_result.forced_esi_level}."
            )
            result.safety_disclaimers.insert(0, msg)
            logger.critical(
                f"Safety override applied: ESI {old_level} → {risk_result.forced_esi_level}"
            )

        # ── Ensure escalate_immediately consistency ──────────────────────────
        if result.triage_level == 1:
            result.escalate_immediately = True
        if risk_result.requires_immediate_escalation:
            result.escalate_immediately = True

        # ── Confidence-based escalation ──────────────────────────────────────
        if result.confidence_score < config.CONFIDENCE_THRESHOLD_LOW:
            result.safety_disclaimers.append(
                f"Low model confidence ({result.confidence_score:.0%}) — this assessment "
                "may be unreliable. Seek in-person evaluation for confirmation."
            )
        if result.confidence_score < config.CONFIDENCE_THRESHOLD_CRITICAL:
            if result.triage_level > 1:
                result.triage_level -= 1  # escalate (lower number = more severe)
                result.safety_disclaimers.append(
                    "Critical confidence threshold reached — triage level escalated by one "
                    "level due to high model uncertainty."
                )

        # ── Sync ESI label and color after any overrides ─────────────────────
        result.sync_esi_display()

        # ── Dosage stripping ─────────────────────────────────────────────────
        action = result.recommendations.action
        for pattern in DOSAGE_PATTERNS:
            action = pattern.sub(DOSAGE_REPLACEMENT, action)
        result.recommendations.action = action

        # ── Mandatory disclaimers ────────────────────────────────────────────
        for disc in MANDATORY_DISCLAIMERS:
            if disc not in result.safety_disclaimers:
                result.safety_disclaimers.append(disc)

        # ── Conditional disclaimers ──────────────────────────────────────────
        self._conditional_disclaimers(result, patient, risk_result)

        # ── Red flags from risk engine ────────────────────────────────────────
        for flag in risk_result.flags:
            if flag not in result.red_flags_detected:
                result.red_flags_detected.append(flag)

        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(
            f"Safety validator complete in {elapsed}ms, "
            f"final ESI={result.triage_level}, override={result.safety_override_applied}"
        )
        return result

    @staticmethod
    def _conditional_disclaimers(
        result: TriageResult, patient: PatientInput, risk_result: RiskResult
    ) -> None:
        discs = result.safety_disclaimers

        if patient.age is not None and patient.age < 18:
            _add_disc(
                discs,
                "This system has not been specifically validated for pediatric patients. "
                "Pediatric assessment requires specialist evaluation.",
            )
        if patient.age is not None and patient.age > 80:
            _add_disc(
                discs,
                "This system has not been specifically validated for geriatric patients. "
                "Older adults may present atypically.",
            )
        if patient.is_pregnant():
            _add_disc(
                discs,
                "Pregnancy significantly alters normal vital sign ranges and symptom presentation. "
                "Specialist obstetric evaluation is recommended.",
            )

        all_vitals_missing = not any(
            [
                patient.blood_pressure_systolic,
                patient.heart_rate,
                patient.temperature_celsius,
                patient.spo2_percent,
                patient.respiratory_rate,
                patient.glucose_mmol,
            ]
        )
        if all_vitals_missing:
            _add_disc(
                discs,
                "No vital signs were provided. This assessment is based on symptom description "
                "only and has significantly higher uncertainty.",
            )

        if result.safety_override_applied:
            _add_disc(
                discs,
                "The AI model's severity assessment was overridden by the clinical rule engine "
                "due to detected red flags.",
            )

    # ── Hardcoded fallback result ─────────────────────────────────────────────

    @staticmethod
    def emergency_fallback() -> TriageResult:
        """Return when all AI models are unavailable."""
        from models.triage_output import RecommendationDetail

        result = TriageResult(
            triage_level=2,
            esi_label="Emergent",
            esi_color="#EF9F27",
            confidence_score=0.0,
            confidence_label="Low",
            escalate_immediately=True,
            model_used="emergency_fallback",
        )
        result.recommendations = RecommendationDetail(
            action=(
                "Go directly to an emergency department now. "
                "All AI assessment systems are currently unavailable."
            ),
            urgency="Emergent — within 15 minutes.",
            timeframe="Immediately.",
            location_of_care="Emergency department.",
            what_to_bring="ID, list of medications, list of allergies.",
            what_to_avoid="Do not delay seeking care.",
        )
        result.safety_disclaimers = [
            "All AI systems are currently unavailable. This is an automated safety escalation. "
            "Please seek immediate in-person medical evaluation.",
        ] + MANDATORY_DISCLAIMERS
        return result


def _add_disc(discs: List[str], msg: str) -> None:
    if msg not in discs:
        discs.append(msg)
