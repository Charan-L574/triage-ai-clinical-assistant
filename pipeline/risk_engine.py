"""
pipeline/risk_engine.py — Stage 2: Deterministic clinical rule engine.

Runs entirely offline. No API calls. Cannot be overridden.
Evaluates vital sign thresholds and keyword-based red-flag rules.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskResult:
    triggered: bool = False
    flags: List[str] = field(default_factory=list)
    forced_esi_level: Optional[int] = None
    requires_immediate_escalation: bool = False
    rule_count_triggered: int = 0

    def _update_esi(self, level: int) -> None:
        """Keep forced ESI at most severe (lowest number)."""
        if self.forced_esi_level is None or level < self.forced_esi_level:
            self.forced_esi_level = level
        self.triggered = True
        self.rule_count_triggered += 1
        if level == 1:
            self.requires_immediate_escalation = True

    def flag_esi1(self, msg: str) -> None:
        self.flags.append(msg)
        self._update_esi(1)

    def flag_esi2_min(self, msg: str) -> None:
        self.flags.append(msg)
        self._update_esi(2)


class RiskEngine:
    """
    Stage 2: Hard-coded clinical safety rule evaluation.
    The only component that can force ESI upward.
    Never raises exceptions to the caller.
    """

    def evaluate(self, patient, extracted) -> RiskResult:
        t0 = time.monotonic()
        result = RiskResult()
        try:
            self._vital_rules(patient, result)
            self._keyword_rules(patient, result)
            self._pediatric_rules(patient, result)
            self._pain_rule(patient, result)
        except Exception as exc:
            logger.error(f"Risk engine evaluation error: {exc}")

        elapsed = int((time.monotonic() - t0) * 1000)
        if result.triggered:
            logger.warning(
                f"Risk engine triggered {result.rule_count_triggered} rule(s), "
                f"forced ESI={result.forced_esi_level}, "
                f"escalate={result.requires_immediate_escalation}, "
                f"time={elapsed}ms"
            )
        else:
            logger.info(f"Risk engine: no rules triggered ({elapsed}ms)")
        return result

    # ── Vital sign threshold rules ─────────────────────────────────────────────

    @staticmethod
    def _vital_rules(patient, r: RiskResult) -> None:
        # SpO2
        spo2 = patient.spo2_percent
        if spo2 is not None:
            if spo2 < 90:
                r.flag_esi1(
                    f"Critical hypoxia — SpO2 at {spo2}% is life-threatening"
                )
            elif 90 <= spo2 <= 93:
                r.flag_esi2_min(
                    f"Low SpO2 — requires urgent evaluation (SpO2 {spo2}%)"
                )

        # Heart rate
        hr = patient.heart_rate
        if hr is not None:
            if hr > 150:
                r.flag_esi1(
                    f"Critical tachycardia — heart rate at {hr} bpm"
                )
            elif hr < 40:
                r.flag_esi1(
                    f"Critical bradycardia — heart rate at {hr} bpm"
                )
            elif 100 <= hr <= 150:
                r.flag_esi2_min(
                    f"Elevated heart rate — possible tachycardia ({hr} bpm)"
                )

        # Systolic BP
        sbp = patient.blood_pressure_systolic
        if sbp is not None:
            if sbp > 200:
                r.flag_esi1(
                    f"Hypertensive crisis — systolic BP at {sbp} mmHg"
                )
            elif sbp < 80:
                r.flag_esi1(
                    f"Hypotensive shock — systolic BP at {sbp} mmHg"
                )

        # Temperature
        temp = patient.temperature_celsius
        if temp is not None:
            if temp > 40.0:
                r.flag_esi1(
                    f"Hyperpyrexia — temperature at {temp}°C"
                )
            elif temp < 35.0:
                r.flag_esi1(
                    f"Hypothermia — temperature at {temp}°C"
                )
            elif 38.5 <= temp <= 40.0:
                r.flag_esi2_min(
                    f"High fever present ({temp}°C)"
                )

        # Respiratory rate
        rr = patient.respiratory_rate
        if rr is not None:
            if rr > 30:
                r.flag_esi1(
                    f"Critical tachypnea — {rr} breaths per minute"
                )
            elif rr < 8:
                r.flag_esi1(
                    f"Critical bradypnea — {rr} breaths per minute"
                )

        # Blood glucose
        glucose = patient.glucose_mmol
        if glucose is not None:
            if glucose < 3.0:
                r.flag_esi1(
                    f"Critical hypoglycemia — {glucose} mmol/L"
                )
            elif glucose > 25.0:
                r.flag_esi1(
                    f"Critical hyperglycemia — {glucose} mmol/L"
                )

    # ── Pain rule ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pain_rule(patient, r: RiskResult) -> None:
        if patient.pain_scale is not None and patient.pain_scale == 10:
            r.flag_esi2_min("Pain scale 10/10 — maximum reported pain intensity")

    # ── Keyword red-flag rules ─────────────────────────────────────────────────

    @staticmethod
    def _keyword_rules(patient, r: RiskResult) -> None:
        texts = []
        if patient.chief_complaint:
            texts.append(patient.chief_complaint.lower())
        if patient.additional_notes:
            texts.append(patient.additional_notes.lower())
        combined = " ".join(texts)

        def has(*phrases) -> bool:
            return any(p in combined for p in phrases)

        # ACS pattern
        chest = has("chest pain", "chest tightness")
        acs_assoc = has(
            "shortness of breath", "sweating", "left arm", "jaw pain", "diaphoresis"
        )
        if chest and acs_assoc:
            r.flag_esi1(
                "Possible acute coronary syndrome — call emergency services immediately"
            )

        # Stroke / neuro emergency
        if has(
            "stroke", "facial droop", "face drooping", "arm weakness",
            "sudden severe headache", "worst headache", "thunderclap headache",
            "slurred speech", "sudden vision loss",
        ):
            r.flag_esi1(
                "Possible cerebrovascular accident or neurological emergency — "
                "call emergency services"
            )

        # Cardiac/respiratory arrest
        if has(
            "unconscious", "unresponsive", "not breathing", "no pulse",
            "collapsed", "fell unconscious",
        ):
            r.flag_esi1(
                "Possible cardiorespiratory arrest — call emergency services now"
            )

        # Mental health emergency
        if has(
            "suicidal", "want to die", "end my life", "kill myself",
            "self harm", "overdose on purpose",
        ):
            r.flag_esi1(
                "Mental health emergency — immediate psychiatric evaluation required"
            )

        # Anaphylaxis
        anaphylaxis_core = has("anaphylaxis")
        allergic = has("allergic reaction")
        airway_sx = has(
            "throat closing", "can't breathe", "cannot breathe", "swelling", "hives"
        )
        if anaphylaxis_core or (allergic and airway_sx):
            r.flag_esi1(
                "Possible anaphylaxis — epinephrine may be required"
            )

        # Surgical abdomen
        if has("severe abdominal pain") and has("rigid", "board-like", "rebound tenderness"):
            r.flag_esi1(
                "Possible surgical abdomen — immediate evaluation required"
            )

        # Toxic ingestion
        if has("overdose", "took too many", "ingested", "swallowed"):
            r.flag_esi1(
                "Possible toxic ingestion — contact poison control and emergency services"
            )

        # Airway emergency
        if has("drowning", "near drowning", "choking", "foreign body airway"):
            r.flag_esi1(
                "Airway emergency — call emergency services"
            )

        # Burns
        if has(
            "severe burn", "burns to face", "burns to airway", "chemical burn"
        ):
            r.flag_esi1(
                "Significant burn injury — emergency evaluation required"
            )

    # ── Pediatric rules ────────────────────────────────────────────────────────

    @staticmethod
    def _pediatric_rules(patient, r: RiskResult) -> None:
        age = patient.age
        if age is None:
            return

        # Neonatal fever
        temp = patient.temperature_celsius
        if age < 1 and temp is not None and temp > 38.0:  # age in years, <1 = under 12 months
            # More granular: spec says under 3 months
            # We store age as integer years. Under 1 year → flag
            r.flag_esi1(
                "Neonatal fever — any fever in infants under 1 year requires "
                "immediate evaluation regardless of other signs"
            )

        # Pediatric tachypnea
        rr = patient.respiratory_rate
        if age < 2 and rr is not None and rr > 50:
            r.flag_esi1(
                f"Pediatric tachypnea in infant — respiratory rate {rr} bpm"
            )
