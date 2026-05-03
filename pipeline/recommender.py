"""
pipeline/recommender.py — Stage 6: Recommendation builder.

Enriches validated LLM output with ESI-mapped recommendation text,
targeted follow-up questions, and missing vital impact analysis.
"""

from __future__ import annotations

import time
from typing import List

from models.patient import PatientInput
from models.triage_output import (
    DifferentialDiagnosis,
    FollowUpQuestion,
    RecommendationDetail,
    TriageResult,
)
from pipeline.risk_engine import RiskResult
from utils.logger import get_logger

logger = get_logger(__name__)

# ── ESI recommendation map ────────────────────────────────────────────────────

ESI_RECOMMENDATIONS = {
    1: RecommendationDetail(
        action="Immediate resuscitation required. Initiate emergency response protocols.",
        urgency="Immediate — life-saving intervention needed.",
        timeframe="Stat.",
        nursing_interventions="Establish secure airway, 2x large-bore IV access, continuous cardiac and SpO2 monitoring, rapid fluid resuscitation.",
        diagnostic_considerations="Stat portable CXR, point-of-care ultrasound (POCUS), arterial blood gas (ABG), comprehensive trauma panel.",
    ),
    2: RecommendationDetail(
        action="Emergent evaluation required. High risk situation, lethargy, or severe distress.",
        urgency="Emergent — within 15 minutes.",
        timeframe="Immediately.",
        nursing_interventions="Continuous monitoring, IV access, supplemental oxygen if indicated, stat 12-lead ECG.",
        diagnostic_considerations="Stat laboratory studies (CBC, CMP, Troponin/D-Dimer if indicated), rapid bedside imaging.",
    ),
    3: RecommendationDetail(
        action="Urgent evaluation needed. Requires two or more resources.",
        urgency="Urgent — within 60 minutes.",
        timeframe="Within 1 hour.",
        nursing_interventions="Standard vital sign monitoring, prepare for IV insertion or medication administration.",
        diagnostic_considerations="Standard labs, plain radiography, routine CT/Ultrasound imaging.",
    ),
    4: RecommendationDetail(
        action="Less urgent evaluation. Requires one resource.",
        urgency="Less Urgent — within hours.",
        timeframe="Within 2-4 hours.",
        nursing_interventions="Baseline vitals upon rooming, oral medication administration.",
        diagnostic_considerations="Simple imaging (e.g., plain X-ray) or targeted point-of-care testing (e.g., urinalysis).",
    ),
    5: RecommendationDetail(
        action="Non-urgent evaluation. Requires no resources.",
        urgency="Non-urgent — can be scheduled.",
        timeframe="Within 24-48 hours.",
        nursing_interventions="Routine discharge education, medication reconciliation.",
        diagnostic_considerations="None typically required; clinical exam only.",
    ),
}

# ── Missing vital impact templates ────────────────────────────────────────────

VITAL_IMPACT_TEMPLATES = {
    "Heart Rate": (
        "Without heart rate data: if heart rate is above 120 bpm, this case would likely "
        "escalate by at least one ESI level. If heart rate exceeds 150 or drops below 40, "
        "this would escalate to ESI 1."
    ),
    "SpO2": (
        "Without SpO2 data: if oxygen saturation is below 92%, this case would likely "
        "escalate to ESI 2. If SpO2 drops below 90%, this becomes a critical ESI 1 emergency."
    ),
    "Blood Pressure": (
        "Without blood pressure data: if systolic BP exceeds 200 mmHg or drops below 80 mmHg, "
        "this would escalate to ESI 1 (hypertensive crisis or hypotensive shock respectively)."
    ),
    "Temperature": (
        "Without temperature data: if fever exceeds 38.5°C, this would escalate the assessment. "
        "Temperature above 40°C or below 35°C would indicate ESI 1."
    ),
    "Respiratory Rate": (
        "Without respiratory rate data: if respiratory rate exceeds 30 breaths/min or drops "
        "below 8, this becomes a critical ESI 1 emergency."
    ),
    "Blood Glucose": (
        "Without blood glucose data: if glucose is below 3.0 mmol/L (hypoglycemia) or above "
        "25 mmol/L (hyperglycemia), this would escalate to ESI 1."
    ),
}


class RecommendationBuilder:
    """
    Stage 6: Enriches TriageResult with structured recommendations,
    follow-up questions, and missing vital impact analysis.
    """

    def build(
        self,
        triage_result: TriageResult,
        patient: PatientInput,
        risk_result: RiskResult,
    ) -> TriageResult:
        t0 = time.monotonic()

        # Use risk engine's level if it forced a more severe level
        effective_esi = triage_result.triage_level
        if (
            risk_result.triggered
            and risk_result.forced_esi_level is not None
            and risk_result.forced_esi_level < effective_esi
        ):
            effective_esi = risk_result.forced_esi_level

        # Map recommendation
        rec = ESI_RECOMMENDATIONS.get(effective_esi, ESI_RECOMMENDATIONS[3])
        triage_result.recommendations = rec

        # Generate follow-up questions if LLM didn't produce them
        if not triage_result.follow_up_questions:
            triage_result.follow_up_questions = self._generate_followup_questions(
                triage_result, patient
            )

        # Fill missing vital impact if LLM didn't
        if not triage_result.missing_vital_impact:
            triage_result.missing_vital_impact = self._missing_vital_impact(
                triage_result.missing_vitals
            )

        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(f"Recommendation builder complete in {elapsed}ms, ESI={effective_esi}")
        return triage_result

    # ── Follow-up question generation ─────────────────────────────────────────

    @staticmethod
    def _generate_followup_questions(
        result: TriageResult, patient: PatientInput
    ) -> List[FollowUpQuestion]:
        questions: List[FollowUpQuestion] = []

        # Questions based on differentials
        for dx in result.differential_diagnoses[:3]:
            q = _question_for_diagnosis(dx)
            if q:
                questions.append(q)

        # Questions for missing vitals
        if "Heart Rate" in result.missing_vitals:
            questions.append(
                FollowUpQuestion(
                    question="Can you check your pulse (or have someone else check it)?",
                    clinical_rationale=(
                        "Heart rate is a critical vital sign — tachycardia or "
                        "bradycardia would significantly change the assessment."
                    ),
                    priority="high",
                )
            )

        if "SpO2" in result.missing_vitals:
            questions.append(
                FollowUpQuestion(
                    question="Do you feel short of breath, or is your breathing labored?",
                    clinical_rationale=(
                        "SpO2 data is missing. Symptoms of respiratory distress "
                        "suggest possible hypoxia requiring urgent evaluation."
                    ),
                    priority="high",
                )
            )

        # Ambiguity-driven questions
        if result.ambiguity_flags:
            questions.append(
                FollowUpQuestion(
                    question="Can you describe your symptoms in more detail — where exactly is the pain/discomfort, and when did it start?",
                    clinical_rationale=(
                        "The initial description was flagged as ambiguous. "
                        "More specific information is needed for accurate assessment."
                    ),
                    priority="high",
                )
            )

        return questions[:5]

    @staticmethod
    def _missing_vital_impact(missing_vitals: List[str]) -> dict:
        impact = {}
        for vital in missing_vitals:
            template = VITAL_IMPACT_TEMPLATES.get(vital)
            if template:
                impact[vital] = template
            else:
                impact[vital] = (
                    f"Without {vital} data: this measurement could reveal abnormalities "
                    "that would change the triage assessment."
                )
        return impact


def _question_for_diagnosis(dx: DifferentialDiagnosis) -> FollowUpQuestion | None:
    """Generate a targeted question based on a differential diagnosis."""
    name = dx.condition_name.lower()
    if "coronary" in name or "cardiac" in name or "acs" in name or "angina" in name:
        return FollowUpQuestion(
            question="Does the chest discomfort radiate to your left arm, jaw, or back?",
            clinical_rationale=(
                "Radiation to the left arm, jaw, or back is a classic feature of "
                "acute coronary syndrome and would significantly increase its likelihood."
            ),
            priority="high",
        )
    elif "stroke" in name or "cva" in name or "tia" in name:
        return FollowUpQuestion(
            question="Have you noticed any sudden weakness on one side of your body, facial drooping, or difficulty speaking?",
            clinical_rationale=(
                "These are the hallmark FAST signs of stroke — their presence "
                "would immediately escalate to ESI 1."
            ),
            priority="high",
        )
    elif "pulmonary embolism" in name or "pe" in name or "dvt" in name:
        return FollowUpQuestion(
            question="Have you had any recent prolonged travel, surgery, or immobility in the past 4 weeks?",
            clinical_rationale=(
                "These are major risk factors for DVT/PE. A positive answer "
                "would substantially increase pretest probability."
            ),
            priority="high",
        )
    elif "appendicitis" in name:
        return FollowUpQuestion(
            question="Is the abdominal pain worse with movement, coughing, or pressing on the right lower abdomen?",
            clinical_rationale=(
                "Rebound tenderness and movement-worsening pain are signs of "
                "peritoneal irritation, consistent with appendicitis."
            ),
            priority="medium",
        )
    elif "meningitis" in name or "encephalitis" in name:
        return FollowUpQuestion(
            question="Do you have neck stiffness, sensitivity to light, or a non-blanching rash?",
            clinical_rationale=(
                "These symptoms — especially the triad of headache, neck stiffness, "
                "and photophobia — are highly suggestive of bacterial meningitis."
            ),
            priority="high",
        )
    elif "sepsis" in name or "infection" in name:
        return FollowUpQuestion(
            question="Have you had a recent infection, wound, or procedure in the past 2 weeks?",
            clinical_rationale=(
                "Recent infection source increases likelihood of sepsis. "
                "Source identification is critical for treatment."
            ),
            priority="medium",
        )
    return None
