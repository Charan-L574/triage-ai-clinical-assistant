"""
models/triage_output.py — Output data models for TriageAI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Sub-models ────────────────────────────────────────────────────────────────


class DifferentialDiagnosis(BaseModel):
    condition_name: str
    likelihood: str  # "High" | "Moderate" | "Low"
    supporting_findings: List[str] = Field(default_factory=list)
    against_findings: List[str] = Field(default_factory=list)
    icd_code: Optional[str] = None
    rule_out_priority: str = "Standard"  # "RULE OUT FIRST" | "Standard"


class RecommendationDetail(BaseModel):
    action: str = ""
    urgency: str = ""
    timeframe: str = ""
    nursing_interventions: str = ""
    diagnostic_considerations: str = ""


class FollowUpQuestion(BaseModel):
    question: str
    clinical_rationale: str
    priority: str = "medium"  # "high" | "medium" | "low"


# ── Primary output ────────────────────────────────────────────────────────────


class TriageResult(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ESI assessment
    triage_level: int = Field(default=2, ge=1, le=5)
    esi_label: str = "Emergent"
    esi_color: str = "#EF9F27"
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_label: str = "Moderate"

    # Escalation
    escalate_immediately: bool = False
    red_flags_detected: List[str] = Field(default_factory=list)

    # Data quality
    missing_vitals: List[str] = Field(default_factory=list)
    ambiguity_flags: List[str] = Field(default_factory=list)
    data_completeness_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # NLP entities
    extracted_entities: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "symptoms": [],
            "body_parts": [],
            "conditions": [],
            "medications": [],
            "procedures": [],
        }
    )

    # Diagnoses
    differential_diagnoses: List[DifferentialDiagnosis] = Field(default_factory=list)

    # Reasoning
    reasoning_steps: List[str] = Field(default_factory=list)
    symptom_timeline_analysis: Optional[str] = None

    # Recommendations
    recommendations: RecommendationDetail = Field(
        default_factory=RecommendationDetail
    )
    follow_up_questions: List[FollowUpQuestion] = Field(default_factory=list)
    missing_vital_impact: Dict[str, str] = Field(default_factory=dict)

    # Safety
    safety_disclaimers: List[str] = Field(default_factory=list)

    # Pipeline metadata
    model_used: str = "unknown"
    model_response_time_ms: int = 0
    pipeline_stage_timings: Dict[str, int] = Field(default_factory=dict)
    raw_llm_output: str = ""
    rag_context_used: bool = False
    rag_extracted_diseases: List[Dict[str, Any]] = Field(default_factory=list)
    rule_engine_triggered: bool = False
    safety_override_applied: bool = False
    fast_tracked: bool = False
    hitl_applied: bool = False

    # ── Helper: map triage level to display properties ────────────────────────
    ESI_MAP: ClassVar[Dict[int, Dict[str, str]]] = {
        1: {"label": "Immediate", "color": "#E24B4A"},
        2: {"label": "Emergent", "color": "#EF9F27"},
        3: {"label": "Urgent", "color": "#FAC775"},
        4: {"label": "Less Urgent", "color": "#378ADD"},
        5: {"label": "Non-Urgent", "color": "#639922"},
    }

    model_config = {"arbitrary_types_allowed": True}

    def sync_esi_display(self) -> None:
        """Ensure esi_label and esi_color match triage_level."""
        mapping = self.ESI_MAP.get(self.triage_level, {"label": "Unknown", "color": "#888888"})
        self.esi_label = mapping["label"]
        self.esi_color = mapping["color"]
        if self.confidence_score >= 0.75:
            self.confidence_label = "High"
        elif self.confidence_score >= 0.50:
            self.confidence_label = "Moderate"
        else:
            self.confidence_label = "Low"


# ── Comparison output ─────────────────────────────────────────────────────────


class ModelComparisonResult(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    patient_summary: str = ""
    results_by_model: Dict[str, Any] = Field(default_factory=dict)
    agreement_score: float = 0.0
    consensus_triage_level: int = 2
    consensus_esi_label: str = "Emergent"
    disagreement_details: List[str] = Field(default_factory=list)
    most_conservative_model: str = ""
    fastest_model: str = ""
    highest_confidence_model: str = ""
    recommendation: str = ""
