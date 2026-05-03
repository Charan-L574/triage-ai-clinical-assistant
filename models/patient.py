"""
models/patient.py — PatientInput Pydantic v2 data model.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SymptomEvent(BaseModel):
    """A single timestamped event in the patient's symptom timeline."""

    timestamp_description: str = Field(
        ..., description="Human-readable time reference, e.g. '3 days ago'"
    )
    description: str = Field(..., description="What happened at this point in time")


class PatientInput(BaseModel):
    """Complete patient intake data. All optional fields default to None."""

    # ── Core complaint ────────────────────────────────────────────────────────
    chief_complaint: str = Field(
        ..., min_length=1, description="Patient's primary symptom description"
    )
    symptom_duration: Optional[str] = None
    symptom_onset: Optional[str] = None  # "sudden" | "gradual" | free text
    symptom_progression: Optional[str] = None  # "worsening" | "stable" | "improving"
    pain_scale: Optional[int] = Field(default=None, ge=0, le=10)

    # ── Vital signs ───────────────────────────────────────────────────────────
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    temperature_celsius: Optional[float] = None
    spo2_percent: Optional[int] = Field(default=None, ge=0, le=100)
    respiratory_rate: Optional[int] = None
    glucose_mmol: Optional[float] = None

    # ── Demographics ──────────────────────────────────────────────────────────
    age: Optional[int] = Field(default=None, ge=0, le=150)
    sex: Optional[str] = None  # "male" | "female" | "other" | "prefer_not_to_say"
    weight_kg: Optional[float] = None

    # ── History ───────────────────────────────────────────────────────────────
    comorbidities: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    surgical_history: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    smoking_status: Optional[str] = None
    alcohol_use: Optional[str] = None
    additional_notes: Optional[str] = None

    # ── Timeline ──────────────────────────────────────────────────────────────
    symptom_timeline: Optional[List[SymptomEvent]] = None

    # ── Meta ──────────────────────────────────────────────────────────────────
    preferred_language: str = "en"
    input_mode: str = "form"  # "form" | "freetext" | "timeline"

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"male", "female", "other", "prefer_not_to_say"}
        if v.lower() not in allowed:
            return v  # Accept free-form — don't block the pipeline
        return v.lower()

    @field_validator("symptom_onset")
    @classmethod
    def validate_onset(cls, v: Optional[str]) -> Optional[str]:
        return v  # Accept any string — normalized downstream

    def is_pregnant(self) -> bool:
        """Heuristic: check comorbidities and medications for pregnancy indicators."""
        pregnancy_terms = {"pregnant", "pregnancy", "gravid", "prenatal", "antenatal"}
        all_text = " ".join(self.comorbidities + self.current_medications).lower()
        return any(term in all_text for term in pregnancy_terms)
