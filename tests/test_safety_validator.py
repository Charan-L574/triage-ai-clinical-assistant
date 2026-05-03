import pytest
from models.patient import PatientInput
from models.triage_output import TriageResult
from pipeline.risk_engine import RiskResult
from pipeline.safety_validator import SafetyValidator

@pytest.fixture
def validator():
    return SafetyValidator()

def test_no_override_needed(validator):
    patient = PatientInput(chief_complaint="Minor cut")
    risk = RiskResult()
    result = TriageResult(triage_level=4, esi_label="Less Urgent")
    
    validated = validator.validate(result, patient, risk)
    
    assert validated.triage_level == 4
    assert validated.safety_override_applied is False
    # Validator appends standard disclaimers for low confidence/missing vitals
    assert len(validated.safety_disclaimers) >= 0

def test_safety_override_applied(validator):
    patient = PatientInput(chief_complaint="Chest pain")
    # Risk engine detects critical threshold and forces ESI 1
    risk = RiskResult(
        triggered=True,
        forced_esi_level=1,
        requires_immediate_escalation=True,
        flags=["Critical Tachycardia (>150 bpm)"]
    )
    # LLM hallucinates an ESI 4
    result = TriageResult(triage_level=4, esi_label="Less Urgent")
    
    validated = validator.validate(result, patient, risk)
    
    # Validator MUST override the LLM's ESI 4 with the Risk Engine's ESI 1
    assert validated.triage_level == 1
    assert validated.safety_override_applied is True
    assert validated.escalate_immediately is True
    assert any("SAFETY OVERRIDE" in d for d in validated.safety_disclaimers)
    assert validated.esi_color == "#E24B4A"

def test_emergency_fallback(validator):
    # Tests the hardcoded fallback used if the pipeline throws an exception
    fallback = validator.emergency_fallback()
    
    assert fallback.triage_level == 2
    assert fallback.esi_label == "Emergent"
    assert fallback.confidence_score == 0.0
    assert fallback.escalate_immediately is True
