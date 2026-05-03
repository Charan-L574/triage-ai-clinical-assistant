import pytest
from models.patient import PatientInput
from pipeline.risk_engine import RiskEngine
from pipeline.extractor import ExtractionResult

@pytest.fixture
def risk_engine():
    return RiskEngine()

def test_normal_vitals(risk_engine):
    patient = PatientInput(
        chief_complaint="I have a minor headache.",
        heart_rate=80,
        spo2_percent=98,
        blood_pressure_systolic=120,
        temperature_celsius=37.0,
        respiratory_rate=16
    )
    extraction = ExtractionResult()
    result = risk_engine.evaluate(patient, extraction)
    
    assert result.triggered is False
    assert result.forced_esi_level is None
    assert len(result.flags) == 0

def test_critical_tachycardia(risk_engine):
    patient = PatientInput(
        chief_complaint="Chest pain",
        heart_rate=155  # Critical threshold is >150
    )
    extraction = ExtractionResult()
    result = risk_engine.evaluate(patient, extraction)
    
    assert result.triggered is True
    assert result.forced_esi_level == 1
    assert result.requires_immediate_escalation is True
    assert any("tachycardia" in flag.lower() for flag in result.flags)

def test_severe_hypoxia(risk_engine):
    patient = PatientInput(
        chief_complaint="Shortness of breath",
        spo2_percent=88  # Critical threshold is <90
    )
    extraction = ExtractionResult()
    result = risk_engine.evaluate(patient, extraction)
    
    assert result.triggered is True
    assert result.forced_esi_level == 1
    assert any("hypoxia" in flag.lower() for flag in result.flags)

def test_red_flag_keyword_detection(risk_engine):
    patient = PatientInput(
        chief_complaint="Patient presents with severe stroke symptoms including facial droop."
    )
    extraction = ExtractionResult(entities={"symptoms": ["stroke", "facial droop"]})
    result = risk_engine.evaluate(patient, extraction)
    
    # "stroke" forces ESI 1 based on Risk Engine rules
    assert result.triggered is True
    assert result.forced_esi_level == 1
    assert any("cerebrovascular" in flag.lower() for flag in result.flags)
