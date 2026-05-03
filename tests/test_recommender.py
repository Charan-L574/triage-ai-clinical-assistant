import pytest
from models.patient import PatientInput
from models.triage_output import TriageResult
from pipeline.risk_engine import RiskResult
from pipeline.recommender import RecommendationBuilder

@pytest.fixture
def recommender():
    return RecommendationBuilder()

def test_esi_1_recommendation(recommender):
    patient = PatientInput(chief_complaint="Cardiac arrest")
    risk = RiskResult()
    result = TriageResult(triage_level=1)
    
    enriched = recommender.build(result, patient, risk)
    
    rec = enriched.recommendations
    assert rec is not None
    assert "Immediate resuscitation required" in rec.action
    assert "Establish secure airway" in rec.nursing_interventions
    assert "Stat portable CXR" in rec.diagnostic_considerations

def test_missing_vital_impact_generation(recommender):
    patient = PatientInput(chief_complaint="Chest pain")
    # Simulation: Risk engine sees missing vitals
    risk = RiskResult()
    result = TriageResult(
        triage_level=3,
        missing_vitals=["Heart Rate", "SpO2"]
    )
    
    enriched = recommender.build(result, patient, risk)
    
    impacts = enriched.missing_vital_impact
    assert "Heart Rate" in impacts
    assert "SpO2" in impacts
    assert "above 120 bpm" in impacts["Heart Rate"]
    assert "below 92%" in impacts["SpO2"]
