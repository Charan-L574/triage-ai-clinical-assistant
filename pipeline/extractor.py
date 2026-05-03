"""
pipeline/extractor.py — Stage 1: NLP entity extraction and data-quality scoring.

Primary: scispacy en_core_sci_sm model.
Fallback: keyword dictionary matching (200+ medical terms).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Fallback keyword dictionary ───────────────────────────────────────────────
KEYWORD_DICT: Dict[str, List[str]] = {
    "symptoms": [
        "pain", "ache", "fever", "nausea", "vomiting", "dizziness", "fatigue",
        "weakness", "swelling", "redness", "itching", "rash", "cough", "wheeze",
        "dyspnea", "shortness of breath", "chest tightness", "palpitations",
        "headache", "migraine", "confusion", "seizure", "tremor", "syncope",
        "fainting", "numbness", "tingling", "bleeding", "bruising", "jaundice",
        "diarrhea", "constipation", "abdominal pain", "back pain", "neck pain",
        "joint pain", "muscle pain", "sore throat", "ear pain", "eye pain",
        "dysuria", "hematuria", "frequency", "urgency", "incontinence",
        "diaphoresis", "sweating", "chills", "rigors", "weight loss",
        "anorexia", "insomnia", "anxiety", "depression", "hallucination",
        "photophobia", "phonophobia", "diplopia", "blurred vision", "tinnitus",
        "hearing loss", "epistaxis", "hemoptysis", "hematemesis", "melena",
        "edema", "pallor", "cyanosis", "flushing", "urticaria", "dysphagia",
        "odynophagia", "hoarseness", "stridor", "snoring", "apnea",
        "palpitation", "arrhythmia", "syncope", "presyncope",
    ],
    "body_parts": [
        "chest", "abdomen", "head", "neck", "back", "arm", "leg", "knee",
        "hip", "shoulder", "elbow", "wrist", "hand", "finger", "foot", "toe",
        "ankle", "spine", "thorax", "pelvis", "groin", "thigh", "calf",
        "heart", "lung", "liver", "kidney", "spleen", "stomach", "bowel",
        "intestine", "colon", "rectum", "bladder", "uterus", "ovary",
        "testicle", "prostate", "breast", "throat", "airway", "trachea",
        "esophagus", "pancreas", "gallbladder", "appendix", "brain",
        "eye", "ear", "nose", "mouth", "tongue", "jaw", "forehead",
        "temple", "occiput", "scalp", "face", "cheek", "lip",
    ],
    "conditions": [
        "diabetes", "hypertension", "asthma", "copd", "heart failure",
        "atrial fibrillation", "stroke", "tia", "pneumonia", "sepsis",
        "appendicitis", "cholecystitis", "pancreatitis", "uti", "pyelonephritis",
        "dvt", "pe", "pulmonary embolism", "mi", "acs", "angina",
        "anaphylaxis", "allergy", "fracture", "dislocation", "sprain",
        "laceration", "concussion", "tbi", "meningitis", "encephalitis",
        "epilepsy", "seizure", "migraine", "depression", "anxiety",
        "schizophrenia", "overdose", "poisoning", "dehydration",
        "electrolyte imbalance", "hypothyroidism", "hyperthyroidism",
        "kidney failure", "liver failure", "cirrhosis", "hepatitis",
        "cancer", "tumor", "malignancy", "anemia", "thrombocytopenia",
        "leukemia", "lymphoma", "hiv", "aids", "tuberculosis", "covid",
        "influenza", "cellulitis", "abscess", "osteomyelitis",
    ],
    "medications": [
        "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "morphine",
        "codeine", "tramadol", "metformin", "insulin", "lisinopril",
        "amlodipine", "atorvastatin", "simvastatin", "omeprazole",
        "lansoprazole", "warfarin", "apixaban", "rivaroxaban", "heparin",
        "amoxicillin", "azithromycin", "ciprofloxacin", "metronidazole",
        "fluconazole", "prednisone", "dexamethasone", "salbutamol",
        "albuterol", "budesonide", "fluticasone", "sertraline", "fluoxetine",
        "escitalopram", "lorazepam", "diazepam", "haloperidol",
        "metoprolol", "bisoprolol", "carvedilol", "furosemide",
        "spironolactone", "enalapril", "ramipril", "valsartan",
        "levothyroxine", "prednisolone", "methylprednisolone",
        "epinephrine", "adrenaline", "naloxone", "atropine",
        "nitroglycerin", "nitrate", "clopidogrel", "ticagrelor",
    ],
    "procedures": [
        "surgery", "operation", "biopsy", "colonoscopy", "endoscopy",
        "catheterization", "intubation", "ventilation", "dialysis",
        "transfusion", "amputation", "appendectomy", "cholecystectomy",
        "hysterectomy", "mastectomy", "bypass", "stent", "pacemaker",
        "defibrillation", "cardioversion", "chemotherapy", "radiation",
        "ct scan", "mri", "x-ray", "ultrasound", "ecg", "ekg",
        "echocardiogram", "angiogram", "lumbar puncture", "bone marrow",
        "suture", "debridement", "drainage", "aspiration",
    ],
}

# ── Language detection heuristics ─────────────────────────────────────────────
NON_ENGLISH_PATTERN = re.compile(
    r"[\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u309F]"
)

TEST_STRINGS = {"test", "lorem ipsum", "hello world", "testing", "asdf", "qwerty"}

# ── Vital sign labels for completeness scoring ────────────────────────────────
VITAL_FIELDS = [
    "blood_pressure_systolic",
    "heart_rate",
    "temperature_celsius",
    "spo2_percent",
    "respiratory_rate",
    "glucose_mmol",
]
VITAL_LABELS = {
    "blood_pressure_systolic": "Blood Pressure",
    "blood_pressure_diastolic": "Blood Pressure (Diastolic)",
    "heart_rate": "Heart Rate",
    "temperature_celsius": "Temperature",
    "spo2_percent": "SpO2",
    "respiratory_rate": "Respiratory Rate",
    "glucose_mmol": "Blood Glucose",
}


@dataclass
class ExtractionResult:
    entities: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "symptoms": [],
            "body_parts": [],
            "conditions": [],
            "medications": [],
            "procedures": [],
        }
    )
    ambiguity_flags: List[str] = field(default_factory=list)
    missing_vitals: List[str] = field(default_factory=list)
    data_completeness_score: float = 0.0
    extraction_method: str = "keyword"
    duration_ms: int = 0


class EntityExtractor:
    """
    Stage 1: NLP entity extraction.
    Tries scispacy first; falls back to keyword dictionary.
    """

    def __init__(self) -> None:
        self._nlp = None
        self._spacy_available = False
        self._try_load_spacy()

    def _try_load_spacy(self) -> None:
        try:
            import spacy  # noqa: F401
            import scispacy  # noqa: F401

            import spacy
            self._nlp = spacy.load("en_core_sci_sm")
            self._spacy_available = True
            logger.info("scispacy en_core_sci_sm loaded successfully")
        except Exception as exc:
            logger.warning(
                f"scispacy not available ({exc}). Using keyword fallback."
            )
            self._spacy_available = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, patient) -> ExtractionResult:
        """Run full extraction pipeline on a PatientInput object."""
        from models.patient import PatientInput  # local import to avoid circular

        t0 = time.monotonic()
        result = ExtractionResult()

        text = self._build_text(patient)

        # Entity extraction
        if self._spacy_available and self._nlp is not None:
            result.entities = self._extract_spacy(text)
            result.extraction_method = "scispacy"
        else:
            result.entities = self._extract_keywords(text)
            result.extraction_method = "keyword"

        # Completeness score
        result.data_completeness_score = self._completeness_score(patient)

        # Missing vitals
        result.missing_vitals = self._missing_vitals(patient)

        # Ambiguity flags
        result.ambiguity_flags = self._ambiguity_flags(patient, result.entities, text)

        result.duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            f"Extraction complete: method={result.extraction_method}, "
            f"entities={sum(len(v) for v in result.entities.values())}, "
            f"completeness={result.data_completeness_score:.2f}, "
            f"ambiguity={len(result.ambiguity_flags)}, "
            f"time={result.duration_ms}ms"
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_text(patient) -> str:
        parts = [patient.chief_complaint or ""]
        if patient.additional_notes:
            parts.append(patient.additional_notes)
        return " ".join(parts)

    def _extract_spacy(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {k: [] for k in KEYWORD_DICT}
        label_map = {
            "DISEASE": "conditions",
            "SIGN_OR_SYMPTOM": "symptoms",
            "BODY_PART": "body_parts",
            "MEDICATION": "medications",
            "PROCEDURE": "procedures",
            "CHEMICAL": "medications",
            "ORGANISM": "conditions",
        }
        try:
            doc = self._nlp(text[:5000])  # cap to avoid memory issues
            for ent in doc.ents:
                cat = label_map.get(ent.label_, "symptoms")
                val = ent.text.strip().lower()
                if val and val not in entities[cat]:
                    entities[cat].append(val)
        except Exception as exc:
            logger.warning(f"scispacy NER failed: {exc}. Falling back to keywords.")
            return self._extract_keywords(text)
        # Always supplement with keyword matching (increases recall)
        kw = self._extract_keywords(text)
        for cat in entities:
            for item in kw.get(cat, []):
                if item not in entities[cat]:
                    entities[cat].append(item)
        return entities

    @staticmethod
    def _extract_keywords(text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        entities: Dict[str, List[str]] = {k: [] for k in KEYWORD_DICT}
        for cat, terms in KEYWORD_DICT.items():
            for term in terms:
                if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
                    entities[cat].append(term)
        return entities

    @staticmethod
    def _completeness_score(patient) -> float:
        score = 0.0
        # Vitals: 0.40 total weight (each of 6 vitals = ~0.067)
        vitals_provided = sum(
            1 for f in VITAL_FIELDS if getattr(patient, f, None) is not None
        )
        score += (vitals_provided / len(VITAL_FIELDS)) * 0.40

        # Age + sex: 0.15
        if patient.age is not None:
            score += 0.075
        if patient.sex is not None:
            score += 0.075

        # Comorbidities: 0.10
        if patient.comorbidities:
            score += 0.10

        # Duration + onset: 0.20
        if patient.symptom_duration:
            score += 0.10
        if patient.symptom_onset:
            score += 0.10

        # Additional notes or timeline: 0.15
        if patient.additional_notes:
            score += 0.075
        if patient.symptom_timeline:
            score += 0.075

        return round(min(score, 1.0), 3)

    @staticmethod
    def _missing_vitals(patient) -> List[str]:
        missing = []
        for f in VITAL_FIELDS:
            if getattr(patient, f, None) is None:
                missing.append(VITAL_LABELS.get(f, f))
        return missing

    @staticmethod
    def _ambiguity_flags(patient, entities: Dict[str, List[str]], text: str) -> List[str]:
        flags: List[str] = []
        complaint = patient.chief_complaint or ""

        # Too short
        if len(complaint.strip()) < 15:
            flags.append(
                "Chief complaint is very brief (under 15 characters) — assessment may be inaccurate"
            )

        # No entities extracted
        total_entities = sum(len(v) for v in entities.values())
        if total_entities == 0:
            flags.append(
                "No medical terms detected in the complaint — entity extraction found nothing"
            )

        # Contradictory terms
        text_lower = text.lower()
        if "no pain" in text_lower and (
            "severe pain" in text_lower or "worst pain" in text_lower
        ):
            flags.append(
                "Contradictory pain descriptors detected ('no pain' alongside severity terms)"
            )

        # Test/lorem ipsum strings
        if complaint.strip().lower() in TEST_STRINGS:
            flags.append("Complaint appears to be a test string — results unreliable")

        # Language detection
        if patient.preferred_language and patient.preferred_language.lower() != "en":
            flags.append(
                f"Preferred language is set to '{patient.preferred_language}'. "
                "For best results, please describe symptoms in English."
            )
        elif NON_ENGLISH_PATTERN.search(complaint):
            flags.append(
                "Non-English characters detected in complaint. "
                "Assessment accuracy is reduced for non-English input. "
                "Please describe symptoms in English for best results."
            )

        return flags
