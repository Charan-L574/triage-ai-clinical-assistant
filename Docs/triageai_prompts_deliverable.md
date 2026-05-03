# TriageAI Clinical Assistant —  Prompts Deliverable

# PHASE 1 — PROJECT SCAFFOLDING & ARCHITECTURE

---

## Prompt 1.1 — Initial Project Setup

**Intent:** Establish the project structure, tech stack, and core design philosophy before writing any code.

```
I want to build an AI Clinical Triage Assistant as part of a coding assessment.

The problem statement is:
"Assess patient inputs to provide triage severity, recommendations, and risk flags."

Core requirements:
- Structured output (not freeform text)
- Multi-step reasoning (not a single LLM call)
- Safety guardrails (no hallucinated critical advice goes unchecked)

The prescribed architecture from the spec is:
Input → Extraction → Risk Engine → LLM → Recommendation → Safety Validator

I want to use:
- Python as the primary language
- FastAPI for the backend REST API
- Streamlit for the frontend clinician portal
- LangGraph for orchestrating the multi-step pipeline
- FAISS for vector-based semantic search over a disease ontology
- Pydantic v2 for structured output schemas
- SQLite for case persistence

Create the complete project folder structure for this project, including all module files 
(empty stubs are fine). Explain the purpose of each file in one sentence.
The structure should be production-grade and modular, not a single-file script.
```

---

## Prompt 1.2 — Core Data Models

**Intent:** Define the Pydantic schemas before writing any pipeline logic.

```
Now define the Pydantic v2 data models for this clinical triage system.

I need two main schemas:

1. PatientInput — the incoming payload a clinician submits:
   - patient_id (optional UUID)
   - age (int)
   - sex (literal: M/F/Other)
   - chief_complaint (str) — freeform description
   - symptoms (list of str)
   - vitals: heart_rate, blood_pressure_systolic, blood_pressure_diastolic, 
     respiratory_rate, temperature_celsius, spo2_percent, gcs_score
   - comorbidities (list of str, optional)
   - current_medications (list of str, optional)
   - onset_duration (str, optional — e.g. "2 hours", "sudden")
   - pain_scale (int 0-10, optional)

2. TriageResult — the structured output the system returns:
   - session_id (UUID)
   - triage_level (int 1-5, following the ESI scale)
   - triage_label (str — e.g. "Emergent", "Urgent")
   - confidence_score (float 0.0 to 1.0)
   - primary_diagnosis_hypothesis (str)
   - differential_diagnoses (list of str)
   - risk_flags (list of str — e.g. "Critical SpO2", "Altered consciousness")
   - immediate_actions (list of str)
   - recommended_workup (list of str)
   - disposition_recommendation (str)
   - model_used (str)
   - model_response_time_ms (int)
   - completeness_score (float — how complete the patient data was)
   - rag_matches (list of dict with disease name, ICD code, similarity score)
   - warnings (list of str)
   - timestamp (datetime)

Use Pydantic v2 syntax. Add field validators where clinically appropriate 
(e.g., SpO2 must be between 0 and 100, ESI level must be 1-5).
Add docstrings explaining what each field means clinically.
```

---

## Prompt 1.3 — Configuration & Environment

**Intent:** Set up centralized config before wiring everything together.

```
Create the config.py and config.ini files for this project.

config.py should:
- Load API keys from .env using python-dotenv (GROQ_API_KEY, HF_TOKEN, GEMINI_API_KEY)
- Define model IDs as constants:
  - GROQ_MODEL_ID = "llama-3.3-70b-versatile"
  - MEDGEMMA_MODEL_ID = "google/medgemma-27b-it"
  - GEMINI_MODEL_ID = "gemini-2.5-flash"
- Define thresholds:
  - COMPLETENESS_THRESHOLD = 0.6 (below this, trigger HITL clarification)
  - HITL_AMBIGUITY_KEYWORDS = ["unclear", "vague", "unknown", "unspecified"]
  - BENCHMARK_TIMEOUT_SECONDS = 120
- Define APP_VERSION = "1.0.0"

config.ini should store non-secret operational settings like logging level, 
database path, and FAISS index path.

Also create a .env.example showing exactly what API keys are needed with 
placeholder values and explanatory comments.
```

---

# PHASE 2 — CORE PIPELINE IMPLEMENTATION

---

## Prompt 2.1 — Clinical NLP Extractor

**Intent:** Build the first stage of the pipeline — entity extraction.

```
Build pipeline/extractor.py — the clinical NLP extraction engine.

This module takes a PatientInput object and extracts structured clinical entities from 
the freeform chief_complaint and symptom list fields.

Requirements:
1. Use ScispaCy (en_core_sci_sm model) for named entity recognition on clinical text.
   Fall back to a keyword-matching dictionary if ScispaCy is unavailable.

2. Extract the following entities:
   - Symptoms (both from the symptoms list AND hidden in chief_complaint text)
   - Anatomical locations (e.g., "left arm", "chest", "abdomen")
   - Comorbidities mentioned inline (e.g., "diabetic patient with...")
   - Medications mentioned inline
   - Temporal indicators (e.g., "sudden onset", "3 days ago", "worsening")

3. Compute a Data Completeness Score (float 0.0 to 1.0) based on:
   - Whether HR, BP, SpO2, RR, Temp, GCS are all provided
   - Whether chief_complaint has enough tokens to analyze (>5 words)
   - Whether symptoms list is non-empty
   - Penalize missing critical vitals (SpO2, HR, BP) more than optional ones

4. Return an ExtractionResult dataclass with:
   - extracted_symptoms: list[str]
   - extracted_locations: list[str]
   - extracted_comorbidities: list[str]
   - extracted_medications: list[str]
   - temporal_flags: list[str]
   - completeness_score: float
   - missing_vitals: list[str]

Write clean, well-commented code. The module must work even if ScispaCy fails to load.
```

---

## Prompt 2.2 — Rule-Based Risk Engine

**Intent:** Build the deterministic hard-rule evaluator — the safety backbone.

```
Build pipeline/risk_engine.py — the deterministic vital sign rule evaluation engine.

This is the safety backbone of the system. It must evaluate vitals against hard 
clinical thresholds and can immediately assign a triage level without waiting for LLM 
inference. This is the "fast-track" path for critical patients.

Implement these specific rules (return ESI level 1 or 2 immediately if triggered):

IMMEDIATE RED FLAGS (ESI 1 — Resuscitation):
- SpO2 < 85%
- Heart rate < 30 or > 180 bpm
- Respiratory rate < 6 or > 40 breaths/min
- GCS <= 8 (severe altered consciousness)
- Systolic BP < 70 (profound hypotension)
- Chief complaint contains: ["cardiac arrest", "not breathing", "unresponsive", 
  "pulseless", "anaphylaxis"]

HIGH RISK FLAGS (ESI 2 — Emergent):
- SpO2 85-90%
- Heart rate 150-180 or 30-50 bpm
- Systolic BP 70-80 (critical hypotension)
- GCS 9-12 (moderate altered consciousness)
- Temperature > 40°C or < 35°C
- Chief complaint contains: ["stroke", "STEMI", "sepsis", "overdose", "chest pain 
  with radiation", "suicidal"]

Return a RiskEngineResult dataclass:
- fast_track: bool (True = skip LLM, assign ESI immediately)
- forced_esi_level: Optional[int]
- triggered_rules: list[str] (human-readable descriptions of what fired)
- risk_flags: list[str] (clinical flags to surface in output)
- risk_score: float (0.0-1.0 normalized risk score)

Log every triggered rule. The system must never silently ignore a triggered hard rule.
```

---

## Prompt 2.3 — FAISS RAG Retriever

**Intent:** Build the semantic search layer over the disease ontology.

```
Build rag/retriever.py — the FAISS vector retrieval module.

This module provides semantic search over a disease ontology index to give the LLM 
relevant disease profiles, ICD codes, and symptom-disease mappings as context.

Requirements:
1. Load a FAISS index from disk (index path from config). The index should be built 
   from a disease ontology dataset (use the Human Disease Ontology or ORDO if available).
   Build the index from a CSV with columns: disease_name, icd_code, symptoms, description.

2. Embed a query using sentence-transformers (use "all-MiniLM-L6-v2" as the embedding 
   model — small, fast, and good enough for clinical retrieval).

3. Run a FAISS L2 search and return the top-k matches (default k=5).

4. Return a list of RAGMatch objects:
   - disease_name: str
   - icd_code: str  
   - l2_distance: float (lower = more similar)
   - similarity_score: float (1 / (1 + l2_distance), normalized 0-1)
   - relevant_symptoms: list[str]
   - brief_description: str

5. Implement as a singleton with lazy initialization so the FAISS index 
   only loads once at server startup (not on every request).

6. Handle gracefully if the index file doesn't exist — return empty results 
   with a warning rather than crashing.

Also write rag/build_index.py — a one-time script to build and save the FAISS index 
from a CSV source file.
```

---

## Prompt 2.4 — LLM Router with Waterfall

**Intent:** Build the multi-provider LLM orchestration layer.

```
Build pipeline/llm_router.py — the tiered LLM waterfall router.

This is the most critical module for operational resilience. The system must ALWAYS 
return a response, even if one or two LLM providers are down.

Implement a waterfall with this exact tier order:
1. Tier 1 (Primary): Groq API — llama-3.3-70b-versatile
2. Tier 2 (Medical Fallback): HuggingFace Inference API — google/medgemma-27b-it
3. Tier 3 (Last Resort): Google Gemini API — gemini-2.5-flash

For each tier:
- Wrap the call in try/except
- Set a timeout (Groq: 30s, HF: 60s, Gemini: 45s)
- If the call fails for any reason (network error, rate limit, invalid response, 
  JSON parse failure), log the error and fall through to the next tier
- Track which model was actually used and response time in milliseconds

The router should expose:
- call_best_available(system_prompt: str, user_prompt: str) -> str
  Returns raw text response from whichever tier succeeded

Also implement:
- _extract_json(raw_text: str) -> dict | None
  Tries to parse JSON from LLM output — handles markdown code fences 
  (```json ... ```) and raw JSON blobs. Returns None if unparseable.

Log a warning message at each tier fallback so it's visible in logs 
which models are failing.
```

---

## Prompt 2.5 — Prompts Module

**Intent:** Decouple all LLM prompts into a dedicated file for easy maintenance.

```
Build utils/prompts.py — the centralized prompt management module.

All LLM system prompts and user prompt builders must live here. No prompt strings 
anywhere else in the codebase.

Create the following:

1. TRIAGE_SYSTEM_PROMPT — The main system prompt for the triage LLM node:
   - You are a board-certified emergency medicine AI assistant
   - You assess patient data and assign an ESI triage level (1=most critical, 5=least)
   - You MUST respond with ONLY valid JSON matching the TriageResult schema
   - Never make up vitals or symptoms not provided
   - If data is insufficient, assign a conservative (lower ESI number = more urgent) level
   - You are not a replacement for a physician — flag uncertainty in warnings field
   - Include relevant ICD-10 codes in differential diagnoses when confident

2. build_triage_user_prompt(patient: PatientInput, extraction: ExtractionResult, 
   rag_matches: list[RAGMatch], risk_engine_result: RiskEngineResult) -> str
   — Assembles all context into a single, well-structured user prompt.
   Sections: Patient Demographics | Vitals | Chief Complaint | Extracted Entities | 
   RAG Disease Matches | Risk Engine Flags | Completeness Score | JSON Schema

3. CLARIFICATION_SYSTEM — System prompt for the HITL clarification question generator.
   — Generate exactly 5 targeted clinical clarifying questions as a JSON array.

4. build_clarification_prompt(chief_complaint: str, patient_data: dict) -> str
   — Builds the user prompt for generating clarification questions.

5. TIMELINE_ANALYSIS_SYSTEM — System prompt for analyzing symptom timelines.

6. build_timeline_analysis_prompt(events: list[dict]) -> str

Keep all prompts clinically precise, unambiguous about output format requirements, 
and versioned with a comment showing the prompt version date.
```

---

## Prompt 2.6 — Safety Validator

**Intent:** Build the post-LLM output validation and auto-repair layer.

```
Build pipeline/safety_validator.py — the structured output validator and auto-repair engine.

After the LLM responds, the raw output must be validated against the TriageResult 
Pydantic schema. This module handles all cases where the LLM produces malformed, 
incomplete, or unsafe output.

Implement:

1. validate_and_repair(raw_llm_output: str, patient: PatientInput, 
   risk_engine_result: RiskEngineResult) -> TriageResult

   Steps:
   a. Try to parse the raw string as JSON (handle markdown fences)
   b. Try to validate against TriageResult using Pydantic
   c. If validation fails, attempt auto-repair:
      - Fill missing required fields with safe defaults
      - If triage_level is missing/invalid, default to ESI 2 (conservative)
      - If risk flags are empty but risk engine flagged critical issues, 
        inject the risk engine flags
      - If confidence_score is missing, set to 0.1 (low confidence signal)
   d. If auto-repair also fails, construct a minimal safe fallback TriageResult:
      - triage_level = 2 (conservative)
      - primary_diagnosis_hypothesis = "Unable to process — manual review required"
      - risk_flags = ["AI output failed validation — immediate human review required"]
      - warnings = ["System error: LLM output could not be parsed"]

2. Never allow a completely empty or None response to reach the API caller.
3. Add a "auto_repaired" boolean field to signal to the UI that the output 
   was not from the LLM directly.
4. Log all validation failures and repair actions at WARNING level.
```

---

# PHASE 3 — LANGGRAPH ORCHESTRATION

---

## Prompt 3.1 — LangGraph Graph Definition

**Intent:** Wire all pipeline stages into a stateful LangGraph graph.

```
Build pipeline/langgraph/graph.py and pipeline/langgraph/nodes.py.

I want to use LangGraph to orchestrate the full triage pipeline as a stateful graph 
with Human-in-the-Loop (HITL) support.

Define a TriageState TypedDict with all fields that flow between nodes:
- patient: PatientInput
- extraction: Optional[ExtractionResult]
- risk_result: Optional[RiskEngineResult]
- rag_matches: list[RAGMatch]
- llm_raw_output: Optional[str]
- triage_result: Optional[TriageResult]
- clarification_questions: list[str]
- clarification_answers: dict[str, str]
- thread_id: str
- status: Literal["running", "pending_clarification", "complete", "error"]

Define these graph nodes (each is a Python function taking TriageState → TriageState):
1. extract_node — runs the extractor
2. risk_engine_node — runs the risk engine; if fast_track=True, jump to output node
3. rag_node — runs FAISS retrieval
4. llm_node — calls the LLM router with the full context prompt
5. clarification_node — emits a LangGraph interrupt if completeness < threshold
6. validate_node — runs the safety validator
7. output_node — finalizes and returns the TriageResult

Use LangGraph's SQLite checkpointer for persistence so interrupted graphs 
(pending HITL) can be resumed later by thread_id.

In LangGraphPipeline class, implement:
- start(patient: PatientInput) -> dict  
  Returns {"status": "complete", "result": TriageResult} OR 
  {"status": "pending_clarification", "thread_id": str, "questions": list[str]}
  
- resume(thread_id: str, answers: dict) -> dict
  Resumes from checkpoint with clinician answers injected into state.
```

---

## Prompt 3.2 — FastAPI Backend

**Intent:** Expose the pipeline via a production-ready REST API.

```
Build main.py — the FastAPI backend for TriageAI.

Implement the following endpoints:

POST /api/triage/
  - Accepts PatientInput JSON body
  - Runs the LangGraph pipeline async (run_in_executor to avoid blocking)
  - Returns {"status": "complete", "result": TriageResult} or 
    {"status": "pending_clarification", "thread_id": str, "questions": [...]}
  - Background task: save case to SQLite after completion
  - Timeout: 150 seconds

POST /api/triage/resume  
  - Accepts {thread_id: str, answers: {question_id: answer_text}}
  - Resumes a paused LangGraph run
  - Returns {"status": "complete", "result": TriageResult}

POST /api/clarify
  - Accepts {chief_complaint: str, current_patient_data: dict}
  - Returns {questions: list[str]} — 5 targeted clinical questions


GET /api/cases  
  - Paginated case history with filters: min_esi, max_esi, model_used, date range

GET /api/cases/{session_id}
GET /api/stats — aggregate statistics
DELETE /api/cases/{session_id}
GET /api/health — system health check (DB, RAG index, uptime)
POST /api/timeline/analyze — analyze a symptom timeline without full triage

Add:
- CORS middleware (allow all origins for dev)
- Request logging middleware (log method, path, status, latency)
- Global exception handler (never expose stack traces to client)
- Lifespan handler for startup (create DB tables, init RAG index in background)
```

---

# PHASE 4 — FRONTEND & UX

---

## Prompt 4.1 — Streamlit Frontend

**Intent:** Build the clinician portal with full triage workflow.

```
Build the Streamlit frontend for TriageAI across multiple page files.

frontend/page_triage.py — the main clinical assessment interface:
- Patient demographics form (age, sex, chief complaint)
- Vitals input form with all vital signs (show normal ranges as hints)
- Symptom input (multi-tag input or text area)
- Comorbidities and medications (optional fields)
- "Run Triage Assessment" button
- Display results in a structured clinical report format:
  - Large ESI badge with color coding (red=1, orange=2, yellow=3, green=4, blue=5)
  - Confidence score progress bar
  - Risk flags in red alert boxes
  - Primary hypothesis and differentials
  - Immediate actions as numbered checklist
  - Recommended workup
  - RAG disease matches with ICD codes and similarity scores
  - "Download Clinical Report" button

If the API returns pending_clarification status:
  - Show the clarification questions as a form
  - Wait for clinician answers
  - Submit answers to /api/triage/resume
  - Display the final result

frontend/page_history.py — case log:
- Table of past cases with ESI level, timestamp, primary diagnosis, model used
- Click to expand full case detail
- Delete individual cases
- Filter by ESI level and date range

frontend/page_status.py — system health dashboard:
- Model status cards (which models are enabled, success rates, response times)
- Plotly charts: ESI distribution, model usage over time, response time trends
- System health indicators (DB connected, RAG ready, uptime)

frontend/page_about.py — documentation page:
- Project overview, architecture summary, setup instructions

Use a dark clinical theme. Show the model that was used for each assessment.
```

---

# PHASE 5 — TESTING & VALIDATION

---

## Prompt 5.1 — Test Cases

**Intent:** Create a comprehensive test suite covering all failure modes.

```
Build the test suite for TriageAI in the tests/ directory.

Create tests/test_extractor.py:
- Test extraction of symptoms from freeform chief complaint text
- Test completeness score calculation with full vs. partial vitals
- Test keyword fallback when ScispaCy is unavailable
- Test that missing critical vitals (SpO2, HR) lower the score more than optional ones

Create tests/test_risk_engine.py:
- Test each hard rule independently (all ESI 1 triggers, all ESI 2 triggers)
- Test that SpO2=84% triggers fast_track=True with ESI 1
- Test that normal vitals return fast_track=False
- Test conflicting indicators (e.g., normal HR but critical SpO2)
- Test that chief complaint keyword matching works ("cardiac arrest" → ESI 1)

Create tests/test_safety_validator.py:
- Test auto-repair of missing triage_level field
- Test injection of risk engine flags when LLM output has empty risk_flags
- Test handling of completely unparseable LLM output
- Test that conservative ESI 2 is the fallback when validation fails

Create tests/benchmark_cases.json:
10 benchmark test cases as JSON, each with:
- case_id, description, patient_input, correct_esi_level

Include these clinical scenarios:
1. STEMI presentation (chest pain + diaphoresis + radiation) → ESI 1/2
2. Anaphylaxis (bee sting + hives + throat tightening) → ESI 1
3. SpO2=84% without other symptoms → ESI 1
4. Moderate asthma exacerbation → ESI 2/3
5. UTI in healthy adult (burning on urination, low-grade fever) → ESI 4
6. Minor laceration, no vitals provided → ESI 4/5
7. Diabetic patient with blood glucose 45 mg/dL → ESI 2
8. Tension headache, normal vitals → ESI 4/5
9. Fever + stiff neck + photophobia (meningitis) → ESI 1/2
10. Ankle sprain, stable vitals, no comorbidities → ESI 4/5

Write pytest tests that can run without API keys (mock the LLM calls).
```

---

# PHASE 6 — REFINEMENT PROMPTS (Post-First Iteration)

These are the prompts you would use after your first full working implementation, 
to improve quality, resilience, and clinical accuracy.

---


## Prompt 6.1 — Clinical Report Export

**Intent:** Make the output downloadable as a proper clinical handoff document.

```
Build utils/report_generator.py — the clinical handoff report generator.

When a triage is complete, the clinician should be able to download a plain-text 
clinical summary formatted for handoff to the treating physician.

generate_clinical_report(patient: PatientInput, result: TriageResult) -> str

The report should follow standard SBAR format (Situation, Background, Assessment, 
Recommendation):

TRIAGEAI CLINICAL HANDOFF REPORT
Generated: [timestamp]
Session ID: [id]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

SITUATION
ESI LEVEL: [1-5] — [LABEL] (Confidence: [%])
Chief Complaint: [text]
Primary Hypothesis: [text]

BACKGROUND  
Age/Sex: [age] / [sex]
Vitals: HR [hr] | BP [bp] | RR [rr] | Temp [temp] | SpO2 [spo2]% | GCS [gcs]
Comorbidities: [list]
Medications: [list]

ASSESSMENT
Risk Flags: [list with ⚠ prefix]
Differential Diagnoses: [numbered list]
RAG Ontology Matches: [disease + ICD code + similarity]

RECOMMENDATION
Immediate Actions: [numbered list]
Recommended Workup: [numbered list]
Disposition: [text]

SYSTEM METADATA
Model Used: [model] | Response Time: [ms]ms | Data Completeness: [%]
Warnings: [list]

━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This output is AI-generated clinical decision support only. 
All clinical decisions must be confirmed by a licensed physician.

Add a "Download Report" button to the Streamlit triage page that downloads 
this as a .txt file named triage_[session_id].txt.
```

---

## Prompt 6.2 — Database Layer & Case Persistence

**Intent:** The in-memory case storage was not persisting. Fix it properly.

```
Build the full SQLite database layer for case persistence.

Create db/database.py:
- Use SQLAlchemy 2.0 async with aiosqlite
- Define two tables:
  1. triage_cases: stores full PatientInput + TriageResult per session
  2. model_performance: stores per-request model metrics

Create db/crud.py with async functions:
- save_case(patient, result) — upsert by session_id
- get_case_by_session_id(session_id) -> dict | None
- get_cases_paginated(page, page_size, filters...) -> {cases, total, pages}
- delete_case(session_id) -> bool
- get_aggregate_stats() -> {total_cases, esi_distribution, avg_confidence, 
  model_usage_counts}
- save_model_performance_entry(...)
- get_model_stats() -> list[dict]

Ensure all functions use proper async context managers. 
Tables should be created on startup via create_tables() lifespan call.
Store the full JSON blob of PatientInput and TriageResult in TEXT columns 
(serialize with json.dumps) for simplicity — this avoids complex relational schemas 
for a prototype assessment project.
```

---

## Prompt 6.3 — Transaction Logging

**Intent:** The system needed proper clinical-grade audit logs.

```
Build utils/logger.py — a clinical-grade transaction logger.

Requirements:
1. Use Python's standard logging module with a custom formatter
2. Log to both console (INFO level) and a rotating file (DEBUG level, 
   max 5MB per file, keep 3 backups)
3. Format: [timestamp] [level] [module] — message
4. Add a get_logger(name) function that returns a properly configured logger 
   for any module to import

For clinical audit purposes, all triage transactions should log:
- Session ID
- ESI level assigned
- Model used
- Whether fast-track was triggered
- Whether HITL clarification was needed
- Confidence score
- Processing time

Create a log_triage_transaction(session_id, result, fast_tracked, hitl_triggered) 
helper function that writes a structured audit log entry.

This is important for medical systems — you need a complete audit trail.
```