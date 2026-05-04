# TriageAI Clinical Assistant — Prompts Deliverable

# PHASE 1 — PROJECT SCAFFOLDING & CONFIGURATION

---

## Prompt 1.1 — Setup

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

## Prompt 1.2 — Models

```
Define the Pydantic v2 data models for the clinical triage system.

Create two main schemas:
1. PatientInput: To capture patient demographic details, freeform chief complaint, explicit symptoms, vital signs, comorbidities, medications, and clinical onset details.
2. TriageResult: To return the structured triage assessment, including ESI level and label, diagnosis hypothesis, risk flags, recommended workup, disposition, and model execution metadata.

Include proper field validators for vitals and relevant docstrings.
```

---

## Prompt 1.3 — Config

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

## Prompt 2.1 — Extractor

```
Build pipeline/extractor.py — the clinical NLP extraction engine.

This module takes a PatientInput object and extracts structured clinical entities from the freeform chief_complaint and symptom list fields.

Requirements:
1. Use ScispaCy for clinical named entity recognition, with a keyword-matching fallback dictionary.
2. Extract symptoms, anatomical locations, comorbidities, medications, and temporal indicators.
3. Compute a Data Completeness Score (0.0 to 1.0) by checking the presence of all required vitals and text fields.
4. Return an ExtractionResult dataclass containing the extracted symptoms, locations, comorbidities, medications, temporal flags, completeness score, and missing vitals.
```

---

## Prompt 2.2 — Risk Engine

```
Build pipeline/risk_engine.py — the deterministic vital sign rule evaluation engine.

This is the safety backbone of the system. It evaluates vitals against hard clinical thresholds to assign a triage level without waiting for LLM inference.

Implement these specific rules:
- Immediate red flags (ESI 1) for critical vitals/symptoms (e.g., very low SpO2, extreme heart rates, GCS <= 8).
- High risk flags (ESI 2) for emergent vitals/symptoms (e.g., low SpO2, abnormal vitals, high temperature, acute complaints).

Return a RiskEngineResult dataclass that indicates if the fast-track path was triggered, forced ESI level, triggered rules, risk flags, and a normalized risk score. Log every triggered rule.
```

---

## Prompt 2.3 — Retrieval

```
Build rag/retriever.py — the FAISS vector retrieval module.

This module provides semantic search over a disease ontology index to provide relevant disease profiles, ICD codes, and symptom mappings as context for the LLM.

Requirements:
1. Load a FAISS index from disk built from a CSV source containing disease name, ICD codes, symptoms, and descriptions.
2. Embed the clinical query using a lightweight sentence-transformer model.
3. Perform an L2 distance search to return top matches.
4. Return a list of RAGMatch objects that capture disease name, ICD code, similarity score, symptoms, and descriptions.
```

---

# PHASE 3 — ORCHESTRATION & SAFETY

---

## Prompt 3.1 — Routing

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

## Prompt 3.2 — Prompts

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

## Prompt 3.3 — Validation

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

# PHASE 4 — LANGGRAPH & API ENDPOINTS

---

## Prompt 4.1 — LangGraph

```
Build pipeline/langgraph/graph.py and pipeline/langgraph/nodes.py.

Use LangGraph to orchestrate the full triage pipeline as a stateful graph with Human-in-the-Loop (HITL) support.

Requirements:
1. Define a TriageState TypedDict that tracks all necessary data across nodes (patient, extraction, risk result, RAG matches, LLM output, triage result, clarification, thread_id, status).
2. Create graph nodes for extraction, rule-based risk assessment, RAG retrieval, LLM analysis, clarification, safety validation, and finalizing output.
3. Integrate the SQLite checkpointer for persistence so interrupted runs (HITL) can be resumed later.
4. Expose a pipeline interface to start and resume runs cleanly.
```

---

## Prompt 4.2 — Core API

```
Build main.py — the FastAPI core backend for TriageAI.

Implement the following endpoints:
1. POST /api/triage/: Accept PatientInput, run the LangGraph pipeline async, and return either a completed TriageResult or a pending clarification request. Save the completed case to the database in the background.
2. POST /api/triage/resume: Accept a thread ID and answers to resume a paused LangGraph run.
3. POST /api/clarify: Accept a chief complaint and patient data to generate targeted clinical clarifying questions.
```

---

## Prompt 4.3 — Advanced API

```
Extend main.py to support advanced history and dashboard capabilities.

Implement the following endpoints:
1. GET /api/cases: Returns paginated case history with support for filtering by ESI level, model used, and date ranges.
2. GET /api/cases/{session_id} and DELETE /api/cases/{session_id}: View and delete individual cases.
3. GET /api/stats: Return aggregate system and triage statistics.
4. GET /api/health: Check system health including database connectivity and RAG index status.
5. POST /api/timeline/analyze: Standalone endpoint to analyze symptom timelines without a full triage pipeline.

Add CORS middleware, request logging, a global exception handler, and a lifespan handler.
```

---

# PHASE 5 — FRONTEND & UX

---

## Prompt 5.1 — Triage UI

```
Build frontend/page_triage.py — the main clinical assessment interface.

Create a Streamlit interface that enables:
1. Input forms for patient demographics, all vital signs, symptom descriptions, comorbidities, and medications.
2. An assessment button to trigger triage.
3. A structured clinical report output displaying color-coded ESI badges, a confidence score progress bar, red alert boxes for risk flags, diagnostic hypotheses, checklist of immediate actions, workup, and RAG disease matches.
4. Clarification handling to render follow-up questions from the API and allow the clinician to submit answers to resume the pipeline.
```

---

## Prompt 5.2 — Dashboard

```
Build the reporting and metrics pages for the Streamlit clinician portal.

Requirements:
1. frontend/page_history.py: Create a case log page that displays a table of past triage cases with filtering, expanding detailed views, and case deletion functionality.
2. frontend/page_status.py: Create a system health dashboard showing active models, response times, and interactive charts (e.g., ESI distribution, model usage) using Plotly.
```

---

## Prompt 5.3 — About Page

```
Build frontend/page_about.py — the project documentation page.

Include an overview of the TriageAI Clinical Assistant, the architectural pipeline details, and explicit setup instructions.
```

---

# PHASE 6 — TESTING & PERSISTENCE

---

## Prompt 6.1 — Tests

```
Build the test suite for TriageAI in the tests/ directory using pytest.

1. tests/test_extractor.py: Test symptom extraction, completeness score computation, and keyword fallback logic.
2. tests/test_risk_engine.py: Validate that all critical vitals correctly assign immediate or high risk levels and fast-track triage.
3. tests/test_safety_validator.py: Test parsing logic, auto-repair, and conservative fallback defaults.
4. tests/benchmark_cases.json: Define 10 varied clinical scenario test cases (e.g., STEMI, Anaphylaxis, Asthma, UTI) to validate classification accuracy. Use mocks for LLM calls so tests run without API keys.
```

---

## Prompt 6.2 — Database

```
Build the SQLite database layer using async SQLAlchemy and aiosqlite.

Create db/database.py and db/crud.py:
1. Define async database models for triage cases (storing complete PatientInput and TriageResult) and model performance entries.
2. Create CRUD methods to save cases, fetch a case by ID, paginate cases with filtering, delete cases, and retrieve aggregate stats.
3. Ensure the tables are automatically initialized during startup. Use serialized text columns for the core JSON blobs to keep the prototype architecture simple.
```

---

## Prompt 6.3 — Reports

```
Build the logging and handoff report modules.

1. Build utils/logger.py: Create a clinical-grade transaction logger with console and rotating file handlers. Record all session transactions including thread ID, ESI level, model used, and response times for complete clinical audit trials.
2. Build utils/report_generator.py: Create a plain-text clinical handoff summary using the standard SBAR format (Situation, Background, Assessment, Recommendation). Include all demographics, vitals, risk flags, differentials, and disposition. Expose it via a file download option on the Streamlit interface.
```