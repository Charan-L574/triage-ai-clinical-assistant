import time
from typing import Dict, List

import config
from models.triage_output import TriageResult
from pipeline.extractor import ExtractionResult
from pipeline.llm_router import LLMResponse, LLMRouter
from pipeline.risk_engine import RiskResult
from pipeline.langgraph.state import PipelineState
from pipeline.dependencies import (
    extractor_inst,
    risk_engine_inst,
    retriever_inst,
    llm_router_inst,
    recommender_inst,
    validator_inst,
)
from utils.logger import get_logger
from utils.prompts import SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)

def node_extract(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    patient = state["patient"]
    try:
        extraction = extractor_inst.extract(patient)
    except Exception as exc:
        logger.error(f"[{state['session_id']}] Extraction failed: {exc}")
        extraction = ExtractionResult()
    elapsed = int((time.monotonic() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["extraction"] = elapsed
    logger.info(f"[{state['session_id']}] node_extract done in {elapsed}ms")
    return {**state, "extraction": extraction, "timings": timings}


def node_risk(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    try:
        risk = risk_engine_inst.evaluate(state["patient"], state["extraction"])
    except Exception as exc:
        logger.error(f"[{state['session_id']}] Risk engine failed: {exc}")
        risk = RiskResult()
    elapsed = int((time.monotonic() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["risk_engine"] = elapsed
    logger.info(f"[{state['session_id']}] node_risk done in {elapsed}ms")
    return {**state, "risk": risk, "timings": timings}


def node_rag(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    patient = state["patient"]
    extraction = state["extraction"]
    try:
        query = patient.chief_complaint or ""
        if extraction.entities.get("symptoms"):
            query += " " + " ".join(extraction.entities["symptoms"][:5])
        rag_context, rag_diseases = retriever_inst.retrieve(query)
    except Exception as exc:
        logger.warning(f"[{state['session_id']}] RAG failed: {exc}")
        rag_context = ""
        rag_diseases = []
    elapsed = int((time.monotonic() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["rag_retrieval"] = elapsed
    return {**state, "rag_context": rag_context, "rag_diseases": rag_diseases, "timings": timings}


def node_llm(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    try:
        user_prompt = build_user_prompt(
            state["patient"], state["extraction"], state["risk"], state.get("rag_context", "")
        )
        answers = state.get("clarification_answers", {})
        if answers:
            answer_text = "\n".join(f"- {q}: {a}" for q, a in answers.items())
            user_prompt += f"\n\n[Clinician Clarification Provided]:\n{answer_text}"

        prompt_data = {"system": SYSTEM_PROMPT, "user": user_prompt}
        llm_response = llm_router_inst.route(prompt_data)
    except Exception as exc:
        logger.error(f"[{state['session_id']}] LLM failed: {exc}")
        llm_response = LLMRouter._hardcoded_fallback()
    elapsed = int((time.monotonic() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["llm_reasoning"] = elapsed
    logger.info(f"[{state['session_id']}] node_llm done in {elapsed}ms")
    return {**state, "llm_response": llm_response, "timings": timings}


def node_check_hitl(state: PipelineState) -> PipelineState:
    # Prevent infinite loops: If we already received answers, force completion.
    if state.get("clarification_answers"):
        return {
            **state,
            "needs_clarification": False,
            "clarification_questions": [],
            "hitl_reason": "",
        }

    extraction: ExtractionResult = state["extraction"]
    llm_resp: LLMResponse = state["llm_response"]
    content = llm_resp.content or {}

    reasons: List[str] = []
    questions: List[Dict[str, str]] = []

    if extraction.ambiguity_flags:
        reasons.append("Ambiguous symptom description detected")
        questions.append({
            "id": "ambiguity_clarify",
            "question": "Can you describe the main symptoms in more detail?",
            "reason": "The initial description was flagged as ambiguous.",
            "category": "ambiguity"
        })

    confidence = content.get("confidence_score", 0.5)
    if isinstance(confidence, (int, float)) and confidence < config.CONFIDENCE_THRESHOLD_LOW:
        reasons.append(f"LLM confidence below threshold ({confidence:.0%})")
        questions.append({
            "id": "confidence_symptoms",
            "question": "Can you list all symptoms the patient is currently experiencing?",
            "reason": "Model confidence was below the safety threshold.",
            "category": "confidence"
        })

    triage_level = content.get("triage_level", 3)
    critical_missing = {"SpO2", "Heart Rate", "Blood Pressure"} & set(extraction.missing_vitals)
    if critical_missing and triage_level in (2, 3):
        reasons.append(f"Missing critical vitals ({', '.join(critical_missing)}) on borderline ESI {triage_level}")
        for vital in critical_missing:
            questions.append({
                "id": f"vital_{vital.lower().replace(' ', '_')}",
                "question": f"What is the patient's current {vital}?",
                "reason": f"{vital} is missing and could escalate the case.",
                "category": "vitals"
            })

    diffs = content.get("differential_diagnoses", [])
    if len(diffs) >= 2:
        top_likelihoods = [d.get("likelihood", "") for d in diffs[:2]]
        if len(set(top_likelihoods)) == 1 and top_likelihoods[0] == "High":
            reasons.append("Two diagnoses share equal highest likelihood")
            d1 = diffs[0].get("condition_name", "?")
            d2 = diffs[1].get("condition_name", "?")
            questions.append({
                "id": "differential_discriminator",
                "question": f"Has the patient had any recent: surgery, travel, infection, or similar symptoms?",
                "reason": f"Both {d1} and {d2} have equal High likelihood.",
                "category": "differentials"
            })

    needs_hitl = len(reasons) > 0
    hitl_reason = " | ".join(reasons) if reasons else ""

    return {
        **state,
        "needs_clarification": needs_hitl,
        "clarification_questions": questions,
        "hitl_reason": hitl_reason,
    }


def node_hitl_interrupt(state: PipelineState) -> PipelineState:
    logger.info(f"[{state['session_id']}] HITL interrupt resumed")
    return state

def _validate_and_repair(llm_response: LLMResponse, session_id: str, extraction, risk, rag_used: bool, rag_diseases: list) -> TriageResult:
    content = llm_response.content or {}
    try:
        from models.triage_output import TriageResult
        result = TriageResult(**content)
    except Exception as e:
        logger.error(f"Pydantic parsing failed: {e}")
        from pipeline.safety_validator import SafetyValidator
        result = SafetyValidator().emergency_fallback()
        
    result.session_id = session_id
    result.model_used = llm_response.model_name
    result.model_response_time_ms = llm_response.response_time_ms
    result.raw_llm_output = llm_response.raw_text
    result.rag_context_used = rag_used
    
    # Use standard dict assignment if model attribute is missing (it will be added next)
    result.rag_extracted_diseases = rag_diseases

    result.extracted_entities = extraction.entities
    result.missing_vitals = extraction.missing_vitals
    result.ambiguity_flags = extraction.ambiguity_flags
    result.data_completeness_score = extraction.data_completeness_score
    return result

def node_validate(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    extraction = state["extraction"]
    risk = state["risk"]
    llm_resp = state["llm_response"]
    session_id = state["session_id"]

    answers = state.get("clarification_answers", {})
    if answers:
        extraction.ambiguity_flags = [f for f in extraction.ambiguity_flags if "brief" not in f.lower()]

    try:
        result = _validate_and_repair(
            llm_resp, 
            session_id, 
            extraction, 
            risk, 
            bool(state.get("rag_context")),
            state.get("rag_diseases", [])
        )
    except Exception as exc:
        logger.error(f"[{session_id}] node_validate failed: {exc}")
        result = validator_inst.emergency_fallback()
        result.session_id = session_id

    elapsed = int((time.monotonic() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["validation"] = elapsed
    return {**state, "result": result, "timings": timings}


def node_recommend(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    try:
        result = recommender_inst.build(state["result"], state["patient"], state["risk"])
    except Exception as exc:
        logger.error(f"[{state['session_id']}] node_recommend failed: {exc}")
        result = state["result"]
    elapsed = int((time.monotonic() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["recommendations"] = elapsed
    return {**state, "result": result, "timings": timings}


def node_safety(state: PipelineState) -> PipelineState:
    t0 = time.monotonic()
    try:
        result = validator_inst.validate(state["result"], state["patient"], state["risk"])
    except Exception as exc:
        logger.error(f"[{state['session_id']}] node_safety failed: {exc}")
        result = state["result"]

    result.pipeline_stage_timings = dict(state.get("timings", {}))
    result.rag_context_used = bool(state.get("rag_context"))
    result.rule_engine_triggered = state["risk"].triggered
    if state.get("fast_tracked"):
        result.fast_tracked = True
    if state.get("clarification_answers"):
        result.hitl_applied = True

    elapsed = int((time.monotonic() - t0) * 1000)
    result.pipeline_stage_timings["safety_validation"] = elapsed
    logger.info(f"[{state['session_id']}] node_safety done — final ESI={result.triage_level}")
    return {**state, "result": result}
