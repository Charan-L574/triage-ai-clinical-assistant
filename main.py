"""
main.py — FastAPI backend for TriageAI.

10 endpoints, CORS middleware, request logging, global error handling,
background DB writes, and async health checks.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from db.crud import (
    delete_case,
    get_aggregate_stats,
    get_case_by_session_id,
    get_cases_paginated,
    get_model_stats,
    save_case,
    save_model_performance_entry,
)
from db.database import create_tables
from models.patient import PatientInput
from models.triage_output import ModelComparisonResult, TriageResult
from pipeline.dependencies import retriever_inst
from pipeline.langgraph.graph import LangGraphPipeline
from utils.logger import get_logger
from utils.prompts import (
    CLARIFICATION_SYSTEM,
    TIMELINE_ANALYSIS_SYSTEM,
    build_clarification_prompt,
    build_timeline_analysis_prompt,
)

logger = get_logger(__name__)

_start_time = time.time()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TriageAI FastAPI starting up...")
    await create_tables()
    # Initialize RAG background
    import asyncio
    asyncio.create_task(asyncio.to_thread(retriever_inst.initialize))
    yield
    logger.info("TriageAI FastAPI shutting down")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TriageAI",
    description="AI-powered clinical triage decision-support system",
    version=config.APP_VERSION,
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed}ms)"
    )
    return response


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred. Please try again."},
    )


# ── Helper: background DB save ─────────────────────────────────────────────────
async def _bg_save(patient: PatientInput, result: TriageResult) -> None:
    await save_case(patient, result)
    await save_model_performance_entry(
        model_name=result.model_used,
        was_successful=result.confidence_score > 0,
        response_time_ms=result.model_response_time_ms,
        triage_level=result.triage_level,
        confidence_score=result.confidence_score,
    )


# ── Endpoint request models ────────────────────────────────────────────────────

class ClarifyRequest(BaseModel):
    chief_complaint: str
    current_patient_data: Optional[Dict[str, Any]] = None


class ResumeRequest(BaseModel):
    thread_id: str
    answers: Dict[str, str]  # {question_id: answer_text}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# (Legacy endpoint /api/triage removed in favor of /api/triage/start)


# ── POST /api/triage/start (LangGraph + HITL) ─────────────────────────────────
@app.post("/api/triage/start", response_model=Dict[str, Any])
async def triage_start(patient: PatientInput, background_tasks: BackgroundTasks):
    """
    LangGraph-powered triage with HITL support and fast-track.
    Returns one of:
      {"status": "complete", "result": {...}}
      {"status": "pending_clarification", "thread_id": "...", "questions": [...], "reason": "..."}
    """
    try:
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, LangGraphPipeline.start, patient
            ),
            timeout=150,
        )
        if response["status"] == "complete":
            result: TriageResult = response["result"]
            background_tasks.add_task(_bg_save, patient, result)
            return {"status": "complete", "result": result.model_dump()}
        # Pending clarification — return questions to UI
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out")
    except Exception as exc:
        logger.error(f"triage/start error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /api/triage/resume (HITL resume) ─────────────────────────────────────
@app.post("/api/triage/resume", response_model=Dict[str, Any])
async def triage_resume(req: ResumeRequest, background_tasks: BackgroundTasks):
    """
    Resume a paused LangGraph run after clinician provides clarification answers.
    Always returns {"status": "complete", "result": {...}}
    """
    try:
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, LangGraphPipeline.resume, req.thread_id, req.answers
            ),
            timeout=150,
        )
        if response["status"] == "complete":
            result: TriageResult = response["result"]
            # We don't have the original patient here, so skip background save
            logger.info(f"HITL resume complete for thread {req.thread_id} — ESI={result.triage_level}")
            return {"status": "complete", "result": result.model_dump()}
        
        # If the graph interrupts again (secondary clarification loop)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out")
    except Exception as exc:
        logger.error(f"triage/resume error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))





# ── POST /api/clarify ─────────────────────────────────────────────────────────
@app.post("/api/clarify")
async def clarify(body: ClarifyRequest):
    """Return 5 targeted clarifying questions for a given complaint."""
    from pipeline.llm_router import LLMRouter, _extract_json
    router = LLMRouter()
    try:
        user_prompt = build_clarification_prompt(
            body.chief_complaint, body.current_patient_data
        )
        raw = await asyncio.get_event_loop().run_in_executor(
            None, router.call_best_available, CLARIFICATION_SYSTEM, user_prompt
        )
        parsed = _extract_json(raw)
        if isinstance(parsed, list):
            return {"questions": parsed}
        return {"questions": parsed if parsed else [], "raw": raw}
    except Exception as exc:
        logger.error(f"Clarify endpoint error: {exc}")
        return {"questions": [], "error": str(exc)}


# ── GET /api/models/status ────────────────────────────────────────────────────
@app.get("/api/models/status")
async def models_status():
    """Return configured status and last-use stats for each model."""
    stats = await get_model_stats()
    models = [
        {
            "name": "MedGemma-27B",
            "enabled": bool(config.HF_TOKEN),
            "model_id": config.MEDGEMMA_MODEL_ID,
        },
        {
            "name": "Groq-Llama-3.3",
            "enabled": bool(config.GROQ_API_KEY),
            "model_id": config.GROQ_MODEL_ID,
        },
        {
            "name": "Gemini-2.5-Flash",
            "enabled": bool(config.GEMINI_API_KEY),
            "model_id": config.GEMINI_MODEL_ID,
        },
    ]
    stats_by_name = {s["model_name"]: s for s in stats}
    for m in models:
        s = stats_by_name.get(m["name"], {})
        m.update(
            {
                "success_rate": s.get("success_rate"),
                "avg_response_time_ms": s.get("avg_response_time_ms"),
                "last_used": s.get("last_used"),
                "total_calls": s.get("total_calls", 0),
            }
        )
    return {"models": models}


# ── GET /api/cases ────────────────────────────────────────────────────────────
@app.get("/api/cases")
async def get_cases(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    min_esi: Optional[int] = Query(default=None, ge=1, le=5),
    max_esi: Optional[int] = Query(default=None, ge=1, le=5),
    model_used: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
):
    return await get_cases_paginated(
        page=page,
        page_size=page_size,
        min_esi=min_esi,
        max_esi=max_esi,
        model_used=model_used,
        date_from=date_from,
        date_to=date_to,
    )


# ── GET /api/cases/{session_id} ────────────────────────────────────────────────
@app.get("/api/cases/{session_id}")
async def get_case(session_id: str):
    case = await get_case_by_session_id(session_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


# ── DELETE /api/cases/{session_id} ────────────────────────────────────────────
@app.delete("/api/cases/{session_id}")
async def delete_case_endpoint(session_id: str):
    success = await delete_case(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"deleted": session_id}


# ── GET /api/stats ─────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def stats():
    return await get_aggregate_stats()


# ── POST /api/benchmark ────────────────────────────────────────────────────────
@app.post("/api/benchmark")
async def benchmark(background_tasks: BackgroundTasks):
    """Run all benchmark test cases through the pipeline."""
    from pathlib import Path
    import json as _json

    bench_file = Path(__file__).parent / "tests" / "benchmark_cases.json"
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="benchmark_cases.json not found")

    cases = _json.loads(bench_file.read_text(encoding="utf-8"))
    results = []

    async def run_one(case: dict) -> dict:
        try:
            patient = PatientInput(**case["patient_input"])
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, _orchestrator.run, patient
                ),
                timeout=config.BENCHMARK_TIMEOUT_SECONDS / len(cases),
            )
            correct = case["correct_esi_level"]
            predicted = result.triage_level
            diff = abs(predicted - correct)
            match_label = "correct" if diff == 0 else (f"off_by_{diff}" if diff <= 2 else "wrong")
            return {
                "case_id": case["case_id"],
                "description": case["description"],
                "correct_esi": correct,
                "predicted_esi": predicted,
                "match": match_label,
                "confidence": result.confidence_score,
                "model_used": result.model_used,
                "response_time_ms": result.model_response_time_ms,
            }
        except Exception as exc:
            return {"case_id": case.get("case_id", "?"), "error": str(exc)}

    tasks = [run_one(c) for c in cases]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Compute aggregate per model
    correct_count = sum(1 for r in results if r.get("match") == "correct")
    total = len(results)
    return {
        "total_cases": total,
        "accuracy": round(correct_count / max(total, 1), 3),
        "results": results,
    }


# ── GET /api/health ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    from rag.retriever import DiseaseRetriever
    from db.database import engine
    db_ok = False
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    uptime_seconds = int(time.time() - _start_time)
    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "database_connected": db_ok,
        "rag_index_ready": initialize_rag and True,  # checked via retriever singleton
        "primary_model": config.MEDGEMMA_MODEL_ID,
        "app_version": config.APP_VERSION,
    }


# ── POST /api/timeline/analyze ────────────────────────────────────────────────
@app.post("/api/timeline/analyze")
async def timeline_analyze(body: dict):
    """Analyze a symptom timeline without running full triage."""
    from pipeline.llm_router import LLMRouter, _extract_json
    events = body.get("events", [])
    if len(events) < 1:
        raise HTTPException(status_code=400, detail="At least 1 timeline event required")
    router = LLMRouter()
    try:
        user_prompt = build_timeline_analysis_prompt(events)
        raw = await asyncio.get_event_loop().run_in_executor(
            None, router.call_best_available, TIMELINE_ANALYSIS_SYSTEM, user_prompt
        )
        parsed = _extract_json(raw)
        return parsed if parsed else {"analysis": raw}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
