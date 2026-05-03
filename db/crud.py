"""
db/crud.py — Async CRUD operations for TriageAI database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Integer, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal, ModelPerformance, TriageCase
from models.patient import PatientInput
from models.triage_output import TriageResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Triage Cases ───────────────────────────────────────────────────────────────

async def save_case(patient: PatientInput, result: TriageResult) -> None:
    """Persist a triage case to the database."""
    try:
        async with AsyncSessionLocal() as session:
            ts = datetime.now(timezone.utc)
            case = TriageCase(
                session_id=result.session_id,
                timestamp=ts,
                chief_complaint=patient.chief_complaint[:1000],
                patient_age=patient.age,
                patient_sex=patient.sex,
                triage_level=result.triage_level,
                esi_label=result.esi_label,
                confidence_score=result.confidence_score,
                escalate_immediately=result.escalate_immediately,
                red_flags_count=len(result.red_flags_detected),
                rule_engine_triggered=result.rule_engine_triggered,
                safety_override_applied=result.safety_override_applied,
                model_used=result.model_used,
                response_time_ms=result.model_response_time_ms,
                data_completeness_score=result.data_completeness_score,
                rag_context_used=result.rag_context_used,
                full_input_json=patient.model_dump_json(),
                full_result_json=result.model_dump_json(),
            )
            session.add(case)
            await session.commit()
            logger.info(f"Case saved: {result.session_id}")
    except Exception as exc:
        logger.error(f"Failed to save case {result.session_id}: {exc}")


async def get_case_by_session_id(session_id: str) -> Optional[Dict]:
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(TriageCase).where(TriageCase.session_id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _case_to_dict(row)
    except Exception as exc:
        logger.error(f"get_case_by_session_id error: {exc}")
        return None


async def get_cases_paginated(
    page: int = 1,
    page_size: int = 20,
    min_esi: Optional[int] = None,
    max_esi: Optional[int] = None,
    model_used: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        async with AsyncSessionLocal() as session:
            query = select(TriageCase)

            if min_esi is not None:
                query = query.where(TriageCase.triage_level >= min_esi)
            if max_esi is not None:
                query = query.where(TriageCase.triage_level <= max_esi)
            if model_used:
                query = query.where(TriageCase.model_used == model_used)
            if date_from:
                try:
                    dt = datetime.fromisoformat(date_from)
                    query = query.where(TriageCase.timestamp >= dt)
                except ValueError:
                    pass
            if date_to:
                try:
                    dt = datetime.fromisoformat(date_to)
                    query = query.where(TriageCase.timestamp <= dt)
                except ValueError:
                    pass

            # Count
            count_q = select(func.count()).select_from(query.subquery())
            total = (await session.execute(count_q)).scalar() or 0

            # Paginate
            query = (
                query.order_by(TriageCase.timestamp.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = (await session.execute(query)).scalars().all()

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "cases": [_case_to_dict(r) for r in rows],
            }
    except Exception as exc:
        logger.error(f"get_cases_paginated error: {exc}")
        return {"total": 0, "page": page, "page_size": page_size, "cases": []}


async def delete_case(session_id: str) -> bool:
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                delete(TriageCase).where(TriageCase.session_id == session_id)
            )
            await session.commit()
            return result.rowcount > 0
    except Exception as exc:
        logger.error(f"delete_case error: {exc}")
        return False


async def get_aggregate_stats() -> Dict[str, Any]:
    try:
        async with AsyncSessionLocal() as session:
            # Total cases
            total = (await session.execute(select(func.count(TriageCase.id)))).scalar() or 0

            # Avg confidence
            avg_conf = (
                await session.execute(select(func.avg(TriageCase.confidence_score)))
            ).scalar() or 0.0

            # ESI distribution
            esi_rows = (
                await session.execute(
                    select(TriageCase.triage_level, func.count(TriageCase.id))
                    .group_by(TriageCase.triage_level)
                )
            ).all()
            esi_dist = {str(r[0]): r[1] for r in esi_rows}

            # Model distribution
            model_rows = (
                await session.execute(
                    select(TriageCase.model_used, func.count(TriageCase.id))
                    .group_by(TriageCase.model_used)
                )
            ).all()
            model_dist = {r[0]: r[1] for r in model_rows}

            # Avg response time per model
            rt_rows = (
                await session.execute(
                    select(TriageCase.model_used, func.avg(TriageCase.response_time_ms))
                    .group_by(TriageCase.model_used)
                )
            ).all()
            avg_rt = {r[0]: round(r[1] or 0) for r in rt_rows}

            # Date range
            min_ts = (await session.execute(select(func.min(TriageCase.timestamp)))).scalar()
            max_ts = (await session.execute(select(func.max(TriageCase.timestamp)))).scalar()

            return {
                "total_cases": total,
                "avg_confidence": round(float(avg_conf), 3),
                "esi_distribution": esi_dist,
                "model_distribution": model_dist,
                "avg_response_time_ms": avg_rt,
                "date_from": str(min_ts) if min_ts else None,
                "date_to": str(max_ts) if max_ts else None,
            }
    except Exception as exc:
        logger.error(f"get_aggregate_stats error: {exc}")
        return {}


# ── Model Performance ──────────────────────────────────────────────────────────

async def save_model_performance_entry(
    model_name: str,
    was_successful: bool,
    response_time_ms: int,
    triage_level: Optional[int] = None,
    confidence_score: Optional[float] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    try:
        async with AsyncSessionLocal() as session:
            entry = ModelPerformance(
                timestamp=datetime.now(timezone.utc),
                model_name=model_name,
                was_successful=was_successful,
                response_time_ms=response_time_ms,
                triage_level_produced=triage_level,
                confidence_score=confidence_score,
                error_type=error_type,
                error_message=error_message[:500] if error_message else None,
            )
            session.add(entry)
            await session.commit()
    except Exception as exc:
        logger.error(f"save_model_performance_entry error: {exc}")


async def get_model_stats() -> List[Dict]:
    try:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        ModelPerformance.model_name,
                        func.count(ModelPerformance.id).label("total"),
                        func.sum(
                            func.cast(ModelPerformance.was_successful, Integer)
                        ).label("successes"),
                        func.avg(ModelPerformance.response_time_ms).label("avg_rt"),
                        func.max(ModelPerformance.timestamp).label("last_used"),
                    ).group_by(ModelPerformance.model_name)
                )
            ).all()
            return [
                {
                    "model_name": r.model_name,
                    "total_calls": r.total,
                    "success_rate": round((r.successes or 0) / max(r.total, 1), 3),
                    "avg_response_time_ms": round(r.avg_rt or 0),
                    "last_used": str(r.last_used) if r.last_used else None,
                }
                for r in rows
            ]
    except Exception as exc:
        logger.error(f"get_model_stats error: {exc}")
        return []


# ── Helper ─────────────────────────────────────────────────────────────────────

def _case_to_dict(row: TriageCase) -> Dict:
    return {
        "session_id": row.session_id,
        "timestamp": str(row.timestamp),
        "chief_complaint": row.chief_complaint,
        "patient_age": row.patient_age,
        "patient_sex": row.patient_sex,
        "triage_level": row.triage_level,
        "esi_label": row.esi_label,
        "confidence_score": row.confidence_score,
        "escalate_immediately": row.escalate_immediately,
        "red_flags_count": row.red_flags_count,
        "rule_engine_triggered": row.rule_engine_triggered,
        "safety_override_applied": row.safety_override_applied,
        "model_used": row.model_used,
        "response_time_ms": row.response_time_ms,
        "data_completeness_score": row.data_completeness_score,
        "rag_context_used": row.rag_context_used,
        "full_input_json": row.full_input_json,
        "full_result_json": row.full_result_json,
    }
