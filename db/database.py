"""
db/database.py — Async SQLAlchemy + aiosqlite schema and engine setup.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

import config

# ── Engine ─────────────────────────────────────────────────────────────────────

DATABASE_URL = f"sqlite+aiosqlite:///{config.DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# ── Base and models ────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class TriageCase(Base):
    __tablename__ = "triage_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    chief_complaint = Column(Text, nullable=False)
    patient_age = Column(Integer, nullable=True)
    patient_sex = Column(String, nullable=True)
    triage_level = Column(Integer, nullable=False)
    esi_label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    escalate_immediately = Column(Boolean, nullable=False, default=False)
    red_flags_count = Column(Integer, nullable=False, default=0)
    rule_engine_triggered = Column(Boolean, nullable=False, default=False)
    safety_override_applied = Column(Boolean, nullable=False, default=False)
    model_used = Column(String, nullable=False)
    response_time_ms = Column(Integer, nullable=False, default=0)
    data_completeness_score = Column(Float, nullable=False, default=0.0)
    rag_context_used = Column(Boolean, nullable=False, default=False)
    full_input_json = Column(Text, nullable=True)
    full_result_json = Column(Text, nullable=True)


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    model_name = Column(String, nullable=False, index=True)
    was_successful = Column(Boolean, nullable=False, default=True)
    response_time_ms = Column(Integer, nullable=False, default=0)
    triage_level_produced = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    error_type = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)


# ── Startup ────────────────────────────────────────────────────────────────────

async def create_tables() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
