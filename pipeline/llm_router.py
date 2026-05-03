"""
pipeline/llm_router.py -- Stage 4: LLM routing with waterfall fallback.

Waterfall: MedGemma 27B (featherless-ai) -> Groq Llama-3.3 -> Gemini 2.5 Flash -> hardcoded fallback.
Uses the HuggingFace documented pattern: api_key= and client.chat.completions.create().
Implements 3-attempt JSON extraction and async parallel execution for comparison mode.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import config
from utils.logger import get_logger
from utils.prompts import SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    content: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    model_name: str = ""
    response_time_ms: int = 0
    success: bool = False
    error: Optional[str] = None


# ── JSON extraction helpers ────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """Robust JSON extraction using json_repair."""
    try:
        import json_repair
        # json_repair handles markdown fences, missing quotes, trailing commas, etc.
        parsed = json_repair.loads(text)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            # Sometimes models return an array of objects
            return parsed[0]
        elif isinstance(parsed, list):
            return {"raw_array": parsed}
    except ImportError:
        logger.warning("json_repair not installed; falling back to strict json extraction")
    except Exception as e:
        logger.warning(f"json_repair failed: {e}")

    # Fallback attempt 1: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback attempt 2: extract from triple-backtick fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # Fallback attempt 3: first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except Exception:
            pass

    return None


RETRY_INSTRUCTION = (
    "\n\nYour previous response was not valid JSON. "
    "Respond with ONLY a JSON object. No markdown, no explanation, no preamble."
)


# ── Per-model caller classes ───────────────────────────────────────────────────

class _MedGemmaClient:
    """
    MedGemma 27B via HuggingFace Inference API (featherless-ai provider).
    Uses the documented pattern: api_key= and client.chat.completions.create().
    Model ID format: google/medgemma-27b-text-it:featherless-ai
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(api_key=config.HF_TOKEN)
        return self._client

    def call(self, system: str, user: str) -> str:
        client = self._get_client()
        completion = client.chat.completions.create(
            model=config.MEDGEMMA_MODEL_ID,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
        )
        return completion.choices[0].message.content or ""


class _GroqClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=config.GROQ_API_KEY)
        return self._client

    def call(self, system: str, user: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=config.GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            timeout=config.API_TIMEOUT_SECONDS,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class _GeminiClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        return self._client

    def call(self, system: str, user: str) -> str:
        from google.genai import types
        client = self._get_client()
        full_prompt = f"{system}\n\n{user}"
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_ID,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""


# ── LLMRouter ─────────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Routes LLM calls with waterfall fallback: MedGemma → Groq → Gemini → fallback.
    Also supports parallel execution for comparison mode.
    """

    def __init__(self):
        self._medgemma = _MedGemmaClient()
        self._groq = _GroqClient()
        self._gemini = _GeminiClient()

        # Waterfall order:
        # 1. Groq Llama-3.3   -- fast general fallback
        # 2. MedGemma 27B     -- medical specialist, via featherless-ai provider (free HF token)
        # 3. Gemini 2.5 Flash -- secondary fallback
        self._models = [
            ("Groq-Llama-3.3", self._groq),
            ("MedGemma-27B", self._medgemma),
            ("Gemini-2.5-Flash", self._gemini),
        ]

    # ── Waterfall route ───────────────────────────────────────────────────────

    def route(self, prompt_data: dict) -> LLMResponse:
        system = prompt_data.get("system", SYSTEM_PROMPT)
        user = prompt_data.get("user", "")

        for model_name, client in self._models:
            resp = self._attempt(model_name, client, system, user)
            if resp.success:
                return resp
            logger.warning(f"[{model_name}] failed, trying next model. Error: {resp.error}")

        logger.critical("All LLM models failed — returning hardcoded safe fallback")
        return self._hardcoded_fallback()

    # ── Single-model attempt ──────────────────────────────────────────────────

    def _attempt(self, name: str, client, system: str, user: str) -> LLMResponse:
        t0 = time.monotonic()
        try:
            raw = client.call(system, user)
            elapsed = int((time.monotonic() - t0) * 1000)
            parsed = _extract_json(raw)
            if parsed is not None:
                logger.info(f"[{name}] success in {elapsed}ms")
                return LLMResponse(
                    content=parsed,
                    raw_text=raw,
                    model_name=name,
                    response_time_ms=elapsed,
                    success=True,
                )
            # Retry with correction instruction
            logger.warning(f"[{name}] JSON parse failed, retrying. Raw (first 300 chars): {raw[:300]}")
            raw2 = client.call(system, user + RETRY_INSTRUCTION)
            elapsed2 = int((time.monotonic() - t0) * 1000)
            parsed2 = _extract_json(raw2)
            if parsed2 is not None:
                logger.info(f"[{name}] success after retry in {elapsed2}ms")
                return LLMResponse(
                    content=parsed2,
                    raw_text=raw2,
                    model_name=name,
                    response_time_ms=elapsed2,
                    success=True,
                )
            logger.warning(f"[{name}] JSON parse failed after retry. Raw: {raw2[:300]}")
            return LLMResponse(
                model_name=name,
                response_time_ms=elapsed2,
                success=False,
                error="JSON parsing failed after retry",
                raw_text=raw2,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.warning(f"[{name}] API call exception after {elapsed}ms: {exc}")
            return LLMResponse(
                model_name=name,
                response_time_ms=elapsed,
                success=False,
                error=str(exc),
            )

    # ── Parallel execution (comparison mode) ──────────────────────────────────

    def run_all_parallel(self, prompt_data: dict) -> Dict[str, Any]:
        """Run all models concurrently using thread pool. Returns {model_name: LLMResponse | Exception}."""
        import concurrent.futures
        system = prompt_data.get("system", SYSTEM_PROMPT)
        user = prompt_data.get("user", "")
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            future_to_name = {
                executor.submit(self._attempt, name, client, system, user): name
                for name, client in self._models
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results[name] = exc
                    
        return results

    # ── Hardcoded safe fallback ───────────────────────────────────────────────

    @staticmethod
    def _hardcoded_fallback() -> LLMResponse:
        return LLMResponse(
            content={
                "triage_level": 2,
                "esi_label": "Emergent",
                "confidence_score": 0.0,
                "escalate_immediately": True,
                "red_flags_detected": [],
                "reasoning_steps": [
                    "All AI systems are currently unavailable.",
                    "This is an automated safety escalation to ESI 2.",
                    "Please seek immediate in-person medical evaluation.",
                ],
                "symptom_timeline_analysis": None,
                "differential_diagnoses": [],
                "follow_up_questions": [],
                "missing_vital_impact": {},
            },
            raw_text="SYSTEM_FALLBACK",
            model_name="emergency_fallback",
            response_time_ms=0,
            success=True,
            error="All models unavailable — hardcoded fallback",
        )

    # ── Model-specific public callers (for clarification prompt) ──────────────

    def call_best_available(self, system: str, user: str) -> str:
        """Return raw text from the best available model (for non-triage prompts)."""
        for model_name, client in self._models:
            try:
                raw = client.call(system, user)
                logger.info(f"[{model_name}] clarification call success")
                return raw
            except Exception as exc:
                logger.warning(f"[{model_name}] clarification call failed: {exc}")
        return "[]"
