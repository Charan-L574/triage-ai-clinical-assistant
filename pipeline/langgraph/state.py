from typing import Any, Dict, List
from typing_extensions import TypedDict

class PipelineState(TypedDict, total=False):
    patient: Any
    session_id: str
    extraction: Any
    risk: Any
    rag_context: str
    rag_diseases: List[Dict[str, Any]]
    llm_response: Any
    result: Any
    timings: Dict[str, int]
    fast_tracked: bool
    needs_clarification: bool
    clarification_questions: List[Dict[str, str]]
    clarification_answers: Dict[str, str]
    hitl_reason: str
