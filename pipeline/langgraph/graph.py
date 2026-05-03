import sqlite3
from typing import Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

from models.patient import PatientInput
from pipeline.langgraph.state import PipelineState
from pipeline.langgraph.nodes import (
    node_extract,
    node_risk,
    node_rag,
    node_llm,
    node_check_hitl,
    node_hitl_interrupt,
    node_validate,
    node_recommend,
    node_safety
)

# ── Checkpointer (Persistent to survive Uvicorn reloads) ──────────────────────
_checkpoint_conn = sqlite3.connect("triageai_checkpoints.db", check_same_thread=False)
_checkpointer = SqliteSaver(_checkpoint_conn)
_checkpointer.setup()

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)
    builder.add_node("extract", node_extract)
    builder.add_node("risk", node_risk)
    builder.add_node("rag", node_rag)
    builder.add_node("llm", node_llm)
    builder.add_node("check_hitl", node_check_hitl)
    builder.add_node("hitl_interrupt", node_hitl_interrupt)
    builder.add_node("validate", node_validate)
    builder.add_node("recommend", node_recommend)
    builder.add_node("safety", node_safety)

    builder.set_entry_point("extract")
    builder.add_edge("extract", "risk")
    builder.add_edge("risk", "rag")
    builder.add_edge("rag", "llm")
    builder.add_edge("llm", "check_hitl")

    def route_hitl(state: PipelineState) -> str:
        return "hitl_interrupt" if state.get("needs_clarification") else "validate"

    builder.add_conditional_edges("check_hitl", route_hitl)
    builder.add_edge("hitl_interrupt", "llm")  # Re-run LLM after getting answers
    builder.add_edge("validate", "recommend")
    builder.add_edge("recommend", "safety")
    builder.add_edge("safety", END)

    return builder.compile(
        checkpointer=_checkpointer,
        interrupt_before=["hitl_interrupt"]
    )

_graph = build_graph()

class LangGraphPipeline:
    @staticmethod
    def start(patient: PatientInput, thread_id: str = None) -> Dict[str, Any]:
        if not thread_id:
            import uuid
            thread_id = str(uuid.uuid4())
        config_dict = {"configurable": {"thread_id": thread_id}}
        initial_state: PipelineState = {
            "patient": patient,
            "session_id": thread_id,
            "timings": {},
            "fast_tracked": False,
        }
        
        final_state = _graph.invoke(initial_state, config=config_dict)
        return LangGraphPipeline._handle_output(final_state, thread_id)

    @staticmethod
    def resume(thread_id: str, answers: Dict[str, str]) -> Dict[str, Any]:
        config_dict = {"configurable": {"thread_id": thread_id}}
        _graph.update_state(config_dict, {"clarification_answers": answers})
        final_state = _graph.invoke(None, config=config_dict)
        return LangGraphPipeline._handle_output(final_state, thread_id)

    @staticmethod
    def _handle_output(final_state: PipelineState, thread_id: str) -> Dict[str, Any]:
        next_nodes = _graph.get_state({"configurable": {"thread_id": thread_id}}).next
        if "hitl_interrupt" in next_nodes:
            return {
                "status": "pending_clarification",
                "thread_id": thread_id,
                "reason": final_state.get("hitl_reason", ""),
                "questions": final_state.get("clarification_questions", []),
            }
        
        from models.triage_output import TriageResult
        result: TriageResult = final_state.get("result")
        if result is None:
            raise RuntimeError("Pipeline resumed but result is None")
        return {"status": "complete", "result": result}
