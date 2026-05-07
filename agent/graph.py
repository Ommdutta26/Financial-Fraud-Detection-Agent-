# ============================================================
# graph.py — LangGraph StateGraph pipeline
# ============================================================

from typing import TypedDict, List, Optional
import numpy as np
from langgraph.graph import StateGraph, END

from agent import nodes


# ── State schema ─────────────────────────────────────────────

class FraudState(TypedDict):
    # Input
    transaction:      dict
    features:         np.ndarray
    provenance:       dict          # feature source metadata

    # Scores
    xgb_score:        float
    iso_score:        float
    ensemble_score:   float
    calibrated_score: float         # calibrated honest probability
    risk_level:       str
    threshold:        float

    # Signals
    pattern_flags:    List[str]
    rule_flags:       List[str]
    velocity_flags:   List[str]

    # Explainability
    shap_reasons:     List[str]
    shap_dict:        dict
    counterfactual:   str           # "what would make this safer"

    # Decision
    decision:         str           # renamed from final_decision
    confidence:       float
    risk_summary:     str           # single biggest risk signal
    agent_reasoning:  str

    # Output
    report:           str


# ── Graph construction ────────────────────────────────────────

def build_graph():
    wf = StateGraph(FraudState)

    wf.add_node("score",           nodes.node_score)
    wf.add_node("fast_approve",    nodes.node_fast_approve)
    wf.add_node("memory",          nodes.node_memory)
    wf.add_node("patterns",        nodes.node_patterns)
    wf.add_node("rules",           nodes.node_rules)
    wf.add_node("explain",         nodes.node_explain)
    wf.add_node("decide",          nodes.node_decide)
    wf.add_node("generate_report", nodes.node_report)

    wf.set_entry_point("score")

    wf.add_conditional_edges(
        "score",
        nodes.route_after_score,
        {"fast_approve": "fast_approve", "memory": "memory"},
    )

    wf.add_edge("fast_approve",    END)
    wf.add_edge("memory",          "patterns")
    wf.add_edge("patterns",        "rules")
    wf.add_edge("rules",           "explain")
    wf.add_edge("explain",         "decide")
    wf.add_edge("decide",          "generate_report")
    wf.add_edge("generate_report", END)

    return wf.compile()


_graph = build_graph()


def get_graph():
    return _graph