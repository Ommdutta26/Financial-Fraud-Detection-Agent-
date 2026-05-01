# ============================================================
# graph.py — Assemble the LangGraph StateGraph pipeline
# ============================================================

from typing import TypedDict, List
import numpy as np
from langgraph.graph import StateGraph, END

import nodes


# ── State schema ─────────────────────────────────────────────

class FraudState(TypedDict):
    transaction:     dict
    features:        np.ndarray
    xgb_score:       float
    iso_score:       float
    ensemble_score:  float
    risk_level:      str
    pattern_flags:   List[str]
    rule_flags:      List[str]
    shap_reasons:    List[str]
    shap_dict:       dict
    final_decision:  str
    confidence:      float
    agent_reasoning: str
    report:          str


# ── Graph construction ────────────────────────────────────────

def build_graph():
    wf = StateGraph(FraudState)

    # Nodes
    wf.add_node("score",           nodes.node_score)
    wf.add_node("fast_approve",    nodes.node_fast_approve)
    wf.add_node("patterns",        nodes.node_patterns)
    wf.add_node("rules",           nodes.node_rules)
    wf.add_node("explain",         nodes.node_explain)
    wf.add_node("decide",          nodes.node_decide)
    wf.add_node("generate_report", nodes.node_report)

    # Edges
    wf.set_entry_point("score")
    wf.add_conditional_edges(
        "score",
        nodes.route_after_score,
        {"fast_approve": "fast_approve", "patterns": "patterns"},
    )
    wf.add_edge("fast_approve",    END)
    wf.add_edge("patterns",        "rules")
    wf.add_edge("rules",           "explain")
    wf.add_edge("explain",         "decide")
    wf.add_edge("decide",          "generate_report")
    wf.add_edge("generate_report", END)

    return wf.compile()


# Singleton graph — imported by agent.py
_graph = build_graph()


def get_graph():
    return _graph
