# ============================================================
# nodes.py — LangGraph node functions
# Each function: FraudState → FraudState (partial update)
# ============================================================

import logging

from agent import models_loader
from agent import scoring
from agent import rules
from agent import llm_client
from agent import report as report_builder
from agent.config import FAST_APPROVE_THRESHOLD
logger = logging.getLogger(__name__)


# ── Scoring ──────────────────────────────────────────────────

def node_score(state: dict) -> dict:
    """Score transaction with XGBoost + optional ensemble."""
    scores = scoring.score_transaction(state['features'])
    logger.info(f"Score: ensemble={scores['ensemble_score']:.4f} "
                f"risk={scores['risk_level']}")
    return {**state, **scores}


def node_fast_approve(state: dict) -> dict:
    """Skip full pipeline for very low-risk transactions."""
    score = state['ensemble_score']
    logger.info(f"Fast-approve triggered (score={score:.4f})")
    return {
        **state,
        'pattern_flags':   [],
        'rule_flags':      [],
        'shap_dict':       {},
        'shap_reasons':    ['Score well below threshold'],
        'final_decision':  'APPROVE',
        'confidence':      99.0,
        'agent_reasoning': f'Auto-approved. Very low risk score {score:.4f}.',
        'report':          f'AUTO-APPROVED | Score: {score:.4f} | Risk: LOW',
    }


# ── Pattern & rule detection ─────────────────────────────────

def node_patterns(state: dict) -> dict:
    """Detect suspicious behavioral patterns."""
    flags = rules.get_pattern_flags(state['transaction'], state['ensemble_score'])
    logger.debug(f"Pattern flags: {flags}")
    return {**state, 'pattern_flags': flags}


def node_rules(state: dict) -> dict:
    """Apply business rule engine."""
    flags = rules.get_rule_flags(state['transaction'], state['xgb_score'])
    logger.debug(f"Rule flags: {flags}")
    return {**state, 'rule_flags': flags}


# ── Explainability ───────────────────────────────────────────

def node_explain(state: dict) -> dict:
    """Generate SHAP feature importance explanations."""
    shap_dict, reasons = scoring.get_shap_explanations(state['features'], n=5)
    return {**state, 'shap_dict': shap_dict, 'shap_reasons': reasons}


# ── LLM decision ─────────────────────────────────────────────

def node_decide(state: dict) -> dict:
    """Use Groq LLM to reason and produce final decision."""
    m = models_loader.get()

    llm_state = {**state, 'threshold': m['THRESHOLD']}
    prompt    = llm_client.build_decision_prompt(state['transaction'], llm_state)
    raw       = llm_client.call_groq(prompt)

    parsed = llm_client.parse_llm_response(
        raw,
        fallback_score  = state['ensemble_score'],
        threshold       = m['THRESHOLD'],
        n_pattern_flags = len(state['pattern_flags']),
    )

    logger.info(f"Decision: {parsed['decision']} "
                f"confidence={parsed['confidence']:.1f}%")

    return {
        **state,
        'final_decision':  parsed['decision'],
        'confidence':      parsed['confidence'],
        'agent_reasoning': parsed['reasoning'],
    }


# ── Report ───────────────────────────────────────────────────

def node_report(state: dict) -> dict:
    """Generate structured investigation report."""
    m = models_loader.get()
    full_state = {**state, 'threshold': m['THRESHOLD']}
    return {**state, 'report': report_builder.build_report(full_state)}


# ── Router ───────────────────────────────────────────────────

def route_after_score(state: dict) -> str:
    """Conditional edge: fast-approve very low scores, else full pipeline."""
    if state['ensemble_score'] < FAST_APPROVE_THRESHOLD:
        return "fast_approve"
    return "patterns"
