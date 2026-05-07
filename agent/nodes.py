# ============================================================
# nodes.py — LangGraph node functions
# Each function: FraudState dict → FraudState dict (partial update)
# ============================================================

import logging

from agent import models_loader
from agent import scoring
from agent import rules
from agent import llm_client
from agent import report as report_builder
from agent.config import FAST_APPROVE_THRESHOLD

logger = logging.getLogger(__name__)


# ── Node 1: Score ─────────────────────────────────────────────

def node_score(state: dict) -> dict:
    """
    Run XGBoost + isotonic calibration scoring.
    Also loads threshold so all downstream nodes have it.
    """
    m      = models_loader.get()
    scores = scoring.score_transaction(state['features'])

    logger.info(
        f"[Score] ensemble={scores['ensemble_score']:.4f} "
        f"calibrated={scores['calibrated_score']:.4f} "
        f"risk={scores['risk_level']}"
    )

    return {
        **state,
        **scores,
        'threshold': m['THRESHOLD'],
    }


# ── Node 2: Fast Approve ──────────────────────────────────────

def node_fast_approve(state: dict) -> dict:
    """
    Skip full pipeline for very low-risk transactions.
    Only triggers when score < FAST_APPROVE_THRESHOLD (0.03).
    """
    score = state['ensemble_score']
    cal   = state.get('calibrated_score', score)
    logger.info(f"[FastApprove] score={score:.4f} — bypassing pipeline")

    return {
        **state,
        'pattern_flags':  [],
        'rule_flags':     [],
        'velocity_flags': [],
        'shap_dict':      {},
        'shap_reasons':   ['Score well below detection threshold'],
        'counterfactual': 'N/A — transaction is low risk',
        'decision':       'APPROVE',
        'confidence':     round((1 - score) * 100, 1),
        'risk_summary':   f'Score {score:.4f} far below threshold {state["threshold"]:.4f}',
        'agent_reasoning': (
            f'Auto-approved. Calibrated fraud probability {cal:.1%} '
            f'is far below the {state["threshold"]:.4f} threshold. '
            f'No further investigation required.'
        ),
        'report': (
            f'AUTO-APPROVED | Score: {score:.4f} | '
            f'Calibrated: {cal:.4f} | Risk: LOW'
        ),
    }


# ── Node 3: Memory (velocity tracking) ───────────────────────

def node_memory(state: dict) -> dict:
    """
    Check in-session card velocity before pattern analysis.
    Does NOT record the transaction yet — that happens after decision.
    """
    velocity_flags = rules.get_velocity_flags(state['transaction'])

    if velocity_flags:
        logger.info(f"[Memory] {len(velocity_flags)} velocity flags: {velocity_flags}")
    else:
        logger.debug("[Memory] No velocity flags")

    return {**state, 'velocity_flags': velocity_flags}


# ── Node 4: Patterns ──────────────────────────────────────────

def node_patterns(state: dict) -> dict:
    """
    Detect suspicious behavioral patterns.
    Merges velocity flags into pattern flags.
    """
    pattern_flags = rules.get_pattern_flags(
        state['transaction'],
        state['ensemble_score'],
        provenance=state.get('provenance'),
    )

    # Merge velocity flags in
    all_flags = pattern_flags + state.get('velocity_flags', [])

    logger.debug(f"[Patterns] {len(all_flags)} flags")
    return {**state, 'pattern_flags': all_flags}


# ── Node 5: Rules ─────────────────────────────────────────────

def node_rules(state: dict) -> dict:
    """Apply deterministic business rule engine."""
    rule_flags = rules.get_rule_flags(
        state['transaction'],
        state['xgb_score'],
    )
    logger.debug(f"[Rules] {len(rule_flags)} rules triggered")
    return {**state, 'rule_flags': rule_flags}


# ── Node 6: Explain ───────────────────────────────────────────

def node_explain(state: dict) -> dict:
    """
    Generate SHAP explanations + counterfactual.
    """
    shap_dict, shap_reasons = scoring.get_shap_explanations(
        state['features'], n=5
    )

    counterfactual = scoring.get_counterfactual(
        state['transaction'],
        state['ensemble_score'],
    )

    logger.debug(f"[Explain] top feature: "
                 f"{shap_reasons[0] if shap_reasons else 'N/A'}")

    return {
        **state,
        'shap_dict':     shap_dict,
        'shap_reasons':  shap_reasons,
        'counterfactual': counterfactual,
    }


# ── Node 7: Decide (LLM reasoning) ───────────────────────────

def node_decide(state: dict) -> dict:

    # ── Velocity override — force FLAG if velocity detected ──
    if state.get('velocity_flags'):
        logger.info("[Decide] Velocity flags detected — forcing FLAG")
        rules.record_transaction(state['transaction'])
        return {
            **state,
            'decision':        'FLAG',
            'confidence':      75.0,
            'risk_summary':    state['velocity_flags'][0],
            'agent_reasoning': (
                f"Transaction flagged due to velocity pattern: "
                f"{state['velocity_flags'][0]}. "
                f"Model score {state['ensemble_score']:.3f} is low but "
                f"behavioral pattern overrides for manual review."
            ),
        }

    # ── Normal LLM decision ──────────────────────────────────
    prompt = llm_client.build_decision_prompt(
        tx         = state['transaction'],
        state      = state,
        provenance = state.get('provenance'),
    )
    raw    = llm_client.call_groq(prompt)
    parsed = llm_client.parse_llm_response(
        raw,
        fallback_score  = state['ensemble_score'],
        threshold       = state['threshold'],
        n_pattern_flags = len(state['pattern_flags']),
    )

    logger.info(
        f"[Decide] {parsed['decision']} "
        f"confidence={parsed['confidence']:.1f}%"
    )

    rules.record_transaction(state['transaction'])

    return {
        **state,
        'decision':        parsed['decision'],
        'confidence':      parsed['confidence'],
        'risk_summary':    parsed.get('risk_summary', ''),
        'agent_reasoning': parsed['reasoning'],
    }
# ── Node 8: Report ────────────────────────────────────────────

def node_report(state: dict) -> dict:
    """Generate the full structured investigation report."""
    report = report_builder.build_report(state)
    return {**state, 'report': report}


# ── Router ────────────────────────────────────────────────────

def route_after_score(state: dict) -> str:
    """
    Fast-approve only the most obviously clean transactions.
    Everything else goes through the full 6-node pipeline.
    """
    score = state['ensemble_score']
    if score < FAST_APPROVE_THRESHOLD:
        logger.info(f"[Router] → fast_approve (score={score:.4f})")
        return "fast_approve"
    logger.info(f"[Router] → full pipeline (score={score:.4f})")
    return "memory"