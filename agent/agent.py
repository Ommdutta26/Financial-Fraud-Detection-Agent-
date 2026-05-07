# ============================================================
# agent.py — Public entry point
# ============================================================

import logging
import numpy as np

from agent import features as feat_builder
from agent import graph as pipeline
from agent.graph import FraudState

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)



def run_agent(user_input: dict) -> dict:
    """
    Main entry point for the fraud detection agent.

    Args:
        user_input: Raw transaction dict, e.g.:
            {
                'TransactionAmt': 150.0,
                'hour': 14,
                'P_emaildomain': 'gmail.com',
                'ProductCD': 'W',
                'card1': 12345,
            }

    Returns:
        Dict with decision, scores, flags, explanations, and report.
    """
    logger.info(
        f"Processing transaction: "
        f"amt=${user_input.get('TransactionAmt', 0):.2f} "
        f"hour={user_input.get('hour', '?')}"
    )

    # Build features + provenance (what data source each value came from)
    features, provenance = feat_builder.build_features_with_provenance(user_input)

    initial_state = FraudState(
        transaction      = user_input,
        features         = features,
        provenance       = provenance,
        xgb_score        = 0.0,
        iso_score        = 0.0,
        ensemble_score   = 0.0,
        calibrated_score = 0.0,
        risk_level       = 'LOW',
        threshold        = 0.5,
        pattern_flags    = [],
        rule_flags       = [],
        velocity_flags   = [],
        shap_dict        = {},
        shap_reasons     = [],
        counterfactual   = '',
        decision         = '',
        confidence       = 0.0,
        risk_summary     = '',
        agent_reasoning  = '',
        report           = '',
    )

    result = pipeline.get_graph().invoke(initial_state)

    return {
        'decision':        result.get('decision',
                               result.get('final_decision', 'FLAG')),
        'confidence':      round(result.get('confidence',    50.0), 1),
        'score':           round(result.get('ensemble_score', 0.0), 4),
        'xgb_score':       round(result.get('xgb_score',     0.0), 4),
        'risk_level':      result.get('risk_level',      'MEDIUM'),
        'pattern_flags':   result.get('pattern_flags',   []),
        'rule_flags':      result.get('rule_flags',      []),
        'shap_reasons':    result.get('shap_reasons',    []),
        'shap_dict':       result.get('shap_dict',       {}),
        'agent_reasoning': result.get('agent_reasoning', ''),
        'report':          result.get('report',          ''),
        'threshold':       round(result.get('threshold', 0.5), 4),
        'risk_summary':    result.get('risk_summary',    ''),
    }


if __name__ == '__main__':
    # Quick smoke test
    sample = {
        'TransactionAmt': 4500.00,
        'hour': 2,
        'P_emaildomain': 'guerrillamail.com',
        'ProductCD': 'W',
        'card1': 99999,
    }
    out = run_agent(sample)
    print(out['report'])