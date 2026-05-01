# ============================================================
# agent.py — Public entry point
# ============================================================

import logging
import numpy as np

from agent import features as feat_builder
import graph as pipeline
from graph import FraudState

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
    logger.info(f"Processing transaction: amt=${user_input.get('TransactionAmt',0):.2f} "
                f"hour={user_input.get('hour','?')}")

    features = feat_builder.build_features(user_input)

    initial_state = FraudState(
        transaction      = user_input,
        features         = features,
        xgb_score        = 0.0,
        iso_score        = 0.0,
        ensemble_score   = 0.0,
        risk_level       = 'LOW',
        pattern_flags    = [],
        rule_flags       = [],
        shap_dict        = {},
        shap_reasons     = [],
        final_decision   = '',
        confidence       = 0.0,
        agent_reasoning  = '',
        report           = '',
    )

    result = pipeline.get_graph().invoke(initial_state)

    return {
        'decision':        result['final_decision'],
        'confidence':      round(result['confidence'], 1),
        'score':           round(result['ensemble_score'], 4),
        'xgb_score':       round(result['xgb_score'], 4),
        'risk_level':      result['risk_level'],
        'pattern_flags':   result['pattern_flags'],
        'rule_flags':      result['rule_flags'],
        'shap_reasons':    result['shap_reasons'],
        'shap_dict':       result['shap_dict'],
        'agent_reasoning': result['agent_reasoning'],
        'report':          result['report'],
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
