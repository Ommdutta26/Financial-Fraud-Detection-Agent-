# ============================================================
# scoring.py — ML model inference (XGBoost + optional ensemble)
# ============================================================

import numpy as np
import logging

import models_loader
from config import RISK_BANDS

logger = logging.getLogger(__name__)


def get_risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_BANDS.items():
        if lo <= score < hi:
            return level
    return "CRITICAL"


def score_transaction(features: np.ndarray) -> dict:
    """
    Run ML inference and return a score dict:
      xgb_score, iso_score, ensemble_score, risk_level
    """
    m = models_loader.get()

    xgb_score = float(m['xgb_model'].predict_proba(features)[0][1])
    logger.debug(f"XGBoost score: {xgb_score:.4f}")

    if m['USE_ENSEMBLE']:
        iso_raw   = float(m['iso_model'].score_samples(features)[0])
        iso_score = _normalize_iso(iso_raw, m['iso_norm'])
        weights   = m['ens_weights']
        ensemble  = (xgb_score * weights.get('xgboost', 0.75) +
                     iso_score  * weights.get('isolation_forest', 0.25))
        logger.debug(f"Isolation Forest score: {iso_score:.4f} | Ensemble: {ensemble:.4f}")
    else:
        iso_score = 0.0
        ensemble  = xgb_score

    return {
        'xgb_score':      xgb_score,
        'iso_score':      iso_score,
        'ensemble_score': ensemble,
        'risk_level':     get_risk_level(ensemble),
    }


def get_shap_explanations(features: np.ndarray, n: int = 5) -> tuple[dict, list[str]]:
    """
    Returns (shap_dict, reasons_list) for the top-n impacting features.
    """
    m = models_loader.get()
    try:
        vals    = m['shap_explainer'].shap_values(features)[0]
        impacts = sorted(zip(m['feature_cols'], vals),
                         key=lambda x: abs(x[1]), reverse=True)[:n]

        shap_dict = {f: round(float(v), 4) for f, v in impacts}
        reasons   = [
            f"{feat} {'↑ increases' if v > 0 else '↓ decreases'} "
            f"fraud risk (impact: {abs(v):.3f})"
            for feat, v in impacts
        ]
        return shap_dict, reasons

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {}, [f"Explanation unavailable: {e}"]


# ── Private ──────────────────────────────────────────────────

def _normalize_iso(raw: float, norm_params: dict) -> float:
    mn = norm_params['min']
    mx = norm_params['max']
    return float(1 - (raw - mn) / (mx - mn + 1e-8))
