# ============================================================
# scoring.py — ML model inference (calibrated XGBoost + ensemble)
# ============================================================

import numpy as np
import logging

from agent import models_loader
from agent.config import RISK_BANDS

logger = logging.getLogger(__name__)


def get_risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_BANDS.items():
        if lo <= score < hi:
            return level
    return "CRITICAL"


def score_transaction(features: np.ndarray) -> dict:
    """
    Run ML inference. Returns xgb_score, iso_score,
    ensemble_score, calibrated_score, risk_level.
    """
    m = models_loader.get()

    # ── Raw XGBoost score ────────────────────────────────────
    xgb_score = float(m['xgb_model'].predict_proba(features)[0][1])
    logger.debug(f"XGBoost raw: {xgb_score:.4f}")

    # ── Calibrated score (honest confidence) ─────────────────
    if m['USE_CALIBRATED']:
        cal_score = float(
            m['calibrated_model'].predict_proba(features)[0][1]
        )
        logger.debug(f"Calibrated: {cal_score:.4f}")
    else:
        cal_score = xgb_score

    # ── Isolation Forest ─────────────────────────────────────
    if m['USE_ENSEMBLE']:
        iso_raw   = float(m['iso_model'].score_samples(features)[0])
        iso_score = _normalize_iso(iso_raw, m['iso_norm'])
        weights   = m['ens_weights']
        ensemble  = (
            xgb_score * weights.get('xgboost', 0.75) +
            iso_score  * weights.get('isolation_forest', 0.25)
        )
        logger.debug(f"IsoForest: {iso_score:.4f} | Ensemble: {ensemble:.4f}")
    else:
        iso_score = 0.0
        ensemble  = cal_score  # calibrated IS the final score now

    return {
        'xgb_score':        xgb_score,
        'iso_score':        iso_score,
        'ensemble_score':   ensemble,   # = cal_score when no IsoForest
        'calibrated_score': cal_score,
        'risk_level':       get_risk_level(cal_score),  # based on calibrated
    }

def get_shap_explanations(features: np.ndarray,
                          n: int = 5) -> tuple[dict, list[str]]:
    """
    Returns (shap_dict, reasons_list) for the top-n impacting features.
    """
    m = models_loader.get()
    try:
        vals    = m['shap_explainer'].shap_values(features)[0]
        impacts = sorted(
            zip(m['feature_cols'], vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]

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


def get_counterfactual(tx: dict, ensemble_score: float) -> str:
    """
    Use saved counterfactual baselines to explain what would
    make this transaction safer. Returns a human-readable string.
    """
    m = models_loader.get()
    if not m['HAS_COUNTERFACTUALS']:
        return "Counterfactual analysis not available."

    cf    = m['counterfactuals']
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    lines = []

    # Amount counterfactual
    # Amount counterfactual — sweet spot check
    if 500 <= amt <= 1000:
        lines.append(
            f"${amt:,.2f} is in the $500-$1K fraud sweet spot "
            f"(5.6% fraud rate in training data) — "
            f"amounts in $50-$200 range have lowest risk (2.9-3.1%)"
        )
    else:
        amt_table = cf.get('amount_risk_table', [])
        safest = min(amt_table, key=lambda r: r.get('avg_score', 1.0),
                     default=None)
        if safest:
            lines.append(
                f"Lowest risk amount range is {safest['amount_bin']} "
                f"(avg score {safest['avg_score']:.2%})"
            )
    # Hour counterfactual
    safe_hours = cf.get('safe_hours', [])
    risky_hours = cf.get('risky_hours', [])
    if hour in risky_hours and safe_hours:
        safe_str = ', '.join(f"{h}:00" for h in sorted(safe_hours)[:4])
        lines.append(
            f"Hour {hour}:00 is high-risk. "
            f"Lower-risk hours include: {safe_str}"
        )

    if not lines:
        lines.append(
            "Transaction profile has multiple risk signals — "
            "no single change would bring it to low risk."
        )

    return " | ".join(lines)


# ── Private ──────────────────────────────────────────────────

def _normalize_iso(raw: float, norm_params: dict) -> float:
    mn = norm_params['min']
    mx = norm_params['max']
    return float(1 - (raw - mn) / (mx - mn + 1e-8))