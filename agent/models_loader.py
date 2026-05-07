# ============================================================
# models_loader.py — Load all model artifacts (lazy, singleton)
# ============================================================

import os
import json
import joblib
import logging

logger = logging.getLogger(__name__)

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(BASE, '..', 'models'))


def _path(name: str) -> str:
    return os.path.join(MODELS, name)


def _load(name: str):
    p = _path(name)
    logger.debug(f"Loading: {p}")
    return joblib.load(p)


def _load_optional(name: str, default=None):
    try:
        return _load(name)
    except FileNotFoundError:
        logger.warning(f"Optional artifact not found: {name} — using default")
        return default


def load_all() -> dict:
    m = {}

    # ── Core models (required) ───────────────────────────────
    m['xgb_model']      = _load('xgb_model.pkl')
    m['shap_explainer'] = _load('shap_explainer.pkl')
    m['feature_cols']   = _load('feature_cols.pkl')
    m['threshold_data'] = _load('threshold.pkl')
    m['THRESHOLD']      = m['threshold_data']['threshold']

    # ── Calibrated model (preferred for confidence scores) ───
    cal = _load_optional('calibrated_model.pkl')
    if cal is not None:
        m['calibrated_model'] = cal
        m['USE_CALIBRATED']   = True
        logger.info("Calibrated model loaded ✅")
    else:
        m['USE_CALIBRATED'] = False
        logger.warning("calibrated_model.pkl missing — raw XGBoost scores used")

    # ── Ensemble (IsoForest) ─────────────────────────────────
    # ── Ensemble disabled — XGBoost + calibration only ───────
    m['USE_ENSEMBLE'] = False

    # ── Feature store (real historical aggregates) ───────────
    fs = _load_optional('feature_store.pkl')
    if fs:
        m['feature_store'] = fs
        m['HAS_FEATURE_STORE'] = True
        logger.info(f"Feature store loaded ✅  "
                    f"({len(fs.get('card1_fraud_rate', {})):,} cards, "
                    f"{len(fs.get('email_fraud_rate', {})):,} emails)")
    else:
        m['feature_store']     = {}
        m['HAS_FEATURE_STORE'] = False
        logger.warning("feature_store.pkl missing — heuristic fallbacks used")

    # ── Imputation medians ───────────────────────────────────
    imp = _load_optional('imputation.pkl')
    if imp:
        m['imputation']     = imp
        m['HAS_IMPUTATION'] = True
        logger.info("Imputation medians loaded ✅")
    else:
        m['imputation']     = {'medians_all': {}}
        m['HAS_IMPUTATION'] = False

    # ── Counterfactuals ──────────────────────────────────────
    cf = _load_optional('counterfactuals.pkl')
    if cf:
        m['counterfactuals']     = cf
        m['HAS_COUNTERFACTUALS'] = True
        logger.info("Counterfactuals loaded ✅")
    else:
        m['counterfactuals']     = {}
        m['HAS_COUNTERFACTUALS'] = False

    # ── Fraud examples ───────────────────────────────────────
    fe = _load_optional('fraud_examples.pkl')
    if fe:
        m['fraud_examples']     = fe
        m['HAS_FRAUD_EXAMPLES'] = True
        logger.info("Fraud examples loaded ✅")
    else:
        m['fraud_examples']     = {}
        m['HAS_FRAUD_EXAMPLES'] = False

    # ── Agent config (JSON) ──────────────────────────────────
    cfg_path = _path('agent_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            m['agent_config'] = json.load(f)
        logger.info("Agent config loaded ✅")
    else:
        m['agent_config'] = {}

    # ── Model summary ────────────────────────────────────────
    m['model_summary'] = _load_optional('model_summary.pkl', default={})

    logger.info("All model artifacts loaded")
    return m


# ── Lazy singleton ───────────────────────────────────────────
_models = None

def get() -> dict:
    global _models
    if _models is None:
        _models = load_all()
    return _models


def reload() -> dict:
    """Force reload — useful when models are updated."""
    global _models
    _models = load_all()
    return _models