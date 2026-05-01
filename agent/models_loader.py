# ============================================================
# models_loader.py — Load all pickled models once at startup
# ============================================================

import os
import joblib
import logging

logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))



def _load(name: str):
    path = os.path.join(BASE, '..', 'models', name)
    path = os.path.abspath(path)
    logger.debug(f"Loading model: {path}")
    return joblib.load(path)

def load_all() -> dict:
    """
    Load required and optional models.
    Returns a dict with all model objects and a USE_ENSEMBLE flag.
    """
    models = {}

    # ── Required ────────────────────────────────────────────
    models['xgb_model']      = _load('xgb_model.pkl')
    models['shap_explainer'] = _load('shap_explainer.pkl')
    models['threshold_data'] = _load('threshold.pkl')
    models['feature_cols']   = _load('feature_cols.pkl')
    models['label_encoders'] = _load('label_encoders.pkl')
    models['THRESHOLD']      = models['threshold_data']['threshold']

    # ── Optional ensemble ────────────────────────────────────
    try:
        models['iso_model']   = _load('iso_model.pkl')
        models['iso_norm']    = _load('iso_norm_params.pkl')
        models['ens_weights'] = _load('ensemble_weights.pkl')
        models['USE_ENSEMBLE'] = True
        logger.info("Ensemble models loaded ✓")
    except FileNotFoundError as e:
        models['USE_ENSEMBLE'] = False
        logger.warning(f"Ensemble models not found — XGBoost only. ({e})")

    return models


# Singleton — imported by other modules
_models = load_all()


def get() -> dict:
    """Return the loaded models dict."""
    return _models
