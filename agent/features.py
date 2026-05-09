import numpy as np
import pandas as pd
import logging

from agent.config import (
    EMAIL_RISK_SCORES, EMAIL_RISK_DEFAULT,
    HIGH_RISK_EMAILS,
)
from agent import models_loader

logger = logging.getLogger(__name__)

# High risk emails — data-driven from IEEE-CIS
HIGH_RISK_EMAILS_REAL = {'protonmail.com', 'mail.com', 'outlook.es', 'aim.com'}


# ── Public entry point ───────────────────────────────────────

def build_features(tx: dict) -> np.ndarray:
    m            = models_loader.get()
    feature_cols = m['feature_cols']
    fs           = m['feature_store']
    medians      = m['imputation'].get('medians_all', {})

    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    card  = int(tx.get('card1', 99999))
    email = str(tx.get('P_emaildomain', 'gmail.com'))

    row = _build_row(amt, hour, card, email, fs, medians)

    # Build full vector — start from medians, overlay known values
    full_row = {col: medians.get(col, -999) for col in feature_cols}
    full_row.update(row)

    df = pd.DataFrame([full_row])[feature_cols]
    return df.fillna(-999).values


# ── Feature store helpers ────────────────────────────────────

def _fs_get(fs: dict, store_key: str, lookup_key, fallback: float) -> float:
    return float(fs.get(store_key, {}).get(lookup_key, fallback))


def _global(fs: dict, key: str, default: float) -> float:
    return float(fs.get(key, default))


# ── Row builder ──────────────────────────────────────────────

def _build_row(amt: float, hour: int, card: int,
               email: str, fs: dict, medians: dict) -> dict:

    # ── Global fallbacks ─────────────────────────────────────
    global_fraud = _global(fs, 'global_fraud_rate',  0.035)
    global_avg   = _global(fs, 'global_avg_amount',   135.0)
    global_std   = _global(fs, 'global_std_amount',   216.0)
    global_count = _global(fs, 'global_tx_count',      12.0)
    global_max   = _global(fs, 'global_max_amount',  5000.0)

    # ── Card lookups (real history) ──────────────────────────
    card_fraud  = _fs_get(fs, 'card1_fraud_rate',  card, global_fraud)
    card_count  = _fs_get(fs, 'card1_tx_count',    card, global_count)
    card_avg    = _fs_get(fs, 'card1_avg_amount',  card, global_avg)
    card_std    = _fs_get(fs, 'card1_std_amount',  card, global_std)
    card_max    = _fs_get(fs, 'card1_max_amount',  card, global_max)
    card_unique_emails = _fs_get(fs, 'card1_unique_emails', card, 1.0)

    # ── Email lookups (real history) ─────────────────────────
    email_fraud = _fs_get(fs, 'email_fraud_rate', email, global_fraud)

    # ── Card + email combo ───────────────────────────────────
    card_email_key   = (card, email)
    card_email_fraud = _fs_get(fs, 'card_email_fraud_rate',
                               card_email_key, global_fraud)
    card_email_count = _fs_get(fs, 'card_email_count',
                               card_email_key, 1.0)
    card_device_count = _fs_get(fs, 'card_device_count',
                                card, medians.get('card_device_count', 2.0))
    card_addr_count   = _fs_get(fs, 'card_addr_count',
                                card, medians.get('card_addr_count', 3.0))

    # ── Derived ──────────────────────────────────────────────
    amt_deviation = (amt - card_avg) / (card_std + 1.0)
    card_amount_rank = min(0.99, amt / (card_max + 1.0))

    return {
        # ── Direct inputs ─────────────────────────────────────
        'TransactionAmt': amt,
        'hour':           hour,
        'card1':          card,

        # ── Time ─────────────────────────────────────────────
        'amount_log':         np.log1p(amt),
        'amount_dec':         amt % 1,
        'is_round_amt':       int(amt % 1 == 0),
        'is_night':           int(hour <= 5),
        'is_early_morning':   int(5 <= hour <= 9),
        'is_risky_hour':      int(hour in [5, 6, 7, 8, 9]),
        'is_weekend':         0,   # unknown at inference
        'is_business_hour':   int(9 <= hour <= 17),
        'is_fraud_sweetspot': int(500 <= amt <= 1000),
        'is_card_test_amt':   int(0 < amt <= 50),

        # ── Card aggregates ───────────────────────────────────
        'card1_tx_count':         card_count,
        'card1_avg_amount':       card_avg,
        'card1_std_amount':       card_std,
        'card1_max_amount':       card_max,
        'card1_amount_deviation': amt_deviation,
        'card1_unique_emails':    card_unique_emails,
        'card_amount_rank':       card_amount_rank,
        'card_tx_sequence':       card_count,
        'is_first_tx':            int(card_count == 0),

        # ── Cross-entity ──────────────────────────────────────
        'card_email_count':  card_email_count,
        'card_device_count': card_device_count,
        'card_addr_count':   card_addr_count,

        # ── Label features (feature store) ────────────────────
        'p_email_high_risk':     int(email in HIGH_RISK_EMAILS_REAL),
        'card1_fraud_rate':      card_fraud,
        'email_fraud_rate':      email_fraud,
        'card_email_fraud_rate': card_email_fraud,
    }


# ── Provenance ───────────────────────────────────────────────

def build_features_with_provenance(tx: dict) -> tuple[np.ndarray, dict]:
    m   = models_loader.get()
    fs  = m['feature_store']

    amt   = float(tx.get('TransactionAmt', 0))
    card  = int(tx.get('card1', 99999))
    email = str(tx.get('P_emaildomain', 'gmail.com'))

    global_fraud    = fs.get('global_fraud_rate', 0.035)
    card_key        = (card, email)
    card_fraud_raw  = fs.get('card1_fraud_rate',     {}).get(card)
    email_fraud_raw = fs.get('email_fraud_rate',      {}).get(email)
    combo_fraud_raw = fs.get('card_email_fraud_rate', {}).get(card_key)
    card_count_raw  = fs.get('card1_tx_count',        {}).get(card)

    provenance = {
        'card_seen_before':      card_fraud_raw  is not None,
        'email_seen_before':     email_fraud_raw is not None,
        'combo_seen_before':     combo_fraud_raw is not None,
        'card_fraud_rate':       card_fraud_raw  or global_fraud,
        'email_fraud_rate':      email_fraud_raw or global_fraud,
        'card_email_fraud_rate': combo_fraud_raw or global_fraud,
        'card_tx_count':         card_count_raw  or 0,
        'card_source':  'feature_store' if card_fraud_raw  else 'global_fallback',
        'email_source': 'feature_store' if email_fraud_raw else 'global_fallback',
        'combo_source': 'feature_store' if combo_fraud_raw else 'global_fallback',
        'amt_vs_avg':   amt - fs.get('card1_avg_amount', {}).get(
                            card, fs.get('global_avg_amount', 135)),
    }

    features = build_features(tx)
    return features, provenance