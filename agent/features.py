# ============================================================
# features.py — Raw transaction → feature vector
# ============================================================

import numpy as np
import pandas as pd
import logging

from agent.config import HIGH_RISK_EMAILS, MEDIUM_RISK_EMAILS
from agent import models_loader
logger = logging.getLogger(__name__)


def build_features(tx: dict) -> np.ndarray:
    """
    Convert a raw transaction dict into the model's feature vector.

    In production, behavioral fields (card_fraud_rate, etc.) come from
    a feature store / database. Here they are derived heuristically for
    demo purposes.
    """
    m = models_loader.get()
    feature_cols   = m['feature_cols']
    label_encoders = m['label_encoders']

    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    card  = int(tx.get('card1', 99999))
    email = str(tx.get('P_emaildomain', 'gmail.com'))
    prod  = str(tx.get('ProductCD', 'W'))

    row = _build_row(amt, hour, card, email, prod)

    df = pd.DataFrame([row])

    # Fill any feature columns the row is missing
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -999

    # Label-encode categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                logger.warning(f"Label encoder for '{col}' failed — using -999")
                df[col] = -999

    return df[feature_cols].fillna(-999).values


# ── Private helpers ──────────────────────────────────────────

def _email_fraud_rate(email: str) -> float:
    if email in HIGH_RISK_EMAILS:
        return 0.75
    if email in MEDIUM_RISK_EMAILS:
        return 0.15
    return 0.05


def _card_fraud_rate(amt: float) -> float:
    return min(0.9, max(0.01, (amt / 5000) * 0.6))


def _build_row(amt: float, hour: int, card: int,
               email: str, prod: str) -> dict:
    email_fraud = _email_fraud_rate(email)
    card_fraud  = _card_fraud_rate(amt)
    amt_dev     = amt / max(80, amt * 0.3)

    return {
        # ── Core ───────────────────────────────────────────
        'TransactionAmt': amt,
        'ProductCD':      prod,
        'P_emaildomain':  email,
        'R_emaildomain':  email,

        # ── Time ───────────────────────────────────────────
        'hour':             hour,
        'day_of_week':      3,
        'is_night':         1 if hour <= 5 else 0,
        'is_weekend':       0,
        'is_business_hour': 1 if 9 <= hour <= 17 else 0,

        # ── Amount ─────────────────────────────────────────
        'amount_log':    np.log1p(amt),
        'amount_dec':    amt % 1,
        'is_round_amt':  1 if amt % 1 == 0 else 0,

        # ── Card behavioral ────────────────────────────────
        'card1':                  card,
        'card2':                  111,
        'card1_fraud_rate':       card_fraud,
        'card1_tx_count':         max(1, int(50 - amt / 200)),
        'card1_avg_amount':       max(10, amt * 0.4),
        'card1_std_amount':       max(5, amt * 0.2),
        'card1_max_amount':       amt * 1.5,
        'card1_amount_deviation': amt_dev,
        'card1_unique_cards':     1,
        'card1_unique_emails':    2 if amt > 500 else 1,

        # ── Email ──────────────────────────────────────────
        'email_fraud_rate': email_fraud,
        'p_email_high_risk': 1 if email in HIGH_RISK_EMAILS else 0,
        'same_email':        1,

        # ── Device ─────────────────────────────────────────
        'device_fraud_rate': 0.4 if amt > 1000 else 0.05,
        'DeviceInfo':        'Windows',

        # ── Address ────────────────────────────────────────
        'addr1': 299,
        'addr2': 87,

        # ── Graph / network ────────────────────────────────
        'card_email_count':      1 if amt > 800 else 5,
        'card_device_count':     2,
        'card_addr_count':       3,
        'card_email_fraud_rate': card_fraud * email_fraud,

        # ── Velocity ───────────────────────────────────────
        'card_amount_rank':  min(0.99, amt / 5000),
        'card_tx_sequence':  max(0, int(10 - amt / 500)),
        'time_since_last_tx': 3600 if amt > 500 else 86400,
        'is_first_tx':       0,
    }
