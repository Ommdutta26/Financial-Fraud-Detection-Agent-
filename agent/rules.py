# ============================================================
# rules.py — Business rule engine + behavioral pattern flags
#            + in-memory card velocity tracking
# ============================================================

import time
import logging
from collections import defaultdict

from agent.config import (
    EMAIL_RISK_SCORES, EMAIL_RISK_DEFAULT,
    HIGH_RISK_EMAILS, HIGH_RISK_PRODUCTS, PRODUCT_LABELS,
    RULE_AMOUNT_WARN, RULE_AMOUNT_AML,
    RULE_NIGHT_AMOUNT, RULE_EMAIL_AMOUNT, RULE_MODEL_THRESH,
    PATTERN_HIGH_AMT, PATTERN_MED_AMT, PATTERN_ROUND_AMT,
    PATTERN_EXTREME, PATTERN_ELEVATED,
    VELOCITY_WINDOW_HOURS, VELOCITY_TX_LIMIT, VELOCITY_AMOUNT_FACTOR,
)

logger = logging.getLogger(__name__)

# ── In-memory card velocity store ────────────────────────────
# { card_id: [ {timestamp, amount, hour}, ... ] }
_card_history: dict = defaultdict(list)


# ── Velocity helpers ─────────────────────────────────────────

def record_transaction(tx: dict) -> None:
    """
    Record a transaction in the in-memory velocity store.
    Call this AFTER the decision is made.
    """
    card = int(tx.get('card1', 0))
    _card_history[card].append({
        'ts':     time.time(),
        'amount': float(tx.get('TransactionAmt', 0)),
        'hour':   int(tx.get('hour', 12)),
    })
    # Keep only last 50 records per card
    _card_history[card] = _card_history[card][-50:]


def get_velocity_flags(tx: dict) -> list[str]:
    """
    Check in-session transaction velocity for this card.
    Returns velocity alert strings.
    """
    card    = int(tx.get('card1', 0))
    amt     = float(tx.get('TransactionAmt', 0))
    history = _card_history.get(card, [])
    flags   = []

    if not history:
        return flags

    now        = time.time()
    window_sec = VELOCITY_WINDOW_HOURS * 3600
    recent     = [h for h in history if now - h['ts'] <= window_sec]

    # Velocity: too many transactions in short window
    if len(recent) >= VELOCITY_TX_LIMIT:
        flags.append(
            f"⚡ Velocity: {len(recent)} transactions on this card "
            f"in the last {VELOCITY_WINDOW_HOURS}h"
        )

    # Escalation: each transaction bigger than the last
    if len(history) >= 2:
        amounts = [h['amount'] for h in history[-4:]]
        if all(amounts[i] < amounts[i+1] for i in range(len(amounts)-1)):
            flags.append(
                "📈 Escalating amounts: each transaction larger than the last "
                "— card-testing pattern"
            )

    # Unusual spike: this transaction >> card average this session
    if len(history) >= 3:
        avg_hist = sum(h['amount'] for h in history) / len(history)
        if amt > avg_hist * VELOCITY_AMOUNT_FACTOR:
            flags.append(
                f"💥 Amount spike: ${amt:,.2f} is "
                f"{amt/avg_hist:.1f}× this card's session average "
                f"(${avg_hist:,.2f})"
            )

    return flags


# ── Pattern flags ─────────────────────────────────────────────

def get_pattern_flags(tx: dict, ensemble_score: float,
                      provenance: dict | None = None) -> list[str]:
    """
    Detect suspicious behavioral patterns.
    Returns human-readable flag strings.
    """
    flags = []
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    email = str(tx.get('P_emaildomain', ''))
    prod  = str(tx.get('ProductCD', 'W'))

    # ── Time-based ───────────────────────────────────────────
    if hour <= 5:
        flags.append(f"🌙 Off-hours transaction ({hour}:00 AM)")
        
    if hour in [5, 6, 7, 8, 9]:
        flags.append(
            f"⏰ Early morning transaction ({hour}:00) — "
            "highest fraud risk window (5-9AM peak from training data)"
        )

    # ── Email risk ───────────────────────────────────────────
    email_risk = EMAIL_RISK_SCORES.get(email, EMAIL_RISK_DEFAULT)
    if email_risk >= 0.9:
        flags.append(f"📧 Disposable/anonymous email domain ({email})")
    elif email_risk >= 0.2 and amt > 2000:
        flags.append(
            f"📧 Consumer email ({email}) on large transaction "
            f"(${amt:,.2f}) — elevated risk combination"
        )

    # ── Amount signals ────────────────────────────────────────
    if amt > PATTERN_HIGH_AMT:
        flags.append(f"💰 Very high amount (${amt:,.2f})")
    elif amt > PATTERN_MED_AMT:
        flags.append(f"💵 High amount (${amt:,.2f})")

    if amt % 1 == 0 and amt >= PATTERN_ROUND_AMT:
        flags.append(
            f"🎯 Round amount (${amt:,.0f}) — "
            "common in card-testing and manual fraud"
        )
    if 500 <= amt <= 1000:
        flags.append(
            f"💳 Amount ${amt:,.2f} in fraud sweet spot ($500-$1K) — "
            "highest fraud rate range in training data (5.6%)"
        )

    if amt <= 50:
        flags.append(
            f"🔬 Small amount (${amt:,.2f}) — "
            "possible card-testing transaction"
        )

    # ── Product risk ──────────────────────────────────────────
    if prod in HIGH_RISK_PRODUCTS:
        label = PRODUCT_LABELS.get(prod, prod)
        flags.append(
            f"🛍️ High-risk product: {label} — "
            "digital/cash goods are instant and untraceable"
        )

    # ── Model score signals ───────────────────────────────────
    if ensemble_score > PATTERN_EXTREME:
        flags.append("🚨 Extremely high model fraud confidence")
    elif ensemble_score > PATTERN_ELEVATED:
        flags.append("⚠️ Elevated fraud probability signal")

    # ── Feature store signals (if provenance available) ───────
    if provenance:
        cerf = provenance.get('card_email_fraud_rate', 0)
        if cerf > 0.5:
            flags.append(
                f"🔗 This card+email combination has "
                f"{cerf:.0%} historical fraud rate"
            )
        elif not provenance.get('card_seen_before'):
            flags.append("🆕 Card not seen in training data — unknown history")

    return flags


# ── Rule engine ───────────────────────────────────────────────

def get_rule_flags(tx: dict, xgb_score: float) -> list[str]:
    """
    Apply deterministic business rules.
    Returns triggered rule strings.
    """
    flags = []
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    email = str(tx.get('P_emaildomain', ''))
    prod  = str(tx.get('ProductCD', 'W'))

    email_risk = EMAIL_RISK_SCORES.get(email, EMAIL_RISK_DEFAULT)

    # R01 — Large amount warning
    if amt > RULE_AMOUNT_WARN:
        flags.append(
            f"🔴 R01: Amount ${amt:,.2f} exceeds "
            f"${RULE_AMOUNT_WARN:,} warning threshold"
        )

    # R02 — AML reporting threshold
    if amt > RULE_AMOUNT_AML:
        flags.append(
            f"🔴 R02: Amount ${amt:,.2f} exceeds "
            f"${RULE_AMOUNT_AML:,} — AML reporting required"
        )

    # R03 — Night + high amount
    if hour <= 5 and amt > RULE_NIGHT_AMOUNT:
        flags.append(
            f"🔴 R03: Off-hours transaction ({hour}:00) "
            f"+ high amount (${amt:,.2f})"
        )

    # R04 — Risky email + significant amount
    if email_risk >= 0.9 and amt > RULE_EMAIL_AMOUNT:
        flags.append(
            f"🔴 R04: Anonymous/disposable email ({email}) "
            f"+ amount ${amt:,.2f} > ${RULE_EMAIL_AMOUNT}"
        )

    # R05 — Model exceeds CRITICAL threshold
    if xgb_score > RULE_MODEL_THRESH:
        flags.append(
            f"🔴 R05: XGBoost score {xgb_score:.3f} exceeds "
            f"critical threshold {RULE_MODEL_THRESH}"
        )

    # R06 — High-risk product + risky email combo
    if prod in HIGH_RISK_PRODUCTS and email_risk >= 0.2 and amt > 500:
        flags.append(
            f"🔴 R06: High-risk product ({PRODUCT_LABELS.get(prod, prod)}) "
            f"+ risky email domain — high-fraud combination"
        )

    # R07 — Round amount + digital goods
    if prod in HIGH_RISK_PRODUCTS and amt % 1 == 0 and amt >= PATTERN_ROUND_AMT:
        flags.append(
            "🔴 R07: Round amount + digital goods — "
            "classic card-testing pattern"
        )
    if hour in [5, 6, 7, 8, 9] and amt > RULE_NIGHT_AMOUNT:
        flags.append(
            f"🔴 R08: Early morning high-risk window ({hour}:00) "
            f"+ amount ${amt:,.2f} — peak fraud period"
        )

    return flags