# ============================================================
# rules.py — Business rule engine & behavioral pattern flags
# ============================================================

from config import (
    HIGH_RISK_EMAILS,
    RULE_AMOUNT_WARN, RULE_AMOUNT_AML, RULE_NIGHT_AMOUNT,
    RULE_EMAIL_AMOUNT, RULE_MODEL_THRESH,
    PATTERN_HIGH_AMT, PATTERN_MED_AMT, PATTERN_ROUND_AMT,
    PATTERN_EXTREME, PATTERN_ELEVATED,
)


def get_pattern_flags(tx: dict, ensemble_score: float) -> list[str]:
    """
    Detect suspicious behavioral patterns.
    Returns a list of human-readable flag strings.
    """
    flags = []
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    email = str(tx.get('P_emaildomain', ''))

    if hour <= 5:
        flags.append(f"🌙 Unusual hour ({hour}:00 AM transaction)")

    if email in HIGH_RISK_EMAILS:
        flags.append(f"📧 High-risk email domain ({email})")

    if amt > PATTERN_HIGH_AMT:
        flags.append(f"💰 Very high amount (${amt:,.2f})")
    elif amt > PATTERN_MED_AMT:
        flags.append(f"💵 High amount (${amt:,.2f})")

    if amt % 1 == 0 and amt > PATTERN_ROUND_AMT:
        flags.append("🎯 Suspiciously round amount")

    if ensemble_score > PATTERN_EXTREME:
        flags.append("🚨 Extremely high model confidence in fraud")
    elif ensemble_score > PATTERN_ELEVATED:
        flags.append("⚠️ Elevated fraud probability signal")

    return flags


def get_rule_flags(tx: dict, xgb_score: float) -> list[str]:
    """
    Apply deterministic business rules.
    Returns a list of triggered rule strings.
    """
    flags = []
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    email = str(tx.get('P_emaildomain', ''))

    if amt > RULE_AMOUNT_WARN:
        flags.append(f"🔴 Rule R01: Amount exceeds ${RULE_AMOUNT_WARN:,} limit")

    if amt > RULE_AMOUNT_AML:
        flags.append(f"🔴 Rule R02: Amount exceeds ${RULE_AMOUNT_AML:,} — AML reporting required")

    if hour <= 5 and amt > RULE_NIGHT_AMOUNT:
        flags.append("🔴 Rule R03: Night transaction + high amount")

    if email in HIGH_RISK_EMAILS and amt > RULE_EMAIL_AMOUNT:
        flags.append("🔴 Rule R04: Anonymous email + significant amount")

    if xgb_score > RULE_MODEL_THRESH:
        flags.append(f"🔴 Rule R05: Model score exceeds {RULE_MODEL_THRESH*100:.0f}% fraud threshold")

    return flags
