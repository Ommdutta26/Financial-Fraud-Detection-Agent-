# ============================================================
# llm_client.py — Groq LLM wrapper with rich reasoning prompt
# ============================================================

import os
import logging
from dotenv import load_dotenv
from groq import Groq

from agent.config import (
    GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    PRODUCT_LABELS, EMAIL_RISK_SCORES, EMAIL_RISK_DEFAULT,
)

load_dotenv()
logger = logging.getLogger(__name__)

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_SYSTEM_PROMPT = """You are a senior fraud analyst at a tier-1 bank with 15 years experience.
You have access to ML model scores, SHAP explainability, behavioral patterns, and business rules.
Your job is to synthesise ALL signals into a clear, professional fraud decision.

DECISION FRAMEWORK:
- APPROVE: Low risk. Score well below threshold. No serious flags. Normal behavioral profile.
- FLAG: Medium risk OR conflicting signals. Human review required. Err on side of caution.
- BLOCK: High risk. Score above threshold OR critical rule triggered OR multiple serious flags.

Be concise, precise, and specific. Reference actual values from the data provided.
Never give generic responses. Every word must be grounded in the specific transaction."""


def call_groq(prompt: str) -> str:
    try:
        response = _client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = LLM_MAX_TOKENS,
            temperature = LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return f"LLM_ERROR: {e}"


def build_decision_prompt(tx: dict, state: dict,
                          provenance: dict | None = None) -> str:
    """
    Build a rich, data-grounded prompt that gives the LLM everything
    it needs to reason properly — not just scores but context,
    history signals, and counterfactuals.
    """
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    email = str(tx.get('P_emaildomain', '?'))
    prod  = str(tx.get('ProductCD', '?'))
    card  = tx.get('card1', '?')

    prod_label  = PRODUCT_LABELS.get(prod, prod)
    email_risk  = EMAIL_RISK_SCORES.get(email, EMAIL_RISK_DEFAULT)
    time_label  = "off-hours (night)" if hour <= 5 else \
                  "business hours" if 9 <= hour <= 17 else "evening"

    # ── Historical context from provenance ───────────────────
    history_block = ""
    if provenance:
        card_seen  = provenance.get('card_seen_before', False)
        email_seen = provenance.get('email_seen_before', False)
        combo_seen = provenance.get('combo_seen_before', False)
        cerf       = provenance.get('card_email_fraud_rate', 0)
        crf        = provenance.get('card_fraud_rate', 0)
        ctx        = provenance.get('card_tx_count', 0)
        amt_diff   = provenance.get('amt_vs_avg', 0)

        history_block = f"""
HISTORICAL CONTEXT (from feature store — real training data):
  Card {card} seen before:          {'Yes' if card_seen else 'No — unknown card'}
  Card historical fraud rate:        {crf:.1%}
  Card transaction count (training): {int(ctx)}
  Email {email} seen before:         {'Yes' if email_seen else 'No — unknown email'}
  Card+Email combo seen before:      {'Yes' if combo_seen else 'No — first time'}
  Card+Email historical fraud rate:  {cerf:.1%}
  This amount vs card average:       {'${:+,.2f}'.format(amt_diff)} ({'above' if amt_diff > 0 else 'below'} average)"""

    # ── Calibrated confidence ─────────────────────────────────
    cal_score = state.get('calibrated_score', state['ensemble_score'])

    # ── Pattern and rule summary ──────────────────────────────
    pattern_block = "\n".join(
        f"  • {f}" for f in state['pattern_flags']
    ) or "  None detected"

    rule_block = "\n".join(
        f"  • {r}" for r in state['rule_flags']
    ) or "  None triggered"

    shap_block = "\n".join(
        f"  • {r}" for r in state['shap_reasons'][:5]
    ) or "  Not available"

    cf_block = state.get('counterfactual', "Not available")

    return f"""Analyze this transaction and provide your fraud decision.

══ TRANSACTION ══════════════════════════════════════════
  Amount:      ${amt:,.2f}
  Time:        {hour}:00 ({time_label})
  Product:     {prod} — {prod_label}
  Email:       {email}  [risk score: {email_risk:.2f}/1.0]
  Card ID:     {card}

══ MODEL SCORES ═════════════════════════════════════════
  XGBoost raw:       {state['xgb_score']:.4f}
  Calibrated score:  {cal_score:.4f}   ← honest probability
  Ensemble score:    {state['ensemble_score']:.4f}
  Decision threshold:{state['threshold']:.4f}
  Risk level:        {state['risk_level']}
  Score vs threshold:{'+' if state['ensemble_score'] >= state['threshold'] else ''}{state['ensemble_score'] - state['threshold']:.4f}
{history_block}

══ SHAP — WHY THE MODEL THINKS THIS ════════════════════
{shap_block}

══ BEHAVIORAL PATTERNS ══════════════════════════════════
{pattern_block}

══ BUSINESS RULES TRIGGERED ═════════════════════════════
{rule_block}

══ COUNTERFACTUAL ═══════════════════════════════════════
  {cf_block}

══ YOUR TASK ════════════════════════════════════════════
Step 1 — Assess the ML signal: Is the score above threshold? Is calibrated probability high?
Step 2 — Assess the behavioral signals: Do patterns/rules confirm or contradict the score?
Step 3 — Assess historical context: Does the card/email history raise or lower concern?
Step 4 — Synthesise: What is the single clearest reason for your decision?

Respond EXACTLY in this format (no extra text):
DECISION: [APPROVE or FLAG or BLOCK]
CONFIDENCE: [0-100]
REASONING: [2-3 sentences. Be specific — mention actual values, not generalities.]
RISK_SUMMARY: [One sentence: the single biggest risk signal in this transaction.]"""


def parse_llm_response(raw: str, fallback_score: float,
                       threshold: float,
                       n_pattern_flags: int) -> dict:
    """
    Parse structured LLM response. Falls back to rule-based logic
    if LLM fails or returns unexpected output.
    """
    decision     = 'FLAG'
    confidence   = 50.0
    reasoning    = 'Manual review recommended.'
    risk_summary = ''

    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith('DECISION:'):
            d = line.replace('DECISION:', '').strip().upper()
            decision = ('BLOCK'   if 'BLOCK'   in d else
                        'APPROVE' if 'APPROVE' in d else 'FLAG')

        elif line.startswith('CONFIDENCE:'):
            try:
                confidence = float(
                    line.replace('CONFIDENCE:', '').replace('%', '').strip()
                )
            except ValueError:
                confidence = fallback_score * 100

        elif line.startswith('REASONING:'):
            reasoning = line.replace('REASONING:', '').strip()

        elif line.startswith('RISK_SUMMARY:'):
            risk_summary = line.replace('RISK_SUMMARY:', '').strip()

    # Fallback if LLM failed
    if 'LLM_ERROR' in raw or reasoning == 'Manual review recommended.':
        logger.warning("LLM response unusable — applying rule-based fallback")
        decision, confidence, reasoning = _rule_based_fallback(
            fallback_score, threshold, n_pattern_flags
        )
        risk_summary = f"Score {fallback_score:.3f} vs threshold {threshold:.3f}."

    return {
        'decision':     decision,
        'confidence':   confidence,
        'reasoning':    reasoning,
        'risk_summary': risk_summary,
    }


def _rule_based_fallback(score: float, threshold: float,
                         n_flags: int) -> tuple[str, float, str]:
    if score >= threshold:
        return (
            'BLOCK',
            min(99.0, score * 100),
            f"Score {score:.3f} exceeds threshold {threshold:.3f}. "
            f"{n_flags} behavioral flags detected. Auto-blocked by rule engine."
        )
    if score >= threshold * 0.6:
        return (
            'FLAG',
            65.0,
            f"Score {score:.3f} is {score/threshold:.0%} of threshold {threshold:.3f}. "
            f"{n_flags} flags detected. Routed for manual review."
        )
    return (
        'APPROVE',
        round((1 - score) * 100, 1),
        f"Score {score:.3f} well below threshold {threshold:.3f}. Low risk."
    )