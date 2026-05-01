# ============================================================
# llm_client.py — Groq LLM wrapper
# ============================================================

import os
import logging
from dotenv import load_dotenv
from groq import Groq

from config import GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

load_dotenv()
logger = logging.getLogger(__name__)

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_SYSTEM_PROMPT = (
    "You are a senior fraud analyst at a major bank. "
    "Be concise, precise, and professional. "
    "Always respond in the exact format requested."
)


def call_groq(prompt: str) -> str:
    """
    Send a prompt to Groq and return the raw text response.
    Returns 'LLM_ERROR: <message>' on failure so callers can detect it.
    """
    try:
        response = _client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": prompt},
            ],
            max_tokens  = LLM_MAX_TOKENS,
            temperature = LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return f"LLM_ERROR: {e}"


def build_decision_prompt(tx: dict, state: dict) -> str:
    """Build the fraud-analysis prompt from transaction + state data."""
    return f"""Analyze this transaction for fraud.

TRANSACTION:
  Amount:    ${tx.get('TransactionAmt', 0):.2f}
  Hour:      {tx.get('hour', '?')}:00
  Email:     {tx.get('P_emaildomain', '?')}
  Product:   {tx.get('ProductCD', '?')}

RISK SCORES:
  XGBoost:   {state['xgb_score']:.3f}
  Ensemble:  {state['ensemble_score']:.3f}
  Risk Level:{state['risk_level']}
  Threshold: {state['threshold']:.3f}

PATTERN FLAGS:
{chr(10).join(state['pattern_flags']) or 'None'}

RULE FLAGS:
{chr(10).join(state['rule_flags']) or 'None'}

TOP RISK FACTORS:
{chr(10).join(state['shap_reasons'][:3]) or 'N/A'}

Respond EXACTLY like this:
DECISION: [APPROVE or FLAG or BLOCK]
CONFIDENCE: [0-100]%
REASONING: [1 sentence professional explanation]"""


def parse_llm_response(raw: str, fallback_score: float,
                       threshold: float, n_pattern_flags: int) -> dict:
    """
    Parse the structured LLM response into decision fields.
    Falls back to rule-based logic if LLM fails or returns unexpected output.
    """
    decision   = 'FLAG'
    confidence = 50.0
    reasoning  = 'Manual review recommended.'

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

    # Fallback if LLM failed
    if 'LLM_ERROR' in raw or reasoning == 'Manual review recommended.':
        logger.warning("LLM response unusable — applying rule-based fallback")
        decision, confidence, reasoning = _rule_based_fallback(
            fallback_score, threshold, n_pattern_flags
        )

    return {'decision': decision, 'confidence': confidence, 'reasoning': reasoning}


def _rule_based_fallback(score: float, threshold: float,
                         n_flags: int) -> tuple[str, float, str]:
    if score >= threshold:
        return ('BLOCK', score * 100,
                f"Score {score:.3f} exceeds threshold {threshold:.3f}. {n_flags} pattern flags detected.")
    if score >= threshold * 0.7:
        return ('FLAG', 65.0,
                f"Score {score:.3f} near threshold {threshold:.3f}. {n_flags} pattern flags detected.")
    return ('APPROVE', (1 - score) * 100,
            f"Score {score:.3f} well below threshold {threshold:.3f}.")
