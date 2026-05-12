# ============================================================
# report.py — Structured investigation report generator
# ============================================================

from agent.config import PRODUCT_LABELS

_EMOJI = {'APPROVE': '✅', 'FLAG': '⚠️', 'BLOCK': '🚨'}
_ACTION = {
    'APPROVE': 'Process transaction normally.',
    'FLAG':    'Route to human analyst for manual review.',
    'BLOCK':   'Block transaction. Notify customer and security team.',
}


def build_report(state: dict) -> str:
    """
    Render the full fraud investigation report as a formatted string.
    `state` is the fully-populated FraudState dict.
    """
    tx         = state['transaction']
    decision   = state['decision']
    emoji      = _EMOJI.get(decision, '❓')
    action     = _ACTION.get(decision, 'Unknown action.')
    provenance = state.get('provenance', {})
    
    # ── Format sections ──────────────────────────────────────
    shap_lines = "\n".join(
        f"  {i+1}. {r}"
        for i, r in enumerate(state['shap_reasons'][:5])
    ) or "  N/A"

    pattern_lines = (
        "\n".join(f"  {f}" for f in state['pattern_flags'])
        or "  None"
    )
    rule_lines = (
        "\n".join(f"  {f}" for f in state['rule_flags'])
        or "  None"
    )

    # ── Calibrated score ─────────────────────────────────────
    cal_score = state.get('calibrated_score', state['ensemble_score'])

    # ── Product label ────────────────────────────────────────
    prod       = tx.get('ProductCD', '?')
    prod_label = PRODUCT_LABELS.get(prod, prod)

    # ── Historical context block ─────────────────────────────
    history_lines = ""
    if provenance:
        card_seen  = provenance.get('card_seen_before',  False)
        email_seen = provenance.get('email_seen_before', False)
        combo_seen = provenance.get('combo_seen_before', False)
        cerf       = provenance.get('card_email_fraud_rate', 0)
        crf        = provenance.get('card_fraud_rate', 0)
        ctx        = int(provenance.get('card_tx_count', 0))
        amt_diff   = provenance.get('amt_vs_avg', 0)

        history_lines = f"""
─────────────────────────────────────────────────
HISTORICAL CONTEXT  (feature store)
─────────────────────────────────────────────────
  Card seen in training:      {'Yes' if card_seen  else 'No — unknown card'}
  Card fraud rate:            {crf:.1%}
  Card transaction count:     {ctx:,}
  Email seen in training:     {'Yes' if email_seen else 'No — unknown email'}
  Card+Email combo seen:      {'Yes' if combo_seen else 'No — first time'}
  Card+Email fraud rate:      {cerf:.1%}
  Amount vs card average:     ${ '{:+,.2f}'.format(amt_diff)} \
({'above' if amt_diff > 0 else 'below'} avg)"""

    # ── Counterfactual block ─────────────────────────────────
    counterfactual = state.get('counterfactual', '')
    cf_lines = ""
    if counterfactual and counterfactual not in ('', 'N/A — transaction is low risk'):
        cf_lines = f"""
─────────────────────────────────────────────────
COUNTERFACTUAL ANALYSIS
─────────────────────────────────────────────────
  {counterfactual}"""

    # ── Risk summary ─────────────────────────────────────────
    risk_summary = state.get('risk_summary', '')
    risk_summary_line = (
        f"\n    Primary Risk:  {risk_summary}" if risk_summary else ""
    )

    return f"""
╔══════════════════════════════════════════════════╗
║         FRAUD INVESTIGATION REPORT              ║
╚══════════════════════════════════════════════════╝

{emoji}  DECISION:   {decision}
    Confidence:  {state['confidence']:.1f}%
    Risk Level:  {state['risk_level']}
    Score:       {state['ensemble_score']:.4f}  (threshold {state['threshold']:.4f}){risk_summary_line}

─────────────────────────────────────────────────
TRANSACTION DETAILS
─────────────────────────────────────────────────
  Amount:    ${tx.get('TransactionAmt', 0):.2f}
  Hour:      {tx.get('hour', '?')}:00
  Email:     {tx.get('P_emaildomain', '?')}
  Product:   {prod} — {prod_label}
  Card ID:   {tx.get('card1', '?')}

─────────────────────────────────────────────────
MODEL SCORES
─────────────────────────────────────────────────
  XGBoost raw:    {state['xgb_score']:.4f}
  Calibrated:     {cal_score:.4f}   ← honest probability
  Threshold:      {state['threshold']:.4f}
{history_lines}

─────────────────────────────────────────────────
PATTERN FLAGS  ({len(state['pattern_flags'])} detected)
─────────────────────────────────────────────────
{pattern_lines}

─────────────────────────────────────────────────
RULE FLAGS  ({len(state['rule_flags'])} triggered)
─────────────────────────────────────────────────
{rule_lines}

─────────────────────────────────────────────────
TOP RISK FACTORS (SHAP)
─────────────────────────────────────────────────
{shap_lines}

─────────────────────────────────────────────────
AI REASONING  (Groq — Llama 3.3 70B)
─────────────────────────────────────────────────
  {state['agent_reasoning']}
{cf_lines}

─────────────────────────────────────────────────
RECOMMENDED ACTION
─────────────────────────────────────────────────
  {emoji}  {action}

══════════════════════════════════════════════════
"""