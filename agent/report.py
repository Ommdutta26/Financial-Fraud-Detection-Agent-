# ============================================================
# report.py — Structured investigation report generator
# ============================================================

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
    tx       = state['transaction']
    decision = state['final_decision']
    emoji    = _EMOJI.get(decision, '❓')
    action   = _ACTION.get(decision, 'Unknown action.')

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

    return f"""
╔══════════════════════════════════════════════════╗
║         FRAUD INVESTIGATION REPORT              ║
╚══════════════════════════════════════════════════╝

{emoji}  DECISION:   {decision}
    Confidence:  {state['confidence']:.1f}%
    Risk Level:  {state['risk_level']}
    Score:       {state['ensemble_score']:.4f}  (threshold {state['threshold']:.4f})

─────────────────────────────────────────────────
TRANSACTION DETAILS
─────────────────────────────────────────────────
  Amount:    ${tx.get('TransactionAmt', 0):.2f}
  Hour:      {tx.get('hour', '?')}:00
  Email:     {tx.get('P_emaildomain', '?')}
  Product:   {tx.get('ProductCD', '?')}

─────────────────────────────────────────────────
MODEL SCORES
─────────────────────────────────────────────────
  XGBoost:   {state['xgb_score']:.4f}
  Ensemble:  {state['ensemble_score']:.4f}

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

─────────────────────────────────────────────────
RECOMMENDED ACTION
─────────────────────────────────────────────────
  {emoji}  {action}

══════════════════════════════════════════════════
"""
