# ============================================================
# FRAUD DETECTION AGENT — INDUSTRY LEVEL
# agent.py
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from dotenv import load_dotenv
import os
warnings.filterwarnings('ignore')

from typing          import TypedDict, List
from langgraph.graph import StateGraph, END
from groq            import Groq
load_dotenv()

# ============================================================
# LOAD MODELS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))

def _load(name):
    return joblib.load(os.path.join(BASE, 'models', name))

xgb_model      = _load('xgb_model.pkl')
shap_explainer = _load('shap_explainer.pkl')
threshold_data = _load('threshold.pkl')
feature_cols   = _load('feature_cols.pkl')
label_encoders = _load('label_encoders.pkl')

try:
    iso_model      = _load('iso_model.pkl')
    iso_norm       = _load('iso_norm_params.pkl')
    ens_weights    = _load('ensemble_weights.pkl')
    USE_ENSEMBLE   = True
except:
    USE_ENSEMBLE   = False

THRESHOLD = threshold_data['threshold']

# ============================================================
# GROQ SETUP
# ============================================================

GROQ_API_KEY =os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL   = "llama-3.3-70b-versatile"

# ============================================================
# STATE
# ============================================================

class FraudState(TypedDict):
    transaction:     dict
    features:        np.ndarray
    xgb_score:       float
    iso_score:       float
    ensemble_score:  float
    risk_level:      str
    pattern_flags:   List[str]
    rule_flags:      List[str]
    shap_reasons:    List[str]
    shap_dict:       dict
    final_decision:  str
    confidence:      float
    agent_reasoning: str
    report:          str

# ============================================================
# HELPERS
# ============================================================

# Domain risk scores
HIGH_RISK_EMAILS   = {'anonymous.com', 'guerrillamail.com', 'tempmail.com',
                      'throwam.com', 'mailnull.com'}
MEDIUM_RISK_EMAILS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'}

def build_features(tx: dict) -> np.ndarray:
    """Build full feature vector from raw transaction input."""

    amt  = float(tx.get('TransactionAmt', 0))
    hour = int(tx.get('hour', 12))
    card = int(tx.get('card1', 99999))
    email= str(tx.get('P_emaildomain', 'gmail.com'))
    prod = str(tx.get('ProductCD', 'W'))

    # Simulate realistic behavioral features
    # In production these come from feature store / DB
    email_fraud  = 0.75 if email in HIGH_RISK_EMAILS else \
                   0.15 if email in MEDIUM_RISK_EMAILS else 0.05
    card_fraud   = min(0.9, max(0.01, (amt / 5000) * 0.6))
    amt_deviation= amt / max(80, amt * 0.3)

    row = {
        # Core transaction
        'TransactionAmt':          amt,
        'ProductCD':               prod,
        'P_emaildomain':           email,
        'R_emaildomain':           email,

        # Time features
        'hour':                    hour,
        'day_of_week':             3,
        'is_night':                1 if hour <= 5 else 0,
        'is_weekend':              0,
        'is_business_hour':        1 if 9 <= hour <= 17 else 0,

        # Amount features
        'amount_log':              np.log1p(amt),
        'amount_dec':              amt % 1,
        'is_round_amt':            1 if amt % 1 == 0 else 0,

        # Card behavioral features
        'card1':                   card,
        'card2':                   111,
        'card1_fraud_rate':        card_fraud,
        'card1_tx_count':          max(1, int(50 - amt / 200)),
        'card1_avg_amount':        max(10, amt * 0.4),
        'card1_std_amount':        max(5,  amt * 0.2),
        'card1_max_amount':        amt * 1.5,
        'card1_amount_deviation':  amt_deviation,
        'card1_unique_cards':      1,
        'card1_unique_emails':     2 if amt > 500 else 1,

        # Email features
        'email_fraud_rate':        email_fraud,
        'p_email_high_risk':       1 if email in HIGH_RISK_EMAILS else 0,
        'same_email':              1,

        # Device features
        'device_fraud_rate':       0.4 if amt > 1000 else 0.05,
        'DeviceInfo':              'Windows',

        # Address
        'addr1':                   299,
        'addr2':                   87,

        # Graph / network features
        'card_email_count':        1 if amt > 800 else 5,
        'card_device_count':       2,
        'card_addr_count':         3,
        'card_email_fraud_rate':   card_fraud * email_fraud,

        # Velocity
        'card_amount_rank':        min(0.99, amt / 5000),
        'card_tx_sequence':        max(0, int(10 - amt / 500)),
        'time_since_last_tx':      3600 if amt > 500 else 86400,
        'is_first_tx':             0,
    }

    df = pd.DataFrame([row])

    # Add any missing features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -999

    # Label encode categoricals
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                df[col] = -999

    return df[feature_cols].fillna(-999).values


def normalize_iso(raw: float) -> float:
    mn = iso_norm['min']
    mx = iso_norm['max']
    return float(1 - (raw - mn) / (mx - mn + 1e-8))


def get_risk_level(score: float) -> str:
    if   score < 0.30: return "LOW"
    elif score < 0.50: return "MEDIUM"
    elif score < 0.75: return "HIGH"
    else:              return "CRITICAL"


def get_shap(features: np.ndarray, n: int = 5):
    """Return SHAP explanations as dict and human text."""
    try:
        vals    = shap_explainer.shap_values(features)[0]
        impacts = sorted(zip(feature_cols, vals),
                         key=lambda x: abs(x[1]), reverse=True)[:n]
        d       = {f: round(float(v), 4) for f, v in impacts}
        reasons = [
            f"{f} {'↑ increases' if v > 0 else '↓ decreases'} "
            f"fraud risk (impact: {abs(v):.3f})"
            for f, v in impacts
        ]
        return d, reasons
    except Exception as e:
        return {}, [f"Explanation unavailable: {e}"]


def call_groq(prompt: str) -> str:
    """Call Groq LLM safely."""
    try:
        r = groq_client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [
                {"role": "system",
                 "content": ("You are a senior fraud analyst at a major bank. "
                             "Be concise, precise, and professional. "
                             "Always respond in the exact format requested.")},
                {"role": "user", "content": prompt}
            ],
            max_tokens  = 250,
            temperature = 0.1,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM_ERROR: {e}"


# ============================================================
# AGENT NODES
# ============================================================

def node_score(state: FraudState) -> FraudState:
    """Score transaction with XGBoost + optional ensemble."""
    f = state['features']

    xgb_score = float(xgb_model.predict_proba(f)[0][1])

    if USE_ENSEMBLE:
        iso_raw   = float(iso_model.score_samples(f)[0])
        iso_score = normalize_iso(iso_raw)
        w         = ens_weights
        ensemble  = (xgb_score * w.get('xgboost', 0.75) +
                     iso_score  * w.get('isolation_forest', 0.25))
    else:
        iso_score = 0.0
        ensemble  = xgb_score

    return {
        **state,
        'xgb_score':      xgb_score,
        'iso_score':      iso_score,
        'ensemble_score': ensemble,
        'risk_level':     get_risk_level(ensemble),
    }


def node_patterns(state: FraudState) -> FraudState:
    """Detect suspicious behavioral patterns."""
    tx    = state['transaction']
    flags = []
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))
    email = str(tx.get('P_emaildomain', ''))

    if hour <= 5:
        flags.append(f"🌙 Unusual hour ({hour}:00 AM transaction)")
    if email in HIGH_RISK_EMAILS:
        flags.append(f"📧 High-risk email domain ({email})")
    if amt > 3000:
        flags.append(f"💰 Very high amount (${amt:,.2f})")
    elif amt > 1000:
        flags.append(f"💵 High amount (${amt:,.2f})")
    if amt % 1 == 0 and amt > 100:
        flags.append("🎯 Suspiciously round amount")
    if state['ensemble_score'] > 0.8:
        flags.append("🚨 Extremely high model confidence in fraud")
    if state['ensemble_score'] > 0.6:
        flags.append("⚠️ Elevated fraud probability signal")

    return {**state, 'pattern_flags': flags}


def node_rules(state: FraudState) -> FraudState:
    """Apply business rule engine."""
    tx    = state['transaction']
    flags = []
    amt   = float(tx.get('TransactionAmt', 0))
    hour  = int(tx.get('hour', 12))

    if amt > 5000:
        flags.append("🔴 Rule R01: Amount exceeds $5,000 limit")
    if amt > 10000:
        flags.append("🔴 Rule R02: Amount exceeds $10,000 — AML reporting required")
    if hour <= 5 and amt > 500:
        flags.append("🔴 Rule R03: Night transaction + high amount")
    if str(tx.get('P_emaildomain','')) in HIGH_RISK_EMAILS and amt > 200:
        flags.append("🔴 Rule R04: Anonymous email + significant amount")
    if state['xgb_score'] > 0.9:
        flags.append("🔴 Rule R05: Model score exceeds 90% fraud threshold")

    return {**state, 'rule_flags': flags}


def node_explain(state: FraudState) -> FraudState:
    """Generate SHAP explanations."""
    shap_dict, reasons = get_shap(state['features'], n=5)
    return {**state, 'shap_dict': shap_dict, 'shap_reasons': reasons}


def node_decide(state: FraudState) -> FraudState:
    """Use Groq LLM to reason and make final decision."""
    tx = state['transaction']

    prompt = f"""Analyze this transaction for fraud.

TRANSACTION:
  Amount:    ${tx.get('TransactionAmt', 0):.2f}
  Hour:      {tx.get('hour', '?')}:00
  Email:     {tx.get('P_emaildomain', '?')}
  Product:   {tx.get('ProductCD', '?')}

RISK SCORES:
  XGBoost:   {state['xgb_score']:.3f}
  Ensemble:  {state['ensemble_score']:.3f}
  Risk Level:{state['risk_level']}
  Threshold: {THRESHOLD:.3f}

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

    raw        = call_groq(prompt)
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
                    line.replace('CONFIDENCE:', '')
                        .replace('%', '').strip())
            except:
                confidence = state['ensemble_score'] * 100
        elif line.startswith('REASONING:'):
            reasoning = line.replace('REASONING:', '').strip()

    # Fallback if LLM failed
    if 'LLM_ERROR' in raw or reasoning == 'Manual review recommended.':
        s = state['ensemble_score']
        if   s >= THRESHOLD:      decision, confidence = 'BLOCK',   s * 100
        elif s >= THRESHOLD * 0.7:decision, confidence = 'FLAG',    65.0
        else:                     decision, confidence = 'APPROVE', (1-s)*100
        reasoning = (f"Score {s:.3f} vs threshold {THRESHOLD:.3f}. "
                     f"{len(state['pattern_flags'])} pattern flags detected.")

    return {
        **state,
        'final_decision':  decision,
        'confidence':      confidence,
        'agent_reasoning': reasoning,
    }


def node_report(state: FraudState) -> FraudState:
    """Generate structured investigation report."""
    tx    = state['transaction']
    emoji = {'APPROVE': '✅', 'FLAG': '⚠️', 'BLOCK': '🚨'}.get(
             state['final_decision'], '❓')
    action = {
        'APPROVE': 'Process transaction normally.',
        'FLAG':    'Route to human analyst for manual review.',
        'BLOCK':   'Block transaction. Notify customer and security team.',
    }.get(state['final_decision'], 'Unknown')

    report = f"""
╔══════════════════════════════════════════════════╗
║         FRAUD INVESTIGATION REPORT              ║
╚══════════════════════════════════════════════════╝

{emoji}  DECISION:   {state['final_decision']}
    Confidence:  {state['confidence']:.1f}%
    Risk Level:  {state['risk_level']}
    Score:       {state['ensemble_score']:.4f}  (threshold {THRESHOLD:.4f})

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
{chr(10).join(f'  {f}' for f in state['pattern_flags']) or '  None'}

─────────────────────────────────────────────────
RULE FLAGS  ({len(state['rule_flags'])} triggered)
─────────────────────────────────────────────────
{chr(10).join(f'  {f}' for f in state['rule_flags']) or '  None'}

─────────────────────────────────────────────────
TOP RISK FACTORS (SHAP)
─────────────────────────────────────────────────
{chr(10).join(f'  {i+1}. {r}' for i,r in enumerate(state['shap_reasons'][:5]))}

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
    return {**state, 'report': report}


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def _route(state: FraudState) -> str:
    return "fast_approve" if state['ensemble_score'] < 0.08 else "patterns"


def node_fast_approve(state: FraudState) -> FraudState:
    return {
        **state,
        'pattern_flags':   [],
        'rule_flags':      [],
        'shap_dict':       {},
        'shap_reasons':    ['Score well below threshold'],
        'final_decision':  'APPROVE',
        'confidence':      99.0,
        'agent_reasoning': f'Auto-approved. Very low risk score {state["ensemble_score"]:.4f}.',
        'report':          f'AUTO-APPROVED | Score: {state["ensemble_score"]:.4f} | Risk: LOW',
    }


_wf = StateGraph(FraudState)
_wf.add_node("score",        node_score)
_wf.add_node("patterns",     node_patterns)
_wf.add_node("rules",        node_rules)
_wf.add_node("explain",      node_explain)
_wf.add_node("decide",       node_decide)
_wf.add_node("generate_report", node_report)
_wf.add_node("fast_approve", node_fast_approve)

_wf.set_entry_point("score")
_wf.add_conditional_edges("score", _route,
    {"fast_approve": "fast_approve", "patterns": "patterns"})
_wf.add_edge("patterns",     "rules")
_wf.add_edge("rules",        "explain")
_wf.add_edge("explain",      "decide")
_wf.add_edge("decide",       "generate_report")
_wf.add_edge("generate_report", END)
_wf.add_edge("fast_approve", END)

_agent = _wf.compile()


# ============================================================
# PUBLIC API
# ============================================================

def run_agent(user_input: dict) -> dict:
    """
    Main entry point.
    user_input = {'TransactionAmt': 150.0, 'hour': 14, ...}
    Returns dict with decision, score, explanation, report.
    """
    features = build_features(user_input)

    result = _agent.invoke(FraudState(
        transaction      = user_input,
        features         = features,
        xgb_score        = 0.0,
        iso_score        = 0.0,
        ensemble_score   = 0.0,
        risk_level       = 'LOW',
        pattern_flags    = [],
        rule_flags       = [],
        shap_dict        = {},
        shap_reasons     = [],
        final_decision   = '',
        confidence       = 0.0,
        agent_reasoning  = '',
        report           = '',
    ))

    return {
        'decision':       result['final_decision'],
        'confidence':     round(result['confidence'], 1),
        'score':          round(result['ensemble_score'], 4),
        'xgb_score':      round(result['xgb_score'], 4),
        'risk_level':     result['risk_level'],
        'pattern_flags':  result['pattern_flags'],
        'rule_flags':     result['rule_flags'],
        'shap_reasons':   result['shap_reasons'],
        'shap_dict':      result['shap_dict'],
        'agent_reasoning':result['agent_reasoning'],
        'report':         result['report'],
    }
