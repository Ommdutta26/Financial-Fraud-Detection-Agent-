# ============================================================
# config.py — Central configuration & constants
# ============================================================

# ── Groq / LLM ──────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS   = 250
LLM_TEMPERATURE  = 0.1

# ── Decision thresholds ──────────────────────────────────────
FAST_APPROVE_THRESHOLD  = 0.03   # skip pipeline only if very confident
DECISION_BLOCK_THRESHOLD = 0.70
DECISION_FLAG_THRESHOLD  = 0.30

# ── Risk level bands ─────────────────────────────────────────
RISK_BANDS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.50),
    "HIGH":     (0.50, 0.75),
    "CRITICAL": (0.75, 1.00),
}

# ── Email domain risk scores ─────────────────────────────────
EMAIL_RISK_SCORES = {
    'anonymous.com':     1.0,
    'guerrillamail.com': 1.0,
    'tempmail.com':      1.0,
    'throwam.com':       0.9,
    'mailnull.com':      0.9,
    'yopmail.com':       0.9,
    'gmail.com':         0.2,
    'yahoo.com':         0.25,
    'hotmail.com':       0.25,
    'outlook.com':       0.15,
    'company.com':       0.05,
}
EMAIL_RISK_DEFAULT = 0.3   # unknown domain fallback

# ── Product code risk ────────────────────────────────────────
HIGH_RISK_PRODUCTS = {'W', 'C'}   # Digital goods + Cash — untraceable

# ── Rule thresholds ──────────────────────────────────────────
RULE_AMOUNT_WARN   = 5_000
RULE_AMOUNT_AML    = 10_000
RULE_NIGHT_AMOUNT  = 500
RULE_EMAIL_AMOUNT  = 200
RULE_MODEL_THRESH  = 0.75   # aligned with CRITICAL band

# ── Pattern flag thresholds ──────────────────────────────────
PATTERN_HIGH_AMT   = 3_000
PATTERN_MED_AMT    = 1_000
PATTERN_ROUND_AMT  = 100
PATTERN_EXTREME    = 0.8
PATTERN_ELEVATED   = 0.6