# ============================================================
# config.py — Central configuration & constants
# ============================================================

# ── Groq / LLM ──────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 250
LLM_TEMPERATURE = 0.1

# ── Fast-approve shortcut ────────────────────────────────────
FAST_APPROVE_THRESHOLD = 0.08   # scores below this skip full pipeline

# ── Risk level bands ─────────────────────────────────────────
RISK_BANDS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.50),
    "HIGH":     (0.50, 0.75),
    "CRITICAL": (0.75, 1.00),
}

# ── Email domain risk lists ──────────────────────────────────
HIGH_RISK_EMAILS = {
    'anonymous.com', 'guerrillamail.com', 'tempmail.com',
    'throwam.com', 'mailnull.com',
}
MEDIUM_RISK_EMAILS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'}

# ── Rule thresholds ──────────────────────────────────────────
RULE_AMOUNT_WARN   = 5_000
RULE_AMOUNT_AML    = 10_000
RULE_NIGHT_AMOUNT  = 500
RULE_EMAIL_AMOUNT  = 200
RULE_MODEL_THRESH  = 0.9        # R05 triggers when xgb_score > this

# ── Pattern flag thresholds ──────────────────────────────────
PATTERN_HIGH_AMT   = 3_000
PATTERN_MED_AMT    = 1_000
PATTERN_ROUND_AMT  = 100
PATTERN_EXTREME    = 0.8
PATTERN_ELEVATED   = 0.6
