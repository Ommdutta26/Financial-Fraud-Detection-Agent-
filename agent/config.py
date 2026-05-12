# ============================================================
# config.py — Central configuration & constants
# ============================================================

# ── Groq / LLM ──────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS   = 900
LLM_TEMPERATURE  = 0.1

# ── Decision thresholds ──────────────────────────────────────
FAST_APPROVE_THRESHOLD   = 0.03
DECISION_BLOCK_THRESHOLD = 0.70
DECISION_FLAG_THRESHOLD  = 0.30

# ── Risk level bands ─────────────────────────────────────────
RISK_BANDS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.50),
    "HIGH":     (0.50, 0.75),
    "CRITICAL": (0.75, 1.00),
}

# ── Email domain risk scores — data-driven from IEEE-CIS ─────
EMAIL_RISK_SCORES = {
    'protonmail.com':   1.0,   # 40.8% fraud rate
    'mail.com':         0.8,   # 19.0% fraud rate
    'outlook.es':       0.7,   # 13.0% fraud rate
    'aim.com':          0.6,   # 12.7% fraud rate
    'outlook.com':      0.5,   #  9.5% fraud rate
    'hotmail.es':       0.4,   #  6.6% fraud rate
    'live.com.mx':      0.35,  #  5.5% fraud rate
    'hotmail.com':      0.3,   #  5.3% fraud rate
    'gmail.com':        0.25,  #  4.4% fraud rate
    'icloud.com':       0.2,   #  3.1% fraud rate
    'yahoo.com':        0.15,  #  2.3% fraud rate
    'anonymous.com':    0.15,  #  2.3% fraud rate
    'aol.com':          0.1,   #  2.2% fraud rate
    'att.net':          0.05,  #  0.7% fraud rate
    'verizon.net':      0.05,  #  0.8% fraud rate
    'sbcglobal.net':    0.03,  #  0.4% fraud rate
}
EMAIL_RISK_DEFAULT = 0.25      # unknown domain fallback

# Derived from scores — single source of truth
HIGH_RISK_EMAILS   = {k for k, v in EMAIL_RISK_SCORES.items() if v >= 0.9}
MEDIUM_RISK_EMAILS = {k for k, v in EMAIL_RISK_SCORES.items() if 0.15 <= v < 0.9}

# ── Product code risk ────────────────────────────────────────
HIGH_RISK_PRODUCTS = {'W', 'C'}
PRODUCT_LABELS = {
    'W': 'Digital Goods',
    'H': 'Hotel/Travel',
    'C': 'Cash',
    'S': 'Services',
    'R': 'Retail',
}

# ── Rule thresholds ──────────────────────────────────────────
RULE_AMOUNT_WARN   = 5_000
RULE_AMOUNT_AML    = 10_000
RULE_NIGHT_AMOUNT  = 500
RULE_EMAIL_AMOUNT  = 200
RULE_MODEL_THRESH  = 0.75

# ── Pattern flag thresholds ──────────────────────────────────
PATTERN_HIGH_AMT   = 3_000
PATTERN_MED_AMT    = 1_000
PATTERN_ROUND_AMT  = 100
PATTERN_EXTREME    = 0.80
PATTERN_ELEVATED   = 0.60

# ── Velocity — card memory ───────────────────────────────────
VELOCITY_WINDOW_HOURS  = 1
VELOCITY_TX_LIMIT      = 3
VELOCITY_AMOUNT_FACTOR = 3.0