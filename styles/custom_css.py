CUSTOM_CSS = """
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Decision cards */
    .approve-card {
        background: linear-gradient(135deg, #0d4f2e, #1a7a46);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .flag-card {
        background: linear-gradient(135deg, #4f3a0d, #7a5a1a);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .block-card {
        background: linear-gradient(135deg, #4f0d0d, #7a1a1a);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .step-card {
        background: #1e2130;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 10px 15px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .flag-item {
        background: #2d1f0e;
        border-left: 3px solid #f59e0b;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    .rule-item {
        background: #2d0e0e;
        border-left: 3px solid #ef4444;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    h1 { color: #e2e8f0 !important; }
    h2 { color: #cbd5e1 !important; }
    h3 { color: #94a3b8 !important; }
</style>
"""

# Decision color/emoji/css mappings — single source of truth
DECISION_CSS = {
    'APPROVE': 'approve-card',
    'FLAG':    'flag-card',
    'BLOCK':   'block-card',
}

DECISION_EMOJI = {
    'APPROVE': '✅',
    'FLAG':    '⚠️',
    'BLOCK':   '🚨',
}

DECISION_COLOR = {
    'APPROVE': '#22c55e',
    'FLAG':    '#f59e0b',
    'BLOCK':   '#ef4444',
}

COLORS_MAP = {
    'APPROVE': '#22c55e',
    'FLAG':    '#f59e0b',
    'BLOCK':   '#ef4444',
}
