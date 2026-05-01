"""
history.py — Session transaction history panel.

Renders a score-trend line chart, decision-distribution donut,
and a summary table. Also exposes helpers for managing session state.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from components.charts import build_score_trend, build_decision_pie
from styles.custom_css import COLORS_MAP


# ── Session-state helpers ─────────────────────────────────────────────────────

def init_history() -> None:
    """Initialise session history list if not already present."""
    if 'history' not in st.session_state:
        st.session_state.history = []


def append_history(result: dict, amount: float) -> None:
    """Append one transaction record to session history."""
    st.session_state.history.append({
        'time':     datetime.now().strftime('%H:%M:%S'),
        'amount':   amount,
        'decision': result['decision'],
        'score':    result['score'],
        'flags':    len(result['pattern_flags']) + len(result['rule_flags']),
    })


# ── History panel ─────────────────────────────────────────────────────────────

def render_history() -> None:
    """Render the full session history section (charts + table + clear button)."""
    if not st.session_state.history:
        return

    st.markdown("---")
    st.markdown("### 📈 Session Transaction History")

    hist_df = pd.DataFrame(st.session_state.history)

    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        fig_trend = build_score_trend(hist_df, COLORS_MAP)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_h2:
        fig_pie = build_decision_pie(hist_df, COLORS_MAP)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.dataframe(
        hist_df.rename(columns={
            'time':     'Time',
            'amount':   'Amount ($)',
            'decision': 'Decision',
            'score':    'Score',
            'flags':    'Flags',
        }),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
