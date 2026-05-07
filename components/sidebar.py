"""
sidebar.py — Sidebar input controls + model info panel.
Returns the user inputs and whether the Analyze button was clicked.
"""

import streamlit as st
import json
import os


PRODUCT_LABELS = {
    'W': 'W — Digital Goods',
    'H': 'H — Hotel/Travel',
    'C': 'C — Cash',
    'S': 'S — Services',
    'R': 'R — Retail',
}

DAY_LABELS = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday',
}


def render_sidebar() -> tuple[dict, bool]:
    """
    Renders the sidebar UI and returns (user_input dict, analyze_clicked bool).
    """
    with st.sidebar:
        st.markdown("## 🔍 Transaction Input")
        st.markdown("---")

        amount = st.number_input(
            "💰 Transaction Amount ($)",
            min_value=0.01,
            max_value=50000.0,
            value=150.0,
            step=10.0,
            help="Enter the transaction amount",
        )

        hour = st.slider(
            "🕐 Hour of Day",
            min_value=0,
            max_value=23,
            value=14,
            help="0 = midnight, 12 = noon, 23 = 11PM",
        )

        email = st.selectbox(
            "📧 Email Domain",
            options=[
                # Safe — low fraud rate
                'att.net',           # 0.7%
                'verizon.net',       # 0.8%
                'yahoo.com',         # 2.3%
                'aol.com',           # 2.2%
                # Medium — elevated
                'gmail.com',         # 4.4%
                'hotmail.com',       # 5.3%
                'outlook.com',       # 9.5%
                # High risk — real data
                'aim.com',           # 12.7%
                'outlook.es',        # 13.0%
                'mail.com',          # 19.0%
                'protonmail.com',    # 40.8% ← highest in dataset
            ],
            help="Sender email domain — fraud rate shown in brackets",
        )

        product = st.selectbox(
            "🛍️ Product Code",
            options=list(PRODUCT_LABELS.keys()),
            format_func=lambda x: PRODUCT_LABELS[x],
        )

        card_id = st.number_input(
            "🃏 Card ID",
            min_value=1,
            max_value=99999,
            value=12345,
        )

        # ── Advanced inputs (optional) ───────────────────────
        with st.expander("⚙️ Advanced Inputs (optional)"):
            st.caption("These improve inference accuracy when provided.")

            day_of_week = st.selectbox(
                "📅 Day of Week",
                options=list(DAY_LABELS.keys()),
                index=2,
                format_func=lambda x: DAY_LABELS[x],
            )

            device = st.selectbox(
                "💻 Device Type",
                options=['Windows', 'iOS Device', 'MacOS',
                         'Android', 'Unknown'],
                help="Device used for transaction",
            )

            card2 = st.number_input(
                "💳 Card2 ID",
                min_value=1,
                max_value=999,
                value=111,
                help="Secondary card identifier",
            )

        st.markdown("---")
        analyze_clicked = st.button(
            "🔍 Analyze Transaction",
            use_container_width=True,
            type="primary",
        )

        _render_model_info()

        st.markdown("---")
        st.markdown(
            "**Stack:** XGBoost + LangGraph + Groq LLM  \n"
            "**Dataset:** IEEE-CIS Fraud Detection  \n"
            "**Model:** Calibrated XGBoost (7668 trees)"  # ← fixed
        )

    user_input = {
        'TransactionAmt': amount,
        'hour':           hour,
        'P_emaildomain':  email,
        'ProductCD':      product,
        'card1':          card_id,
        # Advanced
        'day_of_week':    day_of_week,
        'DeviceInfo':     device,
        'card2':          card2,
    }

    return user_input, analyze_clicked


def _render_model_info() -> None:
    """Renders the model metrics panel inside the sidebar."""
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    try:
        # Try pkl first, fall back to json
        import joblib
        summary_path_pkl  = os.path.join('models', 'model_summary.pkl')
        summary_path_json = os.path.join('models', 'model_summary.json')

        if os.path.exists(summary_path_pkl):
            ms = joblib.load(summary_path_pkl)
        else:
            with open(summary_path_json) as f:
                ms = json.load(f)

        st.metric("ROC-AUC",  ms.get('ensemble_roc_auc', 'N/A'))
        st.metric("PR-AUC",   ms.get('ensemble_pr_auc',  'N/A'))
        st.metric("F1 Score", ms.get('best_f1',           'N/A'))
        st.metric("Features", ms.get('features_count',    'N/A'))
        st.metric("Trees",    ms.get('trees',              'N/A'))

    except (FileNotFoundError, Exception):
        st.info("Model summary not found")