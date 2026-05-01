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
                'gmail.com', 'yahoo.com', 'outlook.com',
                'hotmail.com', 'anonymous.com',
                'guerrillamail.com', 'company.com',
            ],
            help="Sender email domain",
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
            "**Model:** Ensemble (XGB + IsoForest)"
        )

    user_input = {
        'TransactionAmt': amount,
        'hour':           hour,
        'P_emaildomain':  email,
        'ProductCD':      product,
        'card1':          card_id,
    }

    return user_input, analyze_clicked


def _render_model_info() -> None:
    """Renders the model metrics panel inside the sidebar."""
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    try:
        summary_path = os.path.join('models', 'model_summary.json')
        with open(summary_path) as f:
            ms = json.load(f)
        st.metric("ROC-AUC",  ms.get('ensemble_roc_auc', 'N/A'))
        st.metric("PR-AUC",   ms.get('ensemble_pr_auc',  'N/A'))
        st.metric("F1 Score", ms.get('best_f1',           'N/A'))
        st.metric("Features", ms.get('features_count',    'N/A'))
    except FileNotFoundError:
        st.info("model_summary.json not found")
