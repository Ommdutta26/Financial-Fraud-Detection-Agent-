"""
app.py — Fraud Detection Dashboard entry point.

Orchestrates page config, CSS, sidebar, agent call, and all render sections.
All heavy lifting is delegated to the components/ and styles/ modules.
"""

import streamlit as st
from datetime import datetime
import requests
from styles.custom_css import CUSTOM_CSS
from components.sidebar import render_sidebar
from components.result_display import render_result
from components.history import init_history, append_history, render_history
from agent.agent import run_agent
from dotenv import load_dotenv
import os


load_dotenv() 

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

init_history()


# ── Sidebar ───────────────────────────────────────────────────────────────────

user_input, analyze_clicked = render_sidebar()


# ── Header ────────────────────────────────────────────────────────────────────

col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("# 🔍 Fraud Detection Agent")
    st.markdown(
        "**Multi-Agent AI System** | Calibrated XGBoost + "
        "LangGraph Orchestration + Groq LLM Reasoning"
    )
with col_time:
    st.markdown(
        f"<br><p style='text-align:right;color:#64748b'>"
        f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ── Analysis ──────────────────────────────────────────────────────────────────

if analyze_clicked:
    with st.spinner("🧠 Agent analyzing transaction through 7-node pipeline..."):
        result = run_agent(user_input)
        
        payload = {
            "transaction_id": f"TXN-{user_input.get('card1', 'UNKNOWN')}",
            "transaction_amount": user_input.get("TransactionAmt"),
            "risk_level": result.get("risk_level"),
            "score": result.get("score"),
            "decision": result.get("decision"),
            "confidence": result.get("confidence"),
            "agent_reasoning": result.get("agent_reasoning"),
            "rule_flags": result.get("rule_flags"),
            "pattern_flags": result.get("pattern_flags"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # ============================================================
        # Send result to n8n automation workflow
        # ============================================================
    if N8N_WEBHOOK_URL:
        try: 
            response = requests.post( N8N_WEBHOOK_URL, json=payload, timeout=5 ) 
            if response.status_code == 200: 
                st.success("✅ Fraud alert sent to automation pipeline")
        except Exception as e: st.warning(f"⚠️ n8n webhook failed: {e}")
    else: st.warning("⚠️ N8N_WEBHOOK_URL not found in .env file")
    append_history(result, amount=user_input['TransactionAmt'])
    render_result(result)

else:
    st.markdown("""
    <div style="text-align:center;padding:60px;color:#475569">
        <h2>👈 Enter transaction details in the sidebar</h2>
        <p>Click <b>Analyze Transaction</b> to run the 7-node AI agent pipeline</p>
        <br>
        <p style="font-size:0.9em">
            <b>Pipeline:</b>
            Risk Scorer → Memory Check → Pattern Analyzer → Rule Engine →
            SHAP Explainer → Groq LLM → Report Writer
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Session history ───────────────────────────────────────────────────────────

render_history()


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8em'>"
    "Fraud Detection Agent | Calibrated XGBoost + LangGraph + Groq LLM | "
    "IEEE-CIS Dataset | Built for placement portfolio"
    "</p>",
    unsafe_allow_html=True,
)