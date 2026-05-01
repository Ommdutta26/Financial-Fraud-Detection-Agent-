# ============================================================
# FRAUD DETECTION DASHBOARD — INDUSTRY LEVEL
# app.py
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json, os
from datetime import datetime
from agent import run_agent

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "Fraud Detection Agent",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE — Transaction History
# ============================================================

if 'history' not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR — INPUT
# ============================================================

with st.sidebar:
    st.markdown("## 🔍 Transaction Input")
    st.markdown("---")

    amount = st.number_input(
        "💰 Transaction Amount ($)",
        min_value  = 0.01,
        max_value  = 50000.0,
        value      = 150.0,
        step       = 10.0,
        help       = "Enter the transaction amount"
    )

    hour = st.slider(
        "🕐 Hour of Day",
        min_value = 0,
        max_value = 23,
        value     = 14,
        help      = "0 = midnight, 12 = noon, 23 = 11PM"
    )

    email = st.selectbox(
        "📧 Email Domain",
        options = ['gmail.com', 'yahoo.com', 'outlook.com',
                   'hotmail.com', 'anonymous.com',
                   'guerrillamail.com', 'company.com'],
        help    = "Sender email domain"
    )

    product = st.selectbox(
        "🛍️ Product Code",
        options     = ['W', 'H', 'C', 'S', 'R'],
        format_func = lambda x: {
            'W': 'W — Digital Goods',
            'H': 'H — Hotel/Travel',
            'C': 'C — Cash',
            'S': 'S — Services',
            'R': 'R — Retail',
        }[x],
    )

    card_id = st.number_input(
        "🃏 Card ID",
        min_value = 1,
        max_value = 99999,
        value     = 12345,
    )

    st.markdown("---")
    analyze_btn = st.button(
        "🔍 Analyze Transaction",
        use_container_width = True,
        type                = "primary",
    )

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    try:
        with open(os.path.join('models', 'model_summary.json')) as f:
            ms = json.load(f)
        st.metric("ROC-AUC",  ms.get('ensemble_roc_auc', 'N/A'))
        st.metric("PR-AUC",   ms.get('ensemble_pr_auc',  'N/A'))
        st.metric("F1 Score", ms.get('best_f1',           'N/A'))
        st.metric("Features", ms.get('features_count',    'N/A'))
    except:
        st.info("model_summary.json not found")

    st.markdown("---")
    st.markdown(
        "**Stack:** XGBoost + LangGraph + Groq LLM  \n"
        "**Dataset:** IEEE-CIS Fraud Detection  \n"
        "**Model:** Ensemble (XGB + IsoForest)"
    )


# ============================================================
# MAIN HEADER
# ============================================================

col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("# 🔍 Fraud Detection Agent")
    st.markdown(
        "**Multi-Agent AI System** | XGBoost Ensemble + "
        "LangGraph Orchestration + Groq LLM Reasoning"
    )
with col_time:
    st.markdown(f"<br><p style='text-align:right;color:#64748b'>"
                f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}</p>",
                unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# ANALYSIS
# ============================================================

if analyze_btn:

    user_input = {
        'TransactionAmt': amount,
        'hour':           hour,
        'P_emaildomain':  email,
        'ProductCD':      product,
        'card1':          card_id,
    }

    with st.spinner("🧠 Agent analyzing transaction through 6-node pipeline..."):
        result = run_agent(user_input)

    # Store in history
    st.session_state.history.append({
        'time':     datetime.now().strftime('%H:%M:%S'),
        'amount':   amount,
        'decision': result['decision'],
        'score':    result['score'],
        'flags':    len(result['pattern_flags']) + len(result['rule_flags']),
    })

    # ── Decision Banner ──
    decision = result['decision']
    css_class = {'APPROVE': 'approve-card',
                 'FLAG':    'flag-card',
                 'BLOCK':   'block-card'}.get(decision, 'flag-card')
    emoji     = {'APPROVE': '✅', 'FLAG': '⚠️', 'BLOCK': '🚨'}.get(decision, '❓')
    color     = {'APPROVE': '#22c55e', 'FLAG': '#f59e0b',
                 'BLOCK':   '#ef4444'}.get(decision, '#94a3b8')

    st.markdown(f"""
    <div class="{css_class}">
        <h1 style="color:{color};margin:0;font-size:2.5em">{emoji} {decision}</h1>
        <p style="color:#e2e8f0;margin:5px 0 0 0;font-size:1.1em">
            Confidence: {result['confidence']:.1f}%  |
            Risk Level: {result['risk_level']}  |
            Score: {result['score']:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Gauge + Metrics + Agent Steps ──
    col1, col2, col3 = st.columns([1.2, 1, 1.5])

    with col1:
        st.markdown("#### 📊 Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = result['score'] * 100,
            title = {'text': "Fraud Risk %", 'font': {'color': '#94a3b8'}},
            delta = {'reference': 50, 'increasing': {'color': '#ef4444'}},
            gauge = {
                'axis':  {'range': [0, 100],
                           'tickcolor': '#94a3b8',
                           'tickfont':  {'color': '#94a3b8'}},
                'bar':   {'color': color},
                'bgcolor': '#1e2130',
                'steps': [
                    {'range': [0,  30], 'color': '#14532d'},
                    {'range': [30, 60], 'color': '#713f12'},
                    {'range': [60,100], 'color': '#7f1d1d'},
                ],
                'threshold': {
                    'line':  {'color': 'white', 'width': 3},
                    'thickness': 0.75,
                    'value': result.get('threshold', 0.54) * 100,
                }
            },
            number = {'suffix': '%', 'font': {'color': color}},
        ))
        fig.update_layout(
            height     = 260,
            paper_bgcolor = '#0e1117',
            font       = {'color': '#94a3b8'},
            margin     = dict(t=50, b=10, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Score Breakdown")
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b;margin:0;font-size:0.8em">XGBoost Score</p>
            <p style="color:#6366f1;margin:0;font-size:1.8em;font-weight:bold">
                {result['xgb_score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b;margin:0;font-size:0.8em">Ensemble Score</p>
            <p style="color:{color};margin:0;font-size:1.8em;font-weight:bold">
                {result['score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b;margin:0;font-size:0.8em">Confidence</p>
            <p style="color:#e2e8f0;margin:0;font-size:1.8em;font-weight:bold">
                {result['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### 🔄 Agent Pipeline Steps")
        steps = [
            ("1", "Risk Scorer",      f"XGBoost: {result['xgb_score']:.3f}"),
            ("2", "Pattern Analyzer", f"{len(result['pattern_flags'])} patterns found"),
            ("3", "Rule Engine",      f"{len(result['rule_flags'])} rules triggered"),
            ("4", "SHAP Explainer",   f"Top: {result['shap_reasons'][0][:35] + '...' if result['shap_reasons'] else 'N/A'}"),
            ("5", "Groq LLM",         f"Decision: {result['decision']}"),
            ("6", "Report Writer",    "Report generated ✅"),
        ]
        for num, name, detail in steps:
            st.markdown(f"""
            <div class="step-card">
                <span style="color:#6366f1;font-weight:bold">Step {num}</span>
                <span style="color:#e2e8f0;margin-left:8px">{name}</span><br>
                <span style="color:#64748b;font-size:0.85em">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Flags + SHAP ──
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("#### 🚩 Pattern & Rule Flags")

        if result['pattern_flags']:
            st.markdown("**Behavioral Patterns:**")
            for f in result['pattern_flags']:
                st.markdown(f'<div class="flag-item">{f}</div>',
                           unsafe_allow_html=True)
        else:
            st.success("No suspicious patterns detected")

        if result['rule_flags']:
            st.markdown("**Business Rules Triggered:**")
            for r in result['rule_flags']:
                st.markdown(f'<div class="rule-item">{r}</div>',
                           unsafe_allow_html=True)
        else:
            st.success("No business rules triggered")

    with col5:
        st.markdown("#### 📊 SHAP Feature Impact")
        if result['shap_dict']:
            shap_df = pd.DataFrame(
                list(result['shap_dict'].items()),
                columns=['Feature', 'Impact']
            ).sort_values('Impact', key=abs, ascending=True)

            shap_df['Color'] = shap_df['Impact'].apply(
                lambda x: '#ef4444' if x > 0 else '#22c55e'
            )

            fig2 = go.Figure(go.Bar(
                x           = shap_df['Impact'],
                y           = shap_df['Feature'],
                orientation = 'h',
                marker_color= shap_df['Color'],
                text        = shap_df['Impact'].apply(lambda x: f"{x:+.3f}"),
                textposition= 'outside',
            ))
            fig2.update_layout(
                height        = 300,
                paper_bgcolor = '#0e1117',
                plot_bgcolor  = '#1e2130',
                font          = {'color': '#94a3b8'},
                margin        = dict(t=20, b=20, l=10, r=60),
                xaxis         = dict(gridcolor='#2d3148',
                                     title='SHAP Impact'),
                yaxis         = dict(gridcolor='#2d3148'),
                showlegend    = False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("SHAP values not available")

    st.markdown("---")

    # ── Row 3: LLM Reasoning + Full Report ──
    col6, col7 = st.columns(2)

    with col6:
        st.markdown("#### 🧠 AI Reasoning (Groq LLM)")
        st.markdown(f"""
        <div style="background:#1e2130;border-left:3px solid #6366f1;
                    border-radius:6px;padding:15px;color:#e2e8f0;
                    font-style:italic;line-height:1.6">
            "{result['agent_reasoning']}"
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Top Risk Factors")
        for i, reason in enumerate(result['shap_reasons'][:5], 1):
            st.markdown(f"**{i}.** {reason}")

    with col7:
        st.markdown("#### 📄 Investigation Report")
        st.text_area(
            label       = "Full Report",
            value       = result['report'],
            height      = 350,
            label_visibility = 'collapsed',
        )
        st.download_button(
            label    = "⬇️ Download Report",
            data     = result['report'],
            file_name= f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime     = "text/plain",
            use_container_width = True,
        )


# ============================================================
# TRANSACTION HISTORY
# ============================================================

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📈 Session Transaction History")

    hist_df = pd.DataFrame(st.session_state.history)

    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        # Score trend chart
        fig3 = go.Figure()
        colors_map = {'APPROVE': '#22c55e', 'FLAG': '#f59e0b', 'BLOCK': '#ef4444'}
        fig3.add_trace(go.Scatter(
            x    = list(range(1, len(hist_df) + 1)),
            y    = hist_df['score'],
            mode = 'lines+markers',
            name = 'Fraud Score',
            line = dict(color='#6366f1', width=2),
            marker = dict(
                color = [colors_map.get(d, '#94a3b8')
                         for d in hist_df['decision']],
                size  = 10,
            )
        ))
        fig3.add_hline(
            y          = 0.54,
            line_dash  = "dash",
            line_color = "#ef4444",
            annotation_text = "Threshold",
        )
        fig3.update_layout(
            title         = "Fraud Score Trend (this session)",
            height        = 250,
            paper_bgcolor = '#0e1117',
            plot_bgcolor  = '#1e2130',
            font          = {'color': '#94a3b8'},
            margin        = dict(t=40, b=20, l=20, r=20),
            xaxis         = dict(gridcolor='#2d3148', title='Transaction #'),
            yaxis         = dict(gridcolor='#2d3148', title='Score',
                                 range=[0, 1]),
            showlegend    = False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_h2:
        # Decision summary
        counts = hist_df['decision'].value_counts()
        fig4   = go.Figure(go.Pie(
            labels = counts.index,
            values = counts.values,
            marker_colors = [colors_map.get(d, '#94a3b8')
                             for d in counts.index],
            hole   = 0.5,
        ))
        fig4.update_layout(
            title         = "Decision Distribution",
            height        = 250,
            paper_bgcolor = '#0e1117',
            font          = {'color': '#94a3b8'},
            margin        = dict(t=40, b=20, l=20, r=20),
            legend        = dict(font=dict(color='#94a3b8')),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # History table
    st.dataframe(
        hist_df.rename(columns={
            'time':     'Time',
            'amount':   'Amount ($)',
            'decision': 'Decision',
            'score':    'Score',
            'flags':    'Flags',
        }),
        use_container_width = True,
        hide_index          = True,
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

else:
    # Show when no analysis done yet
    st.markdown("""
    <div style="text-align:center;padding:60px;color:#475569">
        <h2>👈 Enter transaction details in the sidebar</h2>
        <p>Click <b>Analyze Transaction</b> to run the 6-node AI agent pipeline</p>
        <br>
        <p style="font-size:0.9em">
            <b>Pipeline:</b>
            Risk Scorer → Pattern Analyzer → Rule Engine →
            SHAP Explainer → Groq LLM → Report Writer
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8em'>"
    "Fraud Detection Agent | XGBoost + LangGraph + Groq LLM | "
    "IEEE-CIS Dataset | Built for placement portfolio"
    "</p>",
    unsafe_allow_html=True
)
