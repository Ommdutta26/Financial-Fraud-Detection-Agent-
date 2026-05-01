"""
result_display.py — Renders the analysis result sections after the agent runs.

Sections:
  - Decision banner
  - Row 1: Gauge | Score breakdown | Pipeline steps
  - Row 2: Flags | SHAP chart
  - Row 3: LLM reasoning | Investigation report
"""

import streamlit as st
from datetime import datetime

from components.charts import (
    build_risk_gauge,
    build_shap_bar,
)
from styles.custom_css import (
    DECISION_CSS,
    DECISION_EMOJI,
    DECISION_COLOR,
)


# ── Decision Banner ───────────────────────────────────────────────────────────

def render_decision_banner(result: dict) -> None:
    decision  = result['decision']
    css_class = DECISION_CSS.get(decision, 'flag-card')
    emoji     = DECISION_EMOJI.get(decision, '❓')
    color     = DECISION_COLOR.get(decision, '#94a3b8')

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


# ── Row 1 ─────────────────────────────────────────────────────────────────────

def render_row1(result: dict) -> None:
    """Gauge · Score breakdown · Pipeline steps."""
    col1, col2, col3 = st.columns([1.2, 1, 1.5])
    color = DECISION_COLOR.get(result['decision'], '#94a3b8')

    with col1:
        st.markdown("#### 📊 Risk Gauge")
        fig = build_risk_gauge(
            score=result['score'],
            color=color,
            threshold=result.get('threshold', 0.54),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        _render_score_breakdown(result, color)

    with col3:
        _render_pipeline_steps(result)


def _render_score_breakdown(result: dict, color: str) -> None:
    st.markdown("#### 🎯 Score Breakdown")

    cards = [
        ("#6366f1", "XGBoost Score",  f"{result['xgb_score']:.4f}"),
        (color,     "Ensemble Score", f"{result['score']:.4f}"),
        ("#e2e8f0", "Confidence",     f"{result['confidence']:.1f}%"),
    ]
    for text_color, label, value in cards:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b;margin:0;font-size:0.8em">{label}</p>
            <p style="color:{text_color};margin:0;font-size:1.8em;font-weight:bold">
                {value}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


def _render_pipeline_steps(result: dict) -> None:
    st.markdown("#### 🔄 Agent Pipeline Steps")

    first_shap = result['shap_reasons'][0][:35] + '...' if result['shap_reasons'] else 'N/A'
    steps = [
        ("1", "Risk Scorer",      f"XGBoost: {result['xgb_score']:.3f}"),
        ("2", "Pattern Analyzer", f"{len(result['pattern_flags'])} patterns found"),
        ("3", "Rule Engine",      f"{len(result['rule_flags'])} rules triggered"),
        ("4", "SHAP Explainer",   f"Top: {first_shap}"),
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


# ── Row 2 ─────────────────────────────────────────────────────────────────────

def render_row2(result: dict) -> None:
    """Pattern & rule flags · SHAP bar chart."""
    col4, col5 = st.columns(2)

    with col4:
        _render_flags(result)

    with col5:
        st.markdown("#### 📊 SHAP Feature Impact")
        if result['shap_dict']:
            fig = build_shap_bar(result['shap_dict'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP values not available")


def _render_flags(result: dict) -> None:
    st.markdown("#### 🚩 Pattern & Rule Flags")

    if result['pattern_flags']:
        st.markdown("**Behavioral Patterns:**")
        for f in result['pattern_flags']:
            st.markdown(f'<div class="flag-item">{f}</div>', unsafe_allow_html=True)
    else:
        st.success("No suspicious patterns detected")

    if result['rule_flags']:
        st.markdown("**Business Rules Triggered:**")
        for r in result['rule_flags']:
            st.markdown(f'<div class="rule-item">{r}</div>', unsafe_allow_html=True)
    else:
        st.success("No business rules triggered")


# ── Row 3 ─────────────────────────────────────────────────────────────────────

def render_row3(result: dict) -> None:
    """LLM reasoning · Full investigation report."""
    col6, col7 = st.columns(2)

    with col6:
        _render_llm_reasoning(result)

    with col7:
        _render_report(result)


def _render_llm_reasoning(result: dict) -> None:
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


def _render_report(result: dict) -> None:
    st.markdown("#### 📄 Investigation Report")
    st.text_area(
        label="Full Report",
        value=result['report'],
        height=350,
        label_visibility='collapsed',
    )
    st.download_button(
        label="⬇️ Download Report",
        data=result['report'],
        file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ── Full result render (convenience wrapper) ──────────────────────────────────

def render_result(result: dict) -> None:
    """Renders the complete analysis result — banner + all three rows."""
    render_decision_banner(result)
    render_row1(result)
    st.markdown("---")
    render_row2(result)
    st.markdown("---")
    render_row3(result)
