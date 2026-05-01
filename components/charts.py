"""
charts.py — All Plotly chart builders for the fraud dashboard.
Each function returns a Plotly Figure ready to pass to st.plotly_chart().
"""

import plotly.graph_objects as go
import pandas as pd


# ── Shared layout defaults ────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    paper_bgcolor='#0e1117',
    plot_bgcolor='#1e2130',
    font=dict(color='#94a3b8'),
)


# ── Risk Gauge ────────────────────────────────────────────────────────────────

def build_risk_gauge(score: float, color: str, threshold: float = 0.54) -> go.Figure:
    """Gauge chart showing fraud risk as a percentage."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=score * 100,
        title={'text': "Fraud Risk %", 'font': {'color': '#94a3b8'}},
        delta={'reference': 50, 'increasing': {'color': '#ef4444'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': '#94a3b8',
                'tickfont': {'color': '#94a3b8'},
            },
            'bar': {'color': color},
            'bgcolor': '#1e2130',
            'steps': [
                {'range': [0,  30], 'color': '#14532d'},
                {'range': [30, 60], 'color': '#713f12'},
                {'range': [60, 100], 'color': '#7f1d1d'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.75,
                'value': threshold * 100,
            },
        },
        number={'suffix': '%', 'font': {'color': color}},
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        height=260,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


# ── SHAP Bar Chart ────────────────────────────────────────────────────────────

def build_shap_bar(shap_dict: dict) -> go.Figure:
    """Horizontal bar chart of SHAP feature impacts."""
    shap_df = (
        pd.DataFrame(list(shap_dict.items()), columns=['Feature', 'Impact'])
        .sort_values('Impact', key=abs, ascending=True)
    )
    shap_df['Color'] = shap_df['Impact'].apply(
        lambda x: '#ef4444' if x > 0 else '#22c55e'
    )

    fig = go.Figure(go.Bar(
        x=shap_df['Impact'],
        y=shap_df['Feature'],
        orientation='h',
        marker_color=shap_df['Color'],
        text=shap_df['Impact'].apply(lambda x: f"{x:+.3f}"),
        textposition='outside',
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        height=300,
        margin=dict(t=20, b=20, l=10, r=60),
        xaxis=dict(gridcolor='#2d3148', title='SHAP Impact'),
        yaxis=dict(gridcolor='#2d3148'),
        showlegend=False,
    )
    return fig


# ── Score Trend (session history) ─────────────────────────────────────────────

def build_score_trend(hist_df: pd.DataFrame, colors_map: dict) -> go.Figure:
    """Line + marker chart of fraud scores across the session."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(hist_df) + 1)),
        y=hist_df['score'],
        mode='lines+markers',
        name='Fraud Score',
        line=dict(color='#6366f1', width=2),
        marker=dict(
            color=[colors_map.get(d, '#94a3b8') for d in hist_df['decision']],
            size=10,
        ),
    ))
    fig.add_hline(
        y=0.54,
        line_dash='dash',
        line_color='#ef4444',
        annotation_text='Threshold',
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        title='Fraud Score Trend (this session)',
        height=250,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(gridcolor='#2d3148', title='Transaction #'),
        yaxis=dict(gridcolor='#2d3148', title='Score', range=[0, 1]),
        showlegend=False,
    )
    return fig


# ── Decision Pie ──────────────────────────────────────────────────────────────

def build_decision_pie(hist_df: pd.DataFrame, colors_map: dict) -> go.Figure:
    """Donut chart showing APPROVE / FLAG / BLOCK distribution."""
    counts = hist_df['decision'].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[colors_map.get(d, '#94a3b8') for d in counts.index],
        hole=0.5,
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title='Decision Distribution',
        height=250,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(font=dict(color='#94a3b8')),
    )
    return fig
