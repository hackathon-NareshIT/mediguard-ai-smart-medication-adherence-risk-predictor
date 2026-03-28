"""
MediGuard AI - Utility helpers: charts, medication tracker, alerts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
PALETTE = {
    "bg": "#0A0F1E",
    "surface": "#111827",
    "card": "#1a2235",
    "accent": "#00D4FF",
    "accent2": "#7C3AED",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "text": "#e2e8f0",
    "muted": "#64748b",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": PALETTE["text"], "family": "IBM Plex Sans"},
        "xaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#1e293b"},
        "yaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#1e293b"},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
    }
}


# ─────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────
def risk_gauge(score: float, risk_name: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        delta={"reference": 50, "increasing": {"color": PALETTE["danger"]},
               "decreasing": {"color": PALETTE["success"]}},
        title={"text": f"<b>{risk_name}</b>", "font": {"size": 18, "color": PALETTE["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PALETTE["muted"], "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": PALETTE["card"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#16a34a22"},
                {"range": [40, 70], "color": "#d9770622"},
                {"range": [70, 100], "color": "#dc262622"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────
# PROBABILITY BAR CHART
# ─────────────────────────────────────────────
def probability_bars(probabilities: dict) -> go.Figure:
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"size": 14, "color": PALETTE["text"]},
    ))
    fig.update_layout(
        title={"text": "Risk Class Probabilities", "font": {"size": 16}},
        yaxis={"range": [0, 110], "ticksuffix": "%"},
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────
def feature_importance_chart(top_factors: list) -> go.Figure:
    LABEL_MAP = {
        "missed_doses_last_30d": "Missed Doses (30d)",
        "days_since_last_refill": "Days Since Refill",
        "depression_score": "Depression Score",
        "previous_adherence_rate": "Past Adherence Rate",
        "refill_on_time_rate": "On-Time Refill Rate",
        "medication_complexity_score": "Medication Complexity",
        "side_effects_reported": "Side Effects",
        "health_literacy_score": "Health Literacy",
        "caregiver_support": "Caregiver Support",
        "distance_to_pharmacy": "Pharmacy Distance",
        "num_medications": "No. of Medications",
        "chronic_conditions": "Chronic Conditions",
        "insurance_coverage": "Insurance Coverage",
        "num_hospitalizations_1y": "Hospitalizations (1y)",
        "poly_pharmacy": "Polypharmacy",
        "age": "Age",
        "avg_daily_doses": "Avg Daily Doses",
        "employment_status": "Employment Status",
    }

    names = [LABEL_MAP.get(f["feature"], f["feature"]) for f in top_factors]
    imps = [f["importance"] * 100 for f in top_factors]

    fig = go.Figure(go.Bar(
        x=imps[::-1], y=names[::-1],
        orientation="h",
        marker=dict(
            color=imps[::-1],
            colorscale=[[0, "#7C3AED"], [0.5, "#00D4FF"], [1, "#22d3ee"]],
            showscale=False,
        ),
        text=[f"{i:.1f}%" for i in imps[::-1]],
        textposition="outside",
        textfont={"size": 12, "color": PALETTE["text"]},
    ))
    fig.update_layout(
        title={"text": "Top Risk Drivers", "font": {"size": 16}},
        xaxis={"ticksuffix": "%"},
        height=320,
        margin=dict(l=10, r=60, t=50, b=10),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────
# ADHERENCE TIMELINE (simulated)
# ─────────────────────────────────────────────
def adherence_timeline(days: int = 30, adherence_pct: float = 75.0) -> go.Figure:
    dates = [datetime.today() - timedelta(days=days - i) for i in range(days)]
    taken = np.random.choice([1, 0], size=days, p=[adherence_pct / 100, 1 - adherence_pct / 100])

    colors = [PALETTE["success"] if t else PALETTE["danger"] for t in taken]

    fig = go.Figure(go.Bar(
        x=dates, y=[1] * days,
        marker_color=colors,
        hovertext=["✅ Taken" if t else "❌ Missed" for t in taken],
        hoverinfo="text+x",
    ))
    fig.update_layout(
        title={"text": f"30-Day Dose History (Simulated ~{adherence_pct:.0f}% adherent)", "font": {"size": 15}},
        yaxis={"visible": False, "range": [0, 1.5]},
        xaxis={"tickformat": "%b %d"},
        height=200,
        margin=dict(l=10, r=10, t=50, b=30),
        bargap=0.15,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────
# POPULATION RISK DISTRIBUTION
# ─────────────────────────────────────────────
def population_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_label"].value_counts().sort_index()
    labels = ["Low Risk", "Moderate Risk", "High Risk"]
    colors_list = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        marker_colors=colors_list,
        hole=0.6,
        textinfo="label+percent",
        textfont={"size": 13},
        hovertemplate="%{label}: %{value} patients<extra></extra>",
    ))
    fig.update_layout(
        title={"text": "Population Risk Distribution", "font": {"size": 16}},
        height=300,
        margin=dict(l=0, r=0, t=50, b=0),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────
# MEDICATION SCHEDULE
# ─────────────────────────────────────────────
SAMPLE_MEDICATIONS = [
    {"name": "Metformin 500mg", "time": "08:00 AM", "frequency": "Twice daily", "status": "taken", "days_left": 14},
    {"name": "Lisinopril 10mg", "time": "09:00 AM", "frequency": "Once daily", "status": "taken", "days_left": 7},
    {"name": "Atorvastatin 20mg", "time": "09:00 PM", "frequency": "Once daily", "status": "pending", "days_left": 21},
    {"name": "Aspirin 81mg", "time": "08:00 AM", "frequency": "Once daily", "status": "taken", "days_left": 3},
    {"name": "Amlodipine 5mg", "time": "08:00 AM", "frequency": "Once daily", "status": "missed", "days_left": 18},
]


# ─────────────────────────────────────────────
# ALERT GENERATOR
# ─────────────────────────────────────────────
def generate_alerts(result: dict, patient_data: dict) -> list:
    alerts = []
    score = result["risk_score"]
    label = result["risk_label"]

    if patient_data.get("missed_doses_last_30d", 0) > 10:
        alerts.append({"level": "danger", "icon": "🚨",
                        "msg": f"Critical: {patient_data['missed_doses_last_30d']} missed doses in last 30 days."})
    if patient_data.get("days_since_last_refill", 0) > 60:
        alerts.append({"level": "warning", "icon": "⚠️",
                        "msg": f"Refill overdue by ~{patient_data['days_since_last_refill'] - 30} days."})
    if patient_data.get("depression_score", 0) > 15:
        alerts.append({"level": "warning", "icon": "💊",
                        "msg": "Elevated depression score detected. Consider mental health support."})
    if patient_data.get("side_effects_reported", 0) >= 3:
        alerts.append({"level": "warning", "icon": "⚕️",
                        "msg": "Multiple side effects reported. Review medication tolerability."})
    if patient_data.get("num_medications", 0) >= 8:
        alerts.append({"level": "info", "icon": "📋",
                        "msg": "Polypharmacy detected. Medication reconciliation recommended."})
    if label == 2:
        alerts.append({"level": "danger", "icon": "🏥",
                        "msg": "High Risk: Immediate care coordinator follow-up recommended."})
    if not alerts:
        alerts.append({"level": "success", "icon": "✅",
                        "msg": "No critical alerts. Continue current medication regimen."})
    return alerts