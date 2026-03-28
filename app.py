"""
MediGuard AI — Streamlit Cloud compatible entry point
"""

import sys
import os

# ── Fix import path for Streamlit Cloud ──────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import time

from models.risk_model import (
    get_model, predict_risk, generate_training_data,
    FEATURE_NAMES, RISK_LABELS, RISK_COLORS
)
from utils.charts import (
    risk_gauge, probability_bars, feature_importance_chart,
    adherence_timeline, population_distribution, generate_alerts,
    SAMPLE_MEDICATIONS, PALETTE
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediGuard AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #0A0F1E; --surface: #111827; --card: #1a2235; --border: #1e293b;
  --accent: #00D4FF; --accent2: #7C3AED; --success: #22c55e;
  --warning: #f59e0b; --danger: #ef4444; --text: #e2e8f0; --muted: #64748b;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; background-color: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1526,#111827) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stAppViewContainer"] > .main { background: var(--bg) !important; }
.mg-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
.mg-card-accent { border-left: 3px solid var(--accent); }
.mg-hero { background: linear-gradient(135deg,#0d1a3a,#1a0d3a 50%,#0d2a1a); border: 1px solid var(--border); border-radius: 16px; padding: 32px 36px; margin-bottom: 28px; }
.mg-hero h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg,#00D4FF,#7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px 0; }
.mg-hero p { color: var(--muted); font-size: 1rem; margin: 0; }
.kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi-tile { flex: 1; min-width: 130px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; text-align: center; }
.kpi-tile .val { font-size: 1.9rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1; }
.kpi-tile .lbl { font-size: 0.72rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.07em; }
.risk-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 0.88rem; font-weight: 600; letter-spacing: 0.04em; }
.risk-low    { background: rgba(34,197,94,.15);  color: #22c55e; border: 1px solid #22c55e44; }
.risk-medium { background: rgba(245,158,11,.15); color: #f59e0b; border: 1px solid #f59e0b44; }
.risk-high   { background: rgba(239,68,68,.15);  color: #ef4444; border: 1px solid #ef4444aa; }
.alert-danger  { background: rgba(239,68,68,.10);  border-left: 3px solid #ef4444; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.alert-warning { background: rgba(245,158,11,.10); border-left: 3px solid #f59e0b; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.alert-info    { background: rgba(0,212,255,.08);  border-left: 3px solid #00D4FF; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.alert-success { background: rgba(34,197,94,.10);  border-left: 3px solid #22c55e; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.med-row { display: flex; align-items: center; gap: 14px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.med-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.med-name { font-weight: 600; font-size: 0.93rem; }
.med-sub  { font-size: 0.78rem; color: var(--muted); }
.med-badge { margin-left: auto; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.med-taken   { background: rgba(34,197,94,.15);  color: #22c55e; }
.med-pending { background: rgba(0,212,255,.12);  color: #00D4FF; }
.med-missed  { background: rgba(239,68,68,.15);  color: #ef4444; }
.stButton > button { background: linear-gradient(135deg,#00D4FF22,#7C3AED22) !important; border: 1px solid var(--accent) !important; color: var(--accent) !important; font-weight: 600 !important; border-radius: 8px !important; }
.stButton > button:hover { background: linear-gradient(135deg,#00D4FF44,#7C3AED44) !important; }
.sec-header { font-size: 1.1rem; font-weight: 600; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; margin: 20px 0 12px 0; display: flex; align-items: center; gap: 8px; }
.sec-header::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg,var(--border),transparent); }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'IBM Plex Mono' !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--card) !important; border-radius: 10px !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; background: rgba(0,212,255,.1) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION & CACHE
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Training MediGuard AI model...")
def load_model_cached():
    return get_model()

@st.cache_data(show_spinner=False)
def get_population_data():
    return generate_training_data(n_samples=1000)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 22px 0;">
      <div style="font-size:2.6rem;margin-bottom:6px;">💊</div>
      <div style="font-size:1.35rem;font-weight:700;background:linear-gradient(135deg,#00D4FF,#7C3AED);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">MediGuard AI</div>
      <div style="font-size:0.72rem;color:#64748b;letter-spacing:0.1em;margin-top:2px;">MEDICATION ADHERENCE INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav = st.radio("Navigation", [
        "🏠  Dashboard", "🔬  Risk Predictor",
        "📋  Medication Tracker", "📊  Analytics", "⚙️  Model Info"
    ], label_visibility="collapsed")
    page = nav.split("  ")[1].lower().replace(" ", "_")

    st.markdown("---")

    if st.session_state.prediction_result:
        r = st.session_state.prediction_result
        color = r["risk_color"]
        st.markdown(f"""
        <div class="mg-card" style="border-left:3px solid {color};">
          <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Last Assessment</div>
          <div style="font-size:1.6rem;font-weight:700;color:{color};font-family:'IBM Plex Mono';">
            {r['risk_score']:.0f}<span style="font-size:.9rem;">%</span>
          </div>
          <div class="risk-badge risk-{'low' if r['risk_label']==0 else ('medium' if r['risk_label']==1 else 'high')}" style="margin-top:6px;">
            {r['risk_name']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:18px;left:0;width:260px;text-align:center;font-size:.68rem;color:#334155;">
      MediGuard AI v1.0 · Demo only<br>Not a substitute for clinical advice
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
model, metrics = load_model_cached()
pop_df = get_population_data()


# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "dashboard":
    st.markdown("""
    <div class="mg-hero">
      <h1>💊 MediGuard AI</h1>
      <p>Intelligent Medication Adherence Risk Prediction · ML-Powered · Real-Time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    risk_counts = pop_df["risk_label"].value_counts().sort_index()
    total = len(pop_df)
    high_risk_pct = risk_counts.get(2, 0) / total * 100

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-tile"><div class="val" style="color:#00D4FF;">{total:,}</div><div class="lbl">Patients Tracked</div></div>
      <div class="kpi-tile"><div class="val" style="color:#ef4444;">{high_risk_pct:.1f}%</div><div class="lbl">High-Risk Patients</div></div>
      <div class="kpi-tile"><div class="val" style="color:#22c55e;">{metrics['accuracy']*100:.1f}%</div><div class="lbl">Model Accuracy</div></div>
      <div class="kpi-tile"><div class="val" style="color:#7C3AED;">{metrics['roc_auc']:.3f}</div><div class="lbl">ROC-AUC Score</div></div>
      <div class="kpi-tile"><div class="val" style="color:#f59e0b;">18</div><div class="lbl">Risk Features</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="sec-header">📈 Population Risk Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(population_distribution(pop_df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="sec-header">📅 30-Day Adherence Calendar</div>', unsafe_allow_html=True)
        st.plotly_chart(adherence_timeline(30, 72.0), use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div class="sec-header">💊 Today\'s Medications</div>', unsafe_allow_html=True)
        for med in SAMPLE_MEDICATIONS:
            dot_color = "#22c55e" if med["status"] == "taken" else ("#ef4444" if med["status"] == "missed" else "#00D4FF")
            badge_cls = f"med-{med['status']}"
            refill_warn = "🔴 " if med["days_left"] <= 5 else ("🟡 " if med["days_left"] <= 10 else "")
            st.markdown(f"""
            <div class="med-row">
              <div class="med-dot" style="background:{dot_color};"></div>
              <div>
                <div class="med-name">{med['name']}</div>
                <div class="med-sub">🕐 {med['time']} · {med['frequency']} · {refill_warn}{med['days_left']}d left</div>
              </div>
              <div class="med-badge {badge_cls}">{'✅' if med['status']=='taken' else '❌' if med['status']=='missed' else '⏰'} {med['status'].title()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header" style="margin-top:24px;">🔔 Alerts</div>', unsafe_allow_html=True)
        for a in [
            {"level":"danger",  "icon":"🚨","msg":"Patient #1042: 12 missed doses this month."},
            {"level":"warning", "icon":"⚠️","msg":"Patient #0873: Refill overdue by 15 days."},
            {"level":"info",    "icon":"💡","msg":"3 patients scheduled for follow-up today."},
            {"level":"success", "icon":"✅","msg":"Weekly adherence report generated."},
        ]:
            st.markdown(f'<div class="alert-{a["level"]}">{a["icon"]} {a["msg"]}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: RISK PREDICTOR
# ═════════════════════════════════════════════════════════════
elif page == "risk_predictor":
    st.markdown("""
    <div class="mg-hero">
      <h1>🔬 Risk Predictor</h1>
      <p>Enter patient details to generate a real-time AI-powered medication adherence risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown('<div class="sec-header">👤 Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age                 = c1.slider("Age", 18, 90, 55)
        employment_lbl      = c2.selectbox("Employment", ["Unemployed","Part-time","Full-time","Retired"])
        chronic_conditions  = c3.slider("Chronic Conditions", 0, 7, 2)
        emp_map = {"Unemployed":0,"Part-time":1,"Full-time":2,"Retired":3}

        st.markdown('<div class="sec-header">💊 Medication Profile</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        num_medications             = c1.slider("No. of Medications", 1, 15, 4)
        avg_daily_doses             = c2.slider("Avg Daily Doses", 0.5, 6.0, 2.0, 0.5)
        medication_complexity_score = c3.slider("Complexity (1-10)", 1, 10, 4)
        side_effects_reported       = c4.slider("Side Effects", 0, 5, 1)

        st.markdown('<div class="sec-header">📊 Adherence History</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        missed_doses_last_30d   = c1.slider("Missed Doses (30d)", 0, 30, 4)
        days_since_last_refill  = c2.slider("Days Since Refill", 0, 120, 28)
        previous_adherence_rate = c3.slider("Past Adherence (%)", 0, 100, 75)
        refill_on_time_rate     = c4.slider("On-Time Refill (%)", 0, 100, 70)

        st.markdown('<div class="sec-header">🏥 Clinical Factors</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        depression_score        = c1.slider("Depression Score (0-27)", 0, 27, 5)
        health_literacy_score   = c2.slider("Health Literacy (1-10)", 1, 10, 6)
        caregiver_lbl           = c3.selectbox("Caregiver Support", ["None","Partial","Full"])
        insurance_lbl           = c4.selectbox("Insurance Coverage", ["None","Partial","Full"])
        support_map = {"None":0,"Partial":1,"Full":2}
        c1, c2 = st.columns(2)
        distance_to_pharmacy    = c1.slider("Pharmacy Distance (km)", 0.1, 50.0, 3.0, 0.5)
        num_hospitalizations_1y = c2.slider("Hospitalizations (1yr)", 0, 5, 0)

        submitted = st.form_submit_button("🔍  Run Risk Assessment", use_container_width=True)

    if submitted:
        patient_data = {
            "age": age, "num_medications": num_medications,
            "chronic_conditions": chronic_conditions,
            "days_since_last_refill": days_since_last_refill,
            "missed_doses_last_30d": missed_doses_last_30d,
            "avg_daily_doses": avg_daily_doses,
            "medication_complexity_score": medication_complexity_score,
            "side_effects_reported": side_effects_reported,
            "caregiver_support": support_map[caregiver_lbl],
            "health_literacy_score": health_literacy_score,
            "insurance_coverage": support_map[insurance_lbl],
            "distance_to_pharmacy": distance_to_pharmacy,
            "employment_status": emp_map[employment_lbl],
            "depression_score": depression_score,
            "previous_adherence_rate": previous_adherence_rate,
            "num_hospitalizations_1y": num_hospitalizations_1y,
            "poly_pharmacy": 1 if num_medications >= 5 else 0,
            "refill_on_time_rate": refill_on_time_rate,
        }
        with st.spinner("Analyzing patient risk profile..."):
            time.sleep(0.5)
            result = predict_risk(model, patient_data)
            st.session_state.prediction_result = result
            st.session_state.patient_data = patient_data

        st.markdown("---")
        st.markdown('<div class="sec-header">📊 Assessment Results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            st.plotly_chart(risk_gauge(result["risk_score"], result["risk_name"], result["risk_color"]),
                            use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(probability_bars(result["probabilities"]),
                            use_container_width=True, config={"displayModeBar": False})
        with col3:
            lvl = 'low' if result['risk_label']==0 else ('medium' if result['risk_label']==1 else 'high')
            st.markdown(f"""
            <div class="mg-card mg-card-accent" style="margin-top:10px;">
              <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">Summary</div>
              <div class="risk-badge risk-{lvl}" style="font-size:1rem;padding:8px 20px;">{result['risk_name']}</div>
              <div style="margin-top:14px;font-size:.85rem;color:#94a3b8;line-height:1.6;">
                {"✅ Continue current regimen." if result['risk_label']==0 else
                 ("⚠️ Increase monitoring frequency." if result['risk_label']==1 else
                  "🚨 Immediate intervention required.")}
              </div>
            </div>
            """, unsafe_allow_html=True)
            alerts = generate_alerts(result, patient_data)
            for a in alerts[:3]:
                st.markdown(f'<div class="alert-{a["level"]}">{a["icon"]} {a["msg"]}</div>', unsafe_allow_html=True)

        st.plotly_chart(feature_importance_chart(result["top_factors"]),
                        use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════
# PAGE: MEDICATION TRACKER
# ═════════════════════════════════════════════════════════════
elif page == "medication_tracker":
    st.markdown("""
    <div class="mg-hero">
      <h1>📋 Medication Tracker</h1>
      <p>Monitor scheduled medications, refill status, and daily adherence patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    taken   = sum(1 for m in SAMPLE_MEDICATIONS if m["status"] == "taken")
    missed  = sum(1 for m in SAMPLE_MEDICATIONS if m["status"] == "missed")
    pending = sum(1 for m in SAMPLE_MEDICATIONS if m["status"] == "pending")

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-tile"><div class="val" style="color:#22c55e;">{taken}</div><div class="lbl">Taken Today</div></div>
      <div class="kpi-tile"><div class="val" style="color:#ef4444;">{missed}</div><div class="lbl">Missed</div></div>
      <div class="kpi-tile"><div class="val" style="color:#00D4FF;">{pending}</div><div class="lbl">Pending</div></div>
      <div class="kpi-tile"><div class="val" style="color:#e2e8f0;">{taken/len(SAMPLE_MEDICATIONS)*100:.0f}%</div><div class="lbl">Today's Rate</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<div class="sec-header">💊 Medication Schedule</div>', unsafe_allow_html=True)
        for med in SAMPLE_MEDICATIONS:
            dot_color = "#22c55e" if med["status"]=="taken" else ("#ef4444" if med["status"]=="missed" else "#00D4FF")
            refill_warn = "🔴 " if med["days_left"]<=5 else ("🟡 " if med["days_left"]<=10 else "✅ ")
            st.markdown(f"""
            <div class="med-row">
              <div class="med-dot" style="background:{dot_color};width:12px;height:12px;"></div>
              <div style="flex:1;">
                <div class="med-name">{med['name']}</div>
                <div class="med-sub">⏰ {med['time']} · 🔁 {med['frequency']}</div>
              </div>
              <div style="text-align:right;">
                <div class="med-badge med-{med['status']}" style="margin-bottom:4px;">{med['status'].title()}</div>
                <div style="font-size:.72rem;color:#64748b;">{refill_warn}{med['days_left']} days left</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="sec-header">📅 Adherence History</div>', unsafe_allow_html=True)
        st.plotly_chart(adherence_timeline(30, 72.0), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="sec-header">⚠️ Refill Alerts</div>', unsafe_allow_html=True)
        for med in SAMPLE_MEDICATIONS:
            if med["days_left"] <= 10:
                lvl = "danger" if med["days_left"] <= 5 else "warning"
                st.markdown(f'<div class="alert-{lvl}">💊 <b>{med["name"]}</b> — refill in {med["days_left"]} days</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════
elif page == "analytics":
    st.markdown("""
    <div class="mg-hero">
      <h1>📊 Analytics</h1>
      <p>Population-level risk trends, feature correlations, and model performance.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Risk Overview", "🔍 Feature Analysis", "📉 Model Performance"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(population_distribution(pop_df), use_container_width=True, config={"displayModeBar": False})
        with col2:
            import plotly.express as px
            fig = px.histogram(pop_df, x="age", color="risk_label",
                color_discrete_map={0:"#22c55e",1:"#f59e0b",2:"#ef4444"},
                title="Age Distribution by Risk Class", nbins=25)
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color":"#e2e8f0"}, xaxis={"gridcolor":"#1e293b"}, yaxis={"gridcolor":"#1e293b"},
                legend={"bgcolor":"rgba(0,0,0,0)"}, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="sec-header">📐 Average Feature Values by Risk Class</div>', unsafe_allow_html=True)
        summary = pop_df.groupby("risk_label")[
            ["missed_doses_last_30d","previous_adherence_rate","depression_score","num_medications","days_since_last_refill"]
        ].mean().round(2)
        summary.index = ["Low Risk","Moderate Risk","High Risk"]
        st.dataframe(summary.style.background_gradient(cmap="RdYlGn_r", axis=0), use_container_width=True)

    with tab2:
        import plotly.express as px
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(pop_df.sample(300), x="missed_doses_last_30d", y="previous_adherence_rate",
                color="risk_label", color_discrete_map={0:"#22c55e",1:"#f59e0b",2:"#ef4444"},
                title="Missed Doses vs Adherence Rate", opacity=0.7)
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color":"#e2e8f0"}, xaxis={"gridcolor":"#1e293b"}, yaxis={"gridcolor":"#1e293b"},
                legend={"bgcolor":"rgba(0,0,0,0)"}, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            fig2 = px.box(pop_df, x="risk_label", y="depression_score", color="risk_label",
                color_discrete_map={0:"#22c55e",1:"#f59e0b",2:"#ef4444"},
                title="Depression Score by Risk Class")
            fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color":"#e2e8f0"}, xaxis={"gridcolor":"#1e293b"}, yaxis={"gridcolor":"#1e293b"},
                legend={"bgcolor":"rgba(0,0,0,0)"}, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="kpi-row">
              <div class="kpi-tile"><div class="val" style="color:#22c55e;">{metrics['accuracy']*100:.1f}%</div><div class="lbl">Accuracy</div></div>
              <div class="kpi-tile"><div class="val" style="color:#00D4FF;">{metrics['roc_auc']:.4f}</div><div class="lbl">ROC-AUC</div></div>
            </div>
            """, unsafe_allow_html=True)
            cv = metrics["cv_scores"]
            st.markdown(f"""
            <div class="mg-card">
              <div style="font-size:.75rem;color:#64748b;margin-bottom:10px;text-transform:uppercase;">5-Fold Cross Validation</div>
              {"".join([f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b;"><span style="color:#94a3b8;">Fold {i+1}</span><span style="color:#00D4FF;font-family:IBM Plex Mono;">{s*100:.2f}%</span></div>' for i,s in enumerate(cv)])}
              <div style="display:flex;justify-content:space-between;padding:8px 0 0 0;font-weight:600;">
                <span>Mean</span><span style="color:#22c55e;font-family:IBM Plex Mono;">{np.mean(cv)*100:.2f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            with st.expander("📄 Classification Report"):
                st.code(metrics["report"], language=None)


# ═════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ═════════════════════════════════════════════════════════════
elif page == "model_info":
    st.markdown("""
    <div class="mg-hero">
      <h1>⚙️ Model Architecture</h1>
      <p>Technical details, feature engineering, and training pipeline documentation.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="mg-card mg-card-accent">
          <div class="sec-header" style="margin-top:0;">🤖 Algorithm</div>
          <div style="line-height:1.9;color:#94a3b8;font-size:.9rem;">
            <div>• <b style="color:#e2e8f0;">Model:</b> Gradient Boosting Classifier</div>
            <div>• <b style="color:#e2e8f0;">Estimators:</b> 200 trees · Depth: 4</div>
            <div>• <b style="color:#e2e8f0;">Learning Rate:</b> 0.08 · Subsample: 80%</div>
            <div>• <b style="color:#e2e8f0;">Preprocessing:</b> StandardScaler</div>
            <div>• <b style="color:#e2e8f0;">Classes:</b> Low · Moderate · High Risk</div>
            <div>• <b style="color:#e2e8f0;">Serialization:</b> joblib + st.cache_resource</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="mg-card">
          <div class="sec-header" style="margin-top:0;">⚠️ Disclaimer</div>
          <div style="font-size:.82rem;color:#64748b;line-height:1.7;">
            MediGuard AI is a <b style="color:#f59e0b;">demonstration system</b> built on synthetic data.
            It is <b>not</b> validated for clinical use. Always consult a licensed healthcare provider.
          </div>
        </div>
        """, unsafe_allow_html=True)