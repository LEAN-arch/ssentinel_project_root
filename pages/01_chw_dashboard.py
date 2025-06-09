# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - FIELD OPERATIONS DASHBOARD

import logging
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from analytics import generate_chw_alerts
from config import settings
from data_processing import get_cached_trend, load_health_records
from visualization import (create_empty_figure, plot_line_chart,
                           render_kpi_card)

# --- Page Setup ---
st.set_page_config(page_title="Field Operations", page_icon="🧑‍⚕️", layout="wide")
logger = logging.getLogger(__name__)

# --- Data Loading ---
@st.cache_data(ttl=settings.WEB_CACHE_TTL_SECONDS)
def get_data() -> pd.DataFrame:
    """Loads and caches the base health records for the dashboard."""
    return load_health_records()

# --- Helper Functions ---
def get_summary_kpis(df: pd.DataFrame) -> dict:
    """Calculates summary KPIs from a daily filtered DataFrame."""
    if df.empty:
        return {"visits": 0, "high_prio": 0, "crit_spo2": 0, "high_fever": 0}
    
    return {
        "visits": df['patient_id'].nunique(),
        "high_prio": (df['ai_followup_priority_score'] >= 80).sum(),
        "crit_spo2": (df['min_spo2_pct'] < settings.ANALYTICS.spo2_critical_threshold_pct).sum(),
        "high_fever": (df['temperature'] >= settings.ANALYTICS.temp_high_fever_threshold_c).sum()
    }

def display_alerts(alerts: list):
    """Renders a list of patient alerts in a structured format."""
    st.subheader("🚨 Priority Patient Alerts")
    if not alerts:
        st.success("✅ No significant patient alerts for this selection.")
        return
    
    for alert in alerts:
        level = alert.get('alert_level', 'INFO')
        icon = '🔴' if level == 'CRITICAL' else '🟠' if level == 'WARNING' else 'ℹ️'
        expander_title = f"{icon} {level}: {alert.get('reason')} for Pt. {alert.get('patient_id')}"
        with st.expander(expander_title, expanded=(level == 'CRITICAL')):
            st.markdown(f"**Details:** {alert.get('details', 'N/A')}")
            st.markdown(f"**Context:** {alert.get('context', 'N/A')}")
            st.markdown(f"**Priority Score:** {alert.get('priority', 0):.0f}")

# --- Main Page ---
st.title("🧑‍⚕️ Field Operations Dashboard")
st.markdown("Monitor zone-level performance, patient risk signals, and field activity.")
st.divider()

full_df = get_data()

if full_df.empty:
    st.error("No health data available. Dashboard cannot be rendered.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")
    min_date, max_date = full_df['encounter_date'].min().date(), full_df['encounter_date'].max().date()
    
    zone_options = ["All Zones"] + sorted(full_df['zone_id'].dropna().unique())
    selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
    
    view_date = st.date_input("View Daily Activity For:", value=max_date, min_value=min_date, max_value=max_date)
    
    trend_start = st.date_input("Trend Start Date:", value=max(min_date, view_date - timedelta(days=29)))

# --- Data Filtering ---
daily_mask = (full_df['encounter_date'].dt.date == view_date)
trend_mask = (full_df['encounter_date'].dt.date.between(trend_start, view_date))
if selected_zone != "All Zones":
    zone_mask = (full_df['zone_id'] == selected_zone)
    daily_mask &= zone_mask
    trend_mask &= zone_mask
    
daily_df = full_df[daily_mask]
trend_df = full_df[trend_mask]

# --- UI Rendering ---
st.info(f"**Viewing:** {view_date:%A, %d %b %Y} | **Zone:** {selected_zone}")

# 1. Daily Performance Snapshot
st.header(f"📊 Daily Snapshot for {view_date:%d %b}")
if daily_df.empty:
    st.markdown("ℹ️ No activity recorded for the selected date and filters.")
else:
    kpis = get_summary_kpis(daily_df)
    cols = st.columns(4)
    with cols[0]: render_kpi_card("Visits Today", kpis["visits"], icon="👥", status_level="NEUTRAL")
    with cols[1]: render_kpi_card("High Priority Follow-ups", kpis["high_prio"], icon="🎯", status_level="MODERATE_CONCERN" if kpis["high_prio"] > 0 else "GOOD_PERFORMANCE")
    with cols[2]: render_kpi_card("Critical SpO2 Cases", kpis["crit_spo2"], icon="💨", status_level="HIGH_CONCERN" if kpis["crit_spo2"] > 0 else "GOOD_PERFORMANCE")
    with cols[3]: render_kpi_card("High Fever Cases", kpis["high_fever"], icon="🔥", status_level="HIGH_CONCERN" if kpis["high_fever"] > 0 else "GOOD_PERFORMANCE")

st.divider()

# 2. Key Alerts & Trends
col1, col2 = st.columns([1, 2])
with col1:
    patient_alerts = generate_chw_alerts(patient_df=daily_df)
    display_alerts(patient_alerts)

with col2:
    st.subheader("📈 Activity Trends")
    if not trend_df.empty:
        visits_trend = get_cached_trend(df=trend_df, value_col='patient_id', date_col='encounter_date', freq='D', agg_func='nunique')
        fig = plot_line_chart(visits_trend, title="Daily Unique Patient Visits", y_title="Unique Patients")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.altair_chart(create_empty_figure("Activity Trends"), use_container_width=True)

st.divider()
st.caption(settings.APP_FOOTER_TEXT)
