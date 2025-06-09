# ssentinel_project_root/pages/01_chw_dashboard.py
import streamlit as st
import pandas as pd
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List

# --- Configuration and Centralized Module Imports ---
# SME NOTE: This is the corrected import block. It directly imports the
# new, correct functions from their final locations, resolving all `ImportError` issues.
try:
    from config import settings
    from data_processing.loaders import load_health_records
    from data_processing.helpers import hash_dataframe_safe
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, create_empty_figure
    
    # Directly import the new, centralized functions
    from analytics.alerting import generate_chw_patient_alerts
    
    # Import the CORRECT functions from the component files
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import get_chw_activity_trends
except ImportError as e:
    st.error(
        "FATAL IMPORT ERROR: A required application component could not be loaded.\n\n"
        f"**Details:**\n`{e}`\n\n"
        "Please ensure all component files (e.g., `analytics/alerting.py`, `pages/chw_components/*`) "
        "are correctly saved on the server and that all `__init__.py` files exist."
    )
    st.stop()

# --- Page Setup ---
logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

st.set_page_config(
    page_title=f"CHW Dashboard - {_get_setting('APP_NAME', 'Sentinel')}",
    page_icon="🧑‍🏫", layout="wide"
)

# --- Data Loading ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300), hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_dashboard_data() -> pd.DataFrame:
    """Loads and prepares the base health records once for the entire dashboard."""
    logger.info("CHW Dashboard: Loading base health records.")
    df = load_health_records(source_context="CHWDash/LoadBaseData")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    
    if 'encounter_date' in df.columns:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        df.dropna(subset=['encounter_date'], inplace=True)
    return df

# --- Main UI ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown("Team Performance Monitoring & Field Support")
st.divider()

all_data = get_dashboard_data()

# --- Sidebar UI ---
with st.sidebar:
    st.header("Dashboard Filters")
    if all_data.empty:
        st.warning("No data loaded. Filters are disabled.")
        active_chw, active_zone, daily_date, trend_start, trend_end = None, None, date.today(), date.today() - timedelta(days=29), date.today()
    else:
        chw_options = ["All CHWs"] + sorted(all_data['chw_id'].dropna().astype(str).unique())
        zone_options = ["All Zones"] + sorted(all_data['zone_id'].dropna().astype(str).unique())
        selected_chw = st.selectbox("Filter by CHW ID:", options=chw_options, key="chw_filter")
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options, key="zone_filter")
        min_date, max_date = all_data['encounter_date'].min().date(), all_data['encounter_date'].max().date()
        daily_date = st.date_input("View Daily Activity For:", value=max_date, min_value=min_date, max_value=max_date, key="daily_date_filter")
        default_trend_start = max(min_date, daily_date - timedelta(days=29))
        trend_range = st.date_input("Select Trend Date Range:", value=[default_trend_start, daily_date], min_value=min_date, max_value=max_date, key="trend_date_filter")
        active_chw = None if selected_chw == "All CHWs" else selected_chw
        active_zone = None if selected_zone == "All Zones" else selected_zone
        trend_start, trend_end = trend_range if len(trend_range) == 2 else (default_trend_start, daily_date)

# --- Filter Data ---
if all_data.empty:
    daily_df, trend_df = pd.DataFrame(), pd.DataFrame()
else:
    daily_mask = (all_data['encounter_date'].dt.date == daily_date)
    trend_mask = (all_data['encounter_date'].dt.date.between(trend_start, trend_end))
    if active_chw:
        daily_mask &= (all_data['chw_id'].astype(str) == active_chw)
        trend_mask &= (all_data['chw_id'].astype(str) == active_chw)
    if active_zone:
        daily_mask &= (all_data['zone_id'].astype(str) == active_zone)
        trend_mask &= (all_data['zone_id'].astype(str) == active_zone)
    daily_df = all_data[daily_mask]
    trend_df = all_data[trend_mask]

st.info(f"**Date:** {daily_date:%d %b %Y} | **CHW:** {active_chw or 'All'} | **Zone:** {active_zone or 'All'}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if daily_df.empty:
    st.markdown("ℹ️ No activity for the selected date and filters.")
else:
    summary_kpis = calculate_chw_daily_summary_metrics(daily_df)
    kpi_cols = st.columns(4)
    with kpi_cols[0]: render_kpi_card(title="Visits Today", value_str=str(summary_kpis.get("visits_count", 0)), icon="👥")
    with kpi_cols[1]: render_kpi_card(title="High Prio Follow-ups", value_str=str(summary_kpis.get("high_ai_prio_followups_count", 0)), icon="🎯")
    with kpi_cols[2]: render_kpi_card(title="Critical SpO2 Cases", value_str=str(summary_kpis.get("critical_spo2_cases_identified_count", 0)), icon="💨")
    with kpi_cols[3]: render_kpi_card(title="High Fever Cases", value_str=str(summary_kpis.get("high_fever_cases_identified_count", 0)), icon="🔥")
st.divider()

# --- Section 2: Key Alerts & Tasks ---
st.header("🚦 Key Alerts & Tasks")
if daily_df.empty:
    st.markdown("ℹ️ No activity data to generate alerts or tasks.")
else:
    chw_alerts = generate_chw_patient_alerts(patient_encounter_data_df=daily_df, for_date=daily_date)
    chw_tasks = generate_chw_tasks(daily_df, for_date=daily_date, chw_id=active_chw, zone_id=active_zone)
    
    alert_col, task_col = st.columns(2)
    with alert_col:
        st.subheader("🚨 Priority Patient Alerts")
        if chw_alerts:
            for alert in chw_alerts:
                level = alert.get('alert_level', 'INFO')
                icon = '🔴' if level == 'CRITICAL' else ('🟠' if level == 'WARNING' else 'ℹ️')
                with st.expander(f"{icon} {level}: {alert.get('primary_reason')} for Pt. {alert.get('patient_id')}", expanded=(level == 'CRITICAL')):
                    st.markdown(f"**Details:** {alert.get('brief_details')}")
                    st.markdown(f"**Context:** {alert.get('context_info')}")
        else:
            st.success("✅ No significant patient alerts.")
            
    with task_col:
        st.subheader("📋 Top Priority Tasks")
        if chw_tasks:
            tasks_df = pd.DataFrame(chw_tasks).sort_values(by='priority_score', ascending=False)
            for _, task in tasks_df.head(5).iterrows():
                st.info(f"**Task:** {task.get('task_description')} for Pt. {task.get('patient_id')}\n\n"
                        f"**Due:** {task.get('due_date')} | **Priority:** {task.get('priority_score', 0.0):.1f}")
        else:
            st.info("ℹ️ No high-priority tasks identified.")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if daily_df.empty:
    st.markdown("ℹ️ No activity data for local epi signals.")
else:
    epi_signals = extract_chw_epi_signals(for_date=daily_date, chw_daily_encounter_df=daily_df)
    if epi_signals and epi_signals.get("detected_symptom_clusters"):
        st.markdown("###### Detected Symptom Clusters (Requires Supervisor Verification):")
        for cluster in epi_signals["detected_symptom_clusters"]:
            st.warning(f"⚠️ **Pattern: {cluster.get('symptoms_pattern', 'Unknown')}** - {cluster.get('patient_count', 'N/A')} cases in area. Please verify.")
    else:
        st.info("No unusual symptom clusters detected today.")
st.divider()

# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
st.markdown(f"Displaying trends from **{trend_start:%d %b %Y}** to **{trend_end:%d %b %Y}**.")
if trend_df.empty:
    st.markdown("ℹ️ No historical data available for the selected trend period.")
else:
    activity_trends = get_chw_activity_trends(trend_df)
    
    cols = st.columns(2)
    with cols[0]:
        visits_trend = activity_trends.get("patient_visits_trend")
        if visits_trend is not None and not visits_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(visits_trend, "Daily Patient Visits Trend", "Unique Patients Visited"), use_container_width=True)
        else:
            st.altair_chart(create_empty_figure("Daily Patient Visits", "No trend data available"), use_container_width=True)
    with cols[1]:
        prio_trend = activity_trends.get("high_priority_followups_trend")
        if prio_trend is not None and not prio_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(prio_trend, "High Priority Follow-ups Trend", "High Prio. Patients"), use_container_width=True)
        else:
            st.altair_chart(create_empty_figure("High Priority Follow-ups", "No trend data available"), use_container_width=True)
st.divider()

st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
