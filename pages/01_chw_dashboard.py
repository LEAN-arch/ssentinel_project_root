# ssentinel_project_root/pages/01_chw_dashboard.py
"""
CHW Supervisor Operations View for the Sentinel Health Co-Pilot.
This dashboard provides tools for team performance monitoring and field support.

SME NOTE: This is the complete, unabridged, and debugged version of the script.
It merges the necessary architectural fixes (imports, data loading, state management)
with the full set of original UI features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import sys

# --- Configuration and Custom Module Imports ---
# SME NOTE: This import block is now robust. It correctly imports from the new
# centralized analytics modules and the necessary component wrappers.
try:
    from config import settings
    from data_processing.loaders import load_health_records
    from data_processing.helpers import hash_dataframe_safe
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, create_empty_figure
    
    # Correctly import from the new, centralized analytics location
    from analytics.alerting import generate_chw_patient_alerts
    
    # Restore all necessary component imports
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import get_chw_activity_trends
except ImportError as e:
    # This robust error handling helps diagnose issues if the project structure changes.
    st.error(
        "A required application component could not be loaded. This is often due to a file being moved, renamed, or a missing `__init__.py` file in a directory.\n\n"
        f"**Details:**\n`{e}`\n\n"
        "Please check the project structure and import statements in `analytics/` and `pages/chw_components/`."
    )
    st.stop()

# --- Page Setup ---
logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to get a setting with a fallback value."""
    return getattr(settings, attr_name, default_value)

st.set_page_config(
    page_title=f"CHW Dashboard - {_get_setting('APP_NAME', 'Sentinel')}",
    page_icon="🧑‍🏫", layout="wide"
)

# --- Data Loading ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300), hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_dashboard_data() -> pd.DataFrame:
    """
    SME NOTE: SIMPLIFIED DATA LOADING.
    Loads and prepares the base health records once. Filtering is now handled
    in the main app logic, which is more efficient for user interactions.
    """
    logger.info("CHW Dashboard: Loading base health records for all operations.")
    df = load_health_records(source_context="CHWDash/LoadBaseData")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    
    if 'encounter_date' in df.columns:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        df.dropna(subset=['encounter_date'], inplace=True)
    return df

# --- Main Page UI ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown("Team Performance Monitoring & Field Support")
st.divider()

# Load the base data for the entire dashboard
all_data = get_dashboard_data()

# --- Sidebar UI ---
with st.sidebar:
    st.header("Dashboard Filters")
    
    if all_data.empty:
        st.warning("No data available to populate filters.")
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

# --- Filter Data based on UI selections ---
if all_data.empty:
    daily_df = pd.DataFrame()
    trend_df = pd.DataFrame()
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

filter_context_parts = [f"Snapshot Date: {daily_date.strftime('%d %b %Y')}"]
if active_chw: filter_context_parts.append(f"CHW: {active_chw}")
if active_zone: filter_context_parts.append(f"Zone: {active_zone}")
st.info(f"Displaying data for: {', '.join(filter_context_parts)}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if daily_df.empty:
    st.markdown("ℹ️ No activity for the selected date and filters.")
else:
    # This entire block is restored to its full, original functionality.
    daily_summary_metrics = calculate_chw_daily_summary_metrics(daily_df)
    kpi_cols = st.columns(4)
    with kpi_cols[0]: render_kpi_card(title="Visits Today", value_str=str(daily_summary_metrics.get("visits_count", 0)), icon="👥", help_text="Total unique patients encountered.")
    with kpi_cols[1]:
        prio_followups = daily_summary_metrics.get("high_ai_prio_followups_count", 0)
        prio_status = "ACCEPTABLE" if prio_followups <= 2 else ("MODERATE_CONCERN" if prio_followups <= 5 else "HIGH_CONCERN")
        render_kpi_card(title="High Prio Follow-ups", value_str=str(prio_followups), icon="🎯", status_level=prio_status, help_text=f"Patients needing urgent follow-up (AI prio score ≥ 80).")
    with kpi_cols[2]:
        critical_spo2 = daily_summary_metrics.get("critical_spo2_cases_identified_count", 0)
        spo2_status = "HIGH_CONCERN" if critical_spo2 > 0 else "ACCEPTABLE"
        render_kpi_card(title="Critical SpO2 Cases", value_str=str(critical_spo2), icon="💨", status_level=spo2_status, help_text=f"Patients with SpO2 < 90%.")
    with kpi_cols[3]:
        high_fever = daily_summary_metrics.get("high_fever_cases_identified_count", 0)
        fever_status = "HIGH_CONCERN" if high_fever > 0 else "ACCEPTABLE"
        render_kpi_card(title="High Fever Cases", value_str=str(high_fever), icon="🔥", status_level=fever_status, help_text=f"Patients with body temp ≥ 39.5°C.")
st.divider()

# --- Section 2: Key Alerts & Tasks ---
st.header("🚦 Key Alerts & Tasks")
if daily_df.empty:
    st.markdown("ℹ️ No activity data to generate alerts or tasks.")
else:
    # SME NOTE: This UI section is fully restored.
    chw_alerts = generate_chw_patient_alerts(patient_encounter_data_df=daily_df, for_date=daily_date, max_alerts=15)
    chw_tasks = generate_chw_tasks(daily_df, for_date=daily_date, chw_id=active_chw, zone_id=active_zone)

    if chw_alerts:
        st.subheader("🚨 Priority Patient Alerts (Today)")
        critical_alerts = [a for a in chw_alerts if a.get("alert_level") == "CRITICAL"]
        warning_alerts = [a for a in chw_alerts if a.get("alert_level") == "WARNING"]
        
        st.metric("Critical Alerts", len(critical_alerts))
        
        if critical_alerts:
            st.error("CRITICAL ALERTS - IMMEDIATE ATTENTION REQUIRED:")
            for alert in critical_alerts:
                with st.expander(f"🔴 CRITICAL: Pt. {alert.get('patient_id', 'N/A')} - {alert.get('primary_reason', 'Alert')}", expanded=True):
                    st.markdown(f"**Details:** {alert.get('brief_details', 'N/A')}\n\n**Context:** {alert.get('context_info', 'N/A')}")
        if warning_alerts:
            st.warning("WARNING ALERTS - ATTENTION ADVISED:")
            for alert in warning_alerts:
                with st.expander(f"🟠 WARNING: Pt. {alert.get('patient_id', 'N/A')} - {alert.get('primary_reason', 'Warning')}"):
                    st.markdown(f"**Details:** {alert.get('brief_details', 'N/A')}\n\n**Context:** {alert.get('context_info', 'N/A')}")
    else:
        st.success("✅ No significant patient alerts generated for today's selection.")
    
    st.markdown("---")
    if chw_tasks:
        st.subheader("📋 Top Priority Tasks (Today/Next Day)")
        tasks_df = pd.DataFrame(chw_tasks).sort_values(by=['priority_score', 'due_date'], ascending=[False, True])
        st.metric("High Priority Tasks", len(tasks_df[tasks_df['priority_score'] >= 70]))
        for _, task in tasks_df.head().iterrows():
            icon = '🔴' if task['priority_score'] >= 85 else ('🟠' if task['priority_score'] >= 60 else '🟢')
            with st.expander(f"{icon} {task.get('task_description', 'N/A')} for Pt. {task.get('patient_id', 'N/A')}", expanded=True):
                st.markdown(f"**Priority:** `{task.get('priority_score', 0.0):.1f}` | **Due:** `{task.get('due_date', 'N/A')}`")
    else:
        st.info("ℹ️ No high-priority tasks identified.")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if daily_df.empty:
    st.markdown("ℹ️ No activity data for local epi signals.")
else:
    # SME NOTE: This UI section is fully restored.
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
    st.markdown("ℹ️ No historical data available for the selected trend period and filters.")
else:
    activity_trends = get_chw_activity_trends(trend_df)
    cols = st.columns(2)
    with cols[0]:
        visits_trend = activity_trends.get("patient_visits_trend")
        if visits_trend is not None and not visits_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(visits_trend, "Daily Patient Visits Trend", "Unique Patients Visited"), use_container_width=True)
        else:
            st.altair_chart(create_empty_figure("Daily Patient Visits Trend"), use_container_width=True)
    with cols[1]:
        prio_trend = activity_trends.get("high_priority_followups_trend")
        if prio_trend is not None and not prio_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(prio_trend, "High Priority Follow-ups Trend", "High Prio. Patients"), use_container_width=True)
        else:
            st.altair_chart(create_empty_figure("High Priority Follow-ups Trend"), use_container_width=True)
st.divider()

# --- Page Footer ---
st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
