# sentinel_project_root/pages/01_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# --- Configuration and Custom Module Imports ---
try:
    # Sentinel Core Imports
    from config import settings
    from data_processing.loaders import load_health_records
    from data_processing.helpers import hash_dataframe_safe

    # Visualization and UI Element Imports
    from visualization.ui_elements import render_kpi_card, render_collapsible_item, render_progress_bar, render_status_indicator, load_custom_css
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart, create_empty_figure

    # Component-specific Logic Imports
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.alert_generation import generate_chw_alerts
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import calculate_chw_activity_trends_data
except ImportError as e:
    # Provides a more informative error message in the Streamlit app if modules are not found
    project_root = Path(__file__).resolve().parent.parent
    st.error(
        f"**Module Import Error:** {e}. "
        f"Please ensure the project is structured correctly and the root '{project_root}' is in `sys.path`. "
        "This is typically handled by `app.py`. Check for missing `__init__.py` files or installation issues."
    )
    st.stop()

logger = logging.getLogger(__name__)

# --- Constants ---
# Using constants for session state keys prevents typos and improves maintainability.
SS_KEY_CHW_ID = "chw_dashboard_selected_chw_id_v10"
SS_KEY_ZONE_ID = "chw_dashboard_selected_zone_id_v10"
SS_KEY_DAILY_DATE = "chw_dashboard_daily_view_date_v10"
SS_KEY_TREND_RANGE = "chw_dashboard_trend_date_range_v10"

# --- Page Configuration ---
def configure_page():
    """Sets the Streamlit page configuration."""
    page_icon_val = "🧑‍🏫"
    try:
        app_logo_small_path = getattr(settings, 'APP_LOGO_SMALL_PATH', None)
        if app_logo_small_path and Path(app_logo_small_path).is_file():
            page_icon_val = str(Path(app_logo_small_path).resolve())
        st.set_page_config(
            page_title=f"CHW Dashboard - {getattr(settings, 'APP_NAME', 'Sentinel')}",
            page_icon=page_icon_val,
            layout=getattr(settings, 'APP_LAYOUT', "wide")
        )
        # Load custom CSS styles for UI elements
        load_custom_css()
    except Exception as e:
        logger.error(f"Error applying page configuration: {e}", exc_info=True)
        st.set_page_config(page_title="CHW Dashboard", page_icon="🧑‍🏫", layout="wide")

configure_page()

# --- Data Loading Functions ---
@st.cache_data(ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 300))
def load_data_for_filters() -> pd.DataFrame:
    """
    Loads only the necessary columns for populating filter dropdowns.
    This is more memory-efficient than loading the entire health records CSV.
    """
    logger.info("CHW Page: Loading data for filter dropdowns.")
    try:
        # Optimization: Select only the columns needed for filters.
        filter_cols = ['chw_id', 'zone_id', 'encounter_date']
        df_filters = load_health_records(
            source_context="CHWDash/LoadFilterData",
            use_cols=filter_cols
        )
        if not isinstance(df_filters, pd.DataFrame) or df_filters.empty:
            logger.warning("CHW Page: Filter data is empty. Dropdowns may be limited.")
            return pd.DataFrame(columns=filter_cols)
        # Ensure encounter_date is datetime for min/max operations
        if 'encounter_date' in df_filters.columns and not pd.api.types.is_datetime64_any_dtype(df_filters['encounter_date']):
             df_filters['encounter_date'] = pd.to_datetime(df_filters['encounter_date'], errors='coerce')
        return df_filters
    except Exception as e:
        logger.error(f"Failed to load data for filters: {e}", exc_info=True)
        return pd.DataFrame()


@st.cache_data(
    ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 300),
    show_spinner="Loading CHW operational data...",
    hash_funcs={pd.DataFrame: hash_dataframe_safe}
)
def load_chw_dashboard_data(
    view_date: date, trend_start_date: date, trend_end_date: date,
    chw_id_filter: Optional[str], zone_id_filter: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and filters the main health records for the dashboard based on user selections.
    Separates data into a daily snapshot and a trend period.
    The logic for pre-calculating KPIs was moved to the respective analytics
    modules for better separation of concerns.
    """
    log_context = f"CHWDash/LoadData(date={view_date}, chw={chw_id_filter or 'All'})"
    logger.info(f"Executing data load for CHW Dashboard: {log_context}")
    all_health_df = load_health_records(source_context=log_context)

    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        st.error("Critical: Health records data could not be loaded. Dashboard cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()

    if 'encounter_date' not in all_health_df.columns:
        st.error("Critical: 'encounter_date' column missing in data. Dashboard cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Ensure encounter_date is timezone-naive datetime for consistent comparison
    if not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
        all_health_df['encounter_date'] = pd.to_datetime(all_health_df['encounter_date'], errors='coerce')
    if all_health_df['encounter_date'].dt.tz is not None:
        all_health_df['encounter_date'] = all_health_df['encounter_date'].dt.tz_localize(None)

    # Filter for daily activity
    daily_mask = (all_health_df['encounter_date'].dt.date == view_date)
    if chw_id_filter:
        daily_mask &= (all_health_df['chw_id'].astype(str) == str(chw_id_filter))
    if zone_id_filter:
        daily_mask &= (all_health_df['zone_id'].astype(str) == str(zone_id_filter))
    daily_df = all_health_df[daily_mask].copy()

    # Filter for trend period
    trend_mask = (all_health_df['encounter_date'].dt.date >= trend_start_date) & \
                 (all_health_df['encounter_date'].dt.date <= trend_end_date)
    if chw_id_filter:
        trend_mask &= (all_health_df['chw_id'].astype(str) == str(chw_id_filter))
    if zone_id_filter:
        trend_mask &= (all_health_df['zone_id'].astype(str) == str(zone_id_filter))
    trend_df = all_health_df[trend_mask].copy()

    logger.info(f"Data loaded. Daily records: {len(daily_df)}, Trend records: {len(trend_df)}.")
    return daily_df, trend_df

# --- Sidebar and Filters ---
def setup_sidebar(df_filters: pd.DataFrame):
    """Configures and displays the sidebar with all dashboard filters."""
    st.sidebar.markdown("---")
    try:
        logo_path = getattr(settings, 'APP_LOGO_SMALL_PATH', None)
        if logo_path and Path(logo_path).is_file():
            st.sidebar.image(str(Path(logo_path).resolve()), width=230)
        st.sidebar.markdown("---")
    except Exception as e:
        logger.warning(f"Could not display sidebar logo: {e}")

    st.sidebar.header("Dashboard Filters")

    # Filter options generation
    chw_options = ["All CHWs"] + sorted(df_filters['chw_id'].dropna().astype(str).unique()) if 'chw_id' in df_filters.columns else ["All CHWs"]
    zone_options = ["All Zones"] + sorted(df_filters['zone_id'].dropna().astype(str).unique()) if 'zone_id' in df_filters.columns else ["All Zones"]

    # Date range from data
    min_date, max_date = date(2022, 1, 1), date.today()
    if 'encounter_date' in df_filters.columns and df_filters['encounter_date'].notna().any():
        min_date_from_data = df_filters['encounter_date'].min().date()
        max_date_from_data = df_filters['encounter_date'].max().date()
        if min_date_from_data and max_date_from_data and min_date_from_data <= max_date_from_data:
            min_date, max_date = min_date_from_data, max_date_from_data


    # --- Filter Widgets ---
    # CHW ID Filter
    selected_chw_ui = st.sidebar.selectbox("Filter by CHW ID:", options=chw_options, key=SS_KEY_CHW_ID)
    active_chw_filter = None if selected_chw_ui == "All CHWs" else selected_chw_ui

    # Zone Filter
    selected_zone_ui = st.sidebar.selectbox("Filter by Zone:", options=zone_options, key=SS_KEY_ZONE_ID)
    active_zone_filter = None if selected_zone_ui == "All Zones" else selected_zone_ui

    # Daily Snapshot Date
    selected_daily_date = st.sidebar.date_input("View Daily Activity For:", value=max_date, min_value=min_date, max_value=max_date, key=SS_KEY_DAILY_DATE)

    # Trend Date Range
    default_trend_start = max(min_date, selected_daily_date - timedelta(days=29))
    selected_trend_range = st.sidebar.date_input("Select Trend Date Range:", value=(default_trend_start, selected_daily_date), min_value=min_date, max_value=max_date, key=SS_KEY_TREND_RANGE)

    trend_start, trend_end = (default_trend_start, selected_daily_date)
    if isinstance(selected_trend_range, (tuple, list)) and len(selected_trend_range) == 2:
        trend_start, trend_end = selected_trend_range
    else: # Handle case where user deselects one end of the range
        st.session_state[SS_KEY_TREND_RANGE] = (default_trend_start, selected_daily_date)


    return active_chw_filter, active_zone_filter, selected_daily_date, trend_start, trend_end

# --- Main Dashboard Rendering ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support** | {getattr(settings, 'APP_NAME', 'Sentinel')}")
st.divider()

# --- Load data and setup filters ---
df_for_filters = load_data_for_filters()
active_chw, active_zone, daily_date, trend_start, trend_end = setup_sidebar(df_for_filters)

# Display context message
filter_context_parts = [f"Snapshot Date: **{daily_date.strftime('%d %b %Y')}**"]
if active_chw: filter_context_parts.append(f"CHW: **{active_chw}**")
if active_zone: filter_context_parts.append(f"Zone: **{active_zone}**")
st.info(f"Displaying data for: {', '.join(filter_context_parts)}")

# --- Load main data based on filters ---
try:
    daily_activity_df, trend_activity_df = load_chw_dashboard_data(daily_date, trend_start, trend_end, active_chw, active_zone)
    data_load_successful = True
except Exception as e:
    data_load_successful = False
    logger.error(f"Main data loading failed catastrophically: {e}", exc_info=True)
    st.error(f"🛑 Critical Error during data loading: {e}. Dashboard may be incomplete.")


# ===== Main Layout with Tabs =====
if data_load_successful:
    tab1, tab2 = st.tabs(["**Today's Snapshot**", "**Trends & Analytics**"])

    # --- TAB 1: Today's Snapshot ---
    with tab1:
        if daily_activity_df.empty:
            st.markdown("ℹ️ _No activity data found for the selected date and filters._")
        else:
            # --- Calculate Daily Metrics ---
            daily_summary_metrics = calculate_chw_daily_summary_metrics(daily_activity_df, daily_date)
            chw_alerts = generate_chw_alerts(daily_activity_df, daily_date, active_zone or "All Zones")
            chw_tasks = generate_chw_tasks(daily_activity_df, daily_date, active_chw, active_zone or "All Zones")
            epi_signals = extract_chw_epi_signals(daily_date, active_zone or "All Zones", daily_activity_df)

            # --- Section 1: Top-Level KPIs ---
            st.subheader("📊 Daily Performance Snapshot")
            kpi_cols = st.columns(6)
            with kpi_cols[0]:
                render_kpi_card("Visits Today", daily_summary_metrics.get("visits_count", 0), icon="👥", value_is_count=True, help_text="Total unique patients encountered today.")
            with kpi_cols[1]:
                prio_followups = daily_summary_metrics.get("high_ai_prio_followups_count", 0)
                p_status = "HIGH_CONCERN" if prio_followups > 5 else "MODERATE_CONCERN" if prio_followups > 2 else "ACCEPTABLE"
                render_kpi_card("High Prio Follow-ups", prio_followups, icon="🎯", status=p_status, value_is_count=True, help_text=f"Patients needing urgent follow-up (AI Prio ≥ {getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80)}).")
            with kpi_cols[2]:
                critical_spo2 = daily_summary_metrics.get("critical_spo2_cases_identified_count", 0)
                s_status = "HIGH_CONCERN" if critical_spo2 > 0 else "ACCEPTABLE"
                render_kpi_card("Critical SpO2 Cases", critical_spo2, icon="💨", status=s_status, value_is_count=True, help_text=f"Patients with SpO2 < {getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90)}%.")
            with kpi_cols[3]:
                high_fever = daily_summary_metrics.get("high_fever_cases_identified_count", 0)
                f_status = "HIGH_CONCERN" if high_fever > 0 else "ACCEPTABLE"
                render_kpi_card("High Fever Cases", high_fever, icon="🔥", status=f_status, value_is_count=True, help_text=f"Patients with body temp ≥ {getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)}°C.")
            with kpi_cols[4]:
                avg_risk = daily_summary_metrics.get("avg_risk_of_visited_patients")
                r_status = "HIGH_CONCERN" if avg_risk and avg_risk > 75 else "MODERATE_CONCERN" if avg_risk and avg_risk > 60 else "ACCEPTABLE"
                render_kpi_card("Avg. Patient Risk", avg_risk, icon="🛡️", status=r_status, help_text="Average AI risk score of all patients visited today.")
            with kpi_cols[5]:
                latency = daily_summary_metrics.get("avg_data_sync_latency_hours")
                l_status = "HIGH_CONCERN" if latency and latency > 24 else "MODERATE_CONCERN" if latency and latency > 8 else "ACCEPTABLE"
                render_kpi_card("Data Sync Latency", latency, icon="📡", units="hrs", status=l_status, delta_is_inverted=True, help_text="Average time since last data sync from CHW devices.")
            st.divider()

            col_main, col_sidebar = st.columns([3, 1.5], gap="large")

            with col_main:
                # --- Section 2: Alerts and Tasks ---
                st.subheader("🚦 Key Alerts & Tasks")
                if not chw_alerts and not chw_tasks:
                    st.success("✅ No new high-priority alerts or tasks for today.")
                
                if chw_alerts:
                    st.markdown("**Priority Patient Alerts**")
                    for alert in chw_alerts:
                        alert_details = {
                            "Patient ID": alert.get('patient_id', 'N/A'),
                            "Details": alert.get('brief_details', 'N/A'),
                            "Context": alert.get('context_info', 'N/A'),
                            "Action Code": alert.get('suggested_action_code', 'REVIEW')
                        }
                        render_collapsible_item(f"{alert.get('primary_reason', 'Alert')}", alert.get('alert_level', 'INFO'), alert_details, is_expanded=(alert.get("alert_level") == "CRITICAL"))
                
                if chw_tasks:
                    st.markdown("**Top Priority Tasks**")
                    for task in chw_tasks:
                        prio = task.get('priority_score', 0)
                        task_level = "TASK_HIGH" if prio >= 80 else "TASK_MEDIUM" if prio >= 60 else "TASK_LOW"
                        task_details = {
                            "Patient ID": task.get('patient_id', 'N/A'),
                            "Assigned CHW": task.get('assigned_chw_id', 'N/A'),
                            "Due Date": task.get('due_date', 'N/A'),
                            "Status": task.get('status', 'PENDING'),
                            "Context": task.get('key_patient_context', 'N/A')
                        }
                        render_collapsible_item(task.get('task_description', 'Task'), task_level, task_details, is_expanded=(prio>=85))
            
            with col_sidebar:
                # --- Section 3: CHW Wellness & Device ---
                st.subheader("❤️‍🩹 CHW Wellness & Device")
                fatigue_score = daily_summary_metrics.get('worker_self_fatigue_index_today')
                fatigue_status = "HIGH_CONCERN" if fatigue_score and fatigue_score >= 80 else "MODERATE_CONCERN" if fatigue_score and fatigue_score >= 60 else "ACCEPTABLE"
                render_progress_bar("CHW Fatigue Index", fatigue_score, status=fatigue_status, display_text=f"{fatigue_score or 0:.0f} / 100")
                
                st.metric("Steps Today", f"{daily_summary_metrics.get('chw_steps_today', 0):,}", help="Total steps recorded by the CHW's device today.")
                st.metric("Device Battery", f"{daily_summary_metrics.get('device_battery_pct', 0)}%", help="Last reported device battery level.")
                st.divider()

                # --- Section 4: Local Epi Signals ---
                st.subheader("🔬 Local Epi Signals")
                clusters = epi_signals.get("detected_symptom_clusters", [])
                if clusters:
                    cluster_df = pd.DataFrame(clusters).rename(columns={'symptoms_pattern': 'Symptom Cluster', 'patient_count': 'Cases'})
                    fig_symptoms = plot_bar_chart(cluster_df, x_col_name='Symptom Cluster', y_col_name='Cases', chart_title="", y_values_are_counts_flag=True, height=250)
                    fig_symptoms.update_layout(title_text="", margin=dict(t=20, l=40, b=20, r=20), yaxis_title="Cases", showlegend=False)
                    st.plotly_chart(fig_symptoms, use_container_width=True)
                else:
                    st.info("No significant symptom clusters detected today.")

                new_malaria = epi_signals.get("newly_identified_malaria_patients_count", 0)
                if new_malaria > 0:
                    render_status_indicator(f"{new_malaria} New Malaria Case(s)", status="MODERATE_CONCERN", icon="🦟")

    # --- TAB 2: Trends & Analytics ---
    with tab2:
        st.subheader("📈 Activity & Performance Trends")
        trend_period_str = f"{trend_start.strftime('%d %b %Y')} to {trend_end.strftime('%d %b %Y')}"
        st.markdown(f"Displaying trends from **{trend_period_str}**.")

        if trend_activity_df.empty:
            st.markdown("ℹ️ _No historical data available for the selected trend period and filters._")
        else:
            activity_trends = calculate_chw_activity_trends_data(trend_activity_df, trend_start, trend_end, active_zone)

            col1, col2 = st.columns(2)
            with col1:
                visits_trend = activity_trends.get("patient_visits_trend")
                if isinstance(visits_trend, pd.Series) and not visits_trend.empty:
                    fig_visits = plot_annotated_line_chart(visits_trend, "Daily Patient Visits", "Unique Patients", y_values_are_counts=True)
                    st.plotly_chart(fig_visits, use_container_width=True)
                else:
                    st.plotly_chart(create_empty_figure("Daily Patient Visits"), use_container_width=True)
            
            with col2:
                prio_trend = activity_trends.get("high_priority_followups_trend")
                if isinstance(prio_trend, pd.Series) and not prio_trend.empty:
                    fig_prio = plot_annotated_line_chart(prio_trend, "High Priority Follow-ups", "High Prio Patients", y_values_are_counts=True)
                    st.plotly_chart(fig_prio, use_container_width=True)
                else:
                    st.plotly_chart(create_empty_figure("High Priority Follow-ups"), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                workload_trend = activity_trends.get("new_high_priority_tasks_trend")
                if isinstance(workload_trend, pd.Series) and not workload_trend.empty:
                    fig_workload = plot_annotated_line_chart(workload_trend, "CHW Workload Trend", "New High-Prio Tasks", y_values_are_counts=True)
                    st.plotly_chart(fig_workload, use_container_width=True)
                else:
                    st.plotly_chart(create_empty_figure("CHW Workload Trend"), use_container_width=True)

            with col4:
                acuity_trend = activity_trends.get("patient_acuity_trend")
                if isinstance(acuity_trend, pd.Series) and not acuity_trend.empty:
                    fig_acuity = plot_annotated_line_chart(acuity_trend, "Patient Acuity Trend", "Avg. AI Risk Score")
                    st.plotly_chart(fig_acuity, use_container_width=True)
                else:
                    st.plotly_chart(create_empty_figure("Patient Acuity Trend"), use_container_width=True)

else:
    st.warning("Data could not be loaded. Please check the data source and application logs.")

st.divider()
st.caption(f"© {date.today().year} {getattr(settings, 'ORGANIZATION_NAME', 'Sentinel')}. All rights reserved.")
