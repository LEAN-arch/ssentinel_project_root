import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# --- Configuration and Custom Module Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records
    from data_processing.helpers import hash_dataframe_safe
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, create_empty_figure
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.alert_generation import generate_chw_alerts
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import calculate_chw_activity_trends_data
except ImportError as e_chw_dash_import:
    import sys
    # FIXED: Use the correct `__file__` magic variable.
    current_file_path = Path(__file__).resolve()
    project_root_dir = current_file_path.parent.parent
    error_message = (
        f"CHW Dashboard Import Error: {e_chw_dash_import}. "
        f"Ensure project root ('{project_root_dir}') is in sys.path (typically handled by app.py) "
        f"and all modules/packages (including pages.chw_components) have __init__.py files. "
        f"Check for typos in import paths or missing dependencies. "
        f"Current Python Path: {sys.path}"
    )
    # This block will attempt to display the error in Streamlit if it's running.
    st.error(error_message)
    st.stop()

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to get a setting with a fallback value."""
    return getattr(settings, attr_name, default_value)


# --- Page Configuration ---
try:
    page_icon_value = "🧑‍🏫"
    app_logo_small_path_str = _get_setting('APP_LOGO_SMALL_PATH', None)
    if app_logo_small_path_str:
        favicon_path = Path(app_logo_small_path_str)
        if favicon_path.is_file():
            page_icon_value = str(favicon_path)
        else:
            logger.warning(f"Favicon for CHW Dashboard not found at path from setting APP_LOGO_SMALL_PATH: {favicon_path}")
    
    page_layout_value = _get_setting('APP_LAYOUT', "wide")
    st.set_page_config(
        page_title=f"CHW Dashboard - {_get_setting('APP_NAME', 'Sentinel App')}",
        page_icon=page_icon_value, layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for CHW Dashboard: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="CHW Dashboard", page_icon="🧑‍🏫", layout="wide")


# --- Main Page UI ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"Team Performance Monitoring & Field Support - {_get_setting('APP_NAME', 'Sentinel Health Co-Pilot')}")
st.divider()


def _create_filter_options(
    df: Optional[pd.DataFrame], column_name: str, default_options: List[str], options_plural_name: str
) -> List[str]:
    """Dynamically creates filter options for a dropdown from a DataFrame."""
    options = [f"All {options_plural_name}"]
    if isinstance(df, pd.DataFrame) and not df.empty and column_name in df.columns:
        unique_values = sorted(list(set(str(val).strip() for val in df[column_name].dropna() if str(val).strip())))
        if unique_values:
            options.extend(unique_values)
        else:
            options.extend(default_options)
    else:
        options.extend(default_options)
    return options


@st.cache_data(
    ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300),
    show_spinner="Loading CHW operational data...", hash_funcs={pd.DataFrame: hash_dataframe_safe}
)
def load_chw_dashboard_data(
    view_date: date, trend_start_date: date, trend_end_date: date,
    chw_id_filter: Optional[str], zone_id_filter: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Loads, processes, and filters the necessary data for the dashboard."""
    log_context = "CHWDash/LoadDashboardData"
    logger.info(
        f"({log_context}) Loading data. View Date: {view_date}, "
        f"Trend Period: {trend_start_date} to {trend_end_date}, "
        f"CHW Filter: {chw_id_filter or 'All'}, Zone Filter: {zone_id_filter or 'All'}"
    )
    all_health_df = load_health_records(source_context=f"{log_context}/LoadHealthRecs")

    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        logger.error(f"({log_context}) CRITICAL: Health records failed to load or are empty.")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    if 'encounter_date' not in all_health_df.columns:
        logger.error(f"({log_context}) Critical: 'encounter_date' column missing in loaded health records.")
        return pd.DataFrame(), pd.DataFrame(), {}

    try:
        if not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
            all_health_df['encounter_date'] = pd.to_datetime(all_health_df['encounter_date'], errors='coerce')
        if all_health_df['encounter_date'].dt.tz is not None:
            all_health_df['encounter_date'] = all_health_df['encounter_date'].dt.tz_localize(None)
        if all_health_df['encounter_date'].isnull().all():
            logger.error(f"({log_context}) Critical: All 'encounter_date' values are null after conversion.")
            return pd.DataFrame(), pd.DataFrame(), {}
    except Exception as e_date_conv:
        logger.error(f"({log_context}) Error processing 'encounter_date': {e_date_conv}. Returning empty.", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), {}

    try:
        daily_df = all_health_df[all_health_df['encounter_date'].dt.date == view_date].copy()
        if chw_id_filter and 'chw_id' in daily_df.columns:
            daily_df = daily_df[daily_df['chw_id'].astype(str) == str(chw_id_filter)]
        if zone_id_filter and 'zone_id' in daily_df.columns:
            daily_df = daily_df[daily_df['zone_id'].astype(str) == str(zone_id_filter)]
    except Exception as e_daily_filter:
        logger.error(f"({log_context}) Error during daily data filtering: {e_daily_filter}", exc_info=True)
        daily_df = pd.DataFrame()

    try:
        trend_df = all_health_df[
            (all_health_df['encounter_date'].dt.date >= trend_start_date) &
            (all_health_df['encounter_date'].dt.date <= trend_end_date)
        ].copy()
        if chw_id_filter and 'chw_id' in trend_df.columns:
            trend_df = trend_df[trend_df['chw_id'].astype(str) == str(chw_id_filter)]
        if zone_id_filter and 'zone_id' in trend_df.columns:
            trend_df = trend_df[trend_df['zone_id'].astype(str) == str(zone_id_filter)]
    except Exception as e_trend_filter:
        logger.error(f"({log_context}) Error during trend data filtering: {e_trend_filter}", exc_info=True)
        trend_df = pd.DataFrame()

    pre_calculated_kpis: Dict[str, Any] = {}
    if chw_id_filter and not daily_df.empty:
        self_check_records = daily_df[
            (daily_df.get('chw_id', pd.Series(dtype=str)).astype(str) == str(chw_id_filter)) &
            (daily_df.get('encounter_type', pd.Series(dtype=str)).astype(str).str.contains('WORKER_SELF_CHECK', case=False, na=False))
        ]
        if not self_check_records.empty:
            fatigue_potential_cols = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            chosen_fatigue_col = next((col for col in fatigue_potential_cols if col in self_check_records.columns and self_check_records[col].notna().any()), None)
            if chosen_fatigue_col:
                try:
                    pre_calculated_kpis['worker_self_fatigue_index_today'] = float(self_check_records[chosen_fatigue_col].max())
                except (ValueError, TypeError):
                    logger.warning(f"({log_context}) Could not convert fatigue metric from '{chosen_fatigue_col}' to float.")

    logger.info(f"({log_context}) Data loaded. Daily records: {len(daily_df)}, Trend records: {len(trend_df)}.")
    return daily_df, trend_df, pre_calculated_kpis


@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300))
def load_data_for_filters() -> pd.DataFrame:
    """Loads a minimal, cached dataset just for populating filter dropdowns."""
    logger.info("CHW Page: Loading data for filter dropdowns.")
    df_filters = load_health_records(source_context="CHWDash/LoadFilterData")
    if not isinstance(df_filters, pd.DataFrame) or df_filters.empty:
        logger.warning("CHW Page: Failed to load data for filters. Dropdowns might be limited.")
        return pd.DataFrame()
    if 'encounter_date' in df_filters.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_filters['encounter_date']):
                df_filters['encounter_date'] = pd.to_datetime(df_filters['encounter_date'], errors='coerce')
            if df_filters['encounter_date'].dt.tz is not None:
                df_filters['encounter_date'] = df_filters['encounter_date'].dt.tz_localize(None)
        except Exception as e_filter_date:
            logger.warning(f"Error processing encounter_date in filter data: {e_filter_date}. May affect date pickers.")
    return df_filters


# --- Sidebar UI ---
with st.sidebar:
    st.markdown("---")
    try:
        project_root_dir_val = _get_setting('PROJECT_ROOT_DIR', '.')
        app_logo_path_val = _get_setting('APP_LOGO_SMALL_PATH', 'assets/logo_placeholder.png')
        logo_path_sidebar = Path(project_root_dir_val) / app_logo_path_val
        if logo_path_sidebar.is_file():
            st.image(str(logo_path_sidebar.resolve()), width=230)
        else:
            logger.warning(f"Sidebar logo for CHW Dashboard not found: {logo_path_sidebar.resolve()}")
            st.caption("Logo not found.")
    except Exception as e_logo_chw:
        logger.error(f"Unexpected error displaying CHW sidebar logo: {e_logo_chw}", exc_info=True)
        st.caption("Error loading logo.")

    st.markdown("---")
    st.header("Dashboard Filters")
    df_for_filters = load_data_for_filters()

    # CHW Filter
    chw_options = _create_filter_options(df_for_filters, 'chw_id', ["CHW001"], "CHWs")
    chw_ss_key = "chw_dashboard_selected_chw_id_v9"
    if chw_ss_key not in st.session_state or st.session_state[chw_ss_key] not in chw_options:
        st.session_state[chw_ss_key] = chw_options[0]
    selected_chw_ui = st.selectbox(
        "Filter by CHW ID:", options=chw_options,
        index=chw_options.index(st.session_state[chw_ss_key]),
        key=f"{chw_ss_key}_widget"
    )
    st.session_state[chw_ss_key] = selected_chw_ui
    active_chw_filter = None if selected_chw_ui.startswith("All ") else selected_chw_ui

    # Zone Filter
    zone_options = _create_filter_options(df_for_filters, 'zone_id', ["ZoneA"], "Zones")
    zone_ss_key = "chw_dashboard_selected_zone_id_v9"
    if zone_ss_key not in st.session_state or st.session_state[zone_ss_key] not in zone_options:
        st.session_state[zone_ss_key] = zone_options[0]
    selected_zone_ui = st.selectbox(
        "Filter by Zone:", options=zone_options,
        index=zone_options.index(st.session_state[zone_ss_key]),
        key=f"{zone_ss_key}_widget"
    )
    st.session_state[zone_ss_key] = selected_zone_ui
    active_zone_filter = None if selected_zone_ui.startswith("All ") else selected_zone_ui

    # Date Filters
    abs_min_fallback_date, abs_max_fallback_date = date(2022, 1, 1), date.today()
    abs_min_data_date, abs_max_data_date = abs_min_fallback_date, abs_max_fallback_date
    if isinstance(df_for_filters, pd.DataFrame) and 'encounter_date' in df_for_filters.columns and df_for_filters['encounter_date'].notna().any():
        try:
            min_from_data, max_from_data = df_for_filters['encounter_date'].min().date(), df_for_filters['encounter_date'].max().date()
            if min_from_data <= max_from_data:
                abs_min_data_date, abs_max_data_date = min_from_data, max_from_data
        except Exception as e_minmax_date:
            logger.warning(f"Error determining min/max dates from filter data: {e_minmax_date}")

    # Daily View Date Filter
    daily_date_ss_key = "chw_dashboard_daily_view_date_v9"
    if daily_date_ss_key not in st.session_state or not (abs_min_data_date <= st.session_state[daily_date_ss_key] <= abs_max_data_date):
        st.session_state[daily_date_ss_key] = abs_max_data_date
    selected_daily_date = st.date_input(
        "View Daily Activity For:", value=st.session_state[daily_date_ss_key],
        min_value=abs_min_data_date, max_value=abs_max_data_date, key=f"{daily_date_ss_key}_widget"
    )
    st.session_state[daily_date_ss_key] = selected_daily_date

    # Trend Date Range Filter
    trend_range_ss_key = "chw_dashboard_trend_date_range_v9"
    default_trend_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    default_trend_end = selected_daily_date
    default_trend_start = max(abs_min_data_date, default_trend_end - timedelta(days=default_trend_days - 1))
    if trend_range_ss_key not in st.session_state:
        st.session_state[trend_range_ss_key] = [default_trend_start, default_trend_end]
    
    selected_trend_range = st.date_input(
        "Select Trend Date Range:", value=st.session_state[trend_range_ss_key],
        min_value=abs_min_data_date, max_value=abs_max_data_date, key=f"{trend_range_ss_key}_widget"
    )
    
    # FIXED: Corrected indentation for the else clause.
    if isinstance(selected_trend_range, (list, tuple)) and len(selected_trend_range) == 2:
        trend_start_date_filter, trend_end_date_filter = selected_trend_range
        st.session_state[trend_range_ss_key] = [trend_start_date_filter, trend_end_date_filter]
    else:
        trend_start_date_filter, trend_end_date_filter = st.session_state[trend_range_ss_key]


# --- Load Data based on Filters ---
daily_activity_df, trend_activity_df, daily_kpis_precalculated = pd.DataFrame(), pd.DataFrame(), {}
data_load_successful = False
try:
    daily_activity_df, trend_activity_df, daily_kpis_precalculated = load_chw_dashboard_data(
        selected_daily_date, trend_start_date_filter, trend_end_date_filter,
        active_chw_filter, active_zone_filter
    )
    data_load_successful = True
except Exception as e_load_main_data:
    logger.error(f"CHW Dashboard: Main data loading/processing failed catastrophically: {e_load_main_data}", exc_info=True)
    st.error(f"🛑 Critical Error during data loading: {str(e_load_main_data)}. Dashboard may be incomplete.")

filter_context_parts = [f"Snapshot Date: {selected_daily_date.strftime('%d %b %Y')}"]
if active_chw_filter: filter_context_parts.append(f"CHW: {active_chw_filter}")
if active_zone_filter: filter_context_parts.append(f"Zone: {active_zone_filter}")
st.info(f"Displaying data for: {', '.join(filter_context_parts)}")


# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not data_load_successful or daily_activity_df.empty:
    st.markdown("ℹ️ No activity data for selected date/filters to display daily performance snapshot.")
else:
    try:
        daily_summary_metrics = calculate_chw_daily_summary_metrics(
            daily_activity_df, selected_daily_date, daily_kpis_precalculated, "CHWDash/DailySummary"
        )
        if daily_summary_metrics:
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                render_kpi_card(title="Visits Today", value_str=str(daily_summary_metrics.get("visits_count", 0)), icon="👥", help_text="Total unique patients encountered.")
            with kpi_cols[1]:
                prio_followups = daily_summary_metrics.get("high_ai_prio_followups_count", 0)
                prio_status = "ACCEPTABLE" if prio_followups <= 2 else ("MODERATE_CONCERN" if prio_followups <= 5 else "HIGH_CONCERN")
                render_kpi_card(title="High Prio Follow-ups", value_str=str(prio_followups), icon="🎯", status_level=prio_status, help_text=f"Patients needing urgent follow-up (AI prio score ≥ {_get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 80)}).")
            with kpi_cols[2]:
                critical_spo2 = daily_summary_metrics.get("critical_spo2_cases_identified_count", 0)
                spo2_status = "HIGH_CONCERN" if critical_spo2 > 0 else "ACCEPTABLE"
                render_kpi_card(title="Critical SpO2 Cases", value_str=str(critical_spo2), icon="💨", status_level=spo2_status, help_text=f"Patients with SpO2 < {_get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90)}%.")
            with kpi_cols[3]:
                high_fever = daily_summary_metrics.get("high_fever_cases_identified_count", 0)
                fever_status = "HIGH_CONCERN" if high_fever > 0 else "ACCEPTABLE"
                render_kpi_card(title="High Fever Cases", value_str=str(high_fever), icon="🔥", status_level=fever_status, help_text=f"Patients with body temp ≥ {_get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)}°C.")
        else:
            st.markdown("ℹ️ Could not generate daily performance snapshot with available data.")
    except Exception as e_daily_summary:
        logger.error(f"Error calculating CHW daily summary: {e_daily_summary}", exc_info=True)
        st.warning("⚠️ Could not calculate daily summary metrics.")
st.divider()


# --- Section 2: Key Alerts & Tasks ---
st.header("🚦 Key Alerts & Tasks")
if not data_load_successful or daily_activity_df.empty:
    st.markdown("ℹ️ No activity data to generate patient alerts or tasks for today.")
else:
    chw_alerts, chw_tasks = [], []
    try:
        chw_alerts = generate_chw_alerts(daily_activity_df, selected_daily_date, active_zone_filter or "All Zones", 15)
    except Exception as e_alerts:
        logger.error(f"CHW Dashboard: Error generating patient alerts: {e_alerts}", exc_info=True)
        st.warning("⚠️ Could not generate patient alerts for display.")
    try:
        chw_tasks = generate_chw_tasks(daily_activity_df, selected_daily_date, active_chw_filter, active_zone_filter or "All Zones", 20)
    except Exception as e_tasks:
        logger.error(f"CHW Dashboard: Error generating CHW tasks: {e_tasks}", exc_info=True)
        st.warning("⚠️ Could not generate tasks list for display.")

    if chw_alerts:
        st.subheader("🚨 Priority Patient Alerts (Today)")
        critical_alerts = [a for a in chw_alerts if a.get("alert_level") == "CRITICAL"]
        warning_alerts = [a for a in chw_alerts if a.get("alert_level") == "WARNING"]
        info_alerts = [a for a in chw_alerts if a.get("alert_level") == "INFO"]
        
        col1_alert_sum, col2_alert_sum, col3_alert_sum = st.columns(3)
        with col1_alert_sum: st.metric("Critical Alerts", len(critical_alerts))
        with col2_alert_sum: st.metric("Warning Alerts", len(warning_alerts))
        with col3_alert_sum: st.metric("Info Alerts", len(info_alerts))
        
        st.markdown("---")
        if critical_alerts:
            st.error("CRITICAL ALERTS - IMMEDIATE ATTENTION REQUIRED:")
            for alert in critical_alerts:
                with st.expander(f"🔴 CRITICAL: Pt. {alert.get('patient_id', 'N/A')} - {alert.get('primary_reason', 'Alert')}", expanded=True):
                    st.markdown(f"**Details:** {alert.get('brief_details', 'N/A')}\n\n**Context:** {alert.get('context_info', 'N/A')}\n\n**Action Code:** `{alert.get('suggested_action_code', 'REVIEW')}`")
        if warning_alerts:
            st.warning("WARNING ALERTS - ATTENTION ADVISED:")
            for alert in warning_alerts:
                with st.expander(f"🟠 WARNING: Pt. {alert.get('patient_id', 'N/A')} - {alert.get('primary_reason', 'Warning')}"):
                    st.markdown(f"**Details:** {alert.get('brief_details', 'N/A')}\n\n**Context:** {alert.get('context_info', 'N/A')}\n\n**Action Code:** `{alert.get('suggested_action_code', 'MONITOR')}`")
        if info_alerts and not (critical_alerts or warning_alerts):
            st.info("INFORMATIONAL ALERTS:")
            for alert in info_alerts:
                with st.expander(f"ℹ️ INFO: Pt. {alert.get('patient_id', 'N/A')} - {alert.get('primary_reason', 'Information')}"):
                     st.markdown(f"**Details:** {alert.get('brief_details', 'N/A')}\n\n**Context:** {alert.get('context_info', 'N/A')}")
    else:
        st.success("✅ No significant patient alerts needing immediate attention generated for today's selection.")
    
    st.markdown("---")
    if chw_tasks:
        st.subheader("📋 Top Priority Tasks (Today/Next Day)")
        tasks_df = pd.DataFrame(chw_tasks).sort_values(by=['priority_score', 'due_date'], ascending=[False, True])
        prio_threshold_high = _get_setting('TASK_PRIORITY_HIGH_THRESHOLD', 70)
        st.metric("High Priority Tasks", len(tasks_df[tasks_df['priority_score'] >= prio_threshold_high]), help=f"Tasks with priority score ≥ {prio_threshold_high}")
        st.markdown("---")
        for _, task in tasks_df.iterrows():
            prio_icon = '🔴' if task['priority_score'] >= 85 else ('🟠' if task['priority_score'] >= 60 else '🟢')
            with st.expander(f"{prio_icon} {task.get('task_description', 'N/A')} for Pt. {task.get('patient_id', 'N/A')}", expanded=(task['priority_score'] >= 85)):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Assigned CHW:** {task.get('assigned_chw_id', 'N/A')} | **Zone:** {task.get('zone_id', 'N/A')}")
                    st.markdown(f"**Patient Context:** {task.get('key_patient_context', 'N/A')}")
                with c2:
                    st.markdown(f"**Priority:** `{task.get('priority_score', 0.0):.1f}`")
                    st.markdown(f"**Due:** `{task.get('due_date', 'N/A')}`")
    else:
        st.info("ℹ️ No high-priority tasks identified based on current data.")
st.divider()


# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not data_load_successful or daily_activity_df.empty:
    st.markdown("ℹ️ No activity data for local epi signals for selected date/filters.")
else:
    try:
        epi_signals = extract_chw_epi_signals(
            for_date=selected_daily_date, chw_zone_context=active_zone_filter or "All Zones",
            chw_daily_encounter_df=daily_activity_df, pre_calculated_chw_kpis=daily_kpis_precalculated,
            max_symptom_clusters_to_report=3
        )
        if epi_signals:
            epi_kpi_cols = st.columns(3)
            with epi_kpi_cols[0]:
                render_kpi_card(title="Symptomatic (Key Cond.)", value_str=str(epi_signals.get("symptomatic_patients_key_conditions_count", "N/A")), icon="🤒", units="cases today", help_text="Patients with symptoms for key conditions.")
            with epi_kpi_cols[1]:
                new_malaria = epi_signals.get("newly_identified_malaria_patients_count", 0)
                render_kpi_card(title="New Malaria Cases", value_str=str(new_malaria), icon="🦟", status_level="HIGH_CONCERN" if new_malaria > 0 else "ACCEPTABLE", units="cases today", help_text="New malaria cases identified today.")
            with epi_kpi_cols[2]:
                pending_tb = epi_signals.get("pending_tb_contact_tracing_tasks_count", 0)
                render_kpi_card(title="Pending TB Contacts", value_str=str(pending_tb), icon="👥", status_level="MODERATE_CONCERN" if pending_tb > 0 else "ACCEPTABLE", units="to trace", help_text="TB contacts needing follow-up.")
            
            if epi_signals.get("detected_symptom_clusters"):
                st.markdown("###### Detected Symptom Clusters (Requires Supervisor Verification):")
                for cluster in epi_signals["detected_symptom_clusters"]:
                    st.warning(f"⚠️ **Pattern: {cluster.get('symptoms_pattern', 'Unknown')}** - {cluster.get('patient_count', 'N/A')} cases in {cluster.get('location_hint', 'CHW area')}. Please verify.")
    except Exception as e_epi:
        logger.error(f"CHW Dashboard: Error extracting epi signals: {e_epi}", exc_info=True)
        st.warning("⚠️ Could not extract epi signals for display.")
st.divider()


# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_str = f"{trend_start_date_filter.strftime('%d %b %Y')} - {trend_end_date_filter.strftime('%d %b %Y')}"
st.markdown(f"Displaying trends from {trend_period_str}.")
if not data_load_successful or trend_activity_df.empty:
    st.markdown("ℹ️ No historical data available for the selected trend period and/or filters.")
else:
    try:
        activity_trends_data = calculate_chw_activity_trends_data(
            trend_activity_df, trend_start_date_filter, trend_end_date_filter,
            active_zone_filter, time_period_aggregation='D'
        )
        if activity_trends_data:
            trend_plot_cols = st.columns(2)
            with trend_plot_cols[0]:
                visits_trend_series = activity_trends_data.get("patient_visits_trend")
                if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
                    st.plotly_chart(plot_annotated_line_chart(visits_trend_series, "Daily Patient Visits Trend", "Unique Patients Visited"), use_container_width=True)
                else:
                    st.caption("ℹ️ No patient visit trend data to display for this selection.")
            with trend_plot_cols[1]:
                prio_trend_series = activity_trends_data.get("high_priority_followups_trend")
                if isinstance(prio_trend_series, pd.Series) and not prio_trend_series.empty:
                    st.plotly_chart(plot_annotated_line_chart(prio_trend_series, "Daily High Prio. Follow-ups Trend", "High Prio. Follow-ups"), use_container_width=True)
                else:
                    st.caption("ℹ️ No high-priority follow-up trend data to display for this selection.")
    except Exception as e_trends_calc:
        logger.error(f"CHW Dashboard: CRITICAL Error calling calculate_chw_activity_trends_data: {e_trends_calc}", exc_info=True)
        st.error("⚠️ An error occurred while calculating activity trends.")
st.divider()

# --- Page Footer ---
st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
