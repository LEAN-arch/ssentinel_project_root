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
    current_file_path = Path(__file__).resolve()
    project_root_dir = current_file_path.parent.parent 
    error_message = (
        f"CHW Dashboard Import Error: {e_chw_dash_import}. "
        f"Ensure project root ('{project_root_dir}') is in sys.path (typically handled by app.py) "
        f"and all modules/packages (including `pages.chw_components`) have `__init__.py` files. "
        f"Check for typos in import paths or missing dependencies. "
        f"Current Python Path: {sys.path}"
    )
    try:
        st.error(error_message)
        st.stop()
    except NameError: 
        print(error_message, file=sys.stderr)
        raise

logger = logging.getLogger(__name__)

try:
    page_icon_value = "🧑‍🏫" 
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_FAVICON_PATH'):
        favicon_path = Path(settings.PROJECT_ROOT_DIR) / settings.APP_FAVICON_PATH
        if favicon_path.is_file(): page_icon_value = str(favicon_path)
        else: logger.warning(f"Favicon for CHW Dashboard not found: {favicon_path}")
    page_layout_value = "wide" 
    if hasattr(settings, 'APP_LAYOUT'): page_layout_value = settings.APP_LAYOUT
    st.set_page_config(
        page_title=f"CHW Dashboard - {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'App'}",
        page_icon=page_icon_value, layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for CHW Dashboard: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="CHW Dashboard", page_icon="🧑‍🏫", layout="wide")

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'Sentinel Health Co-Pilot'}**")
st.divider()

def _create_filter_options(
    df: Optional[pd.DataFrame], column_name: str, default_options: List[str], options_plural_name: str
) -> List[str]:
    options = [f"All {options_plural_name}"]
    if isinstance(df, pd.DataFrame) and not df.empty and column_name in df.columns:
        unique_values = sorted(list(set(str(val).strip() for val in df[column_name].dropna() if str(val).strip())))
        if unique_values: options.extend(unique_values)
        else:
            logger.warning(f"CHW Filters: Column '{column_name}' for '{options_plural_name}' is empty. Using defaults.")
            options.extend(default_options)
    else:
        logger.warning(f"CHW Filters: Column '{column_name}' not in DF or DF empty for '{options_plural_name}'. Using defaults.")
        options.extend(default_options)
    return options

@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS if hasattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS') else 300,
    show_spinner="Loading CHW operational data...", hash_funcs={pd.DataFrame: hash_dataframe_safe}
)
def load_chw_dashboard_data(
    view_date: date, trend_start_date: date, trend_end_date: date,
    chw_id_filter: Optional[str], zone_id_filter: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_context = "CHWDash/LoadDashboardData"
    logger.info(
        f"({log_context}) Loading data. View Date: {view_date}, "
        f"Trend Period: {trend_start_date} to {trend_end_date}, "
        f"CHW Filter: {chw_id_filter or 'All'}, Zone Filter: {zone_id_filter or 'All'}"
    )
    all_health_df = load_health_records(source_context=f"{log_context}/LoadHealthRecs")

    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        csv_path_setting = settings.HEALTH_RECORDS_CSV_PATH if hasattr(settings, 'HEALTH_RECORDS_CSV_PATH') else "health_records_expanded.csv"
        data_dir_setting = settings.DATA_DIR if hasattr(settings, 'DATA_DIR') else "data_sources"
        expected_filename = Path(csv_path_setting).name 
        full_expected_path = Path(data_dir_setting) / expected_filename
        logger.error(f"({log_context}) CRITICAL: Health records failed to load or are empty. Expected at: {full_expected_path}")
        st.error(
            f"CRITICAL DATA ERROR: Could not load health records. Ensure '{expected_filename}' is in '{Path(data_dir_setting).resolve()}' and is not empty. Dashboard functionality will be severely limited."
        )
        return pd.DataFrame(), pd.DataFrame(), {}

    if 'encounter_date' not in all_health_df.columns:
        logger.error(f"({log_context}) 'encounter_date' column missing in loaded health records.")
        st.error("Data Integrity Error: 'encounter_date' column is missing.")
        return pd.DataFrame(), pd.DataFrame(), {}

    if not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
        all_health_df['encounter_date'] = pd.to_datetime(all_health_df['encounter_date'], errors='coerce')
    if all_health_df['encounter_date'].dt.tz is not None:
        all_health_df['encounter_date'] = all_health_df['encounter_date'].dt.tz_localize(None)
    if all_health_df['encounter_date'].isnull().all():
        logger.error(f"({log_context}) All 'encounter_date' values are null after conversion.")
        st.error("Data Integrity Error: All 'encounter_date' values are invalid.")
        return pd.DataFrame(), pd.DataFrame(), {}

    daily_df = all_health_df[all_health_df['encounter_date'].dt.date == view_date].copy()
    if chw_id_filter and 'chw_id' in daily_df.columns:
        daily_df = daily_df[daily_df['chw_id'] == chw_id_filter]
    if zone_id_filter and 'zone_id' in daily_df.columns:
        daily_df = daily_df[daily_df['zone_id'] == zone_id_filter]

    trend_df = all_health_df[
        (all_health_df['encounter_date'].dt.date >= trend_start_date) &
        (all_health_df['encounter_date'].dt.date <= trend_end_date)
    ].copy()
    if chw_id_filter and 'chw_id' in trend_df.columns:
        trend_df = trend_df[trend_df['chw_id'] == chw_id_filter]
    if zone_id_filter and 'zone_id' in trend_df.columns:
        trend_df = trend_df[trend_df['zone_id'] == zone_id_filter]

    pre_calculated_kpis: Dict[str, Any] = {}
    if chw_id_filter and not daily_df.empty: 
        chw_id_col_data = daily_df.get('chw_id', pd.Series(dtype=str))
        encounter_type_col_data = daily_df.get('encounter_type', pd.Series(dtype=str)).astype(str) 
        self_check_records = daily_df[
            (chw_id_col_data == chw_id_filter) &
            (encounter_type_col_data.str.contains('WORKER_SELF_CHECK', case=False, na=False)) 
        ]
        if not self_check_records.empty:
            fatigue_potential_cols = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            chosen_fatigue_col = next(
                (col for col in fatigue_potential_cols if col in self_check_records.columns and self_check_records[col].notna().any()), None
            )
            if chosen_fatigue_col:
                pre_calculated_kpis['worker_self_fatigue_index_today'] = self_check_records[chosen_fatigue_col].max()
            else:
                pre_calculated_kpis['worker_self_fatigue_index_today'] = np.nan
                logger.debug(f"({log_context}) No suitable fatigue column found for CHW {chw_id_filter}.")
        else:
            pre_calculated_kpis['worker_self_fatigue_index_today'] = np.nan
            logger.debug(f"({log_context}) No self-check records for CHW {chw_id_filter} on {view_date}.")
    
    logger.info(f"({log_context}) Data loaded. Daily records: {len(daily_df)}, Trend records: {len(trend_df)}.")
    return daily_df, trend_df, pre_calculated_kpis

@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS if hasattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS') else 300
)
def load_data_for_filters() -> pd.DataFrame:
    logger.info("CHW Page: Loading data for filter dropdowns.")
    df_filters = load_health_records(source_context="CHWDash/LoadFilterData")
    if not isinstance(df_filters, pd.DataFrame) or df_filters.empty:
        logger.warning("CHW Page: Failed to load data for filters.")
        return pd.DataFrame()
    if 'encounter_date' in df_filters.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_filters['encounter_date']):
            df_filters['encounter_date'] = pd.to_datetime(df_filters['encounter_date'], errors='coerce')
        if df_filters['encounter_date'].dt.tz is not None:
            df_filters['encounter_date'] = df_filters['encounter_date'].dt.tz_localize(None)
    return df_filters

st.sidebar.markdown("---") 
try:
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_LOGO_SMALL_PATH'):
        project_root_path = Path(settings.PROJECT_ROOT_DIR)
        logo_path_sidebar = project_root_path / settings.APP_LOGO_SMALL_PATH
        if logo_path_sidebar.is_file(): st.sidebar.image(str(logo_path_sidebar.resolve()), width=230)
        else:
            logger.warning(f"Sidebar logo for CHW Dashboard not found: {logo_path_sidebar.resolve()}")
            st.sidebar.caption("Logo not found.")
    else:
        logger.warning("Settings for CHW sidebar logo missing.")
        st.sidebar.caption("Logo config missing.")
except Exception as e_logo_chw: 
    logger.error(f"Unexpected error displaying CHW sidebar logo: {e_logo_chw}", exc_info=True)
    st.sidebar.caption("Error loading logo.")
st.sidebar.markdown("---") 
st.sidebar.header("Dashboard Filters")

df_for_filters = load_data_for_filters()

chw_options = _create_filter_options(df_for_filters, 'chw_id', ["CHW001", "CHW002"], "CHWs")
chw_ss_key = "chw_dashboard_selected_chw_id_v8" 
if chw_ss_key not in st.session_state or st.session_state[chw_ss_key] not in chw_options:
    st.session_state[chw_ss_key] = chw_options[0] 
selected_chw_ui = st.sidebar.selectbox(
    "Filter by CHW ID:", options=chw_options,
    index=chw_options.index(st.session_state[chw_ss_key]) if st.session_state[chw_ss_key] in chw_options else 0,
    key=f"{chw_ss_key}_widget"
)
st.session_state[chw_ss_key] = selected_chw_ui
active_chw_filter = None if selected_chw_ui.startswith("All ") else selected_chw_ui

zone_options = _create_filter_options(df_for_filters, 'zone_id', ["ZoneA", "ZoneB"], "Zones")
zone_ss_key = "chw_dashboard_selected_zone_id_v8"
if zone_ss_key not in st.session_state or st.session_state[zone_ss_key] not in zone_options:
    st.session_state[zone_ss_key] = zone_options[0]
selected_zone_ui = st.sidebar.selectbox(
    "Filter by Zone:", options=zone_options,
    index=zone_options.index(st.session_state[zone_ss_key]) if st.session_state[zone_ss_key] in zone_options else 0,
    key=f"{zone_ss_key}_widget"
)
st.session_state[zone_ss_key] = selected_zone_ui
active_zone_filter = None if selected_zone_ui.startswith("All ") else selected_zone_ui

abs_min_fallback_date = date(2022, 1, 1) 
abs_max_fallback_date = date.today()    
abs_min_data_date, abs_max_data_date = abs_min_fallback_date, abs_max_fallback_date

if isinstance(df_for_filters, pd.DataFrame) and 'encounter_date' in df_for_filters.columns:
    valid_dates = df_for_filters['encounter_date'].dropna()
    if not valid_dates.empty:
        min_from_data = valid_dates.min().date()
        max_from_data = valid_dates.max().date()
        if min_from_data <= max_from_data: 
            abs_min_data_date, abs_max_data_date = min_from_data, max_from_data
        else: logger.warning("Min date from filter data is after max date. Using fallbacks.")
    else: logger.info("No valid encounter dates in filter data. Using fallbacks.")
else: logger.info("Filter data for dates empty or 'encounter_date' missing. Using fallbacks.")

daily_date_ss_key = "chw_dashboard_daily_view_date_v8"
default_daily_date = abs_max_data_date 
if daily_date_ss_key not in st.session_state: st.session_state[daily_date_ss_key] = default_daily_date
else: 
    persisted_daily_date = st.session_state[daily_date_ss_key]
    st.session_state[daily_date_ss_key] = min(max(persisted_daily_date, abs_min_data_date), abs_max_data_date)

selected_daily_date = st.sidebar.date_input(
    "View Daily Activity For:", value=st.session_state[daily_date_ss_key],
    min_value=abs_min_data_date, max_value=abs_max_data_date, key=f"{daily_date_ss_key}_widget"
)
st.session_state[daily_date_ss_key] = selected_daily_date

trend_range_ss_key = "chw_dashboard_trend_date_range_v8"
default_trend_days_setting = (settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND 
                              if hasattr(settings, 'WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND') else 30)
default_trend_end_date = selected_daily_date 
default_trend_start_date = max(abs_min_data_date, default_trend_end_date - timedelta(days=default_trend_days_setting - 1))

if trend_range_ss_key not in st.session_state:
    st.session_state[trend_range_ss_key] = [default_trend_start_date, default_trend_end_date]
else: 
    persisted_start, persisted_end = st.session_state[trend_range_ss_key]
    current_start = min(max(persisted_start, abs_min_data_date), abs_max_data_date)
    current_end = min(max(persisted_end, abs_min_data_date), abs_max_data_date)
    if current_start > current_end: current_start = current_end 
    st.session_state[trend_range_ss_key] = [current_start, current_end]

selected_trend_range = st.sidebar.date_input(
    "Select Trend Date Range:", value=st.session_state[trend_range_ss_key],
    min_value=abs_min_data_date, max_value=abs_max_data_date, key=f"{trend_range_ss_key}_widget"
)

trend_start_date_filter, trend_end_date_filter = default_trend_start_date, default_trend_end_date
if isinstance(selected_trend_range, (list, tuple)) and len(selected_trend_range) == 2:
    trend_start_date_filter, trend_end_date_filter = selected_trend_range
    if trend_start_date_filter > trend_end_date_filter:
        st.sidebar.error("Trend Start date cannot be after End date. Adjusting.")
        trend_end_date_filter = trend_start_date_filter
    st.session_state[trend_range_ss_key] = [trend_start_date_filter, trend_end_date_filter]
else: 
    trend_start_date_filter, trend_end_date_filter = st.session_state.get(trend_range_ss_key, [default_trend_start_date, default_trend_end_date])

daily_activity_df, trend_activity_df, daily_kpis_precalculated = pd.DataFrame(), pd.DataFrame(), {}
try:
    daily_activity_df, trend_activity_df, daily_kpis_precalculated = load_chw_dashboard_data(
        selected_daily_date, trend_start_date_filter, trend_end_date_filter,
        active_chw_filter, active_zone_filter
    )
except Exception as e_load_main_data:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_load_main_data}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {str(e_load_main_data)}. Please check logs and data sources.")

filter_context_parts = [f"Snapshot Date: **{selected_daily_date.strftime('%d %b %Y')}**"]
if active_chw_filter: filter_context_parts.append(f"CHW: **{active_chw_filter}**")
if active_zone_filter: filter_context_parts.append(f"Zone: **{active_zone_filter}**")
st.info(f"Displaying data for: {', '.join(filter_context_parts)}")

st.header("📊 Daily Performance Snapshot")
if not daily_activity_df.empty:
    daily_summary_metrics = {}
    try:
        daily_summary_metrics = calculate_chw_daily_summary_metrics(
            daily_activity_df, selected_daily_date, daily_kpis_precalculated, "CHWDash/DailySummary"
        )
    except Exception as e_daily_summary:
        logger.error(f"Error calculating CHW daily summary: {e_daily_summary}", exc_info=True)
        st.warning("⚠️ Could not calculate daily summary metrics.")

    kpi_cols = st.columns(4)
    visits_today = daily_summary_metrics.get("visits_count", 0)
    # Assuming render_kpi_card expects: title, value (str), icon, status (optional), units (optional), help_text (optional), container (optional)
    render_kpi_card(title="Visits Today", value=str(visits_today), icon="👥", help_text="Total unique patients encountered.", container=kpi_cols[0])

    prio_followups = daily_summary_metrics.get("high_ai_prio_followups_count", 0)
    prio_threshold_setting = settings.FATIGUE_INDEX_HIGH_THRESHOLD if hasattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD') else 0.7 
    prio_status = "ACCEPTABLE" if prio_followups <= 2 else ("MODERATE_CONCERN" if prio_followups <= 5 else "HIGH_CONCERN")
    render_kpi_card(title="High Prio Follow-ups", value=str(prio_followups), icon="🎯", status=prio_status, help_text=f"Patients needing urgent follow-up (AI prio score ≥ {prio_threshold_setting:.1f}).", container=kpi_cols[1])

    critical_spo2_threshold_setting = settings.ALERT_SPO2_CRITICAL_LOW_PCT if hasattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT') else 90
    critical_spo2_cases = daily_summary_metrics.get("critical_spo2_cases_identified_count", 0)
    spo2_status = "HIGH_CONCERN" if critical_spo2_cases > 0 else "ACCEPTABLE"
    render_kpi_card(title="Critical SpO2 Cases", value=str(critical_spo2_cases), icon="💨", status=spo2_status, help_text=f"Patients with SpO2 < {critical_spo2_threshold_setting}%.", container=kpi_cols[2])
    
    high_fever_threshold_setting = settings.ALERT_BODY_TEMP_HIGH_FEVER_C if hasattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C') else 39.0
    high_fever_cases = daily_summary_metrics.get("high_fever_cases_identified_count", 0)
    fever_status = "HIGH_CONCERN" if high_fever_cases > 0 else "ACCEPTABLE"
    render_kpi_card(title="High Fever Cases", value=str(high_fever_cases), icon="🔥", status=fever_status, help_text=f"Patients with body temp ≥ {high_fever_threshold_setting}°C.", container=kpi_cols[3])
else:
    st.markdown("ℹ️ _No activity data for selected date/filters for daily performance snapshot._")
st.divider()

st.header("🚦 Key Alerts & Tasks")
chw_alerts = []
if not daily_activity_df.empty:
    try:
        chw_alerts = generate_chw_alerts(daily_activity_df, selected_daily_date, active_zone_filter or "All Zones", max_alerts=10)
    except Exception as e_alerts:
        logger.error(f"CHW Dashboard: Error generating patient alerts: {e_alerts}", exc_info=True)
        st.warning("⚠️ Could not generate patient alerts.")

if chw_alerts:
    st.subheader("Priority Patient Alerts (Today):")
    critical_alerts_found = False
    for alert in chw_alerts: 
        if alert.get("alert_level") == "CRITICAL":
            critical_alerts_found = True
            render_traffic_light_indicator(
                title=f"Pt. {alert.get('patient_id', 'N/A')}: {alert.get('primary_reason', 'Critical Alert')}",
                level="HIGH_RISK", 
                details=(f"Details: {alert.get('brief_details','N/A')} | Context: {alert.get('context_info','N/A')} | Action: {alert.get('suggested_action_code','REVIEW')}")
            )
    if not critical_alerts_found: st.info("ℹ️ No CRITICAL patient alerts identified for this selection today.")
    warning_alerts = [alert for alert in chw_alerts if alert.get("alert_level") == "WARNING"]
    if warning_alerts:
        st.markdown("###### Warning Level Alerts:")
        for alert in warning_alerts:
            render_traffic_light_indicator(
                title=f"Pt. {alert.get('patient_id', 'N/A')}: {alert.get('primary_reason', 'Warning')}",
                level="MODERATE_CONCERN", details=f"Details: {alert.get('brief_details','N/A')} | Context: {alert.get('context_info','N/A')}"
            )
    elif not critical_alerts_found: st.info("ℹ️ Only informational alerts (if any) were generated.")
elif not daily_activity_df.empty: st.success("✅ No significant patient alerts needing immediate attention generated for today's selection.")
else: st.markdown("ℹ️ _No activity data to generate patient alerts for today._")

chw_tasks = []
if not daily_activity_df.empty:
    try:
        chw_tasks = generate_chw_tasks(daily_activity_df, selected_daily_date, active_chw_filter, active_zone_filter or "All Zones", max_tasks=10)
    except Exception as e_tasks:
        logger.error(f"CHW Dashboard: Error generating CHW tasks: {e_tasks}", exc_info=True)
        st.warning("⚠️ Could not generate tasks list.")

if chw_tasks:
    st.subheader("Top Priority Tasks (Today/Next Day):")
    tasks_df = pd.DataFrame(chw_tasks)
    task_display_cols_ordered = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id']
    actual_task_cols = [col for col in task_display_cols_ordered if col in tasks_df.columns]
    if not tasks_df.empty and actual_task_cols:
        st.dataframe(
            tasks_df[actual_task_cols], use_container_width=True, height=min(420, len(tasks_df) * 38 + 58), 
            hide_index=True, column_config={"priority_score": st.column_config.NumberColumn(format="%.1f"), "due_date": st.column_config.DateColumn(format="YYYY-MM-DD")}
        )
    elif not tasks_df.empty: st.warning("⚠️ Task data available but cannot display correctly due to column configuration issues.")
elif not daily_activity_df.empty: st.info("ℹ️ No high-priority tasks identified based on current data.")
else: st.markdown("ℹ️ _No activity data to generate tasks for today._")
st.divider()

st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_activity_df.empty:
    epi_signals = {}
    try:
        epi_signals = extract_chw_epi_signals(daily_activity_df, daily_kpis_precalculated, selected_daily_date, active_zone_filter or "All Zones", min_cluster_size=3)
    except Exception as e_epi:
        logger.error(f"CHW Dashboard: Error extracting epi signals: {e_epi}", exc_info=True)
        st.warning("⚠️ Could not extract epi signals.")

    epi_kpi_cols = st.columns(3)
    render_kpi_card(title="Symptomatic (Key Cond.)", value=str(epi_signals.get("symptomatic_patients_key_conditions_count", "N/A")), icon="🤒", units="cases today", help_text="Patients seen today with symptoms related to key conditions.", container=epi_kpi_cols[0])
    
    new_malaria_cases = epi_signals.get("newly_identified_malaria_patients_count", 0)
    malaria_status = "HIGH_CONCERN" if new_malaria_cases > 1 else ("MODERATE_CONCERN" if new_malaria_cases == 1 else "ACCEPTABLE")
    render_kpi_card(title="New Malaria Cases", value=str(new_malaria_cases), icon="🦟", status=malaria_status, units="cases today", help_text="New malaria cases identified today.", container=epi_kpi_cols[1])

    pending_tb_contacts = epi_signals.get("pending_tb_contact_tracing_tasks_count", 0)
    tb_status = "MODERATE_CONCERN" if pending_tb_contacts > 0 else "ACCEPTABLE"
    render_kpi_card(title="Pending TB Contacts", value=str(pending_tb_contacts), icon="👥", status=tb_status, units="to trace", help_text="TB contacts needing follow-up.", container=epi_kpi_cols[2])

    symptom_clusters = epi_signals.get("detected_symptom_clusters", [])
    if symptom_clusters:
        st.markdown("###### Detected Symptom Clusters (Requires Supervisor Verification):")
        for cluster in symptom_clusters:
            st.warning(f"⚠️ **Pattern: {cluster.get('symptoms_pattern', 'Unknown')}** - {cluster.get('patient_count', 'N/A')} cases in {cluster.get('location_hint', 'CHW area')}. Supervisor to verify.")
    elif isinstance(daily_activity_df, pd.DataFrame) and 'patient_reported_symptoms' in daily_activity_df.columns and daily_activity_df['patient_reported_symptoms'].notna().any():
        st.info("ℹ️ No significant symptom clusters detected today based on current data and criteria.")
else:
    st.markdown("ℹ️ _No activity data for local epi signals for selected date/filters._")
st.divider()

st.header("📈 CHW Team Activity Trends")
trend_period_str = f"{trend_start_date_filter.strftime('%d %b %Y')} - {trend_end_date_filter.strftime('%d %b %Y')}"
trend_filter_str = f" for CHW **{active_chw_filter}**" if active_chw_filter else ""
trend_filter_str += f" in Zone **{active_zone_filter}**" if active_zone_filter else ""
trend_filter_str = trend_filter_str or " (All CHWs/Zones)" 
st.markdown(f"Displaying trends from **{trend_period_str}**{trend_filter_str}.")

if not trend_activity_df.empty:
    activity_trends = {}
    try:
        activity_trends = calculate_chw_activity_trends_data(trend_activity_df, trend_start_date_filter, trend_end_date_filter, active_zone_filter, freq_alias='D')
    except Exception as e_trends:
        logger.error(f"CHW Dashboard: Error calculating activity trends: {e_trends}", exc_info=True)
        st.warning("⚠️ Could not calculate activity trends.")

    trend_plot_cols = st.columns(2)
    visits_trend_series = activity_trends.get("patient_visits_trend")
    if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
        with trend_plot_cols[0]:
            try:
                fig_visits = plot_annotated_line_chart(visits_trend_series, "Daily Patient Visits Trend", "Unique Patients Visited", y_values_are_counts=True)
                st.plotly_chart(fig_visits, use_container_width=True)
            except Exception as e_plot_visits:
                logger.error(f"Error plotting patient visits trend: {e_plot_visits}", exc_info=True)
                st.caption("⚠️ Error displaying patient visits trend plot.")
    else:
        with trend_plot_cols[0]: st.caption("ℹ️ No patient visit trend data for this selection.")

    prio_followups_trend_series = activity_trends.get("high_priority_followups_trend")
    if isinstance(prio_followups_trend_series, pd.Series) and not prio_followups_trend_series.empty:
        with trend_plot_cols[1]:
            try:
                fig_prio = plot_annotated_line_chart(prio_followups_trend_series, "Daily High Prio. Follow-ups Trend", "High Prio. Follow-ups", y_values_are_counts=True)
                st.plotly_chart(fig_prio, use_container_width=True)
            except Exception as e_plot_prio:
                logger.error(f"Error plotting high priority followups trend: {e_plot_prio}", exc_info=True)
                st.caption("⚠️ Error displaying high priority follow-ups trend plot.")
    else:
        with trend_plot_cols[1]: st.caption("ℹ️ No high-priority follow-up trend data for this selection.")
else:
    st.markdown("ℹ️ _No historical data available for the selected trend period and/or filters._")

st.divider()
footer_text = settings.APP_FOOTER_TEXT if hasattr(settings, 'APP_FOOTER_TEXT') else "Sentinel Health Co-Pilot."
st.caption(footer_text)

logger.info(
    f"CHW Supervisor Dashboard page fully rendered. Filters: "
    f"DailyDate={selected_daily_date.isoformat()}, Trend=({trend_start_date_filter.isoformat()} to {trend_end_date_filter.isoformat()}), "
    f"CHW='{active_chw_filter or 'All'}', Zone='{active_zone_filter or 'All'}'. "
    f"Data: Daily={not daily_activity_df.empty}, Trend={not trend_activity_df.empty}"
)
