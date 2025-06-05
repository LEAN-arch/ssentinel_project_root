# sentinel_project_root/pages/chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta # Import 'date' directly
from typing import Optional, Dict, Any, List, Tuple
import os # Keep for os.path.exists if settings paths are strings initially
from pathlib import Path 

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart
    
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.alert_generation import generate_chw_alerts
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import calculate_chw_activity_trends_data
except ImportError as e_chw_dash_abs_final: # Unique exception variable name
    import sys 
    _current_file_chw_final = Path(__file__).resolve()
    _pages_dir_chw_final = _current_file_chw_final.parent
    _project_root_chw_assumption_final = _pages_dir_chw_final.parent 

    error_msg_chw_detail_final = (
        f"CHW Dashboard Import Error (using absolute imports): {e_chw_dash_abs_final}. "
        f"Ensure project root ('{_project_root_chw_assumption_final}') is in sys.path (done by app.py) "
        f"and all modules/packages (e.g., 'pages', 'pages.chw_components') have `__init__.py` files. "
        f"Check for typos in import paths. Current Python Path: {sys.path}"
    )
    try:
        st.error(error_msg_chw_detail_final)
        st.stop()
    except NameError: 
        print(error_msg_chw_detail_final, file=sys.stderr)
        raise

logger = logging.getLogger(__name__)

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME}**")
st.divider()

def _create_filter_dropdown_options(
    df_for_options: Optional[pd.DataFrame],
    column_name_in_df: str,
    default_fallback_options: List[str],
    display_name_plural_for_all: str
) -> List[str]:
    all_option_label = f"All {display_name_plural_for_all}"
    options_list = [all_option_label]
    if isinstance(df_for_options, pd.DataFrame) and not df_for_options.empty and column_name_in_df in df_for_options.columns:
        # Ensure values are strings for consistent sorting and display
        unique_values_from_df = sorted([str(val) for val in df_for_options[column_name_in_df].dropna().unique()])
        if unique_values_from_df: options_list.extend(unique_values_from_df)
        else:
            logger.warning(f"Filter options: Col '{column_name_in_df}' for '{display_name_plural_for_all}' has no unique values. Using defaults.")
            options_list.extend(default_fallback_options)
    else:
        logger.warning(f"Filter options: Col '{column_name_in_df}' not found or DF empty for '{display_name_plural_for_all}'. Using defaults.")
        options_list.extend(default_fallback_options)
    return options_list

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...")
def get_chw_dashboard_display_data(
    selected_view_date: date, 
    selected_trend_start_date: date,
    selected_trend_end_date: date,
    selected_chw_id_filter: Optional[str] = None,
    selected_zone_id_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "CHWDashboardDataLoad"
    logger.info(
        f"({log_ctx}) Loading data for view: {selected_view_date}, trend: {selected_trend_start_date}-{selected_trend_end_date}, "
        f"CHW: {selected_chw_id_filter or 'All'}, Zone: {selected_zone_id_filter or 'All'}"
    )
    all_health_records_df = load_health_records(source_context=f"{log_ctx}/LoadAllRecs") # Path handled by settings
    if not isinstance(all_health_records_df, pd.DataFrame) or all_health_records_df.empty:
        logger.warning(f"({log_ctx}) No health records loaded. Dashboard will display no data.")
        st.warning("No health data available. Please check data sources. Specifically, 'health_records_expanded.csv' might be missing or empty.")
        return pd.DataFrame(), pd.DataFrame(), {}
    if 'encounter_date' not in all_health_records_df.columns or not pd.api.types.is_datetime64_any_dtype(all_health_records_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' invalid in loaded health records. Cannot filter data effectively.")
        st.error("Data Integrity Error: 'encounter_date' in health records is invalid. Dashboard cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), {}

    df_daily = all_health_records_df[all_health_records_df['encounter_date'].dt.date == selected_view_date].copy()
    if selected_chw_id_filter and 'chw_id' in df_daily.columns: df_daily = df_daily[df_daily['chw_id'] == selected_chw_id_filter]
    if selected_zone_id_filter and 'zone_id' in df_daily.columns: df_daily = df_daily[df_daily['zone_id'] == selected_zone_id_filter]

    df_trend = all_health_records_df[
        (all_health_records_df['encounter_date'].dt.date >= selected_trend_start_date) &
        (all_health_records_df['encounter_date'].dt.date <= selected_trend_end_date)
    ].copy()
    if selected_chw_id_filter and 'chw_id' in df_trend.columns: df_trend = df_trend[df_trend['chw_id'] == selected_chw_id_filter]
    if selected_zone_id_filter and 'zone_id' in df_trend.columns: df_trend = df_trend[df_trend['zone_id'] == selected_zone_id_filter]

    pre_calc_daily_kpis: Dict[str, Any] = {}
    if selected_chw_id_filter and not df_daily.empty:
        # Robustly check for 'encounter_type' and 'chw_id' before filtering
        encounter_type_series = df_daily.get('encounter_type', pd.Series(dtype=str)).astype(str)
        chw_id_series = df_daily.get('chw_id', pd.Series(dtype=str))
        
        worker_self_checks = df_daily[
            (chw_id_series == selected_chw_id_filter) &
            (encounter_type_series.str.contains('WORKER_SELF_CHECK', case=False, na=False))
        ]
        if not worker_self_checks.empty:
            fatigue_cols = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            actual_fatigue_col = next((c for c in fatigue_cols if c in worker_self_checks.columns and worker_self_checks[c].notna().any()), None)
            pre_calc_daily_kpis['worker_self_fatigue_index_today'] = worker_self_checks[actual_fatigue_col].max() if actual_fatigue_col else np.nan
        else: pre_calc_daily_kpis['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(f"({log_ctx}) Data loading complete. Daily: {len(df_daily)} records, Trend: {len(df_trend)} records.")
    return df_daily, df_trend, pre_calc_daily_kpis

# --- Sidebar Filters ---
# Construct path to logo using PROJECT_ROOT_DIR from settings
# settings.APP_LOGO_SMALL_PATH is already a string path from settings.py
logo_path_sidebar_chw_final = Path(settings.APP_LOGO_SMALL_PATH) # Path object from string
if not logo_path_sidebar_chw_final.is_absolute(): 
    logo_path_sidebar_chw_final = (Path(settings.PROJECT_ROOT_DIR) / settings.APP_LOGO_SMALL_PATH).resolve()

if logo_path_sidebar_chw_final.exists() and logo_path_sidebar_chw_final.is_file(): 
    st.sidebar.image(str(logo_path_sidebar_chw_final), width=120)
else: 
    logger.warning(f"Sidebar logo not found at {logo_path_sidebar_chw_final}")
    st.sidebar.markdown(f"#### {settings.APP_NAME}")
st.sidebar.header("Dashboard Filters")

# Load data for filters *once*
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_data_for_chw_filters():
    return load_health_records(source_context="CHWDash/FilterPopulation")

df_for_chw_filters = load_data_for_chw_filters()

chw_filter_options_val = _create_filter_dropdown_options(df_for_chw_filters, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
chw_session_key = "chw_dashboard_chw_id_v3" # Ensure unique session state keys
if chw_session_key not in st.session_state or st.session_state[chw_session_key] not in chw_filter_options_val : st.session_state[chw_session_key] = chw_filter_options_val[0]
selected_chw_id_ui_val = st.sidebar.selectbox("Filter by CHW ID:", options=chw_filter_options_val, key=f"{chw_session_key}_widget", index=chw_filter_options_val.index(st.session_state[chw_session_key]))
st.session_state[chw_session_key] = selected_chw_id_ui_val
actual_chw_id_query_param_val = None if selected_chw_id_ui_val.startswith("All ") else selected_chw_id_ui_val

zone_filter_options_val = _create_filter_dropdown_options(df_for_chw_filters, 'zone_id', ["ZoneA", "ZoneB", "ZoneC"], "Zones")
zone_session_key = "chw_dashboard_zone_id_v3"
if zone_session_key not in st.session_state or st.session_state[zone_session_key] not in zone_filter_options_val: st.session_state[zone_session_key] = zone_filter_options_val[0]
selected_zone_id_ui_val = st.sidebar.selectbox("Filter by Zone:", options=zone_filter_options_val, key=f"{zone_session_key}_widget", index=zone_filter_options_val.index(st.session_state[zone_session_key]))
st.session_state[zone_session_key] = selected_zone_id_ui_val
actual_zone_id_query_param_val = None if selected_zone_id_ui_val.startswith("All ") else selected_zone_id_ui_val

abs_min_date_val = date.today() - timedelta(days=180); abs_max_date_val = date.today()
daily_date_session_key_val = "chw_dashboard_daily_date_v3"
if daily_date_session_key_val not in st.session_state: st.session_state[daily_date_session_key_val] = abs_max_date_val
selected_daily_date_ui_val = st.sidebar.date_input("View Daily Activity For:", value=st.session_state[daily_date_session_key_val], min_value=abs_min_date_val, max_value=abs_max_date_val, key=f"{daily_date_session_key_val}_widget")
st.session_state[daily_date_session_key_val] = selected_daily_date_ui_val

trend_date_range_session_key_val = "chw_dashboard_trend_date_range_v3"
default_trend_end_ui_val = selected_daily_date_ui_val
default_trend_start_ui_val = max(abs_min_date_val, default_trend_end_ui_val - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))
if trend_date_range_session_key_val not in st.session_state: st.session_state[trend_date_range_session_key_val] = [default_trend_start_ui_val, default_trend_end_ui_val]
selected_trend_date_range_ui_val = st.sidebar.date_input("Select Trend Date Range:", value=st.session_state[trend_date_range_session_key_val], min_value=abs_min_date_val, max_value=abs_max_date_val, key=f"{trend_date_range_session_key_val}_widget")

trend_start_date_query_param_val: date; trend_end_date_query_param_val: date
if isinstance(selected_trend_date_range_ui_val, (list, tuple)) and len(selected_trend_date_range_ui_val) == 2:
    st.session_state[trend_date_range_session_key_val] = selected_trend_date_range_ui_val
    trend_start_date_query_param_val, trend_end_date_query_param_val = selected_trend_date_range_ui_val
else: 
    trend_start_date_query_param_val, trend_end_date_query_param_val = st.session_state[trend_date_range_session_key_val]
    st.sidebar.warning("Trend date range selection error. Using previous/default range.")
if trend_start_date_query_param_val > trend_end_date_query_param_val: 
    st.sidebar.error("Trend Start Date must be <= Trend End Date."); trend_end_date_query_param_val = trend_start_date_query_param_val
    st.session_state[trend_date_range_session_key_val][1] = trend_end_date_query_param_val

# --- Load Data Based on Filters ---
daily_df_display, period_df_display, daily_pre_calculated_kpis_map = pd.DataFrame(), pd.DataFrame(), {}
try:
    daily_df_display, period_df_display, daily_pre_calculated_kpis_map = get_chw_dashboard_display_data(
        selected_daily_date_ui_val, trend_start_date_query_param_val, trend_end_date_query_param_val, 
        actual_chw_id_query_param_val, actual_zone_id_query_param_val
    )
except Exception as e_main_data_load_chw:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_main_data_load_chw}", exc_info=True)
    st.error(f"An error occurred while loading CHW dashboard data: {str(e_main_data_load_chw)}. Please check logs or contact support.")

# --- Display Filter Context ---
filter_context_display_parts_val = [f"Snapshot Date: **{selected_daily_date_ui_val.strftime('%d %b %Y')}**"]
if actual_chw_id_query_param_val: filter_context_display_parts_val.append(f"CHW: **{actual_chw_id_query_param_val}**")
if actual_zone_id_query_param_val: filter_context_display_parts_val.append(f"Zone: **{actual_zone_id_query_param_val}**")
st.info(f"Displaying data for: {', '.join(filter_context_display_parts_val)}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df_display.empty:
    chw_daily_summary_metrics_map_val = {}
    try:
        chw_daily_summary_metrics_map_val = calculate_chw_daily_summary_metrics(daily_df_display, selected_daily_date_ui_val, daily_pre_calculated_kpis_map, "CHWDash/DailySummary")
    except Exception as e_daily_summary_chw: 
        logger.error(f"Error calculating CHW daily summary metrics for dashboard: {e_daily_summary_chw}", exc_info=True)
        st.warning("Could not calculate daily summary metrics.")
    
    kpi_cols_daily_snapshot_val = st.columns(4)
    with kpi_cols_daily_snapshot_val[0]: render_kpi_card("Visits Today", str(chw_daily_summary_metrics_map_val.get("visits_count", 0)), "👥", help_text="Total unique patients encountered by the CHW/team for the selected date.")
    high_prio_followups_val_chw = chw_daily_summary_metrics_map_val.get("high_ai_prio_followups_count", 0)
    prio_status_level_chw = "ACCEPTABLE" if high_prio_followups_val_chw <= 2 else ("MODERATE_CONCERN" if high_prio_followups_val_chw <= 5 else "HIGH_CONCERN")
    with kpi_cols_daily_snapshot_val[1]: render_kpi_card("High Prio Follow-ups", str(high_prio_followups_val_chw), "🎯", prio_status_level_chw, help_text=f"Patients needing urgent follow-up (AI prio ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}).")
    critical_spo2_cases_val_chw = chw_daily_summary_metrics_map_val.get("critical_spo2_cases_identified_count", 0)
    spo2_status_level_chw = "HIGH_CONCERN" if critical_spo2_cases_val_chw > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot_val[2]: render_kpi_card("Critical SpO2 Cases", str(critical_spo2_cases_val_chw), "💨", spo2_status_level_chw, help_text=f"Patients identified with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}% on the selected date.")
    high_fever_cases_val_chw = chw_daily_summary_metrics_map_val.get("high_fever_cases_identified_count", 0)
    fever_status_level_chw = "HIGH_CONCERN" if high_fever_cases_val_chw > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot_val[3]: render_kpi_card("High Fever Cases", str(high_fever_cases_val_chw), "🔥", fever_status_level_chw, help_text=f"Patients identified with temperature ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C on the selected date.")
else: st.markdown("_No activity data available for the selected date and/or filters to display daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
chw_alerts_list_for_display_val = []
try:
    chw_alerts_list_for_display_val = generate_chw_alerts(daily_df_display, selected_daily_date_ui_val, actual_zone_id_query_param_val or "All Zones", 10)
except Exception as e_alerts_display_chw: logger.error(f"CHW Dashboard: Error generating patient alerts for display: {e_alerts_display_chw}", exc_info=True); st.warning("Could not generate patient alerts.")
if chw_alerts_list_for_display_val:
    st.subheader("Priority Patient Alerts (Today):")
    critical_alerts_exist_chw = False
    for alert_item_disp_chw in chw_alerts_list_for_display_val:
        if alert_item_disp_chw.get("alert_level") == "CRITICAL":
            critical_alerts_exist_chw = True
            render_traffic_light_indicator(f"Pt. {alert_item_disp_chw.get('patient_id', 'N/A')}: {alert_item_disp_chw.get('primary_reason', 'Critical Alert')}", "HIGH_RISK", f"Details: {alert_item_disp_chw.get('brief_details','N/A')} | Context: {alert_item_disp_chw.get('context_info','N/A')} | Action: {alert_item_disp_chw.get('suggested_action_code','REVIEW_IMMEDIATELY')}")
    if not critical_alerts_exist_chw: st.info("No CRITICAL patient alerts identified for this selection today.")
    warning_alerts_for_display_chw = [a_chw for a_chw in chw_alerts_list_for_display_val if a_chw.get("alert_level") == "WARNING"]
    if warning_alerts_for_display_chw:
        st.markdown("###### Warning Level Alerts:")
        for warn_item_disp_chw in warning_alerts_for_display_chw: render_traffic_light_indicator(f"Pt. {warn_item_disp_chw.get('patient_id', 'N/A')}: {warn_item_disp_chw.get('primary_reason', 'Warning')}", "MODERATE_CONCERN", f"Details: {warn_item_disp_chw.get('brief_details','N/A')} | Context: {warn_item_disp_chw.get('context_info','N/A')}")
    elif not critical_alerts_exist_chw : st.info("Only informational alerts (if any) were generated. No urgent patient alerts.")
elif not daily_df_display.empty: st.success("✅ No significant patient alerts needing immediate attention were generated for today's selection.")
else: st.markdown("_No activity data to generate patient alerts for today._")

chw_tasks_list_for_display_val = []
try:
    chw_tasks_list_for_display_val = generate_chw_tasks(daily_df_display, selected_daily_date_ui_val, actual_chw_id_query_param_val, actual_zone_id_query_param_val or "All Zones", 10)
except Exception as e_tasks_display_chw: logger.error(f"CHW Dashboard: Error generating CHW tasks for display: {e_tasks_display_chw}", exc_info=True); st.warning("Could not generate the prioritized tasks list.")
if chw_tasks_list_for_display_val:
    st.subheader("Top Priority Tasks (Today/Next Day):")
    tasks_df_for_table_val = pd.DataFrame(chw_tasks_list_for_display_val)
    task_table_cols_order_val = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id'] 
    actual_cols_for_task_table_val = [col_task for col_task in task_table_cols_order_val if col_task in tasks_df_for_table_val.columns]
    if not tasks_df_for_table_val.empty and actual_cols_for_task_table_val:
        st.dataframe(tasks_df_for_table_val[actual_cols_for_task_table_val], use_container_width=True, height=min(420, len(tasks_df_for_table_val) * 38 + 58), hide_index=True,
                     column_config={"priority_score": st.column_config.NumberColumn(format="%.1f"), "due_date": st.column_config.DateColumn(format="YYYY-MM-DD")})
    elif not tasks_df_for_table_val.empty: st.warning("Task data available but cannot be displayed due to column configuration issues.")
elif not daily_df_display.empty: st.info("No high-priority tasks identified requiring action today or tomorrow based on current data.")
else: st.markdown("_No activity data to generate tasks for today._")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_df_display.empty:
    chw_epi_signals_map_val = {}
    try:
        chw_epi_signals_map_val = extract_chw_epi_signals(daily_df_display, daily_pre_calculated_kpis_map, selected_daily_date_ui_val, actual_zone_id_query_param_val or "All Zones", 3)
    except Exception as e_epi_display_chw: logger.error(f"CHW Dashboard: Error extracting epi signals: {e_epi_display_chw}", exc_info=True); st.warning("Could not extract local epidemiological signals.")
    
    epi_kpi_cols_display_val = st.columns(3)
    with epi_kpi_cols_display_val[0]: render_kpi_card("Symptomatic (Key Cond.)", str(chw_epi_signals_map_val.get("symptomatic_patients_key_conditions_count", 0)), "🤒", units="cases today", help_text=f"Patients seen today with symptoms related to key conditions.")
    new_malaria_val_chw = chw_epi_signals_map_val.get("newly_identified_malaria_patients_count", 0)
    malaria_status_level_chw_epi = "HIGH_CONCERN" if new_malaria_val_chw > 1 else ("MODERATE_CONCERN" if new_malaria_val_chw == 1 else "ACCEPTABLE")
    with epi_kpi_cols_display_val[1]: render_kpi_card("New Malaria Cases", str(new_malaria_val_chw), "🦟", malaria_status_level_chw_epi, units="cases today", help_text="New malaria cases identified today.")
    pending_tb_contacts_val_chw = chw_epi_signals_map_val.get("pending_tb_contact_tracing_tasks_count", 0)
    tb_contact_status_level_chw = "MODERATE_CONCERN" if pending_tb_contacts_val_chw > 0 else "ACCEPTABLE"
    with epi_kpi_cols_display_val[2]: render_kpi_card("Pending TB Contacts", str(pending_tb_contacts_val_chw), "👥", tb_contact_status_level_chw, units="to trace", help_text="TB contacts needing follow-up.")

    detected_symptom_clusters_list_val = chw_epi_signals_map_val.get("detected_symptom_clusters", [])
    if detected_symptom_clusters_list_val:
        st.markdown("###### Detected Symptom Clusters (Requires Verification by Supervisor):")
        for cluster_item_data_val in detected_symptom_clusters_list_val: st.warning(f"⚠️ **Pattern: {cluster_item_data_val.get('symptoms_pattern', 'Unknown')}** - {cluster_item_data_val.get('patient_count', 'N/A')} cases in {cluster_item_data_val.get('location_hint', 'CHW area')}. Supervisor to verify.")
    elif 'patient_reported_symptoms' in daily_df_display.columns and daily_df_display['patient_reported_symptoms'].notna().any():
        st.info("No significant symptom clusters detected today based on current data and criteria.")
else: st.markdown("_No activity data available for selected date/filters to extract local epi signals._")
st.divider()

# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_text_val = f"{trend_start_date_query_param_val.strftime('%d %b %Y')} - {trend_end_date_query_param_val.strftime('%d %b %Y')}"
trend_filter_context_text_val = f" for CHW **{actual_chw_id_query_param_val}**" if actual_chw_id_query_param_val else ""
trend_filter_context_text_val += f" in Zone **{actual_zone_id_query_param_val}**" if actual_zone_id_query_param_val else ""
trend_filter_context_text_val = trend_filter_context_text_val or " (All CHWs/Zones in selected period)"
st.markdown(f"Displaying trends from **{trend_period_display_text_val}**{trend_filter_context_text_val}.")

if not period_df_display.empty:
    chw_activity_trends_map_val = {}
    try:
        chw_activity_trends_map_val = calculate_chw_activity_trends_data(period_df_display, trend_start_date_query_param_val, trend_end_date_query_param_val, actual_zone_id_query_param_val, 'D')
    except Exception as e_trends_display_chw: logger.error(f"CHW Dashboard: Error calculating activity trends: {e_trends_display_chw}", exc_info=True); st.warning("Could not calculate CHW activity trends.")
    
    trend_plot_cols_display_val = st.columns(2)
    with trend_plot_cols_display_val[0]:
        patient_visits_trend_series_val = chw_activity_trends_map_val.get("patient_visits_trend")
        if isinstance(patient_visits_trend_series_val, pd.Series) and not patient_visits_trend_series_val.empty:
            st.plotly_chart(plot_annotated_line_chart(patient_visits_trend_series_val, "Daily Patient Visits Trend", "Unique Patients Visited", y_values_are_counts=True), use_container_width=True)
        else: st.caption("No patient visit trend data available for this selection.")
    with trend_plot_cols_display_val[1]:
        high_prio_followups_trend_srs_val = chw_activity_trends_map_val.get("high_priority_followups_trend")
        if isinstance(high_prio_followups_trend_srs_val, pd.Series) and not high_prio_followups_trend_srs_val.empty:
            st.plotly_chart(plot_annotated_line_chart(high_prio_followups_trend_srs_val, "Daily High Prio. Follow-ups Trend", "High Prio. Follow-ups", y_values_are_counts=True), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available for this selection.")
else: st.markdown("_No historical data available for the selected trend period and/or filters._")

logger.info(f"CHW Supervisor Dashboard page loaded. Filters: Date={selected_daily_date_ui_val}, CHW={actual_chw_id_query_param_val or 'All'}, Zone={actual_zone_id_query_param_val or 'All'}, Trend=({trend_start_date_query_param_val} to {trend_end_date_query_param_val}).")
