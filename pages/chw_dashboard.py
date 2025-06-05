# sentinel_project_root/pages/chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta # Import 'date' directly
from typing import Optional, Dict, Any, List, Tuple
import os
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
except ImportError as e_chw_dash_abs:
    import sys
    _current_file_chw = Path(__file__).resolve()
    _pages_dir_chw = _current_file_chw.parent
    _project_root_chw_assumption = _pages_dir_chw.parent 
    error_msg_chw_detail = (
        f"CHW Dashboard Import Error (using absolute imports): {e_chw_dash_abs}. "
        f"Project root ('{_project_root_chw_assumption}') might not be in sys.path or module typo. "
        f"Ensure `__init__.py` in packages. Current Python Path: {sys.path}"
    )
    try:
        st.error(error_msg_chw_detail)
        st.stop()
    except NameError:
        print(error_msg_chw_detail, file=sys.stderr)
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
    selected_view_date: date, # Use 'date' directly
    selected_trend_start_date: date, # Use 'date'
    selected_trend_end_date: date,   # Use 'date'
    selected_chw_id_filter: Optional[str] = None,
    selected_zone_id_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "CHWDashboardDataLoad"
    logger.info(
        f"({log_ctx}) Loading data for view: {selected_view_date}, trend: {selected_trend_start_date}-{selected_trend_end_date}, "
        f"CHW: {selected_chw_id_filter or 'All'}, Zone: {selected_zone_id_filter or 'All'}"
    )
    all_health_records_df = load_health_records(source_context=f"{log_ctx}/LoadAllRecs")
    if not isinstance(all_health_records_df, pd.DataFrame) or all_health_records_df.empty:
        logger.warning(f"({log_ctx}) No health records loaded."); st.warning("No health data available.")
        return pd.DataFrame(), pd.DataFrame(), {}
    if 'encounter_date' not in all_health_records_df.columns or not pd.api.types.is_datetime64_any_dtype(all_health_records_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' invalid."); st.error("Data Integrity Error: 'encounter_date' invalid.")
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
        worker_self_checks = df_daily[
            (df_daily.get('chw_id') == selected_chw_id_filter) &
            (df_daily.get('encounter_type', pd.Series(dtype=str)).astype(str).str.contains('WORKER_SELF_CHECK', case=False, na=False))
        ]
        if not worker_self_checks.empty:
            fatigue_cols = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            actual_fatigue_col = next((c for c in fatigue_cols if c in worker_self_checks.columns and worker_self_checks[c].notna().any()), None)
            pre_calc_daily_kpis['worker_self_fatigue_index_today'] = worker_self_checks[actual_fatigue_col].max() if actual_fatigue_col else np.nan
        else: pre_calc_daily_kpis['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(f"({log_ctx}) Data loading complete. Daily: {len(df_daily)} recs, Trend: {len(df_trend)} recs.")
    return df_daily, df_trend, pre_calc_daily_kpis

# --- Sidebar Filters ---
project_root_path = Path(settings.PROJECT_ROOT_DIR) # Use PROJECT_ROOT_DIR from settings
logo_path_sidebar_chw = project_root_path / settings.APP_LOGO_SMALL_PATH # Construct full path
if logo_path_sidebar_chw.exists() and logo_path_sidebar_chw.is_file(): st.sidebar.image(str(logo_path_sidebar_chw), width=120)
else: st.sidebar.markdown(f"#### {settings.APP_NAME}")
st.sidebar.header("Dashboard Filters")

df_filters = load_health_records(source_context="CHWDash/FilterData")

chw_opts = _create_filter_dropdown_options(df_filters, 'chw_id', ["CHW001", "CHW002"], "CHWs")
chw_key = "chw_dashboard_chw_id_v2" # Ensure unique session state keys
if chw_key not in st.session_state or st.session_state[chw_key] not in chw_opts : st.session_state[chw_key] = chw_opts[0]
selected_chw_ui = st.sidebar.selectbox("Filter by CHW ID:", options=chw_opts, key=f"{chw_key}_widget", index=chw_opts.index(st.session_state[chw_key]))
st.session_state[chw_key] = selected_chw_ui
actual_chw_filter = None if selected_chw_ui.startswith("All ") else selected_chw_ui

zone_opts = _create_filter_dropdown_options(df_filters, 'zone_id', ["ZoneA", "ZoneB"], "Zones")
zone_key = "chw_dashboard_zone_id_v2"
if zone_key not in st.session_state or st.session_state[zone_key] not in zone_opts: st.session_state[zone_key] = zone_opts[0]
selected_zone_ui = st.sidebar.selectbox("Filter by Zone:", options=zone_opts, key=f"{zone_key}_widget", index=zone_opts.index(st.session_state[zone_key]))
st.session_state[zone_key] = selected_zone_ui
actual_zone_filter = None if selected_zone_ui.startswith("All ") else selected_zone_ui

abs_min_date_chw = date.today() - timedelta(days=180); abs_max_date_chw = date.today()
daily_date_key_chw = "chw_dashboard_daily_date_v2"
if daily_date_key_chw not in st.session_state: st.session_state[daily_date_key_chw] = abs_max_date_chw
selected_daily_date = st.sidebar.date_input("View Daily Activity For:", value=st.session_state[daily_date_key_chw], min_value=abs_min_date_chw, max_value=abs_max_date_chw, key=f"{daily_date_key_chw}_widget")
st.session_state[daily_date_key_chw] = selected_daily_date

trend_date_key_chw = "chw_dashboard_trend_date_range_v2"
def_trend_end_chw = selected_daily_date
def_trend_start_chw = max(abs_min_date_chw, def_trend_end_chw - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))
if trend_date_key_chw not in st.session_state: st.session_state[trend_date_key_chw] = [def_trend_start_chw, def_trend_end_chw]
selected_trend_range = st.sidebar.date_input("Select Trend Date Range:", value=st.session_state[trend_date_key_chw], min_value=abs_min_date_chw, max_value=abs_max_date_chw, key=f"{trend_date_key_chw}_widget")

trend_start_filt: date; trend_end_filt: date # Explicitly type
if isinstance(selected_trend_range, (list, tuple)) and len(selected_trend_range) == 2:
    st.session_state[trend_date_key_chw] = selected_trend_range
    trend_start_filt, trend_end_filt = selected_trend_range
else: 
    trend_start_filt, trend_end_filt = st.session_state[trend_date_key_chw]
    st.sidebar.warning("Trend date range selection error. Using previous/default range.")
if trend_start_filt > trend_end_filt: 
    st.sidebar.error("Trend Start Date must be <= Trend End Date."); trend_end_filt = trend_start_filt
    st.session_state[trend_date_key_chw][1] = trend_end_filt

# --- Load Data Based on Filters ---
daily_df, period_df, daily_kpis_precalc = pd.DataFrame(), pd.DataFrame(), {} # Initialize
try:
    daily_df, period_df, daily_kpis_precalc = get_chw_dashboard_display_data(selected_daily_date, trend_start_filt, trend_end_filt, actual_chw_filter, actual_zone_filter)
except Exception as e_load_chw:
    logger.error(f"CHW Dashboard: Main data loading failed: {e_load_chw}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {e_load_chw}. Please check logs.")

# --- Display Filter Context ---
filter_ctx_parts_chw = [f"Snapshot Date: **{selected_daily_date.strftime('%d %b %Y')}**"]
if actual_chw_filter: filter_ctx_parts_chw.append(f"CHW: **{actual_chw_filter}**")
if actual_zone_filter: filter_ctx_parts_chw.append(f"Zone: **{actual_zone_filter}**")
st.info(f"Displaying data for: {', '.join(filter_ctx_parts_chw)}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df.empty:
    try:
        daily_summary = calculate_chw_daily_summary_metrics(daily_df, selected_daily_date, daily_kpis_precalc, "CHWDash/DailySummary")
    except Exception as e_sum_chw: logger.error(f"Error calculating daily summary: {e_sum_chw}", exc_info=True); daily_summary = {}
    
    kpi_cols_chw = st.columns(4)
    with kpi_cols_chw[0]: render_kpi_card("Visits Today", str(daily_summary.get("visits_count", 0)), "👥", help_text="Total unique patients encountered.")
    prio_fups_chw = daily_summary.get("high_ai_prio_followups_count", 0)
    prio_stat_chw = "ACCEPTABLE" if prio_fups_chw <= 2 else ("MODERATE_CONCERN" if prio_fups_chw <= 5 else "HIGH_CONCERN")
    with kpi_cols_chw[1]: render_kpi_card("High Prio Follow-ups", str(prio_fups_chw), "🎯", prio_stat_chw, help_text=f"Patients needing urgent follow-up (AI prio ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}).")
    crit_spo2_chw = daily_summary.get("critical_spo2_cases_identified_count", 0)
    spo2_stat_chw = "HIGH_CONCERN" if crit_spo2_chw > 0 else "ACCEPTABLE"
    with kpi_cols_chw[2]: render_kpi_card("Critical SpO2 Cases", str(crit_spo2_chw), "💨", spo2_stat_chw, help_text=f"Patients with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    high_fever_chw = daily_summary.get("high_fever_cases_identified_count", 0)
    fever_stat_chw = "HIGH_CONCERN" if high_fever_chw > 0 else "ACCEPTABLE"
    with kpi_cols_chw[3]: render_kpi_card("High Fever Cases", str(high_fever_chw), "🔥", fever_stat_chw, help_text=f"Patients with temp ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else: st.markdown("_No activity data for selected date/filters for daily snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
chw_alerts_list = [] # Initialize
try:
    chw_alerts_list = generate_chw_alerts(daily_df, selected_daily_date, actual_zone_filter or "All Zones", 10)
except Exception as e_alert_chw: logger.error(f"Error generating CHW alerts: {e_alert_chw}", exc_info=True)
if chw_alerts_list:
    st.subheader("Priority Patient Alerts (Today):")
    crit_alerts_found_chw = False
    for alert_item in chw_alerts_list:
        if alert_item.get("alert_level") == "CRITICAL":
            crit_alerts_found_chw = True
            render_traffic_light_indicator(f"Pt. {alert_item.get('patient_id', 'N/A')}: {alert_item.get('primary_reason', 'Critical Alert')}", "HIGH_RISK", f"Details: {alert_item.get('brief_details','N/A')} | Context: {alert_item.get('context_info','N/A')} | Action: {alert_item.get('suggested_action_code','REVIEW')}")
    if not crit_alerts_found_chw: st.info("No CRITICAL patient alerts identified for this selection today.")
    warn_alerts_list = [a for a in chw_alerts_list if a.get("alert_level") == "WARNING"]
    if warn_alerts_list:
        st.markdown("###### Warning Level Alerts:")
        for warn_item in warn_alerts_list: render_traffic_light_indicator(f"Pt. {warn_item.get('patient_id', 'N/A')}: {warn_item.get('primary_reason', 'Warning')}", "MODERATE_CONCERN", f"Details: {warn_item.get('brief_details','N/A')}")
    elif not crit_alerts_found_chw: st.info("Only informational alerts (if any) generated.")
elif not daily_df.empty: st.success("✅ No significant patient alerts needing immediate attention generated for today.")
else: st.markdown("_No activity data to generate patient alerts for today._")

chw_tasks_list = [] # Initialize
try:
    chw_tasks_list = generate_chw_tasks(daily_df, selected_daily_date, actual_chw_filter, actual_zone_filter or "All Zones", 10)
except Exception as e_task_chw: logger.error(f"Error generating CHW tasks: {e_task_chw}", exc_info=True)
if chw_tasks_list:
    st.subheader("Top Priority Tasks (Today/Next Day):")
    tasks_df_for_display = pd.DataFrame(chw_tasks_list)
    task_cols_display_order = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id']
    actual_task_cols_display = [c for c in task_cols_display_order if c in tasks_df_for_display.columns]
    if not tasks_df_for_display.empty and actual_task_cols_display:
        st.dataframe(tasks_df_for_display[actual_task_cols_display], use_container_width=True, height=min(420, len(tasks_df_for_display) * 38 + 58), hide_index=True,
                     column_config={"priority_score": st.column_config.NumberColumn(format="%.1f"), "due_date": st.column_config.DateColumn(format="YYYY-MM-DD")})
    elif not tasks_df_for_display.empty: st.warning("Task data available but cannot display due to column config issues.")
elif not daily_df.empty: st.info("No high-priority tasks identified for action based on current data.")
else: st.markdown("_No activity data to generate tasks for today._")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_df.empty:
    epi_signals_map_data = {} # Initialize
    try:
        epi_signals_map_data = extract_chw_epi_signals(daily_df, daily_kpis_precalc, selected_daily_date, actual_zone_filter or "All Zones", 3)
    except Exception as e_epi_chw: logger.error(f"Error extracting epi signals: {e_epi_chw}", exc_info=True)
    
    epi_kpi_cols_chw_page = st.columns(3) # Renamed to avoid conflict
    with epi_kpi_cols_chw_page[0]: render_kpi_card("Symptomatic (Key Cond.)", str(epi_signals_map_data.get("symptomatic_patients_key_conditions_count", 0)), "🤒", units="cases", help_text=f"Patients seen today with symptoms related to key conditions.")
    new_malaria_chw = epi_signals_map_data.get("newly_identified_malaria_patients_count", 0)
    malaria_stat_chw = "HIGH_CONCERN" if new_malaria_chw > 1 else ("MODERATE_CONCERN" if new_malaria_chw == 1 else "ACCEPTABLE")
    with epi_kpi_cols_chw_page[1]: render_kpi_card("New Malaria Cases", str(new_malaria_chw), "🦟", malaria_stat_chw, units="cases", help_text="New malaria cases identified today.")
    pending_tb_chw = epi_signals_map_data.get("pending_tb_contact_tracing_tasks_count", 0)
    tb_stat_chw = "MODERATE_CONCERN" if pending_tb_chw > 0 else "ACCEPTABLE"
    with epi_kpi_cols_chw_page[2]: render_kpi_card("Pending TB Contacts", str(pending_tb_chw), "👥", tb_stat_chw, units="to trace", help_text="TB contacts needing follow-up.")

    symptom_clusters_list = epi_signals_map_data.get("detected_symptom_clusters", [])
    if symptom_clusters_list:
        st.markdown("###### Detected Symptom Clusters (Supervisor to verify):")
        for cluster_item in symptom_clusters_list: st.warning(f"⚠️ **Pattern: {cluster_item.get('symptoms_pattern', 'Unknown')}** - {cluster_item.get('patient_count', 'N/A')} cases in {cluster_item.get('location_hint', 'CHW area')}. Supervisor to verify/escalate.")
    elif 'patient_reported_symptoms' in daily_df.columns and daily_df['patient_reported_symptoms'].notna().any():
        st.info("No significant symptom clusters detected today based on current criteria.")
else: st.markdown("_No activity data for local epi signals for selected date/filters._")
st.divider()

# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_chw = f"{trend_start_filt.strftime('%d %b %Y')} - {trend_end_filt.strftime('%d %b %Y')}"
trend_filter_context_chw = f" for CHW **{actual_chw_filter}**" if actual_chw_filter else ""
trend_filter_context_chw += f" in Zone **{actual_zone_filter}**" if actual_zone_filter else ""
trend_filter_context_chw = trend_filter_context_chw or " (All CHWs/Zones)"
st.markdown(f"Displaying trends from **{trend_period_display_chw}**{trend_filter_context_chw}.")

if not period_df.empty:
    activity_trends_map_data = {} # Initialize
    try:
        activity_trends_map_data = calculate_chw_activity_trends_data(period_df, trend_start_filt, trend_end_filt, actual_zone_filter, 'D') # Daily aggregation
    except Exception as e_trends_chw: logger.error(f"Error calculating activity trends: {e_trends_chw}", exc_info=True)
    
    trend_plot_cols_chw_page = st.columns(2)
    with trend_plot_cols_chw_page[0]:
        visits_trend_series = activity_trends_map_data.get("patient_visits_trend")
        if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart(visits_trend_series, "Daily Patient Visits Trend", "Unique Patients Visited", y_values_are_counts=True), use_container_width=True)
        else: st.caption("No patient visit trend data available for this selection.")
    with trend_plot_cols_chw_page[1]:
        prio_trend_series = activity_trends_map_data.get("high_priority_followups_trend")
        if isinstance(prio_trend_series, pd.Series) and not prio_trend_series.empty:
            st.plotly_chart(plot_annotated_line_chart(prio_trend_series, "Daily High Prio. Follow-ups Trend", "High Prio. Follow-ups", y_values_are_counts=True), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available for this selection.")
else: st.markdown("_No historical data available for the selected trend period and/or filters._")

logger.info(f"CHW Supervisor Dashboard page loaded. Filters: Date={selected_daily_date}, CHW={actual_chw_filter or 'All'}, Zone={actual_zone_filter or 'All'}, Trend=({trend_start_filt} to {trend_end_filt}).")
