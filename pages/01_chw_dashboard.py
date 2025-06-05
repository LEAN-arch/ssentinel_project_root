# sentinel_project_root/pages/01_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta 
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path 

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
except ImportError as e_chw_dash_final_fix_v4: 
    import sys
    _current_file_chw_final_v4 = Path(__file__).resolve()
    _project_root_chw_assumption_final_v4 = _current_file_chw_final_v4.parent.parent 
    error_msg_chw_detail_final_v4 = (
        f"CHW Dashboard Import Error: {e_chw_dash_final_fix_v4}. "
        f"Ensure project root ('{_project_root_chw_assumption_final_v4}') is in sys.path (handled by app.py) "
        f"and all modules/packages have `__init__.py` files. Check for typos in import paths. "
        f"Current Python Path: {sys.path}"
    )
    try: st.error(error_msg_chw_detail_final_v4); st.stop()
    except NameError: print(error_msg_chw_detail_final_v4, file=sys.stderr); raise

logger = logging.getLogger(__name__)

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME}**")
st.divider()

def _create_filter_dropdown_options_chw_page_v4( 
    df: Optional[pd.DataFrame], col: str, defaults: List[str], name_plural: str
) -> List[str]:
    opts = [f"All {name_plural}"]
    if isinstance(df, pd.DataFrame) and not df.empty and col in df.columns:
        unique_vals = sorted(list(set(str(v) for v in df[col].dropna() if str(v).strip()))) 
        if unique_vals: opts.extend(unique_vals)
        else: logger.warning(f"CHW Filters: Col '{col}' for '{name_plural}' empty. Using defaults."); opts.extend(defaults)
    else: logger.warning(f"CHW Filters: Col '{col}' not in DF or DF empty for '{name_plural}'. Using defaults."); opts.extend(defaults)
    return opts

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...", hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_chw_dashboard_page_data_v4( 
    view_date: date, 
    trend_start: date, 
    trend_end: date, 
    chw_filter: Optional[str], 
    zone_filter: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "CHWDashPageDataV4" 
    logger.info(f"({log_ctx}) Loading data. View: {view_date}, Trend: {trend_start}-{trend_end}, CHW: {chw_filter or 'All'}, Zone: {zone_filter or 'All'}")
    
    all_health_df = load_health_records(source_context=f"{log_ctx}/LoadRecs")
    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        logger.error(f"({log_ctx}) CRITICAL: Health records failed to load or are empty. CSV expected at: {settings.HEALTH_RECORDS_CSV_PATH}")
        st.error(f"CRITICAL DATA ERROR: Could not load health records. Ensure '{Path(settings.HEALTH_RECORDS_CSV_PATH).name}' is in 'data_sources/' and is not empty. Dashboard functionality will be severely limited.")
        return pd.DataFrame(), pd.DataFrame(), {}
    if 'encounter_date' not in all_health_df.columns or not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' invalid in loaded health records."); st.error("Data Integrity Error: 'encounter_date' invalid.")
        return pd.DataFrame(), pd.DataFrame(), {}

    if all_health_df['encounter_date'].dt.tz is not None:
        logger.debug(f"({log_ctx}) Converting encounter_date to timezone-naive for filtering.")
        all_health_df['encounter_date'] = all_health_df['encounter_date'].dt.tz_localize(None)

    df_daily_page_v4 = all_health_df[all_health_df['encounter_date'].dt.date == view_date].copy()
    if chw_filter and 'chw_id' in df_daily_page_v4.columns: df_daily_page_v4 = df_daily_page_v4[df_daily_page_v4['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_daily_page_v4.columns: df_daily_page_v4 = df_daily_page_v4[df_daily_page_v4['zone_id'] == zone_filter]

    df_trend_page_v4 = all_health_df[(all_health_df['encounter_date'].dt.date >= trend_start) & (all_health_df['encounter_date'].dt.date <= trend_end)].copy()
    if chw_filter and 'chw_id' in df_trend_page_v4.columns: df_trend_page_v4 = df_trend_page_v4[df_trend_page_v4['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_trend_page_v4.columns: df_trend_page_v4 = df_trend_page_v4[df_trend_page_v4['zone_id'] == zone_filter]

    pre_calc_kpis_page_v4: Dict[str, Any] = {}
    if chw_filter and not df_daily_page_v4.empty:
        enc_type_series_v4 = df_daily_page_v4.get('encounter_type', pd.Series(dtype=str)).astype(str)
        chw_id_series_page_v4 = df_daily_page_v4.get('chw_id', pd.Series(dtype=str))
        self_checks_page_v4 = df_daily_page_v4[(chw_id_series_page_v4 == chw_filter) & (enc_type_series_v4.str.contains('WORKER_SELF_CHECK', case=False, na=False))]
        if not self_checks_page_v4.empty:
            fatigue_cols_page_v4 = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            actual_fatigue_col_page_v4 = next((c for c in fatigue_cols_page_v4 if c in self_checks_page_v4.columns and self_checks_page_v4[c].notna().any()), None)
            pre_calc_kpis_page_v4['worker_self_fatigue_index_today'] = self_checks_page_v4[actual_fatigue_col_page_v4].max() if actual_fatigue_col_page_v4 else np.nan
        else: pre_calc_kpis_page_v4['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(f"({log_ctx}) Data loaded for CHW page. Daily: {len(df_daily_page_v4)} recs, Trend: {len(df_trend_page_v4)} recs.")
    return df_daily_page_v4, df_trend_page_v4, pre_calc_kpis_page_v4

# --- Sidebar Filters ---
project_root_chw_dash_page_v4 = Path(settings.PROJECT_ROOT_DIR)
logo_path_chw_sidebar_page_v4 = Path(settings.APP_LOGO_SMALL_PATH) 
if logo_path_chw_sidebar_page_v4.is_file(): st.sidebar.image(str(logo_path_chw_sidebar_page_v4), width=230)
else: logger.warning(f"Sidebar logo for CHW Dash not found: {logo_path_chw_sidebar_page_v4}")
st.sidebar.header("Dashboard Filters")

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS) 
def load_chw_filter_options_data_v4():
    logger.info("CHW Page: Loading data for filter dropdowns.")
    df_filt = load_health_records(source_context="CHWDashPage/FilterPopulationV4")
    if not isinstance(df_filt, pd.DataFrame): return pd.DataFrame() 
    if 'encounter_date' in df_filt.columns and not pd.api.types.is_datetime64_any_dtype(df_filt['encounter_date']):
        df_filt['encounter_date'] = pd.to_datetime(df_filt['encounter_date'], errors='coerce')
    if 'encounter_date' in df_filt.columns and df_filt['encounter_date'].dt.tz is not None: 
        df_filt['encounter_date'] = df_filt['encounter_date'].dt.tz_localize(None)
    return df_filt

df_for_chw_filters_page_v4 = load_chw_filter_options_data_v4()

chw_options_list_page_v4 = _create_filter_dropdown_options_chw_page_v4(df_for_chw_filters_page_v4, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
chw_session_key_page_v4 = "chw_dashboard_chw_id_v6" # Ensure unique key
if chw_session_key_page_v4 not in st.session_state or st.session_state[chw_session_key_page_v4] not in chw_options_list_page_v4 : st.session_state[chw_session_key_page_v4] = chw_options_list_page_v4[0]
selected_chw_ui_page_v4 = st.sidebar.selectbox("Filter by CHW ID:", options=chw_options_list_page_v4, key=f"{chw_session_key_page_v4}_widget", index=chw_options_list_page_v4.index(st.session_state[chw_session_key_page_v4]))
st.session_state[chw_session_key_page_v4] = selected_chw_ui_page_v4
actual_chw_filter_page_v4 = None if selected_chw_ui_page_v4.startswith("All ") else selected_chw_ui_page_v4

zone_options_list_page_v4 = _create_filter_dropdown_options_chw_page_v4(df_for_chw_filters_page_v4, 'zone_id', ["ZoneA", "ZoneB", "ZoneC"], "Zones")
zone_session_key_page_v4 = "chw_dashboard_zone_id_v6"
if zone_session_key_page_v4 not in st.session_state or st.session_state[zone_session_key_page_v4] not in zone_options_list_page_v4: st.session_state[zone_session_key_page_v4] = zone_options_list_page_v4[0]
selected_zone_ui_page_v4 = st.sidebar.selectbox("Filter by Zone:", options=zone_options_list_page_v4, key=f"{zone_session_key_page_v4}_widget", index=zone_options_list_page_v4.index(st.session_state[zone_session_key_page_v4]))
st.session_state[zone_session_key_page_v4] = selected_zone_ui_page_v4
actual_zone_filter_page_v4 = None if selected_zone_ui_page_v4.startswith("All ") else selected_zone_ui_page_v4

abs_min_date_chw_dash_page_v4 = date(2022, 1, 1); abs_max_date_chw_dash_page_v4 = date.today()
data_min_date_chw_page_val_v2, data_max_date_chw_page_val_v2 = abs_min_date_chw_dash_page_v4, abs_max_date_chw_dash_page_v4
if not df_for_chw_filters_page_v4.empty and 'encounter_date' in df_for_chw_filters_page_v4.columns and \
   pd.api.types.is_datetime64_any_dtype(df_for_chw_filters_page_v4['encounter_date']) and \
   df_for_chw_filters_page_v4['encounter_date'].notna().any():
    try:
        data_min_date_chw_page_val_v2 = df_for_chw_filters_page_v4['encounter_date'].min().date()
        data_max_date_chw_page_val_v2 = df_for_chw_filters_page_v4['encounter_date'].max().date()
        if data_min_date_chw_page_val_v2 > data_max_date_chw_page_val_v2: data_min_date_chw_page_val_v2 = data_max_date_chw_page_val_v2
    except Exception as e_date_minmax_chw_v4: logger.warning(f"Error deriving min/max dates for CHW Dash: {e_date_minmax_chw_v4}")

daily_date_key_chw_dash_page_v4 = "chw_dashboard_daily_date_v6"
default_daily_date_v4 = data_max_date_chw_page_val_v2
if daily_date_key_chw_dash_page_v4 not in st.session_state: st.session_state[daily_date_key_chw_dash_page_v4] = default_daily_date_v4
selected_daily_date_page_val_v4 = st.sidebar.date_input("View Daily Activity For:", value=st.session_state[daily_date_key_chw_dash_page_v4], min_value=data_min_date_chw_page_val_v2, max_value=data_max_date_chw_page_val_v2, key=f"{daily_date_key_chw_dash_page_v4}_widget")
st.session_state[daily_date_key_chw_dash_page_v4] = selected_daily_date_page_val_v4

trend_date_key_chw_dash_page_v4 = "chw_dashboard_trend_date_range_v6"
def_trend_end_chw_dash_page_v4 = selected_daily_date_page_val_v4
def_trend_start_chw_dash_page_v4 = max(data_min_date_chw_page_val_v2, def_trend_end_chw_dash_page_v4 - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))
if trend_date_key_chw_dash_page_v4 not in st.session_state: st.session_state[trend_date_key_chw_dash_page_v4] = [def_trend_start_chw_dash_page_v4, def_trend_end_chw_dash_page_v4]
selected_trend_range_page_val_v4 = st.sidebar.date_input("Select Trend Date Range:", value=st.session_state[trend_date_key_chw_dash_page_v4], min_value=data_min_date_chw_page_val_v2, max_value=data_max_date_chw_page_val_v2, key=f"{trend_date_key_chw_dash_page_v4}_widget")

trend_start_filt_page_val_v4: date; trend_end_filt_page_val_v4: date 
if isinstance(selected_trend_range_page_val_v4, (list, tuple)) and len(selected_trend_range_page_val_v4) == 2:
    st.session_state[trend_date_key_chw_dash_page_v4] = selected_trend_range_page_val_v4; trend_start_filt_page_val_v4, trend_end_filt_page_val_v4 = selected_trend_range_page_val_v4
else: trend_start_filt_page_val_v4, trend_end_filt_page_val_v4 = st.session_state[trend_date_key_chw_dash_page_v4]; st.sidebar.warning("Trend date range error.")
if trend_start_filt_page_val_v4 > trend_end_filt_page_val_v4: st.sidebar.error("Trend Start <= End."); trend_end_filt_page_val_v4 = trend_start_filt_page_val_v4; st.session_state[trend_date_key_chw_dash_page_v4][1] = trend_end_filt_page_val_v4

# --- Load Data Based on Filters ---
daily_df_chw_page_display_v4, period_df_chw_page_display_v4, daily_kpis_precalc_chw_page_v4 = pd.DataFrame(), pd.DataFrame(), {} 
try:
    daily_df_chw_page_display_v4, period_df_chw_page_display_v4, daily_kpis_precalc_chw_page_v4 = get_chw_dashboard_page_data_v4(
        selected_daily_date_page_val_v4, trend_start_filt_page_val_v4, trend_end_filt_page_val_v4, 
        actual_chw_filter_page_v4, actual_zone_filter_page_v4
    )
except Exception as e_load_chw_main_page_v4:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_load_chw_main_page_v4}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {str(e_load_chw_main_page_v4)}. Please check logs.")

# --- Display Filter Context ---
filter_ctx_parts_chw_page_val_v4 = [f"Snapshot Date: **{selected_daily_date_page_val_v4.strftime('%d %b %Y')}**"]
if actual_chw_filter_page_v4: filter_ctx_parts_chw_page_val_v4.append(f"CHW: **{actual_chw_filter_page_v4}**")
if actual_zone_filter_page_v4: filter_ctx_parts_chw_page_val_v4.append(f"Zone: **{actual_zone_filter_page_v4}**")
st.info(f"Displaying data for: {', '.join(filter_ctx_parts_chw_page_val_v4)}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df_chw_page_display_v4.empty:
    daily_summary_page_val_v4 = {} 
    try:
        daily_summary_page_val_v4 = calculate_chw_daily_summary_metrics(daily_df_chw_page_display_v4, selected_daily_date_page_val_v4, daily_kpis_precalc_chw_page_v4, "CHWDash/DailySummaryV4")
    except Exception as e_daily_summary_chw_page_v4: logger.error(f"Error calculating CHW daily summary: {e_daily_summary_chw_page_v4}", exc_info=True); st.warning("Could not calculate daily summary metrics.")
    
    kpi_cols_daily_snapshot_page_val_v4 = st.columns(4)
    with kpi_cols_daily_snapshot_page_val_v4[0]: render_kpi_card("Visits Today", str(daily_summary_page_val_v4.get("visits_count", "0")), "👥", help_text="Total unique patients encountered.")
    prio_fups_page_val_v4 = daily_summary_page_val_v4.get("high_ai_prio_followups_count", 0)
    prio_status_level_page_val_v4 = "ACCEPTABLE" if prio_fups_page_val_v4 <= 2 else ("MODERATE_CONCERN" if prio_fups_page_val_v4 <= 5 else "HIGH_CONCERN")
    with kpi_cols_daily_snapshot_page_val_v4[1]: render_kpi_card("High Prio Follow-ups", str(prio_fups_page_val_v4), "🎯", prio_status_level_page_val_v4, help_text=f"Patients needing urgent follow-up (AI prio ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}).")
    critical_spo2_cases_page_val_v4 = daily_summary_page_val_v4.get("critical_spo2_cases_identified_count", 0)
    spo2_status_level_page_val_v4 = "HIGH_CONCERN" if critical_spo2_cases_page_val_v4 > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot_page_val_v4[2]: render_kpi_card("Critical SpO2 Cases", str(critical_spo2_cases_page_val_v4), "💨", spo2_status_level_page_val_v4, help_text=f"Patients with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    high_fever_cases_page_val_v4 = daily_summary_page_val_v4.get("high_fever_cases_identified_count", 0)
    fever_status_level_page_val_v4 = "HIGH_CONCERN" if high_fever_cases_page_val_v4 > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot_page_val_v4[3]: render_kpi_card("High Fever Cases", str(high_fever_cases_page_val_v4), "🔥", fever_status_level_page_val_v4, help_text=f"Patients with temp ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else: st.markdown("_No activity data for selected date/filters for daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
chw_alerts_list_s2_v4 = []
if not daily_df_chw_page_display_v4.empty:
    try: chw_alerts_list_s2_v4 = generate_chw_alerts(daily_df_chw_page_display_v4, selected_daily_date_page_val_v4, actual_zone_filter_page_v4 or "All Zones", 10)
    except Exception as e_alerts_page_chw_v4: logger.error(f"CHW Dashboard: Error generating patient alerts: {e_alerts_page_chw_v4}", exc_info=True); st.warning("Could not generate patient alerts.")
if chw_alerts_list_s2_v4:
    st.subheader("Priority Patient Alerts (Today):")
    critical_alerts_exist_chw_v4 = False
    for alert_item_v4 in chw_alerts_list_s2_v4:
        if alert_item_v4.get("alert_level") == "CRITICAL":
            critical_alerts_exist_chw_v4 = True
            render_traffic_light_indicator(f"Pt. {alert_item_v4.get('patient_id', 'N/A')}: {alert_item_v4.get('primary_reason', 'Critical Alert')}", "HIGH_RISK", f"Details: {alert_item_v4.get('brief_details','N/A')} | Context: {alert_item_v4.get('context_info','N/A')} | Action: {alert_item_v4.get('suggested_action_code','REVIEW')}")
    if not critical_alerts_exist_chw_v4: st.info("No CRITICAL patient alerts identified for this selection today.")
    warning_alerts_list_v4 = [a_chw_v4 for a_chw_v4 in chw_alerts_list_s2_v4 if a_chw_v4.get("alert_level") == "WARNING"]
    if warning_alerts_list_v4:
        st.markdown("###### Warning Level Alerts:")
        for warn_item_v4 in warning_alerts_list_v4: render_traffic_light_indicator(f"Pt. {warn_item_v4.get('patient_id', 'N/A')}: {warn_item_v4.get('primary_reason', 'Warning')}", "MODERATE_CONCERN", f"Details: {warn_item_v4.get('brief_details','N/A')} | Context: {warn_item_v4.get('context_info','N/A')}")
    elif not critical_alerts_exist_chw_v4 : st.info("Only informational alerts (if any) generated.")
elif not daily_df_chw_page_display_v4.empty : st.success("✅ No significant patient alerts needing immediate attention generated for today's selection.")
else: st.markdown("_No activity data to generate patient alerts for today._")

chw_tasks_list_s2_v4 = []
if not daily_df_chw_page_display_v4.empty:
    try: chw_tasks_list_s2_v4 = generate_chw_tasks(daily_df_chw_page_display_v4, selected_daily_date_page_val_v4, actual_chw_filter_page_v4, actual_zone_filter_page_v4 or "All Zones", 10)
    except Exception as e_tasks_page_chw_v4: logger.error(f"CHW Dashboard: Error generating CHW tasks: {e_tasks_page_chw_v4}", exc_info=True); st.warning("Could not generate tasks list.")
if chw_tasks_list_s2_v4:
    st.subheader("Top Priority Tasks (Today/Next Day):")
    tasks_df_for_table_val_v4 = pd.DataFrame(chw_tasks_list_s2_v4)
    task_table_cols_order_val_v4 = ['patient_id', 'task_description', 'priority_score', 'due_date', 'status', 'key_patient_context', 'assigned_chw_id'] 
    actual_cols_for_task_table_val_v4 = [col_task_v4 for col_task_v4 in task_table_cols_order_val_v4 if col_task_v4 in tasks_df_for_table_val_v4.columns]
    if not tasks_df_for_table_val_v4.empty and actual_cols_for_task_table_val_v4:
        st.dataframe(tasks_df_for_table_val_v4[actual_cols_for_task_table_val_v4], use_container_width=True, height=min(420, len(tasks_df_for_table_val_v4) * 38 + 58), hide_index=True,
                     column_config={"priority_score": st.column_config.NumberColumn(format="%.1f"), "due_date": st.column_config.DateColumn(format="YYYY-MM-DD")})
    elif not tasks_df_for_table_val_v4.empty: st.warning("Task data available but cannot display due to column config issues.")
elif not daily_df_chw_page_display_v4.empty : st.info("No high-priority tasks identified based on current data.")
else: st.markdown("_No activity data to generate tasks for today._")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_df_chw_page_display_v4.empty:
    epi_signals_map_s3_v4 = {} 
    try: epi_signals_map_s3_v4 = extract_chw_epi_signals(daily_df_chw_page_display_v4, daily_kpis_precalc_chw_page_v4, selected_daily_date_page_val_v4, actual_zone_filter_page_v4 or "All Zones", 3)
    except Exception as e_epi_display_chw_page_v4: logger.error(f"CHW Dashboard: Error extracting epi signals: {e_epi_display_chw_page_v4}", exc_info=True); st.warning("Could not extract epi signals.")
    
    epi_kpi_cols_s3_v4 = st.columns(3)
    with epi_kpi_cols_s3_v4[0]: render_kpi_card("Symptomatic (Key Cond.)", str(epi_signals_map_s3_v4.get("symptomatic_patients_key_conditions_count", "N/A")), "🤒", units="cases today", help_text=f"Patients seen today with symptoms related to key conditions.")
    new_malaria_val_chw_v4 = epi_signals_map_s3_v4.get("newly_identified_malaria_patients_count", 0)
    malaria_status_level_chw_epi_v4 = "HIGH_CONCERN" if new_malaria_val_chw_v4 > 1 else ("MODERATE_CONCERN" if new_malaria_val_chw_v4 == 1 else "ACCEPTABLE")
    with epi_kpi_cols_s3_v4[1]: render_kpi_card("New Malaria Cases", str(new_malaria_val_chw_v4), "🦟", malaria_status_level_chw_epi_v4, units="cases today", help_text="New malaria cases identified today.")
    pending_tb_contacts_val_chw_v4 = epi_signals_map_s3_v4.get("pending_tb_contact_tracing_tasks_count", 0)
    tb_contact_status_level_chw_v4 = "MODERATE_CONCERN" if pending_tb_contacts_val_chw_v4 > 0 else "ACCEPTABLE"
    with epi_kpi_cols_s3_v4[2]: render_kpi_card("Pending TB Contacts", str(pending_tb_contacts_val_chw_v4), "👥", tb_contact_status_level_chw_v4, units="to trace", help_text="TB contacts needing follow-up.")

    detected_symptom_clusters_list_val_v4 = epi_signals_map_s3_v4.get("detected_symptom_clusters", [])
    if detected_symptom_clusters_list_val_v4:
        st.markdown("###### Detected Symptom Clusters (Requires Verification by Supervisor):")
        for cluster_item_data_val_v4 in detected_symptom_clusters_list_val_v4: st.warning(f"⚠️ **Pattern: {cluster_item_data_val_v4.get('symptoms_pattern', 'Unknown')}** - {cluster_item_data_val_v4.get('patient_count', 'N/A')} cases in {cluster_item_data_val_v4.get('location_hint', 'CHW area')}. Supervisor to verify.")
    elif isinstance(daily_df_chw_page_display_v4, pd.DataFrame) and 'patient_reported_symptoms' in daily_df_chw_page_display_v4.columns and daily_df_chw_page_display_v4['patient_reported_symptoms'].notna().any():
        st.info("No significant symptom clusters detected today based on current data and criteria.")
else: st.markdown("_No activity data for local epi signals for selected date/filters._")
st.divider()

# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_text_page_val_v4 = f"{trend_start_filt_page_val_v4.strftime('%d %b %Y')} - {trend_end_filt_page_val_v4.strftime('%d %b %Y')}"
trend_filter_context_text_page_val_v2 = f" for CHW **{actual_chw_filter_page_v4}**" if actual_chw_filter_page_v4 else "" 
trend_filter_context_text_page_val_v2 += f" in Zone **{actual_zone_filter_page_v4}**" if actual_zone_filter_page_v4 else ""
trend_filter_context_text_page_val_v2 = trend_filter_context_text_page_val_v2 or " (All CHWs/Zones)" 
st.markdown(f"Displaying trends from **{trend_period_display_text_page_val_v4}**{trend_filter_context_text_page_val_v2}.")
if not period_df_chw_page_display_v4.empty:
    activity_trends_map_s4_v4 = {}
    try: activity_trends_map_s4_v4 = calculate_chw_activity_trends_data(period_df_chw_page_display_v4, trend_start_filt_page_val_v4, trend_end_filt_page_val_v4, actual_zone_filter_page_v4, 'D')
    except Exception as e_trends_display_chw_page_v4: logger.error(f"CHW Dashboard: Error calculating activity trends: {e_trends_display_chw_page_v4}", exc_info=True); st.warning("Could not calculate activity trends.")
    
    trend_plot_cols_display_val_v4 = st.columns(2)
    with trend_plot_cols_display_val_v4[0]:
        patient_visits_trend_series_val_v4 = activity_trends_map_s4_v4.get("patient_visits_trend")
        if isinstance(patient_visits_trend_series_val_v4, pd.Series) and not patient_visits_trend_series_val_v4.empty:
            st.plotly_chart(plot_annotated_line_chart(patient_visits_trend_series_val_v4, "Daily Patient Visits Trend", "Unique Patients Visited", y_values_are_counts=True), use_container_width=True)
        else: st.caption("No patient visit trend data available for this selection.")
    with trend_plot_cols_display_val_v4[1]:
        high_prio_followups_trend_srs_val_v4 = activity_trends_map_s4_v4.get("high_priority_followups_trend")
        if isinstance(high_prio_followups_trend_srs_val_v4, pd.Series) and not high_prio_followups_trend_srs_val_v4.empty:
            st.plotly_chart(plot_annotated_line_chart(high_prio_followups_trend_srs_val_v4, "Daily High Prio. Follow-ups Trend", "High Prio. Follow-ups", y_values_are_counts=True), use_container_width=True)
        else: st.caption("No high-priority follow-up trend data available for this selection.")
else: st.markdown("_No historical data available for the selected trend period and/or filters._")

logger.info(f"CHW Supervisor Dashboard page fully rendered. Data available: Daily={not daily_df_chw_page_display_v4.empty}, Period={not period_df_chw_page_display_v4.empty}")
