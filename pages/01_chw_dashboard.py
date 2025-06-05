# sentinel_project_root/pages/01_chw_dashboard.py (assuming prefixed filename)
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
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.alert_generation import generate_chw_alerts
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import calculate_chw_activity_trends_data
except ImportError as e_chw_dash_final_fix:
    import sys
    _root_dir_chw_assumption = Path(__file__).resolve().parent.parent
    st.error(f"CHW Dashboard Import Error: {e_chw_dash_final_fix}. Project Root: '{_root_dir_chw_assumption}', SysPath: {sys.path}")
    st.stop()

logger = logging.getLogger(__name__)

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME}**")
st.divider()

def _create_filter_dropdown_options_chw(df: Optional[pd.DataFrame], col: str, defaults: List[str], name: str) -> List[str]:
    opts = [f"All {name}"]
    if isinstance(df, pd.DataFrame) and not df.empty and col in df.columns:
        unique_vals = sorted([str(v) for v in df[col].dropna().unique()])
        if unique_vals: opts.extend(unique_vals)
        else: logger.warning(f"CHW Filters: Col '{col}' for '{name}' empty. Using defaults."); opts.extend(defaults)
    else: logger.warning(f"CHW Filters: Col '{col}' not in DF or DF empty for '{name}'. Using defaults."); opts.extend(defaults)
    return opts

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...")
def get_chw_dashboard_data(view_date: date, trend_start: date, trend_end: date, chw_filter: Optional[str], zone_filter: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "CHWDashData"
    logger.info(f"({log_ctx}) Loading data. View: {view_date}, Trend: {trend_start}-{trend_end}, CHW: {chw_filter or 'All'}, Zone: {zone_filter or 'All'}")
    
    all_health_df = load_health_records(source_context=f"{log_ctx}/LoadRecs") # Path from settings
    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        logger.error(f"({log_ctx}) CRITICAL: Health records failed to load or are empty. CSV expected at: {settings.HEALTH_RECORDS_CSV_PATH}")
        st.error(f"CRITICAL DATA ERROR: Could not load health records. Please ensure '{Path(settings.HEALTH_RECORDS_CSV_PATH).name}' is in the 'data_sources' directory and is not empty. Dashboard functionality will be severely limited.")
        return pd.DataFrame(), pd.DataFrame(), {}
    if 'encounter_date' not in all_health_df.columns or not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' invalid in health records."); st.error("Data Integrity Error: 'encounter_date' invalid.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Daily Snapshot
    df_daily_chw = all_health_df[all_health_df['encounter_date'].dt.date == view_date].copy()
    if chw_filter and 'chw_id' in df_daily_chw.columns: df_daily_chw = df_daily_chw[df_daily_chw['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_daily_chw.columns: df_daily_chw = df_daily_chw[df_daily_chw['zone_id'] == zone_filter]

    # Trend Period
    df_trend_chw = all_health_df[(all_health_df['encounter_date'].dt.date >= trend_start) & (all_health_df['encounter_date'].dt.date <= trend_end)].copy()
    if chw_filter and 'chw_id' in df_trend_chw.columns: df_trend_chw = df_trend_chw[df_trend_chw['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_trend_chw.columns: df_trend_chw = df_trend_chw[df_trend_chw['zone_id'] == zone_filter]

    pre_calc_kpis_chw: Dict[str, Any] = {}
    if chw_filter and not df_daily_chw.empty:
        self_checks = df_daily_chw[(df_daily_chw.get('chw_id') == chw_filter) & (df_daily_chw.get('encounter_type', pd.Series(dtype=str)).astype(str).str.contains('WORKER_SELF_CHECK', case=False, na=False))]
        if not self_checks.empty:
            fatigue_col_name = next((c for c in ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score'] if c in self_checks.columns and self_checks[c].notna().any()), None)
            pre_calc_kpis_chw['worker_self_fatigue_index_today'] = self_checks[fatigue_col_name].max() if fatigue_col_name else np.nan
        else: pre_calc_kpis_chw['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(f"({log_ctx}) Data loaded. Daily: {len(df_daily_chw)} recs, Trend: {len(df_trend_chw)} recs.")
    return df_daily_chw, df_trend_chw, pre_calc_kpis_chw

# --- Sidebar Filters ---
# Resolve logo path using PROJECT_ROOT_DIR from settings
project_root_chw_dash = Path(settings.PROJECT_ROOT_DIR)
logo_path_chw_sidebar = project_root_chw_dash / settings.APP_LOGO_SMALL_PATH
if logo_path_chw_sidebar.is_file(): st.sidebar.image(str(logo_path_chw_sidebar), width=120)
else: logger.warning(f"Sidebar logo for CHW Dash not found: {logo_path_chw_sidebar}")
st.sidebar.header("Dashboard Filters")

# Load data for filter population ONCE.
# If this fails due to missing CSV, subsequent operations will use empty DFs or defaults.
df_for_chw_filter_options = load_health_records(source_context="CHWDash/FilterPopulation")
if not isinstance(df_for_chw_filter_options, pd.DataFrame): df_for_chw_filter_options = pd.DataFrame() # Ensure it's a DataFrame

chw_options_list = _create_filter_dropdown_options_chw(df_for_chw_filter_options, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
chw_session_key_chw = "chw_dashboard_chw_id_v4"
if chw_session_key_chw not in st.session_state or st.session_state[chw_session_key_chw] not in chw_options_list : st.session_state[chw_session_key_chw] = chw_options_list[0]
selected_chw_ui_chw = st.sidebar.selectbox("Filter by CHW ID:", options=chw_options_list, key=f"{chw_session_key_chw}_widget", index=chw_options_list.index(st.session_state[chw_session_key_chw]))
st.session_state[chw_session_key_chw] = selected_chw_ui_chw
actual_chw_filter_val = None if selected_chw_ui_chw.startswith("All ") else selected_chw_ui_chw

zone_options_list = _create_filter_dropdown_options_chw(df_for_chw_filter_options, 'zone_id', ["ZoneA", "ZoneB", "ZoneC"], "Zones")
zone_session_key_chw = "chw_dashboard_zone_id_v4"
if zone_session_key_chw not in st.session_state or st.session_state[zone_session_key_chw] not in zone_options_list: st.session_state[zone_session_key_chw] = zone_options_list[0]
selected_zone_ui_chw = st.sidebar.selectbox("Filter by Zone:", options=zone_options_list, key=f"{zone_session_key_chw}_widget", index=zone_options_list.index(st.session_state[zone_session_key_chw]))
st.session_state[zone_session_key_chw] = selected_zone_ui_chw
actual_zone_filter_val = None if selected_zone_ui_chw.startswith("All ") else selected_zone_ui_chw

# Date Pickers - use hardcoded safe defaults if data-derived dates are problematic
abs_min_date_chw_dash = date(2022, 1, 1) # Further back default
abs_max_date_chw_dash = date.today()
# Try to get min/max from data, but have robust fallbacks
data_min_date_chw, data_max_date_chw = abs_min_date_chw_dash, abs_max_date_chw_dash
if isinstance(df_for_chw_filter_options, pd.DataFrame) and 'encounter_date' in df_for_chw_filter_options.columns and \
   pd.api.types.is_datetime64_any_dtype(df_for_chw_filter_options['encounter_date']) and \
   df_for_chw_filter_options['encounter_date'].notna().any():
    data_min_date_chw = df_for_chw_filter_options['encounter_date'].min().date()
    data_max_date_chw = df_for_chw_filter_options['encounter_date'].max().date()
    if data_min_date_chw > data_max_date_chw: data_min_date_chw = data_max_date_chw # Ensure min <= max

daily_date_key_chw_dash = "chw_dashboard_daily_date_v4"
if daily_date_key_chw_dash not in st.session_state: st.session_state[daily_date_key_chw_dash] = data_max_date_chw # Default to latest data date or today
selected_daily_date_val = st.sidebar.date_input("View Daily Activity For:", value=st.session_state[daily_date_key_chw_dash], min_value=data_min_date_chw, max_value=data_max_date_chw, key=f"{daily_date_key_chw_dash}_widget")
st.session_state[daily_date_key_chw_dash] = selected_daily_date_val

trend_date_key_chw_dash = "chw_dashboard_trend_date_range_v4"
def_trend_end_chw_dash = selected_daily_date_val
def_trend_start_chw_dash = max(data_min_date_chw, def_trend_end_chw_dash - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))
if trend_date_key_chw_dash not in st.session_state: st.session_state[trend_date_key_chw_dash] = [def_trend_start_chw_dash, def_trend_end_chw_dash]
selected_trend_range_val = st.sidebar.date_input("Select Trend Date Range:", value=st.session_state[trend_date_key_chw_dash], min_value=data_min_date_chw, max_value=data_max_date_chw, key=f"{trend_date_key_chw_dash}_widget")

trend_start_filt_val: date; trend_end_filt_val: date 
if isinstance(selected_trend_range_val, (list, tuple)) and len(selected_trend_range_val) == 2:
    st.session_state[trend_date_key_chw_dash] = selected_trend_range_val; trend_start_filt_val, trend_end_filt_val = selected_trend_range_val
else: trend_start_filt_val, trend_end_filt_val = st.session_state[trend_date_key_chw_dash]; st.sidebar.warning("Trend date range error.")
if trend_start_filt_val > trend_end_filt_val: st.sidebar.error("Trend Start <= End."); trend_end_filt_val = trend_start_filt_val; st.session_state[trend_date_key_chw_dash][1] = trend_end_filt_val

# --- Load Data Based on Filters ---
daily_df_chw_display, period_df_chw_display, daily_kpis_precalc_chw = pd.DataFrame(), pd.DataFrame(), {} 
try:
    daily_df_chw_display, period_df_chw_display, daily_kpis_precalc_chw = get_chw_dashboard_data(
        selected_daily_date_val, trend_start_filt_val, trend_end_filt_val, 
        actual_chw_filter_val, actual_zone_filter_val
    )
except Exception as e_load_chw_main:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_load_chw_main}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {str(e_load_chw_main)}. Please check logs.")

# --- Display Filter Context ---
filter_ctx_parts_chw_val = [f"Snapshot Date: **{selected_daily_date_val.strftime('%d %b %Y')}**"]
if actual_chw_filter_val: filter_ctx_parts_chw_val.append(f"CHW: **{actual_chw_filter_val}**")
if actual_zone_filter_val: filter_ctx_parts_chw_val.append(f"Zone: **{actual_zone_filter_val}**")
st.info(f"Displaying data for: {', '.join(filter_ctx_parts_chw_val)}")

# --- Sections (Daily Performance, Alerts, Epi, Trends) ---
# These sections will now use daily_df_chw_display and period_df_chw_display.
# Their internal logic should gracefully handle empty DataFrames if data loading failed.

# Section 1: Daily Performance Snapshot
st.header("📊 Daily Performance Snapshot")
if not daily_df_chw_display.empty:
    daily_summary_val = calculate_chw_daily_summary_metrics(daily_df_chw_display, selected_daily_date_val, daily_kpis_precalc_chw, "CHWDash/DailySummary")
    # ... (KPI rendering logic remains the same, ensure .get() with defaults for robustness) ...
    kpi_cols_chw_s1 = st.columns(4)
    with kpi_cols_chw_s1[0]: render_kpi_card("Visits Today", str(daily_summary_val.get("visits_count", "N/A")), "👥", help_text="Total unique patients encountered.")
    prio_fups_s1 = daily_summary_val.get("high_ai_prio_followups_count", 0)
    prio_stat_s1 = "ACCEPTABLE" if prio_fups_s1 <= 2 else ("MODERATE_CONCERN" if prio_fups_s1 <= 5 else "HIGH_CONCERN")
    with kpi_cols_chw_s1[1]: render_kpi_card("High Prio Follow-ups", str(prio_fups_s1), "🎯", prio_stat_s1, help_text=f"Patients needing urgent follow-up (AI prio ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}).")
    crit_spo2_s1 = daily_summary_val.get("critical_spo2_cases_identified_count", 0)
    spo2_stat_s1 = "HIGH_CONCERN" if crit_spo2_s1 > 0 else "ACCEPTABLE"
    with kpi_cols_chw_s1[2]: render_kpi_card("Critical SpO2 Cases", str(crit_spo2_s1), "💨", spo2_stat_s1, help_text=f"Patients with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    high_fever_s1 = daily_summary_val.get("high_fever_cases_identified_count", 0)
    fever_stat_s1 = "HIGH_CONCERN" if high_fever_s1 > 0 else "ACCEPTABLE"
    with kpi_cols_chw_s1[3]: render_kpi_card("High Fever Cases", str(high_fever_s1), "🔥", fever_stat_s1, help_text=f"Patients with temp ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else: st.markdown("_No activity data for selected date/filters for daily snapshot._")
st.divider()

# Section 2: Key Alerts & Actionable Tasks
st.header("🚦 Key Alerts & Tasks")
# ... (Alert and Task generation using daily_df_chw_display - robustness for empty df handled by components) ...
chw_alerts_list_s2 = generate_chw_alerts(daily_df_chw_display, selected_daily_date_val, actual_zone_filter_val or "All Zones", 10)
if chw_alerts_list_s2: # Similar rendering logic as before
    st.subheader("Priority Patient Alerts (Today):")
    # ... (rest of alert display logic) ...
elif not daily_df_chw_display.empty : st.success("✅ No significant patient alerts generated for today.")
else: st.markdown("_No activity data to generate patient alerts for today._")

chw_tasks_list_s2 = generate_chw_tasks(daily_df_chw_display, selected_daily_date_val, actual_chw_filter_val, actual_zone_filter_val or "All Zones", 10)
if chw_tasks_list_s2: # Similar rendering logic as before
    st.subheader("Top Priority Tasks (Today/Next Day):")
    # ... (rest of task display logic) ...
elif not daily_df_chw_display.empty: st.info("No high-priority tasks identified based on current data.")
else: st.markdown("_No activity data to generate tasks for today._")
st.divider()

# Section 3: Local Epi Signals Watch
st.header("🔬 Local Epi Signals Watch (Today)")
# ... (Epi signal extraction using daily_df_chw_display - robustness for empty df handled by component) ...
if not daily_df_chw_display.empty:
    epi_signals_map_s3 = extract_chw_epi_signals(daily_df_chw_display, daily_kpis_precalc_chw, selected_daily_date_val, actual_zone_filter_val or "All Zones", 3)
    # ... (KPI and cluster rendering logic, ensure .get() for robustness) ...
    epi_kpi_cols_s3 = st.columns(3)
    with epi_kpi_cols_s3[0]: render_kpi_card("Symptomatic (Key Cond.)", str(epi_signals_map_s3.get("symptomatic_patients_key_conditions_count", "N/A")), "🤒", units="cases")
    # ... other epi KPIs ...
else: st.markdown("_No activity data for local epi signals for selected date/filters._")
st.divider()

# Section 4: CHW Team Activity Trends
st.header("📈 CHW Team Activity Trends")
# ... (Trend calculation using period_df_chw_display - robustness for empty df handled by component) ...
if not period_df_chw_display.empty:
    activity_trends_map_s4 = calculate_chw_activity_trends_data(period_df_chw_display, trend_start_filt_val, trend_end_filt_val, actual_zone_filter_val, 'D')
    # ... (Trend plot rendering logic, check if series exist and are not empty) ...
else: st.markdown("_No historical data for selected trend period/filters._")

logger.info(f"CHW Dashboard page fully rendered or attempted. Data available: Daily={not daily_df_chw_display.empty}, Period={not period_df_chw_display.empty}")
