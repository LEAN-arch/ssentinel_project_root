# sentinel_project_root/pages/01_chw_dashboard.py
# CHW Supervisor Operations View for Sentinel Health Co-Pilot

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta 
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path 

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records
    from data_processing.helpers import hash_dataframe_safe # For st.cache_data
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart
    
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.alert_generation import generate_chw_alerts
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import calculate_chw_activity_trends_data
except ImportError as e_chw_dash_final_fix_v2: # Unique exception variable name
    import sys
    _current_file_chw_final_v2 = Path(__file__).resolve()
    _project_root_chw_assumption_final_v2 = _current_file_chw_final_v2.parent.parent 
    error_msg_chw_detail_final_v2 = (
        f"CHW Dashboard Import Error: {e_chw_dash_final_fix_v2}. "
        f"Ensure project root ('{_project_root_chw_assumption_final_v2}') is in sys.path (handled by app.py) "
        f"and all modules/packages have `__init__.py` files. Check for typos in import paths. "
        f"Current Python Path: {sys.path}"
    )
    try: st.error(error_msg_chw_detail_final_v2); st.stop()
    except NameError: print(error_msg_chw_detail_final_v2, file=sys.stderr); raise

logger = logging.getLogger(__name__)

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME}**")
st.divider()

def _create_filter_dropdown_options_chw_page( # Renamed for clarity
    df: Optional[pd.DataFrame], 
    col: str, 
    defaults: List[str], 
    name_plural: str
) -> List[str]:
    opts = [f"All {name_plural}"]
    if isinstance(df, pd.DataFrame) and not df.empty and col in df.columns:
        # Ensure values are strings and unique before sorting
        unique_vals = sorted(list(set(str(v) for v in df[col].dropna())))
        if unique_vals: opts.extend(unique_vals)
        else: logger.warning(f"CHW Filters: Col '{col}' for '{name_plural}' empty. Using defaults."); opts.extend(defaults)
    else: logger.warning(f"CHW Filters: Col '{col}' not in DF or DF empty for '{name_plural}'. Using defaults."); opts.extend(defaults)
    return opts

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...", hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_chw_dashboard_page_data( # Renamed for clarity
    view_date: date, 
    trend_start: date, 
    trend_end: date, 
    chw_filter: Optional[str], 
    zone_filter: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "CHWDashPageData" # Unique context
    logger.info(f"({log_ctx}) Loading data. View: {view_date}, Trend: {trend_start}-{trend_end}, CHW: {chw_filter or 'All'}, Zone: {zone_filter or 'All'}")
    
    all_health_df = load_health_records(source_context=f"{log_ctx}/LoadRecs")
    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        logger.error(f"({log_ctx}) CRITICAL: Health records failed to load or are empty. Expected at: {settings.HEALTH_RECORDS_CSV_PATH}")
        st.error(f"CRITICAL DATA ERROR: Could not load health records. Ensure '{Path(settings.HEALTH_RECORDS_CSV_PATH).name}' is in 'data_sources/' and not empty.")
        return pd.DataFrame(), pd.DataFrame(), {} # Return empty to prevent further errors
    if 'encounter_date' not in all_health_df.columns or not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' invalid in loaded health records."); st.error("Data Integrity Error: 'encounter_date' invalid.")
        return pd.DataFrame(), pd.DataFrame(), {}

    df_daily_page = all_health_df[all_health_df['encounter_date'].dt.date == view_date].copy()
    if chw_filter and 'chw_id' in df_daily_page.columns: df_daily_page = df_daily_page[df_daily_page['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_daily_page.columns: df_daily_page = df_daily_page[df_daily_page['zone_id'] == zone_filter]

    df_trend_page = all_health_df[(all_health_df['encounter_date'].dt.date >= trend_start) & (all_health_df['encounter_date'].dt.date <= trend_end)].copy()
    if chw_filter and 'chw_id' in df_trend_page.columns: df_trend_page = df_trend_page[df_trend_page['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_trend_page.columns: df_trend_page = df_trend_page[df_trend_page['zone_id'] == zone_filter]

    pre_calc_kpis_page: Dict[str, Any] = {}
    if chw_filter and not df_daily_page.empty:
        enc_type_series = df_daily_page.get('encounter_type', pd.Series(dtype=str)).astype(str)
        chw_id_series_page = df_daily_page.get('chw_id', pd.Series(dtype=str))
        self_checks_page = df_daily_page[(chw_id_series_page == chw_filter) & (enc_type_series.str.contains('WORKER_SELF_CHECK', case=False, na=False))]
        if not self_checks_page.empty:
            fatigue_cols_page = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            actual_fatigue_col_page = next((c for c in fatigue_cols_page if c in self_checks_page.columns and self_checks_page[c].notna().any()), None)
            pre_calc_kpis_page['worker_self_fatigue_index_today'] = self_checks_page[actual_fatigue_col_page].max() if actual_fatigue_col_page else np.nan
        else: pre_calc_kpis_page['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(f"({log_ctx}) Data loaded for CHW page. Daily: {len(df_daily_page)} recs, Trend: {len(df_trend_page)} recs.")
    return df_daily_page, df_trend_page, pre_calc_kpis_page

# --- Sidebar Filters ---
project_root_chw_dash_page = Path(settings.PROJECT_ROOT_DIR)
logo_path_chw_sidebar_page = Path(settings.APP_LOGO_SMALL_PATH) # Already absolute from settings
if logo_path_chw_sidebar_page.is_file(): st.sidebar.image(str(logo_path_chw_sidebar_page), width=120)
else: logger.warning(f"Sidebar logo for CHW Dash not found: {logo_path_chw_sidebar_page}")
st.sidebar.header("Dashboard Filters")

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS) # Cache the filter options data
def load_chw_filter_options_data():
    logger.info("CHW Page: Loading data for filter dropdowns.")
    return load_health_records(source_context="CHWDashPage/FilterPopulation")

df_for_chw_filters_page = load_chw_filter_options_data()
if not isinstance(df_for_chw_filter_options, pd.DataFrame): 
    df_for_chw_filter_options = pd.DataFrame() # Ensure it's a DataFrame for _create_filter_dropdown_options

chw_options_list_page = _create_filter_dropdown_options_chw_page(df_for_chw_filter_options, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
chw_session_key_page = "chw_dashboard_chw_id_v5" # Unique key
if chw_session_key_page not in st.session_state or st.session_state[chw_session_key_page] not in chw_options_list_page : st.session_state[chw_session_key_page] = chw_options_list_page[0]
selected_chw_ui_page = st.sidebar.selectbox("Filter by CHW ID:", options=chw_options_list_page, key=f"{chw_session_key_page}_widget", index=chw_options_list_page.index(st.session_state[chw_session_key_page]))
st.session_state[chw_session_key_page] = selected_chw_ui_page
actual_chw_filter_page = None if selected_chw_ui_page.startswith("All ") else selected_chw_ui_page

zone_options_list_page = _create_filter_dropdown_options_chw_page(df_for_chw_filter_options, 'zone_id', ["ZoneA", "ZoneB", "ZoneC"], "Zones")
zone_session_key_page = "chw_dashboard_zone_id_v5"
if zone_session_key_page not in st.session_state or st.session_state[zone_session_key_page] not in zone_options_list_page: st.session_state[zone_session_key_page] = zone_options_list_page[0]
selected_zone_ui_page = st.sidebar.selectbox("Filter by Zone:", options=zone_options_list_page, key=f"{zone_session_key_page}_widget", index=zone_options_list_page.index(st.session_state[zone_session_key_page]))
st.session_state[zone_session_key_page] = selected_zone_ui_page
actual_zone_filter_page = None if selected_zone_ui_page.startswith("All ") else selected_zone_ui_page

# Date Pickers - with robust fallbacks
abs_min_date_chw_dash_page = date(2022, 1, 1); abs_max_date_chw_dash_page = date.today()
data_min_date_chw_page, data_max_date_chw_page = abs_min_date_chw_dash_page, abs_max_date_chw_dash_page # Fallbacks
if isinstance(df_for_chw_filter_options, pd.DataFrame) and 'encounter_date' in df_for_chw_filter_options.columns and \
   pd.api.types.is_datetime64_any_dtype(df_for_chw_filter_options['encounter_date']) and \
   df_for_chw_filter_options['encounter_date'].notna().any():
    try:
        data_min_date_chw_page = df_for_chw_filter_options['encounter_date'].min().date()
        data_max_date_chw_page = df_for_chw_filter_options['encounter_date'].max().date()
        if data_min_date_chw_page > data_max_date_chw_page: data_min_date_chw_page = data_max_date_chw_page
    except Exception as e_date_minmax: logger.warning(f"Error deriving min/max dates for CHW Dash: {e_date_minmax}")


daily_date_key_chw_dash_page = "chw_dashboard_daily_date_v5"
if daily_date_key_chw_dash_page not in st.session_state: st.session_state[daily_date_key_chw_dash_page] = data_max_date_chw_page
selected_daily_date_page_val = st.sidebar.date_input("View Daily Activity For:", value=st.session_state[daily_date_key_chw_dash_page], min_value=data_min_date_chw_page, max_value=data_max_date_chw_page, key=f"{daily_date_key_chw_dash_page}_widget")
st.session_state[daily_date_key_chw_dash_page] = selected_daily_date_page_val

trend_date_key_chw_dash_page = "chw_dashboard_trend_date_range_v5"
def_trend_end_chw_dash_page = selected_daily_date_page_val
def_trend_start_chw_dash_page = max(data_min_date_chw_page, def_trend_end_chw_dash_page - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))
if trend_date_key_chw_dash_page not in st.session_state: st.session_state[trend_date_key_chw_dash_page] = [def_trend_start_chw_dash_page, def_trend_end_chw_dash_page]
selected_trend_range_page_val = st.sidebar.date_input("Select Trend Date Range:", value=st.session_state[trend_date_key_chw_dash_page], min_value=data_min_date_chw_page, max_value=data_max_date_chw_page, key=f"{trend_date_key_chw_dash_page}_widget")

trend_start_filt_page_val: date; trend_end_filt_page_val: date 
if isinstance(selected_trend_range_page_val, (list, tuple)) and len(selected_trend_range_page_val) == 2:
    st.session_state[trend_date_key_chw_dash_page] = selected_trend_range_page_val; trend_start_filt_page_val, trend_end_filt_page_val = selected_trend_range_page_val
else: trend_start_filt_page_val, trend_end_filt_page_val = st.session_state[trend_date_key_chw_dash_page]; st.sidebar.warning("Trend date range error.")
if trend_start_filt_page_val > trend_end_filt_page_val: st.sidebar.error("Trend Start <= End."); trend_end_filt_page_val = trend_start_filt_page_val; st.session_state[trend_date_key_chw_dash_page][1] = trend_end_filt_page_val

# --- Load Data Based on Filters ---
daily_df_chw_page_display, period_df_chw_page_display, daily_kpis_precalc_chw_page = pd.DataFrame(), pd.DataFrame(), {} 
try:
    daily_df_chw_page_display, period_df_chw_page_display, daily_kpis_precalc_chw_page = get_chw_dashboard_page_data(
        selected_daily_date_page_val, trend_start_filt_page_val, trend_end_filt_page_val, 
        actual_chw_filter_page, actual_zone_filter_page
    )
except Exception as e_load_chw_main_page:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_load_chw_main_page}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {str(e_load_chw_main_page)}. Please check logs.")

# --- Display Filter Context ---
filter_ctx_parts_chw_page_val = [f"Snapshot Date: **{selected_daily_date_page_val.strftime('%d %b %Y')}**"]
if actual_chw_filter_page: filter_ctx_parts_chw_page_val.append(f"CHW: **{actual_chw_filter_page}**")
if actual_zone_filter_page: filter_ctx_parts_chw_page_val.append(f"Zone: **{actual_zone_filter_page}**")
st.info(f"Displaying data for: {', '.join(filter_ctx_parts_chw_page_val)}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df_chw_page_display.empty:
    daily_summary_page_val = {} # Initialize
    try:
        daily_summary_page_val = calculate_chw_daily_summary_metrics(daily_df_chw_page_display, selected_daily_date_page_val, daily_kpis_precalc_chw_page, "CHWDash/DailySummary")
    except Exception as e_daily_summary_chw_page: logger.error(f"Error calculating CHW daily summary: {e_daily_summary_chw_page}", exc_info=True); st.warning("Could not calculate daily summary metrics.")
    
    kpi_cols_daily_snapshot_page_val = st.columns(4)
    with kpi_cols_daily_snapshot_page_val[0]: render_kpi_card("Visits Today", str(daily_summary_page_val.get("visits_count", "0")), "👥", help_text="Total unique patients encountered.") # Default to "0" if key missing
    prio_fups_page_val = daily_summary_page_val.get("high_ai_prio_followups_count", 0)
    prio_status_level_page_val = "ACCEPTABLE" if prio_fups_page_val <= 2 else ("MODERATE_CONCERN" if prio_fups_page_val <= 5 else "HIGH_CONCERN")
    with kpi_cols_daily_snapshot_page_val[1]: render_kpi_card("High Prio Follow-ups", str(prio_fups_page_val), "🎯", prio_status_level_page_val, help_text=f"Patients needing urgent follow-up (AI prio ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}).")
    critical_spo2_cases_page_val = daily_summary_page_val.get("critical_spo2_cases_identified_count", 0)
    spo2_status_level_page_val = "HIGH_CONCERN" if critical_spo2_cases_page_val > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot_page_val[2]: render_kpi_card("Critical SpO2 Cases", str(critical_spo2_cases_page_val), "💨", spo2_status_level_page_val, help_text=f"Patients with SpO2 < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%.")
    high_fever_cases_page_val = daily_summary_page_val.get("high_fever_cases_identified_count", 0)
    fever_status_level_page_val = "HIGH_CONCERN" if high_fever_cases_page_val > 0 else "ACCEPTABLE"
    with kpi_cols_daily_snapshot_page_val[3]: render_kpi_card("High Fever Cases", str(high_fever_cases_page_val), "🔥", fever_status_level_page_val, help_text=f"Patients with temp ≥ {settings.ALERT_BODY_TEMP_HIGH_FEVER_C}°C.")
else: st.markdown("_No activity data for selected date/filters for daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
chw_alerts_list_page_display = []
if not daily_df_chw_page_display.empty: # Only generate if there's data
    try: chw_alerts_list_page_display = generate_chw_alerts(daily_df_chw_page_display, selected_daily_date_page_val, actual_zone_filter_page or "All Zones", 10)
    except Exception as e_alerts_page_chw: logger.error(f"CHW Dashboard: Error generating patient alerts: {e_alerts_page_chw}", exc_info=True); st.warning("Could not generate patient alerts.")
if chw_alerts_list_page_display:
    st.subheader("Priority Patient Alerts (Today):") # Render as before
    # ... (full alert rendering logic as previously corrected) ...
elif not daily_df_chw_page_display.empty : st.success("✅ No significant patient alerts generated for today's selection.")
else: st.markdown("_No activity data to generate patient alerts for today._")

chw_tasks_list_page_display = []
if not daily_df_chw_page_display.empty: # Only generate if there's data
    try: chw_tasks_list_page_display = generate_chw_tasks(daily_df_chw_page_display, selected_daily_date_page_val, actual_chw_filter_page, actual_zone_filter_page or "All Zones", 10)
    except Exception as e_tasks_page_chw: logger.error(f"CHW Dashboard: Error generating CHW tasks: {e_tasks_page_chw}", exc_info=True); st.warning("Could not generate tasks list.")
if chw_tasks_list_page_display:
    st.subheader("Top Priority Tasks (Today/Next Day):") # Render as before
    # ... (full task table rendering logic as previously corrected) ...
elif not daily_df_chw_page_display.empty : st.info("No high-priority tasks identified based on current data.")
else: st.markdown("_No activity data to generate tasks for today._")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_df_chw_page_display.empty:
    chw_epi_signals_map_page_val = {} 
    try: chw_epi_signals_map_page_val = extract_chw_epi_signals(daily_df_chw_page_display, daily_kpis_precalc_chw_page, selected_daily_date_page_val, actual_zone_filter_page or "All Zones", 3)
    except Exception as e_epi_display_chw_page: logger.error(f"CHW Dashboard: Error extracting epi signals: {e_epi_display_chw_page}", exc_info=True); st.warning("Could not extract epi signals.")
    # ... (Epi KPI and cluster rendering as previously corrected, using .get() on chw_epi_signals_map_page_val) ...
else: st.markdown("_No activity data for local epi signals for selected date/filters._")
st.divider()

# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_text_page_val = f"{trend_start_filt_page_val.strftime('%d %b %Y')} - {trend_end_filt_page_val.strftime('%d %b %Y')}"
# ... (Trend filter context string and rendering logic as previously corrected, using period_df_chw_page_display) ...
if not period_df_chw_page_display.empty:
    chw_activity_trends_map_page_val = {}
    try: chw_activity_trends_map_page_val = calculate_chw_activity_trends_data(period_df_chw_page_display, trend_start_filt_page_val, trend_end_filt_page_val, actual_zone_filter_page, 'D')
    except Exception as e_trends_page_chw: logger.error(f"CHW Dashboard: Error calculating activity trends: {e_trends_page_chw}", exc_info=True); st.warning("Could not calculate activity trends.")
    # ... (Trend plot rendering) ...
else: st.markdown("_No historical data for selected trend period/filters._")

logger.info(f"CHW Supervisor Dashboard page fully rendered. Data available: Daily={not daily_df_chw_page_display.empty}, Period={not period_df_chw_page_display.empty}")
