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
    from visualization.plots import plot_annotated_line_chart, create_empty_figure # Added create_empty_figure
    
    from pages.chw_components.summary_metrics import calculate_chw_daily_summary_metrics
    from pages.chw_components.alert_generation import generate_chw_alerts
    from pages.chw_components.epi_signals import extract_chw_epi_signals
    from pages.chw_components.task_processing import generate_chw_tasks
    from pages.chw_components.activity_trends import calculate_chw_activity_trends_data
except ImportError as e_chw_dash_resilient: 
    import sys
    _root_dir_chw_resilient = Path(__file__).resolve().parent.parent 
    st.error(f"CHW Dashboard Import Error: {e_chw_dash_resilient}. Project Root: '{_root_dir_chw_resilient}', SysPath: {sys.path}")
    st.stop()

logger = logging.getLogger(__name__)

st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance Monitoring & Field Support - {settings.APP_NAME}**")
st.divider()

def _create_filter_dropdown_options_chw_page_v3( # Renamed for clarity
    df: Optional[pd.DataFrame], col: str, defaults: List[str], name_plural: str
) -> List[str]:
    opts = [f"All {name_plural}"]
    if isinstance(df, pd.DataFrame) and not df.empty and col in df.columns:
        unique_vals = sorted(list(set(str(v) for v in df[col].dropna() if str(v).strip()))) # Ensure non-empty strings
        if unique_vals: opts.extend(unique_vals)
        else: logger.warning(f"CHW Filters: Col '{col}' for '{name_plural}' empty. Using defaults."); opts.extend(defaults)
    else: logger.warning(f"CHW Filters: Col '{col}' not in DF or DF empty for '{name_plural}'. Using defaults."); opts.extend(defaults)
    return opts

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading CHW operational data...", hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_chw_dashboard_page_data_v3( # Renamed for clarity
    view_date: date, trend_start: date, trend_end: date, 
    chw_filter: Optional[str], zone_filter: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log_ctx = "CHWDashPageDataV3"
    logger.info(f"({log_ctx}) Loading data. View: {view_date}, Trend: {trend_start}-{trend_end}, CHW: {chw_filter or 'All'}, Zone: {zone_filter or 'All'}")
    
    all_health_df = load_health_records(source_context=f"{log_ctx}/LoadRecs")
    if not isinstance(all_health_df, pd.DataFrame) or all_health_df.empty:
        logger.error(f"({log_ctx}) CRITICAL: Health records empty. Expected: {settings.HEALTH_RECORDS_CSV_PATH}")
        st.error(f"CRITICAL DATA ERROR: Health records missing or empty. Ensure '{Path(settings.HEALTH_RECORDS_CSV_PATH).name}' is in 'data_sources/'.")
        return pd.DataFrame(), pd.DataFrame(), {}
    if 'encounter_date' not in all_health_df.columns or not pd.api.types.is_datetime64_any_dtype(all_health_df['encounter_date']):
        logger.error(f"({log_ctx}) 'encounter_date' invalid."); st.error("Data Integrity Error: 'encounter_date' invalid.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Ensure encounter_date is timezone-naive for dt.date comparisons
    if all_health_df['encounter_date'].dt.tz is not None:
        logger.debug(f"({log_ctx}) Converting encounter_date to timezone-naive for filtering.")
        all_health_df['encounter_date'] = all_health_df['encounter_date'].dt.tz_localize(None)

    df_daily_page_v3 = all_health_df[all_health_df['encounter_date'].dt.date == view_date].copy()
    if chw_filter and 'chw_id' in df_daily_page_v3.columns: df_daily_page_v3 = df_daily_page_v3[df_daily_page_v3['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_daily_page_v3.columns: df_daily_page_v3 = df_daily_page_v3[df_daily_page_v3['zone_id'] == zone_filter]

    df_trend_page_v3 = all_health_df[(all_health_df['encounter_date'].dt.date >= trend_start) & (all_health_df['encounter_date'].dt.date <= trend_end)].copy()
    if chw_filter and 'chw_id' in df_trend_page_v3.columns: df_trend_page_v3 = df_trend_page_v3[df_trend_page_v3['chw_id'] == chw_filter]
    if zone_filter and 'zone_id' in df_trend_page_v3.columns: df_trend_page_v3 = df_trend_page_v3[df_trend_page_v3['zone_id'] == zone_filter]

    pre_calc_kpis_page_v3: Dict[str, Any] = {}
    if chw_filter and not df_daily_page_v3.empty:
        enc_type_series_v3 = df_daily_page_v3.get('encounter_type', pd.Series(dtype=str)).astype(str)
        chw_id_series_page_v3 = df_daily_page_v3.get('chw_id', pd.Series(dtype=str))
        self_checks_page_v3 = df_daily_page_v3[(chw_id_series_page_v3 == chw_filter) & (enc_type_series_v3.str.contains('WORKER_SELF_CHECK', case=False, na=False))]
        if not self_checks_page_v3.empty:
            fatigue_cols_page_v3 = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            actual_fatigue_col_page_v3 = next((c for c in fatigue_cols_page_v3 if c in self_checks_page_v3.columns and self_checks_page_v3[c].notna().any()), None)
            pre_calc_kpis_page_v3['worker_self_fatigue_index_today'] = self_checks_page_v3[actual_fatigue_col_page_v3].max() if actual_fatigue_col_page_v3 else np.nan
        else: pre_calc_kpis_page_v3['worker_self_fatigue_index_today'] = np.nan
    
    logger.info(f"({log_ctx}) Data loaded. Daily: {len(df_daily_page_v3)} recs, Trend: {len(df_trend_page_v3)} recs.")
    return df_daily_page_v3, df_trend_page_v3, pre_calc_kpis_page_v3

# --- Sidebar Filters ---
project_root_chw_dash_page_v3 = Path(settings.PROJECT_ROOT_DIR)
logo_path_chw_sidebar_page_v3 = Path(settings.APP_LOGO_SMALL_PATH) 
if logo_path_chw_sidebar_page_v3.is_file(): st.sidebar.image(str(logo_path_chw_sidebar_page_v3), width=120)
else: logger.warning(f"Sidebar logo for CHW Dash not found: {logo_path_chw_sidebar_page_v3}")
st.sidebar.header("Dashboard Filters")

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS) 
def load_chw_filter_options_data_v3():
    logger.info("CHW Page: Loading data for filter dropdowns.")
    df = load_health_records(source_context="CHWDashPage/FilterPopulationV3")
    if not isinstance(df, pd.DataFrame): return pd.DataFrame() # Ensure DF return
    if 'encounter_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    if 'encounter_date' in df.columns and df['encounter_date'].dt.tz is not None: # Ensure naive for date ops
        df['encounter_date'] = df['encounter_date'].dt.tz_localize(None)
    return df

df_for_chw_filters_page_v3 = load_chw_filter_options_data_v3()

chw_options_list_page_v3 = _create_filter_dropdown_options_chw_page_v2(df_for_chw_filters_page_v3, 'chw_id', ["CHW001", "CHW002", "CHW003"], "CHWs")
chw_session_key_page_v3 = "chw_dashboard_chw_id_v6"
if chw_session_key_page_v3 not in st.session_state or st.session_state[chw_session_key_page_v3] not in chw_options_list_page_v3 : st.session_state[chw_session_key_page_v3] = chw_options_list_page_v3[0]
selected_chw_ui_page_v3 = st.sidebar.selectbox("Filter by CHW ID:", options=chw_options_list_page_v3, key=f"{chw_session_key_page_v3}_widget", index=chw_options_list_page_v3.index(st.session_state[chw_session_key_page_v3]))
st.session_state[chw_session_key_page_v3] = selected_chw_ui_page_v3
actual_chw_filter_page_v3 = None if selected_chw_ui_page_v3.startswith("All ") else selected_chw_ui_page_v3

zone_options_list_page_v3 = _create_filter_dropdown_options_chw_page_v2(df_for_chw_filters_page_v3, 'zone_id', ["ZoneA", "ZoneB", "ZoneC"], "Zones")
zone_session_key_page_v3 = "chw_dashboard_zone_id_v6"
if zone_session_key_page_v3 not in st.session_state or st.session_state[zone_session_key_page_v3] not in zone_options_list_page_v3: st.session_state[zone_session_key_page_v3] = zone_options_list_page_v3[0]
selected_zone_ui_page_v3 = st.sidebar.selectbox("Filter by Zone:", options=zone_options_list_page_v3, key=f"{zone_session_key_page_v3}_widget", index=zone_options_list_page_v3.index(st.session_state[zone_session_key_page_v3]))
st.session_state[zone_session_key_page_v3] = selected_zone_ui_page_v3
actual_zone_filter_page_v3 = None if selected_zone_ui_page_v3.startswith("All ") else selected_zone_ui_page_v3

# Date Pickers - use hardcoded safe defaults if data-derived dates are problematic
abs_min_date_chw_dash_page_v3 = date(2022, 1, 1) 
abs_max_date_chw_dash_page_v3 = date.today()
data_min_date_chw_page_val, data_max_date_chw_page_val = abs_min_date_chw_dash_page_v3, abs_max_date_chw_dash_page_v3 # Fallbacks
if not df_for_chw_filters_page_v3.empty and 'encounter_date' in df_for_chw_filters_page_v3.columns and \
   pd.api.types.is_datetime64_any_dtype(df_for_chw_filters_page_v3['encounter_date']) and \
   df_for_chw_filters_page_v3['encounter_date'].notna().any():
    try:
        data_min_date_chw_page_val = df_for_chw_filters_page_v3['encounter_date'].min().date()
        data_max_date_chw_page_val = df_for_chw_filters_page_v3['encounter_date'].max().date()
        if data_min_date_chw_page_val > data_max_date_chw_page_val: data_min_date_chw_page_val = data_max_date_chw_page_val
    except Exception as e_date_minmax_chw_v3: logger.warning(f"Error deriving min/max dates for CHW Dash: {e_date_minmax_chw_v3}")

daily_date_key_chw_dash_page_v3 = "chw_dashboard_daily_date_v6"
# Default selected date: latest available data date, or today if no data
default_daily_date = data_max_date_chw_page_val if data_max_date_chw_page_val else date.today()
if daily_date_key_chw_dash_page_v3 not in st.session_state: st.session_state[daily_date_key_chw_dash_page_v3] = default_daily_date
selected_daily_date_page_val_v3 = st.sidebar.date_input("View Daily Activity For:", value=st.session_state[daily_date_key_chw_dash_page_v3], min_value=data_min_date_chw_page_val, max_value=data_max_date_chw_page_val, key=f"{daily_date_key_chw_dash_page_v3}_widget")
st.session_state[daily_date_key_chw_dash_page_v3] = selected_daily_date_page_val_v3

trend_date_key_chw_dash_page_v3 = "chw_dashboard_trend_date_range_v6"
default_trend_end_chw_dash_v3 = selected_daily_date_page_val_v3
default_trend_start_chw_dash_v3 = max(data_min_date_chw_page_val, default_trend_end_chw_dash_v3 - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))
if trend_date_key_chw_dash_page_v3 not in st.session_state: st.session_state[trend_date_key_chw_dash_page_v3] = [default_trend_start_chw_dash_v3, default_trend_end_chw_dash_v3]
selected_trend_range_page_val_v3 = st.sidebar.date_input("Select Trend Date Range:", value=st.session_state[trend_date_key_chw_dash_page_v3], min_value=data_min_date_chw_page_val, max_value=data_max_date_chw_page_val, key=f"{trend_date_key_chw_dash_page_v3}_widget")

trend_start_filt_page_val_v3: date; trend_end_filt_page_val_v3: date 
if isinstance(selected_trend_range_page_val_v3, (list, tuple)) and len(selected_trend_range_page_val_v3) == 2:
    st.session_state[trend_date_key_chw_dash_page_v3] = selected_trend_range_page_val_v3; trend_start_filt_page_val_v3, trend_end_filt_page_val_v3 = selected_trend_range_page_val_v3
else: trend_start_filt_page_val_v3, trend_end_filt_page_val_v3 = st.session_state[trend_date_key_chw_dash_page_v3]; st.sidebar.warning("Trend date range error.")
if trend_start_filt_page_val_v3 > trend_end_filt_page_val_v3: st.sidebar.error("Trend Start <= End."); trend_end_filt_page_val_v3 = trend_start_filt_page_val_v3; st.session_state[trend_date_key_chw_dash_page_v3][1] = trend_end_filt_page_val_v3

# --- Load Data Based on Filters ---
daily_df_chw_page_display_v3, period_df_chw_page_display_v3, daily_kpis_precalc_chw_page_v3 = pd.DataFrame(), pd.DataFrame(), {} 
try:
    daily_df_chw_page_display_v3, period_df_chw_page_display_v3, daily_kpis_precalc_chw_page_v3 = get_chw_dashboard_page_data_v3(
        selected_daily_date_page_val_v3, trend_start_filt_page_val_v3, trend_end_filt_page_val_v3, 
        actual_chw_filter_page_v3, actual_zone_filter_page_v3
    )
except Exception as e_load_chw_main_page_v3:
    logger.error(f"CHW Dashboard: Main data loading/processing failed: {e_load_chw_main_page_v3}", exc_info=True)
    st.error(f"Error loading CHW dashboard data: {str(e_load_chw_main_page_v3)}. Please check logs.")

# --- Display Filter Context ---
filter_ctx_parts_chw_page_val_v3 = [f"Snapshot Date: **{selected_daily_date_page_val_v3.strftime('%d %b %Y')}**"]
if actual_chw_filter_page_v3: filter_ctx_parts_chw_page_val_v3.append(f"CHW: **{actual_chw_filter_page_v3}**")
if actual_zone_filter_page_v3: filter_ctx_parts_chw_page_val_v3.append(f"Zone: **{actual_zone_filter_page_v3}**")
st.info(f"Displaying data for: {', '.join(filter_ctx_parts_chw_page_val_v3)}")

# --- Section 1: Daily Performance Snapshot ---
st.header("📊 Daily Performance Snapshot")
if not daily_df_chw_page_display_v3.empty:
    daily_summary_page_val_v3 = calculate_chw_daily_summary_metrics(daily_df_chw_page_display_v3, selected_daily_date_page_val_v3, daily_kpis_precalc_chw_page_v3, "CHWDash/DailySummaryV3")
    # ... (KPI rendering as before, using daily_summary_page_val_v3)
else: st.markdown("_No activity data for selected date/filters for daily performance snapshot._")
st.divider()

# --- Section 2: Key Alerts & Actionable Tasks ---
st.header("🚦 Key Alerts & Tasks")
if not daily_df_chw_page_display_v3.empty:
    chw_alerts_list_s2_v3 = generate_chw_alerts(daily_df_chw_page_display_v3, selected_daily_date_page_val_v3, actual_zone_filter_page_v3 or "All Zones", 10)
    # ... (Alert rendering as before, using chw_alerts_list_s2_v3)
    chw_tasks_list_s2_v3 = generate_chw_tasks(daily_df_chw_page_display_v3, selected_daily_date_page_val_v3, actual_chw_filter_page_v3, actual_zone_filter_page_v3 or "All Zones", 10)
    # ... (Task rendering as before, using chw_tasks_list_s2_v3)
else: st.markdown("_No activity data to generate alerts or tasks for today._")
st.divider()

# --- Section 3: Local Epi Signals Watch ---
st.header("🔬 Local Epi Signals Watch (Today)")
if not daily_df_chw_page_display_v3.empty:
    epi_signals_map_s3_v3 = extract_chw_epi_signals(daily_df_chw_page_display_v3, daily_kpis_precalc_chw_page_v3, selected_daily_date_page_val_v3, actual_zone_filter_page_v3 or "All Zones", 3)
    # ... (Epi KPI and cluster rendering as before, using epi_signals_map_s3_v3) ...
else: st.markdown("_No activity data for local epi signals for selected date/filters._")
st.divider()

# --- Section 4: CHW Team Activity Trends ---
st.header("📈 CHW Team Activity Trends")
trend_period_display_text_page_val_v3 = f"{trend_start_filt_page_val_v3.strftime('%d %b %Y')} - {trend_end_filt_page_val_v3.strftime('%d %b %Y')}"
trend_filter_context_text_page_val_v3 = f" for CHW **{actual_chw_filter_page_v3}**" if actual_chw_filter_page_v3 else "" # ... (rest of context string)
st.markdown(f"Displaying trends from **{trend_period_display_text_page_val_v3}**{trend_filter_context_text_page_val_v3}.")
if not period_df_chw_page_display_v3.empty:
    activity_trends_map_s4_v3 = calculate_chw_activity_trends_data(period_df_chw_page_display_v3, trend_start_filt_page_val_v3, trend_end_filt_page_val_v3, actual_zone_filter_page_v3, 'D')
    # ... (Trend plot rendering as before, using activity_trends_map_s4_v3 and create_empty_figure for empty series) ...
else: st.markdown("_No historical data for selected trend period/filters._")

logger.info(f"CHW Supervisor Dashboard page fully rendered. Data available: Daily={not daily_df_chw_page_display_v3.empty}, Period={not period_df_chw_page_display_v3.empty}")
