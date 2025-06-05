# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from data_processing.helpers import hash_dataframe_safe 
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart, create_empty_figure

    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash_import:
    import sys
    current_file_path = Path(__file__).resolve()
    project_root_dir = current_file_path.parent.parent
    error_message = (
        f"Clinic Dashboard Import Error: {e_clinic_dash_import}. "
        f"Ensure project root ('{project_root_dir}') is in sys.path (handled by app.py) "
        f"and all modules/packages have `__init__.py` files. Check for typos or missing dependencies. "
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
    page_icon_value = "🏥" 
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_FAVICON_PATH'):
        favicon_path = Path(settings.PROJECT_ROOT_DIR) / settings.APP_FAVICON_PATH
        if favicon_path.is_file(): page_icon_value = str(favicon_path)
        else: logger.warning(f"Favicon not found at resolved path: {favicon_path}")
    page_layout_value = "wide" 
    if hasattr(settings, 'APP_LAYOUT'): page_layout_value = settings.APP_LAYOUT
    st.set_page_config(
        page_title=f"Clinic Console - {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'App'}",
        page_icon=page_icon_value, layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for Clinic Dashboard: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="Clinic Console", page_icon="🏥", layout="wide") 

st.title(f"🏥 {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'Sentinel Health Co-Pilot'} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS if hasattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS') else 300,
    show_spinner="Loading comprehensive clinic operational dataset...",
    hash_funcs={pd.DataFrame: hash_dataframe_safe} 
)
def get_clinic_console_processed_data(
    selected_period_start_date: date, selected_period_end_date: date
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], bool]:
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Loading data for period: {selected_period_start_date.isoformat()} to {selected_period_end_date.isoformat()}")
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadRawHealthRecs")
    raw_iot_df = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadRawIoTData")
    iot_data_available_flag = False
    if hasattr(settings, 'IOT_CLINIC_ENVIRONMENT_CSV_PATH') and hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'DATA_DIR'):
        iot_source_path_str = settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH 
        data_dir = Path(settings.DATA_DIR)
        iot_path = Path(iot_source_path_str)
        if not iot_path.is_absolute(): iot_path = (data_dir / iot_source_path_str).resolve()
        if iot_path.is_file():
            if isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty: iot_data_available_flag = True
            else: logger.warning(f"({log_ctx}) IoT data file found at '{iot_path}', but failed to load into DataFrame or is empty.")
        else: logger.warning(f"({log_ctx}) IoT data file NOT FOUND at resolved path: '{iot_path}'.")
    else: logger.warning(f"({log_ctx}) Settings for IoT path missing. IoT data availability check might be inaccurate.")

    ai_enriched_health_df_full = pd.DataFrame() 
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        try:
            enriched_data, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
            if isinstance(enriched_data, pd.DataFrame): ai_enriched_health_df_full = enriched_data
            else: 
                logger.warning(f"({log_ctx}) AI model application did not return a DataFrame.")
                ai_enriched_health_df_full = raw_health_df 
        except Exception as e_ai_enrich:
            logger.error(f"({log_ctx}) Error during AI model application: {e_ai_enrich}", exc_info=True)
            ai_enriched_health_df_full = raw_health_df 
    else: logger.warning(f"({log_ctx}) Raw health data is empty or invalid. AI enrichment skipped.")

    df_health_period = pd.DataFrame()
    if not ai_enriched_health_df_full.empty and 'encounter_date' in ai_enriched_health_df_full.columns:
        if not pd.api.types.is_datetime64_any_dtype(ai_enriched_health_df_full['encounter_date']):
            ai_enriched_health_df_full['encounter_date'] = pd.to_datetime(ai_enriched_health_df_full['encounter_date'], errors='coerce')
        if ai_enriched_health_df_full['encounter_date'].dt.tz is not None: 
            ai_enriched_health_df_full['encounter_date'] = ai_enriched_health_df_full['encounter_date'].dt.tz_localize(None)
        df_health_period = ai_enriched_health_df_full[
            (ai_enriched_health_df_full['encounter_date'].notna()) &
            (ai_enriched_health_df_full['encounter_date'].dt.date >= selected_period_start_date) &
            (ai_enriched_health_df_full['encounter_date'].dt.date <= selected_period_end_date)
        ].copy()
    elif not ai_enriched_health_df_full.empty: logger.warning(f"({log_ctx}) 'encounter_date' column missing in health data.")

    df_iot_period = pd.DataFrame()
    if iot_data_available_flag and isinstance(raw_iot_df, pd.DataFrame) and 'timestamp' in raw_iot_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df['timestamp']):
            raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')
        if raw_iot_df['timestamp'].dt.tz is not None: 
            raw_iot_df['timestamp'] = raw_iot_df['timestamp'].dt.tz_localize(None)
        df_iot_period = raw_iot_df[
            (raw_iot_df['timestamp'].notna()) &
            (raw_iot_df['timestamp'].dt.date >= selected_period_start_date) & 
            (raw_iot_df['timestamp'].dt.date <= selected_period_end_date)  
        ].copy()
    elif iot_data_available_flag and isinstance(raw_iot_df, pd.DataFrame):
         logger.warning(f"({log_ctx}) 'timestamp' column missing in IoT data.")

    clinic_kpis_period_data: Dict[str, Any] = {"test_summary_details": {}} 
    if not df_health_period.empty:
        try:
            kpis_result = get_clinic_summary_kpis(df_health_period, f"{log_ctx}/PeriodSummaryKPIs")
            if isinstance(kpis_result, dict): clinic_kpis_period_data = kpis_result
            else: logger.warning(f"({log_ctx}) get_clinic_summary_kpis did not return a dictionary.")
        except Exception as e_kpi_clinic:
            logger.error(f"({log_ctx}) Error calculating clinic summary KPIs: {e_kpi_clinic}", exc_info=True)
    else: logger.info(f"({log_ctx}) No health data in selected period for clinic summary KPIs.")
    
    logger.info(f"({log_ctx}) Data processing complete. Health(Full):{len(ai_enriched_health_df_full)}, Health(Period):{len(df_health_period)}, IoT(Period):{len(df_iot_period)}")
    return ai_enriched_health_df_full, df_health_period, df_iot_period, clinic_kpis_period_data, iot_data_available_flag

st.sidebar.markdown("---") 
try:
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_LOGO_SMALL_PATH'):
        project_root_path = Path(settings.PROJECT_ROOT_DIR)
        logo_path_sidebar = project_root_path / settings.APP_LOGO_SMALL_PATH
        if logo_path_sidebar.is_file(): st.sidebar.image(str(logo_path_sidebar.resolve()), width=240) 
        else:
            logger.warning(f"Sidebar logo for Clinic Console not found: {logo_path_sidebar.resolve()}")
            st.sidebar.caption("Logo not found.")
    else:
        logger.warning("Settings for Clinic sidebar logo missing.")
        st.sidebar.caption("Logo config missing.")
except Exception as e_logo_clinic: 
    logger.error(f"Unexpected error displaying Clinic sidebar logo: {e_logo_clinic}", exc_info=True)
    st.sidebar.caption("Error loading logo.")
st.sidebar.markdown("---") 
st.sidebar.header("Console Filters")

abs_min_date_setting = date.today() - timedelta(days=365 * 2) 
abs_max_date_setting = date.today()
default_days_range = (settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND 
                      if hasattr(settings, 'WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND') else 30)
max_query_days = (settings.MAX_QUERY_DAYS_CLINIC if hasattr(settings, 'MAX_QUERY_DAYS_CLINIC') else 90)
default_end_date = abs_max_date_setting
default_start_date = max(abs_min_date_setting, default_end_date - timedelta(days=default_days_range - 1))
date_range_ss_key = "clinic_console_date_range_v5" 

if date_range_ss_key not in st.session_state:
    st.session_state[date_range_ss_key] = [default_start_date, default_end_date]
else: 
    persisted_start, persisted_end = st.session_state[date_range_ss_key]
    current_start = min(max(persisted_start, abs_min_date_setting), abs_max_date_setting)
    current_end = min(max(persisted_end, abs_min_date_setting), abs_max_date_setting)
    if current_start > current_end: current_start = current_end 
    st.session_state[date_range_ss_key] = [current_start, current_end]

selected_range_ui = st.sidebar.date_input(
    "Select Date Range for Clinic Review:", value=st.session_state[date_range_ss_key],
    min_value=abs_min_date_setting, max_value=abs_max_date_setting, key=f"{date_range_ss_key}_widget" 
)

start_date_filter, end_date_filter = default_start_date, default_end_date 
if isinstance(selected_range_ui, (list, tuple)) and len(selected_range_ui) == 2:
    start_date_filter_ui, end_date_filter_ui = selected_range_ui
    start_date_filter = min(max(start_date_filter_ui, abs_min_date_setting), abs_max_date_setting)
    end_date_filter = min(max(end_date_filter_ui, abs_min_date_setting), abs_max_date_setting)
    if start_date_filter > end_date_filter:
        st.sidebar.error("Start date must be ≤ end date. Adjusting end date.")
        end_date_filter = start_date_filter 
    if (end_date_filter - start_date_filter).days + 1 > max_query_days:
        st.sidebar.warning(f"Date range limited to {max_query_days} days for performance. Adjusting end date.")
        end_date_filter = start_date_filter + timedelta(days=max_query_days - 1)
        if end_date_filter > abs_max_date_setting:
             end_date_filter = abs_max_date_setting
             start_date_filter = max(abs_min_date_setting, end_date_filter - timedelta(days=max_query_days -1))
    st.session_state[date_range_ss_key] = [start_date_filter, end_date_filter] 
else: 
    start_date_filter, end_date_filter = st.session_state.get(date_range_ss_key, [default_start_date, default_end_date])
    st.sidebar.warning("Date range selection error. Using previous or default range.")

current_period_str = f"{start_date_filter.strftime('%d %b %Y')} - {end_date_filter.strftime('%d %b %Y')}"
full_hist_health_df, health_df_period, iot_df_period = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
clinic_summary_kpis_data: Dict[str, Any] = {"test_summary_details": {}}
iot_available_flag = False

try:
    full_hist_health_df, health_df_period, iot_df_period, clinic_summary_kpis_data, iot_available_flag = \
        get_clinic_console_processed_data(start_date_filter, end_date_filter)
except Exception as e_load_clinic:
    logger.error(f"Clinic Dashboard: Main data loading failed: {e_load_clinic}", exc_info=True)
    st.error(f"🛑 Error loading clinic dashboard data: {str(e_load_clinic)}. Check logs and data sources.")

if not iot_available_flag: 
    st.sidebar.warning("🔌 IoT environmental data source unavailable or empty. Some metrics may be missing.")
st.info(f"Displaying Clinic Console data for period: **{current_period_str}**")

st.header("🚀 Clinic Performance & Environment Snapshot")
main_kpis_display_data = []
disease_kpis_display_data = []

if clinic_summary_kpis_data and isinstance(clinic_summary_kpis_data.get("test_summary_details"), dict): 
    try:
        main_kpis_display_data = structure_main_clinic_kpis(clinic_summary_kpis_data, current_period_str)
    except Exception as e_struct_main_kpi:
        logger.error(f"Error structuring main clinic KPIs: {e_struct_main_kpi}", exc_info=True)
        st.warning("⚠️ Could not structure main clinic KPIs.")
    try:
        disease_kpis_display_data = structure_disease_specific_clinic_kpis(clinic_summary_kpis_data, current_period_str)
    except Exception as e_struct_disease_kpi:
        logger.error(f"Error structuring disease specific KPIs: {e_struct_disease_kpi}", exc_info=True)
        st.warning("⚠️ Could not structure disease-specific KPIs.")
else:
    st.warning(f"Core clinic summary KPI data not available or in unexpected format for {current_period_str}. Some KPIs might be missing.")

if main_kpis_display_data:
    st.markdown("##### **Overall Service Performance:**")
    kpi_cols_main = st.columns(min(len(main_kpis_display_data), 4))
    for i, kpi_data_item in enumerate(main_kpis_display_data): 
        try:
            render_kpi_card(**kpi_data_item, container=kpi_cols_main[i % 4]) 
        except TypeError as te_main_kpi: # Catch TypeError specifically
            logger.error(f"TypeError rendering main KPI {i}. Data: {kpi_data_item}. Error: {te_main_kpi}", exc_info=True)
            st.error(f"Error displaying a main KPI due to argument mismatch. Check logs. KPI Title: {kpi_data_item.get('title', 'Unknown')}")
        except Exception as e_main_kpi_render: # Catch other potential errors
            logger.error(f"Error rendering main KPI {i}. Data: {kpi_data_item}. Error: {e_main_kpi_render}", exc_info=True)
            st.error(f"Unexpected error displaying a main KPI. KPI Title: {kpi_data_item.get('title', 'Unknown')}")
elif not health_df_period.empty: 
    st.info("ℹ️ Main service performance KPIs could not be fully generated for this period.")

if disease_kpis_display_data:
    st.markdown("##### **Key Disease Testing & Supply Indicators:**")
    kpi_cols_disease = st.columns(min(len(disease_kpis_display_data), 4))
    for i, kpi_data_item in enumerate(disease_kpis_display_data): 
        try:
            render_kpi_card(**kpi_data_item, container=kpi_cols_disease[i % 4])
        except TypeError as te_disease_kpi:
            logger.error(f"TypeError rendering disease KPI {i}. Data: {kpi_data_item}. Error: {te_disease_kpi}", exc_info=True)
            st.error(f"Error displaying a disease KPI due to argument mismatch. Check logs. KPI Title: {kpi_data_item.get('title', 'Unknown')}")
        except Exception as e_disease_kpi_render:
            logger.error(f"Error rendering disease KPI {i}. Data: {kpi_data_item}. Error: {e_disease_kpi_render}", exc_info=True)
            st.error(f"Unexpected error displaying a disease KPI. KPI Title: {kpi_data_item.get('title', 'Unknown')}")
elif not health_df_period.empty:
     st.info("ℹ️ Disease-specific KPIs could not be fully generated for this period.")

st.markdown("##### **Clinic Environment Quick Check:**")
env_summary_kpis_quick_check: Dict[str, Any] = {} 
if not iot_df_period.empty: 
    try:
        env_summary_kpis_quick_check = get_clinic_environmental_summary_kpis(iot_df_period, "ClinicDash/EnvQuickCheck")
    except Exception as e_env_kpi:
        logger.error(f"Error getting environmental summary KPIs: {e_env_kpi}", exc_info=True)
        st.warning("⚠️ Could not calculate environmental summary KPIs for the period.")

has_relevant_env_data = any(
    pd.notna(env_summary_kpis_quick_check.get(key))
    for key in ['avg_co2_overall_ppm', 'avg_pm25_overall_ugm3', 
                'avg_waiting_room_occupancy_overall_persons', 'rooms_noise_high_alert_latest_count']
)

if has_relevant_env_data:
    env_kpi_cols_qc = st.columns(4)
    co2_val = env_summary_kpis_quick_check.get('avg_co2_overall_ppm', np.nan)
    co2_very_high_thresh = settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM if hasattr(settings, 'ALERT_AMBIENT_CO2_VERY_HIGH_PPM') else 1500
    co2_high_thresh = settings.ALERT_AMBIENT_CO2_HIGH_PPM if hasattr(settings, 'ALERT_AMBIENT_CO2_HIGH_PPM') else 1000
    co2_stat = "ACCEPTABLE" 
    if pd.notna(co2_val):
        if co2_val > co2_very_high_thresh: co2_stat = "HIGH_RISK"
        elif co2_val > co2_high_thresh: co2_stat = "MODERATE_CONCERN"
    render_kpi_card("Avg. CO2", f"{co2_val:.0f}" if pd.notna(co2_val) else "N/A", "ppm", "💨", co2_stat, help_text=f"Avg CO2. Target < {co2_high_thresh}ppm.", container=env_kpi_cols_qc[0])

    pm25_val = env_summary_kpis_quick_check.get('avg_pm25_overall_ugm3', np.nan)
    pm25_very_high_thresh = settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 if hasattr(settings, 'ALERT_AMBIENT_PM25_VERY_HIGH_UGM3') else 35.4
    pm25_high_thresh = settings.ALERT_AMBIENT_PM25_HIGH_UGM3 if hasattr(settings, 'ALERT_AMBIENT_PM25_HIGH_UGM3') else 12.0
    pm25_stat = "ACCEPTABLE"
    if pd.notna(pm25_val):
        if pm25_val > pm25_very_high_thresh: pm25_stat = "HIGH_RISK"
        elif pm25_val > pm25_high_thresh: pm25_stat = "MODERATE_CONCERN"
    render_kpi_card("Avg. PM2.5", f"{pm25_val:.1f}" if pd.notna(pm25_val) else "N/A", "µg/m³", "🌫️", pm25_stat, help_text=f"Avg PM2.5. Target < {pm25_high_thresh}µg/m³.", container=env_kpi_cols_qc[1])

    occup_val = env_summary_kpis_quick_check.get('avg_waiting_room_occupancy_overall_persons', np.nan)
    occup_max_thresh = settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX if hasattr(settings, 'TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX') else 10
    occup_stat = "ACCEPTABLE"
    if pd.notna(occup_val) and occup_val > occup_max_thresh: occup_stat = "MODERATE_CONCERN"
    render_kpi_card("Avg. Waiting Occupancy", f"{occup_val:.1f}" if pd.notna(occup_val) else "N/A", "persons", "👨‍👩‍👧‍👦", occup_stat, help_text=f"Avg waiting area occupancy. Target < {occup_max_thresh} persons.", container=env_kpi_cols_qc[2])

    noise_alerts_val = env_summary_kpis_quick_check.get('rooms_noise_high_alert_latest_count', 0) 
    noise_high_thresh_dba = settings.ALERT_AMBIENT_NOISE_HIGH_DBA if hasattr(settings, 'ALERT_AMBIENT_NOISE_HIGH_DBA') else 70
    noise_stat = "ACCEPTABLE"
    if noise_alerts_val > 1: noise_stat = "HIGH_CONCERN"
    elif noise_alerts_val == 1: noise_stat = "MODERATE_CONCERN"
    render_kpi_card("High Noise Alerts", str(noise_alerts_val), "areas", "🔊", noise_stat, help_text=f"Areas with noise > {noise_high_thresh_dba}dBA (latest reading).", container=env_kpi_cols_qc[3])
elif iot_df_period.empty and iot_available_flag: 
    st.info("ℹ️ No environmental IoT data recorded for the selected period for snapshot KPIs.")
elif not iot_available_flag: 
    st.info("🔌 Environmental IoT data source generally unavailable for snapshot.")
st.divider()

st.header("🛠️ Operational Areas Deep Dive")
tab_titles = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tabs_list = st.tabs(tab_titles) 

with tabs_list[0]: 
    st.subheader(f"Local Epidemiological Intelligence ({current_period_str})")
    if not health_df_period.empty:
        try:
            epi_data = calculate_clinic_epidemiological_data(health_df_period, current_period_str)
            symptom_trends_df = epi_data.get("symptom_trends_weekly_top_n_df")
            if isinstance(symptom_trends_df, pd.DataFrame) and not symptom_trends_df.empty:
                st.plotly_chart(plot_bar_chart(symptom_trends_df, 'week_start_date', 'count', "Weekly Symptom Frequency (Top Reported)", 'symptom', 'group', y_values_are_counts_flag=True, x_axis_label_text="Week Starting", y_axis_label_text="Symptom Encounters"), use_container_width=True)
            else: st.caption("ℹ️ No symptom trend data to display for this period.")

            malaria_rdt_name_setting = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT") if hasattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS') else "Malaria RDT"
            malaria_pos_trend_series = epi_data.get("key_test_positivity_trends", {}).get(malaria_rdt_name_setting)
            malaria_target_pos_rate_setting = settings.TARGET_MALARIA_POSITIVITY_RATE if hasattr(settings, 'TARGET_MALARIA_POSITIVITY_RATE') else 0.10
            if isinstance(malaria_pos_trend_series, pd.Series) and not malaria_pos_trend_series.empty:
                st.plotly_chart(plot_annotated_line_chart(malaria_pos_trend_series, f"Weekly {malaria_rdt_name_setting} Positivity Rate", "Positivity %", target_ref_line_val=malaria_target_pos_rate_setting, y_values_are_counts=False), use_container_width=True)
            else: st.caption(f"ℹ️ No {malaria_rdt_name_setting} positivity trend data to display for this period.")
            
            for note in epi_data.get("calculation_notes", []): st.caption(f"Note (Epi Tab): {note}")
        except Exception as e_epi_tab:
            logger.error(f"Error processing Epi Tab content: {e_epi_tab}", exc_info=True)
            st.error("⚠️ An error occurred while generating epidemiological insights.")
    else:
        st.info("ℹ️ No health data available in the selected period for epidemiological analysis.")

with tabs_list[1]: 
    st.subheader(f"Testing & Diagnostics Performance ({current_period_str})")
    if not health_df_period.empty: 
        try:
            testing_insights_map = prepare_clinic_lab_testing_insights_data(health_df_period, clinic_summary_kpis_data, current_period_str, "All Critical Tests Summary")
            crit_tests_summary_df = testing_insights_map.get("all_critical_tests_summary_table_df")
            if isinstance(crit_tests_summary_df, pd.DataFrame) and not crit_tests_summary_df.empty:
                st.markdown("###### **Critical Tests Performance Summary:**")
                st.dataframe(crit_tests_summary_df, use_container_width=True, hide_index=True)
            else: st.caption("ℹ️ No critical tests summary data to display for this period.")

            overdue_tests_df = testing_insights_map.get("overdue_pending_tests_list_df")
            if isinstance(overdue_tests_df, pd.DataFrame) and not overdue_tests_df.empty:
                st.markdown("###### **Overdue Pending Tests (Top 15):**")
                st.dataframe(overdue_tests_df.head(15), use_container_width=True, hide_index=True)
            elif isinstance(overdue_tests_df, pd.DataFrame): 
                st.success("✅ No tests currently flagged as overdue.")
            else: st.caption("ℹ️ No overdue tests data to display for this period.")
            
            for note in testing_insights_map.get("processing_notes", []): st.caption(f"Note (Testing Tab): {note}")
        except Exception as e_testing_tab:
            logger.error(f"Error processing Testing Tab content: {e_testing_tab}", exc_info=True)
            st.error("⚠️ An error occurred while generating testing insights.")
    else:
        st.info("ℹ️ No health data available in the selected period for testing insights.")

with tabs_list[2]: 
    st.subheader(f"Medical Supply Forecast & Status ({current_period_str})")
    if not full_hist_health_df.empty: 
        try:
            use_ai_supply_forecast = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle_v5")
            supply_forecast_map = prepare_clinic_supply_forecast_overview_data(full_hist_health_df, current_period_str, use_ai_supply_forecasting_model=use_ai_supply_forecast)
            st.markdown(f"**Forecast Model Used:** `{supply_forecast_map.get('forecast_model_type_used', 'N/A')}`")
            supply_overview_list = supply_forecast_map.get("forecast_items_overview_list", []) 
            if supply_overview_list: 
                supply_df_display = pd.DataFrame(supply_overview_list) 
                if not supply_df_display.empty:
                    st.dataframe(supply_df_display, use_container_width=True, hide_index=True,
                                 column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
                else: st.info("ℹ️ No supply forecast items generated with current model.")
            else: 
                st.info("ℹ️ No supply forecast data generated or list is empty.")
            for note in supply_forecast_map.get("data_processing_notes", []): st.caption(f"Note (Supply Tab): {note}")
        except Exception as e_supply_tab:
            logger.error(f"Error processing Supply Tab content: {e_supply_tab}", exc_info=True)
            st.error("⚠️ An error occurred while generating supply chain insights.")
    else:
        st.info("ℹ️ Insufficient historical health data available for supply forecasting.")

with tabs_list[3]: 
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_str})")
    if not health_df_period.empty:
        try:
            patient_focus_map = prepare_clinic_patient_focus_overview_data(health_df_period, current_period_str)
            patient_load_plot_df = patient_focus_map.get("patient_load_by_key_condition_df")
            if isinstance(patient_load_plot_df, pd.DataFrame) and not patient_load_plot_df.empty:
                st.markdown("###### **Patient Load by Key Condition (Weekly):**")
                st.plotly_chart(plot_bar_chart(patient_load_plot_df, 'period_start_date', 'unique_patients_count', "Patient Load by Key Condition", 'condition', 'stack', y_values_are_counts_flag=True, x_axis_label_text="Week Starting", y_axis_label_text="Unique Patients Seen"), use_container_width=True)
            else: st.caption("ℹ️ No patient load data to display for this period.")

            flagged_patients_df = patient_focus_map.get("flagged_patients_for_review_df")
            if isinstance(flagged_patients_df, pd.DataFrame) and not flagged_patients_df.empty:
                st.markdown("###### **Flagged Patients for Clinical Review (Top Priority):**")
                st.dataframe(flagged_patients_df.head(15), use_container_width=True, hide_index=True)
            elif isinstance(flagged_patients_df, pd.DataFrame): 
                st.info("ℹ️ No patients currently flagged for clinical review based on criteria.")
            else: st.caption("ℹ️ No flagged patient data to display for this period.")
            for note in patient_focus_map.get("processing_notes", []): st.caption(f"Note (Patient Focus Tab): {note}")
        except Exception as e_patient_tab:
            logger.error(f"Error processing Patient Focus Tab content: {e_patient_tab}", exc_info=True)
            st.error("⚠️ An error occurred while generating patient focus insights.")
    else:
        st.info("ℹ️ No health data available in the selected period for patient focus analysis.")

with tabs_list[4]: 
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_str})")
    if iot_available_flag: 
        try:
            env_details_data_map = prepare_clinic_environmental_detail_data(iot_df_period, iot_available_flag, current_period_str)
            current_env_alerts_list = env_details_data_map.get("current_environmental_alerts_list", [])
            if current_env_alerts_list: 
                st.markdown("###### **Current Environmental Alerts (Latest Readings):**")
                non_acceptable_found = False
                for alert_item in current_env_alerts_list:
                    if alert_item.get("level") != "ACCEPTABLE": 
                        non_acceptable_found = True
                        render_traffic_light_indicator(
                            title=alert_item.get('message', 'Issue detected.'),
                            level=alert_item.get('level', 'UNKNOWN'), 
                            details=alert_item.get('alert_type', 'Env Alert') 
                        )
                if not non_acceptable_found: 
                    if len(current_env_alerts_list) == 1 and current_env_alerts_list[0].get("level") == "ACCEPTABLE":
                        st.success(f"✅ {current_env_alerts_list[0].get('message', 'Environment appears normal.')}")
                    else: 
                        st.success("✅ All monitored environmental parameters appear within acceptable limits based on latest readings.")
            elif not iot_df_period.empty: 
                 st.info("ℹ️ No specific environmental alerts currently active based on latest readings for the period.")

            co2_trend_series = env_details_data_map.get("hourly_avg_co2_trend")
            co2_high_target_setting = settings.ALERT_AMBIENT_CO2_HIGH_PPM if hasattr(settings, 'ALERT_AMBIENT_CO2_HIGH_PPM') else 1000
            if isinstance(co2_trend_series, pd.Series) and not co2_trend_series.empty:
                st.plotly_chart(plot_annotated_line_chart(co2_trend_series, "Hourly Avg. CO2 Levels (Clinic-wide)", "CO2 (ppm)", date_format_hover="%H:%M (%d-%b)", target_ref_line_val=co2_high_target_setting), use_container_width=True)
            elif not iot_df_period.empty: st.caption("ℹ️ CO2 trend data not available for display for this period.")

            latest_room_readings_df = env_details_data_map.get("latest_room_sensor_readings_df")
            if isinstance(latest_room_readings_df, pd.DataFrame) and not latest_room_readings_df.empty:
                st.markdown("###### **Latest Sensor Readings by Room (End of Period):**")
                st.dataframe(latest_room_readings_df, use_container_width=True, hide_index=True)
            elif not iot_df_period.empty: st.caption("ℹ️ Latest room sensor readings not available for display for this period.")
            
            for note in env_details_data_map.get("processing_notes", []): st.caption(f"Note (Env. Detail Tab): {note}")

            if iot_df_period.empty and iot_available_flag: 
                st.info("ℹ️ No IoT environmental data recorded for the selected period.")

        except Exception as e_env_tab:
            logger.error(f"Error processing Environment Detail Tab content: {e_env_tab}", exc_info=True)
            st.error("⚠️ An error occurred while generating environmental details.")
    else: 
        st.warning("🔌 IoT environmental data source is unavailable. Detailed environmental monitoring not possible.")

st.divider()
footer_text = settings.APP_FOOTER_TEXT if hasattr(settings, 'APP_FOOTER_TEXT') else "Sentinel Health Co-Pilot."
st.caption(footer_text)

logger.info(
    f"Clinic Operations Console page fully rendered. Period: {current_period_str}. "
    f"HealthDataPresentAndNotEmpty:{isinstance(health_df_period, pd.DataFrame) and not health_df_period.empty}, "
    f"IoTDataPresentAndNotEmpty:{isinstance(iot_df_period, pd.DataFrame) and not iot_df_period.empty}, "
    f"IoTAvailableFlag:{iot_available_flag}"
)
