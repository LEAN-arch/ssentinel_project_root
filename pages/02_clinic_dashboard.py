# sentinel_project_root/pages/clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta 
from typing import Optional, Dict, Any, Tuple, List
import os 
from pathlib import Path 

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
    
    # Clinic specific components using absolute imports from 'pages' package
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash_abs_final: 
    import sys
    _current_file_clinic_final = Path(__file__).resolve()
    _pages_dir_clinic_final = _current_file_clinic_final.parent
    _project_root_clinic_assumption_final = _pages_dir_clinic_final.parent 

    error_msg_clinic_detail_final = (
        f"Clinic Dashboard Import Error (using absolute imports): {e_clinic_dash_abs_final}. "
        f"Ensure project root ('{_project_root_clinic_assumption_final}') is in sys.path (done by app.py) "
        f"and all modules/packages (e.g., 'pages', 'pages.clinic_components') have `__init__.py` files. "
        f"Check for typos in import paths. Current Python Path: {sys.path}"
    )
    try:
        st.error(error_msg_clinic_detail_final)
        st.stop()
    except NameError: 
        print(error_msg_clinic_detail_final, file=sys.stderr)
        raise

logger = logging.getLogger(__name__)

st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset...",
    hash_funcs={pd.DataFrame: lambda df_cache_clinic: pd.util.hash_pandas_object(df_cache_clinic, index=True) if isinstance(df_cache_clinic, pd.DataFrame) else hash(df_cache_clinic)}
)
def get_clinic_console_processed_data(
    selected_period_start_date: date,
    selected_period_end_date: date
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], bool]:
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Loading data for period: {selected_period_start_date.isoformat()} to {selected_period_end_date.isoformat()}")
    
    raw_health_df_clinic = load_health_records(source_context=f"{log_ctx}/LoadRawHealthRecs")
    raw_iot_df_clinic = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadRawIoTData")
    
    # Resolve IOT path correctly for existence check
    iot_source_path_obj = Path(settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH)
    if not iot_source_path_obj.is_absolute():
        iot_source_path_obj = (Path(settings.PROJECT_ROOT_DIR) / settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH).resolve()
    iot_source_file_exists_flag = iot_source_path_obj.exists() and iot_source_path_obj.is_file()
    
    iot_data_actually_loaded_flag = isinstance(raw_iot_df_clinic, pd.DataFrame) and not raw_iot_df_clinic.empty
    is_iot_data_available_final = iot_source_file_exists_flag and iot_data_actually_loaded_flag

    ai_enriched_health_df_full_clinic: Optional[pd.DataFrame]
    if isinstance(raw_health_df_clinic, pd.DataFrame) and not raw_health_df_clinic.empty:
        ai_enriched_health_df_full_clinic, _ = apply_ai_models(raw_health_df_clinic.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data for clinic is empty or invalid. AI enrichment skipped.")
        ai_enriched_health_df_full_clinic = pd.DataFrame() # Fallback to empty

    df_health_period_final: pd.DataFrame = pd.DataFrame()
    if isinstance(ai_enriched_health_df_full_clinic, pd.DataFrame) and not ai_enriched_health_df_full_clinic.empty and 'encounter_date' in ai_enriched_health_df_full_clinic.columns:
        if not pd.api.types.is_datetime64_any_dtype(ai_enriched_health_df_full_clinic['encounter_date']):
            ai_enriched_health_df_full_clinic['encounter_date'] = pd.to_datetime(ai_enriched_health_df_full_clinic['encounter_date'], errors='coerce')
        df_health_period_final = ai_enriched_health_df_full_clinic[
            (ai_enriched_health_df_full_clinic['encounter_date'].notna()) &
            (ai_enriched_health_df_full_clinic['encounter_date'].dt.date >= selected_period_start_date) &
            (ai_enriched_health_df_full_clinic['encounter_date'].dt.date <= selected_period_end_date)
        ].copy()
    
    df_iot_period_final: pd.DataFrame = pd.DataFrame()
    if is_iot_data_available_final and isinstance(raw_iot_df_clinic, pd.DataFrame) and 'timestamp' in raw_iot_df_clinic.columns:
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df_clinic['timestamp']):
            raw_iot_df_clinic['timestamp'] = pd.to_datetime(raw_iot_df_clinic['timestamp'], errors='coerce')
        df_iot_period_final = raw_iot_df_clinic[
            (raw_iot_df_clinic['timestamp'].notna()) &
            (raw_iot_df_clinic['timestamp'].dt.date >= selected_period_start_date) &
            (raw_iot_df_clinic['timestamp'].dt.date <= selected_period_end_date)
        ].copy()

    clinic_kpis_period_data: Dict[str, Any] = {"test_summary_details": {}}
    if not df_health_period_final.empty:
        try: clinic_kpis_period_data = get_clinic_summary_kpis(df_health_period_final, f"{log_ctx}/PeriodSummaryKPIs")
        except Exception as e_kpi_clinic: logger.error(f"({log_ctx}) Error calculating clinic summary KPIs: {e_kpi_clinic}", exc_info=True)
    else: logger.info(f"({log_ctx}) No health data in selected period for clinic summary KPIs.")
    
    return ai_enriched_health_df_full_clinic, df_health_period_final, df_iot_period_final, clinic_kpis_period_data, is_iot_data_available_final

# --- Sidebar Filters ---
logo_path_sidebar_clinic_final = Path(settings.APP_LOGO_SMALL_PATH)
if not logo_path_sidebar_clinic_final.is_absolute(): logo_path_sidebar_clinic_final = (Path(settings.PROJECT_ROOT_DIR) / settings.APP_LOGO_SMALL_PATH).resolve()
if logo_path_sidebar_clinic_final.exists() and logo_path_sidebar_clinic_final.is_file(): st.sidebar.image(str(logo_path_sidebar_clinic_final), width=240)
else: logger.warning(f"Sidebar logo not found: {logo_path_sidebar_clinic_final}")
st.sidebar.header("Console Filters")

abs_min_date_clinic_val = date.today() - timedelta(days=365); abs_max_date_clinic_val = date.today()
def_end_date_clinic_val = abs_max_date_clinic_val
def_start_date_clinic_val = max(abs_min_date_clinic_val, def_end_date_clinic_val - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND -1))
date_range_key_clinic_ss_val = "clinic_console_date_range_v3" # Ensure unique key
if date_range_key_clinic_ss_val not in st.session_state: st.session_state[date_range_key_clinic_ss_val] = [def_start_date_clinic_val, def_end_date_clinic_val]

selected_range_clinic_ui_val = st.sidebar.date_input("Select Date Range for Clinic Review:", value=st.session_state[date_range_key_clinic_ss_val], min_value=abs_min_date_clinic_val, max_value=abs_max_date_clinic_val, key=f"{date_range_key_clinic_ss_val}_widget")
start_date_clinic_filt_val: date; end_date_clinic_filt_val: date 
if isinstance(selected_range_clinic_ui_val, (list, tuple)) and len(selected_range_clinic_ui_val) == 2:
    st.session_state[date_range_key_clinic_ss_val] = selected_range_clinic_ui_val
    start_date_clinic_filt_val, end_date_clinic_filt_val = selected_range_clinic_ui_val
else: 
    start_date_clinic_filt_val, end_date_clinic_filt_val = st.session_state[date_range_key_clinic_ss_val]
    st.sidebar.warning("Date range selection error. Using previous/default.")
if start_date_clinic_filt_val > end_date_clinic_filt_val:
    st.sidebar.error("Start date must be <= end date. Adjusting end date."); end_date_clinic_filt_val = start_date_clinic_filt_val
    st.session_state[date_range_key_clinic_ss_val][1] = end_date_clinic_filt_val

MAX_QUERY_DAYS_CLINIC_VAL = 90 
if (end_date_clinic_filt_val - start_date_clinic_filt_val).days + 1 > MAX_QUERY_DAYS_CLINIC_VAL: 
    st.sidebar.warning(f"Date range limited to {MAX_QUERY_DAYS_CLINIC_VAL} days for performance.")
    end_date_clinic_filt_val = start_date_clinic_filt_val + timedelta(days=MAX_QUERY_DAYS_CLINIC_VAL -1)
    if end_date_clinic_filt_val > abs_max_date_clinic_val: end_date_clinic_filt_val = abs_max_date_clinic_val
    st.session_state[date_range_key_clinic_ss_val] = [start_date_clinic_filt_val, end_date_clinic_filt_val]

# --- Load Data ---
current_period_str_clinic_val = f"{start_date_clinic_filt_val.strftime('%d %b %Y')} - {end_date_clinic_filt_val.strftime('%d %b %Y')}"
full_hist_health_df_val, health_df_period_val, iot_df_period_val, clinic_summary_kpis_data_val, iot_available_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"test_summary_details": {}}, False
try:
    full_hist_health_df_val, health_df_period_val, iot_df_period_val, clinic_summary_kpis_data_val, iot_available_val = get_clinic_console_processed_data(start_date_clinic_filt_val, end_date_clinic_filt_val)
except Exception as e_load_clinic_val:
    logger.error(f"Clinic Dashboard: Main data loading failed: {e_load_clinic_val}", exc_info=True)
    st.error(f"Error loading clinic dashboard data: {e_load_clinic_val}. Check logs.")
if not iot_available_val: st.sidebar.warning("IoT environmental data source unavailable. Some metrics may be missing.")
st.info(f"Displaying Clinic Console data for period: **{current_period_str_clinic_val}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Clinic Performance & Environment Snapshot")
if clinic_summary_kpis_data_val and isinstance(clinic_summary_kpis_data_val.get("test_summary_details"), dict):
    main_kpis_clinic_val = structure_main_clinic_kpis(clinic_summary_kpis_data_val, current_period_str_clinic_val)
    disease_kpis_clinic_val = structure_disease_specific_clinic_kpis(clinic_summary_kpis_data_val, current_period_str_clinic_val)
    if main_kpis_clinic_val:
        st.markdown("##### **Overall Service Performance:**"); kpi_cols_main_val = st.columns(min(len(main_kpis_clinic_val), 4))
        for i_main, kpi_data_main in enumerate(main_kpis_clinic_val):
            with kpi_cols_main_val[i_main % 4]: render_kpi_card(**kpi_data_main)
    if disease_kpis_clinic_val:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**"); kpi_cols_disease_val = st.columns(min(len(disease_kpis_clinic_val), 4))
        for i_disease, kpi_data_disease in enumerate(disease_kpis_clinic_val):
            with kpi_cols_disease_val[i_disease % 4]: render_kpi_card(**kpi_data_disease)
else: st.warning(f"Core clinic KPIs could not be generated for {current_period_str_clinic_val}.")

st.markdown("##### **Clinic Environment Quick Check:**")
env_summary_kpis_qc_val = get_clinic_environmental_summary_kpis(iot_df_period_val, "ClinicDash/EnvQuickCheck")
has_env_data_qc_val = env_summary_kpis_qc_val and any(pd.notna(v_env) and (v_env != 0 if "count" in k_env else True) for k_env,v_env in env_summary_kpis_qc_val.items() if isinstance(v_env, (int,float)) and ("avg_" in k_env or "_count" in k_env or "_flag" in k_env))
if has_env_data_qc_val:
    env_kpi_cols_qc_val = st.columns(4)
    co2_val_qc = env_summary_kpis_qc_val.get('avg_co2_overall_ppm', np.nan)
    co2_stat_qc = "HIGH_RISK" if pd.notna(co2_val_qc) and co2_val_qc > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM else ("MODERATE_CONCERN" if pd.notna(co2_val_qc) and co2_val_qc > settings.ALERT_AMBIENT_CO2_HIGH_PPM else "ACCEPTABLE")
    with env_kpi_cols_qc_val[0]: render_kpi_card("Avg. CO2", f"{co2_val_qc:.0f}" if pd.notna(co2_val_qc) else "N/A", "ppm", "💨", co2_stat_qc, help_text=f"Avg CO2. Target < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm.")
    pm25_val_qc = env_summary_kpis_qc_val.get('avg_pm25_overall_ugm3', np.nan)
    pm25_stat_qc = "HIGH_RISK" if pd.notna(pm25_val_qc) and pm25_val_qc > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 else ("MODERATE_CONCERN" if pd.notna(pm25_val_qc) and pm25_val_qc > settings.ALERT_AMBIENT_PM25_HIGH_UGM3 else "ACCEPTABLE")
    with env_kpi_cols_qc_val[1]: render_kpi_card("Avg. PM2.5", f"{pm25_val_qc:.1f}" if pd.notna(pm25_val_qc) else "N/A", "µg/m³", "🌫️", pm25_stat_qc, help_text=f"Avg PM2.5. Target < {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³.")
    occup_val_qc = env_summary_kpis_qc_val.get('avg_waiting_room_occupancy_overall_persons', np.nan)
    occup_stat_qc = "MODERATE_CONCERN" if pd.notna(occup_val_qc) and occup_val_qc > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX else "ACCEPTABLE"
    with env_kpi_cols_qc_val[2]: render_kpi_card("Avg. Waiting Occupancy", f"{occup_val_qc:.1f}" if pd.notna(occup_val_qc) else "N/A", "persons", "👨‍👩‍👧‍👦", occup_stat_qc, help_text=f"Avg waiting area occupancy. Target < {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.")
    noise_alerts_val_qc = env_summary_kpis_qc_val.get('rooms_noise_high_alert_latest_count', 0)
    noise_stat_qc = "HIGH_CONCERN" if noise_alerts_val_qc > 1 else ("MODERATE_CONCERN" if noise_alerts_val_qc == 1 else "ACCEPTABLE")
    with env_kpi_cols_qc_val[3]: render_kpi_card("High Noise Alerts", str(noise_alerts_val_qc), "areas", "🔊", noise_stat_qc, help_text=f"Areas with noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA (latest).")
else: st.info("No significant environmental IoT data for this period for snapshot KPIs." if iot_available_val else "Environmental IoT data source generally unavailable for snapshot.")
st.divider()

# --- Tabbed Interface ---
st.header("🛠️ Operational Areas Deep Dive")
tab_titles_clinic_val = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi_clinic_val, tab_testing_clinic_val, tab_supply_clinic_val, tab_patient_clinic_val, tab_env_clinic_val = st.tabs(tab_titles_clinic_val)

with tab_epi_clinic_val:
    st.subheader(f"Local Epidemiological Intelligence ({current_period_str_clinic_val})")
    if not health_df_period_val.empty:
        epi_data_clinic_val = calculate_clinic_epidemiological_data(health_df_period_val, current_period_str_clinic_val)
        symptom_trends_df_val = epi_data_clinic_val.get("symptom_trends_weekly_top_n_df")
        if isinstance(symptom_trends_df_val, pd.DataFrame) and not symptom_trends_df_val.empty:
            st.plotly_chart(plot_bar_chart(symptom_trends_df_val, 'week_start_date', 'count', "Weekly Symptom Frequency (Top Reported)", 'symptom', 'group', y_values_are_counts_flag=True, x_axis_label_text="Week Starting", y_axis_label_text="Symptom Encounters"), use_container_width=True)
        malaria_rdt_name_val = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
        malaria_pos_trend_val = epi_data_clinic_val.get("key_test_positivity_trends", {}).get(malaria_rdt_name_val)
        if isinstance(malaria_pos_trend_val, pd.Series) and not malaria_pos_trend_val.empty:
            st.plotly_chart(plot_annotated_line_chart(malaria_pos_trend_val, f"Weekly {malaria_rdt_name_val} Positivity Rate", "Positivity %", target_ref_line_val=settings.TARGET_MALARIA_POSITIVITY_RATE, y_values_are_counts=False), use_container_width=True)
        for note_epi_val in epi_data_clinic_val.get("calculation_notes", []): st.caption(f"Note (Epi Tab): {note_epi_val}")
    else: st.info("No health data in period for epidemiological analysis.")

with tab_testing_clinic_val:
    st.subheader(f"Testing & Diagnostics Performance ({current_period_str_clinic_val})")
    testing_insights_map_val = prepare_clinic_lab_testing_insights_data(health_df_period_val, clinic_summary_kpis_data_val, current_period_str_clinic_val, "All Critical Tests Summary")
    crit_tests_summary_df_val = testing_insights_map_val.get("all_critical_tests_summary_table_df")
    if isinstance(crit_tests_summary_df_val, pd.DataFrame) and not crit_tests_summary_df_val.empty:
        st.markdown("###### **Critical Tests Performance Summary:**"); st.dataframe(crit_tests_summary_df_val, use_container_width=True, hide_index=True)
    overdue_tests_df_val = testing_insights_map_val.get("overdue_pending_tests_list_df")
    if isinstance(overdue_tests_df_val, pd.DataFrame) and not overdue_tests_df_val.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15):**"); st.dataframe(overdue_tests_df_val.head(15), use_container_width=True, hide_index=True)
    elif isinstance(overdue_tests_df_val, pd.DataFrame): st.success("✅ No tests currently flagged as overdue.")
    for note_test_val in testing_insights_map_val.get("processing_notes", []): st.caption(f"Note (Testing Tab): {note_test_val}")

with tab_supply_clinic_val:
    st.subheader(f"Medical Supply Forecast & Status ({current_period_str_clinic_val})")
    use_ai_supply_val = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle_v3") # Ensure unique key
    supply_forecast_map_val = prepare_clinic_supply_forecast_overview_data(full_hist_health_df_val, current_period_str_clinic_val, use_ai_supply_forecasting_model=use_ai_supply_val)
    st.markdown(f"**Forecast Model Used:** `{supply_forecast_map_val.get('forecast_model_type_used', 'N/A')}`")
    supply_overview_list_val = supply_forecast_map_val.get("forecast_items_overview_list", [])
    if supply_overview_list_val:
        st.dataframe(pd.DataFrame(supply_overview_list_val), use_container_width=True, hide_index=True, column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
    else: st.info("No supply forecast data generated.")
    for note_supply_val in supply_forecast_map_val.get("data_processing_notes", []): st.caption(f"Note (Supply Tab): {note_supply_val}")

with tab_patient_clinic_val:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_period_str_clinic_val})")
    if not health_df_period_val.empty:
        patient_focus_map_val = prepare_clinic_patient_focus_overview_data(health_df_period_val, current_period_str_clinic_val)
        patient_load_plot_df_val = patient_focus_map_val.get("patient_load_by_key_condition_df")
        if isinstance(patient_load_plot_df_val, pd.DataFrame) and not patient_load_plot_df_val.empty:
            st.markdown("###### **Patient Load by Key Condition (Weekly):**")
            st.plotly_chart(plot_bar_chart(patient_load_plot_df_val, 'period_start_date', 'unique_patients_count', "Patient Load by Key Condition", 'condition', 'stack', y_values_are_counts_flag=True, x_axis_label_text="Week Starting", y_axis_label_text="Unique Patients Seen"), use_container_width=True)
        flagged_patients_df_val = patient_focus_map_val.get("flagged_patients_for_review_df")
        if isinstance(flagged_patients_df_val, pd.DataFrame) and not flagged_patients_df_val.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top Priority):**"); st.dataframe(flagged_patients_df_val.head(15), use_container_width=True, hide_index=True)
        elif isinstance(flagged_patients_df_val, pd.DataFrame): st.info("No patients currently flagged for clinical review.")
        for note_patient_val in patient_focus_map_val.get("processing_notes", []): st.caption(f"Note (Patient Focus Tab): {note_patient_val}")
    else: st.info("No health data in period for patient focus analysis.")

with tab_env_clinic_val:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_period_str_clinic_val})")
    env_details_data_map_val = prepare_clinic_environmental_detail_data(iot_df_period_val, iot_available_val, current_period_str_clinic_val)
    list_current_env_alerts_val = env_details_data_map_val.get("current_environmental_alerts_list", [])
    if list_current_env_alerts_val:
        st.markdown("###### **Current Environmental Alerts (Latest Readings):**"); non_acceptable_found_val = False
        for alert_item_env_val in list_current_env_alerts_val:
            if alert_item_env_val.get("level") != "ACCEPTABLE": 
                non_acceptable_found_val = True
                render_traffic_light_indicator(alert_item_env_val.get('message', 'Issue detected.'), alert_item_env_val.get('level', 'UNKNOWN'), alert_item_env_val.get('alert_type', 'Env Alert'))
        if not non_acceptable_found_val and len(list_current_env_alerts_val) == 1 and list_current_env_alerts_val[0].get("level") == "ACCEPTABLE": 
            st.success(f"✅ {list_current_env_alerts_val[0].get('message', 'Environment appears normal.')}")
        elif not non_acceptable_found_val and len(list_current_env_alerts_val) > 1: 
            st.info("Multiple environmental parameters checked; all appear within acceptable limits based on latest readings.")
    co2_trend_clinic_val = env_details_data_map_val.get("hourly_avg_co2_trend")
    if isinstance(co2_trend_clinic_val, pd.Series) and not co2_trend_clinic_val.empty: 
        st.plotly_chart(plot_annotated_line_chart(co2_trend_clinic_val, "Hourly Avg. CO2 Levels (Clinic-wide)", "CO2 (ppm)", date_format_hover="%H:%M (%d-%b)", target_ref_line_val=settings.ALERT_AMBIENT_CO2_HIGH_PPM), use_container_width=True)
    latest_room_df_val = env_details_data_map_val.get("latest_room_sensor_readings_df")
    if isinstance(latest_room_df_val, pd.DataFrame) and not latest_room_df_val.empty: 
        st.markdown("###### **Latest Sensor Readings by Room (End of Period):**"); st.dataframe(latest_room_df_val, use_container_width=True, hide_index=True)
    for note_env_val in env_details_data_map_val.get("processing_notes", []): st.caption(f"Note (Env. Detail Tab): {note_env_val}")
    if not iot_available_val and (not isinstance(iot_df_period_val, pd.DataFrame) or iot_df_period_val.empty):
        st.warning("IoT environmental data source generally unavailable. Detailed environmental monitoring not possible.")

logger.info(f"Clinic Operations Console page loaded for period: {current_period_str_clinic_val}")
