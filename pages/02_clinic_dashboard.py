# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
import os
import sys

# --- Page Specific Logger ---
# Define logger early to be available for all parts of the script, including import error handling.
logger = logging.getLogger(__name__)

# --- Sentinel System Imports from Refactored Structure ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart

    # Clinic specific components from the new structure
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash:
    # Corrected relative import path for Streamlit pages
    st.error(
        f"Clinic Dashboard Import Error: {e_clinic_dash}. This might be due to a missing "
        f"__init__.py file in the 'pages' directory or an incorrect project structure. "
        f"Ensure 'sentinel_project_root' is in the Python path.\n"
        f"Relevant Python Path: {sys.path}"
    )
    logger.error(f"Clinic Dashboard Import Error: {e_clinic_dash}", exc_info=True)
    st.stop()


# --- Page Title and Introduction ---
# Page config is set in app.py
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()


# --- Data Loading Function for this Dashboard ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading comprehensive clinic operational dataset...",
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True)}
)
def get_clinic_console_processed_data(
    selected_period_start_date: date,
    selected_period_end_date: date
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], bool]:
    """
    Loads, enriches (with AI scores), and filters data for the Clinic Console.

    Returns:
        - Full historical health DataFrame (AI enriched, not period filtered).
        - Period-filtered health DataFrame (AI enriched).
        - Period-filtered IoT DataFrame.
        - Clinic summary KPIs for the period.
        - Flag indicating IoT data source availability.
    """
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Loading data for period: {selected_period_start_date.isoformat()} to {selected_period_end_date.isoformat()}")

    # Load raw data
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadRawHealthRecs")
    raw_iot_df = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadRawIoTData")

    iot_source_file_exists = os.path.exists(settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH)
    iot_data_actually_loaded = isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty
    is_iot_data_available = iot_source_file_exists and iot_data_actually_loaded

    # Enrich health data with AI models (e.g., risk scores)
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        ai_enriched_health_df_full, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrichHealth")
    else:
        logger.warning(f"({log_ctx}) Raw health data is empty or invalid. AI enrichment skipped.")
        ai_enriched_health_df_full = pd.DataFrame()

    # Filter AI-enriched health data for the selected period
    df_health_for_period_display = pd.DataFrame()
    if not ai_enriched_health_df_full.empty and 'encounter_date' in ai_enriched_health_df_full.columns:
        if not pd.api.types.is_datetime64_any_dtype(ai_enriched_health_df_full['encounter_date']):
            ai_enriched_health_df_full['encounter_date'] = pd.to_datetime(ai_enriched_health_df_full['encounter_date'], errors='coerce')

        mask = (
            (ai_enriched_health_df_full['encounter_date'].notna()) &
            (ai_enriched_health_df_full['encounter_date'].dt.date >= selected_period_start_date) &
            (ai_enriched_health_df_full['encounter_date'].dt.date <= selected_period_end_date)
        )
        df_health_for_period_display = ai_enriched_health_df_full[mask].copy()

    # Filter IoT data for the selected period
    df_iot_for_period_display = pd.DataFrame()
    if is_iot_data_available and 'timestamp' in raw_iot_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df['timestamp']):
            raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')

        mask_iot = (
            (raw_iot_df['timestamp'].notna()) &
            (raw_iot_df['timestamp'].dt.date >= selected_period_start_date) &
            (raw_iot_df['timestamp'].dt.date <= selected_period_end_date)
        )
        df_iot_for_period_display = raw_iot_df[mask_iot].copy()

    # Get clinic summary KPIs for the period using the period-filtered health data
    clinic_summary_kpis_for_period = {"test_summary_details": {}}  # Default structure
    if not df_health_for_period_display.empty:
        try:
            clinic_summary_kpis_for_period = get_clinic_summary_kpis(
                health_df_period=df_health_for_period_display,
                source_context=f"{log_ctx}/PeriodSummaryKPIs"
            )
        except Exception as e_summary_kpi:
            logger.error(f"({log_ctx}) Error calculating clinic summary KPIs: {e_summary_kpi}", exc_info=True)
    else:
        logger.info(f"({log_ctx}) No health data in selected period for clinic summary KPIs. Using defaults.")

    return ai_enriched_health_df_full, df_health_for_period_display, df_iot_for_period_display, clinic_summary_kpis_for_period, is_iot_data_available


# --- Sidebar Filters Setup ---
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
st.sidebar.header("Console Filters")

# Date Range Picker for Clinic Console
abs_min_date = date.today() - timedelta(days=365)
abs_max_date = date.today()

default_end_date = abs_max_date
default_start_date = default_end_date - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1)
default_start_date = max(default_start_date, abs_min_date)

# Session state for date range picker
date_range_session_key = "clinic_console_date_range_selection"
if date_range_session_key not in st.session_state:
    st.session_state[date_range_session_key] = (default_start_date, default_end_date)

selected_date_range = st.sidebar.date_input(
    "Select Date Range for Clinic Review:",
    value=st.session_state[date_range_session_key],
    min_value=abs_min_date,
    max_value=abs_max_date,
    key=f"{date_range_session_key}_widget"
)

if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date, end_date = default_start_date, default_end_date

# Validate and limit date range
if start_date > end_date:
    st.sidebar.error("Start date must be on or before end date. Adjusting end date.")
    end_date = start_date

MAX_QUERY_DAYS = 90
if (end_date - start_date).days > MAX_QUERY_DAYS:
    st.sidebar.warning(f"Date range is too large. Limiting to {MAX_QUERY_DAYS} days from the start date for performance.")
    end_date = min(start_date + timedelta(days=MAX_QUERY_DAYS - 1), abs_max_date)

# Update session state with validated dates for persistence
st.session_state[date_range_session_key] = (start_date, end_date)
start_date_query_clinic, end_date_query_clinic = start_date, end_date


# --- Load Data Based on Selected Filters ---
current_reporting_period_display_str = f"{start_date_query_clinic.strftime('%d %b %Y')} - {end_date_query_clinic.strftime('%d %b %Y')}"

try:
    (full_historical_health_df_clinic,
     health_df_for_period_clinic_tabs,
     iot_df_for_period_clinic_tabs,
     clinic_summary_kpis_for_period_data,
     iot_data_source_is_available) = get_clinic_console_processed_data(start_date_query_clinic, end_date_query_clinic)
except Exception as e_main_clinic_data_load:
    logger.error(f"Clinic Dashboard: Main data loading failed: {e_main_clinic_data_load}", exc_info=True)
    st.error(f"Error loading clinic dashboard data: {str(e_main_clinic_data_load)}. Please contact support.")
    # Provide empty defaults to allow UI to render without crashing
    full_historical_health_df_clinic, health_df_for_period_clinic_tabs, iot_df_for_period_clinic_tabs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    clinic_summary_kpis_for_period_data = {"test_summary_details": {}}
    iot_data_source_is_available = False
    st.stop()

if not iot_data_source_is_available:
    st.sidebar.warning("IoT environmental data source appears unavailable. Some environmental metrics may be missing.")

st.info(f"Displaying Clinic Console data for period: **{current_reporting_period_display_str}**")

# --- Section 1: Top-Level KPIs ---
st.header("🚀 Clinic Performance & Environment Snapshot")

# Structure and display Main Service KPIs
if clinic_summary_kpis_for_period_data and isinstance(clinic_summary_kpis_for_period_data.get("test_summary_details"), dict):
    main_kpi_cards = structure_main_clinic_kpis(clinic_summary_kpis_for_period_data, current_reporting_period_display_str)
    disease_kpi_cards = structure_disease_specific_clinic_kpis(clinic_summary_kpis_for_period_data, current_reporting_period_display_str)

    if main_kpi_cards:
        st.markdown("##### **Overall Service Performance:**")
        cols = st.columns(len(main_kpi_cards))
        for col, kpi_data in zip(cols, main_kpi_cards):
            with col:
                render_kpi_card(**kpi_data)

    if disease_kpi_cards:
        st.markdown("##### **Key Disease Testing & Supply Indicators:**")
        cols = st.columns(len(disease_kpi_cards))
        for col, kpi_data in zip(cols, disease_kpi_cards):
            with col:
                render_kpi_card(**kpi_data)
else:
    st.warning(f"Core clinic performance KPIs could not be generated for {current_reporting_period_display_str}. Check data sources or component logs.")

# Clinic Environment Quick Check KPIs
st.markdown("##### **Clinic Environment Quick Check:**")
env_summary_kpis = get_clinic_environmental_summary_kpis(iot_df_for_period_clinic_tabs, "ClinicDash/EnvQuickCheckKPIs")
has_meaningful_env_data = env_summary_kpis and any(pd.notna(val) for key, val in env_summary_kpis.items() if "avg" in key or "count" in key)

if has_meaningful_env_data:
    env_kpi_cols = st.columns(4)
    with env_kpi_cols[0]:
        co2_val = env_summary_kpis.get('avg_co2_overall_ppm', np.nan)
        co2_status = "HIGH_RISK" if pd.notna(co2_val) and co2_val > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM else \
                     ("MODERATE_CONCERN" if pd.notna(co2_val) and co2_val > settings.ALERT_AMBIENT_CO2_HIGH_PPM else "ACCEPTABLE")
        render_kpi_card(title="Avg. CO2", value_str=f"{co2_val:.0f}" if pd.notna(co2_val) else "N/A", units="ppm", icon="💨", status_level=co2_status, help_text=f"Target < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm.")
    with env_kpi_cols[1]:
        pm25_val = env_summary_kpis.get('avg_pm25_overall_ugm3', np.nan)
        pm25_status = "HIGH_RISK" if pd.notna(pm25_val) and pm25_val > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 else \
                      ("MODERATE_CONCERN" if pd.notna(pm25_val) and pm25_val > settings.ALERT_AMBIENT_PM25_HIGH_UGM3 else "ACCEPTABLE")
        render_kpi_card(title="Avg. PM2.5", value_str=f"{pm25_val:.1f}" if pd.notna(pm25_val) else "N/A", units="µg/m³", icon="🌫️", status_level=pm25_status, help_text=f"Target < {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³.")
    with env_kpi_cols[2]:
        occupancy_val = env_summary_kpis.get('avg_waiting_room_occupancy_overall_persons', np.nan)
        occupancy_status = "MODERATE_CONCERN" if pd.notna(occupancy_val) and occupancy_val > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX else "ACCEPTABLE"
        render_kpi_card(title="Avg. Waiting Occupancy", value_str=f"{occupancy_val:.1f}" if pd.notna(occupancy_val) else "N/A", units="persons", icon="👨‍👩‍👧‍👦", status_level=occupancy_status, help_text=f"Target < {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.")
    with env_kpi_cols[3]:
        noise_alerts_count = env_summary_kpis.get('rooms_noise_high_alert_latest_count', 0)
        noise_status = "HIGH_RISK" if noise_alerts_count > 1 else ("MODERATE_CONCERN" if noise_alerts_count == 1 else "ACCEPTABLE")
        render_kpi_card(title="High Noise Alerts", value_str=str(noise_alerts_count), units="areas", icon="🔊", status_level=noise_status, help_text=f"Areas with sustained noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA.")
else:
    st.info("No significant environmental IoT data available for this period to display snapshot KPIs." if iot_data_source_is_available else "Environmental IoT data source is unavailable. Monitoring snapshot is limited.")
st.divider()


# --- Tabbed Interface for Detailed Operational Areas ---
st.header("🛠️ Operational Areas Deep Dive")
tab_names = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi, tab_testing, tab_supply, tab_patient, tab_env = st.tabs(tab_names)

with tab_epi:
    st.subheader(f"Local Epidemiological Intelligence ({current_reporting_period_display_str})")
    if not health_df_for_period_clinic_tabs.empty:
        epi_data = calculate_clinic_epidemiological_data(health_df_for_period_clinic_tabs, current_reporting_period_display_str)
        
        df_symptom_trends = epi_data.get("symptom_trends_weekly_top_n_df")
        if isinstance(df_symptom_trends, pd.DataFrame) and not df_symptom_trends.empty:
            st.plotly_chart(plot_bar_chart(df_symptom_trends, 'week_start_date', 'count', "Weekly Symptom Frequency (Top Reported)", 'symptom', 'group', True, "Week Starting", "Symptom Encounters"), use_container_width=True)
        
        malaria_rdt_name = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
        malaria_positivity_trend = epi_data.get("key_test_positivity_trends", {}).get(malaria_rdt_name)
        if isinstance(malaria_positivity_trend, pd.Series) and not malaria_positivity_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(malaria_positivity_trend, f"Weekly {malaria_rdt_name} Positivity Rate", "Positivity %", settings.TARGET_MALARIA_POSITIVITY_RATE, False), use_container_width=True)
        
        for note in epi_data.get("calculation_notes", []): st.caption(f"Note: {note}")
    else:
        st.info("No health data available in the selected period for epidemiological analysis.")

with tab_testing:
    st.subheader(f"Testing & Diagnostics Performance ({current_reporting_period_display_str})")
    if not health_df_for_period_clinic_tabs.empty:
        testing_insights = prepare_clinic_lab_testing_insights_data(health_df_for_period_clinic_tabs, clinic_summary_kpis_for_period_data, current_reporting_period_display_str, "All Critical Tests Summary")
        
        df_critical_summary = testing_insights.get("all_critical_tests_summary_table_df")
        if isinstance(df_critical_summary, pd.DataFrame) and not df_critical_summary.empty:
            st.markdown("###### **Critical Tests Performance Summary:**")
            st.dataframe(df_critical_summary, use_container_width=True, hide_index=True)
        
        df_overdue_tests = testing_insights.get("overdue_pending_tests_list_df")
        if isinstance(df_overdue_tests, pd.DataFrame) and not df_overdue_tests.empty:
            st.markdown("###### **Overdue Pending Tests (Top 15 by Days Pending):**")
            st.dataframe(df_overdue_tests.head(15), use_container_width=True, hide_index=True)
        elif isinstance(df_overdue_tests, pd.DataFrame):
            st.success("✅ No tests currently flagged as overdue based on defined criteria.")

        for note in testing_insights.get("processing_notes", []): st.caption(f"Note: {note}")
    else:
        st.info("No health data available in the selected period for testing and diagnostics analysis.")

with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status ({current_reporting_period_display_str})")
    use_ai_forecast = st.checkbox("Use Advanced AI Supply Forecast (Simulated)", value=False, key="supply_ai_toggle")
    
    supply_data = prepare_clinic_supply_forecast_overview_data(full_historical_health_df_clinic, current_reporting_period_display_str, use_ai_forecast)
    st.markdown(f"**Forecast Model Used:** `{supply_data.get('forecast_model_type_used', 'N/A')}`")
    
    supply_overview = supply_data.get("forecast_items_overview_list", [])
    if supply_overview:
        st.dataframe(pd.DataFrame(supply_overview), use_container_width=True, hide_index=True, column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
    else:
        st.info("No supply forecast data generated for the selected items or model type.")
    
    for note in supply_data.get("data_processing_notes", []): st.caption(f"Note: {note}")

with tab_patient:
    st.subheader(f"Patient Load & High-Interest Case Review ({current_reporting_period_display_str})")
    if not health_df_for_period_clinic_tabs.empty:
        patient_data = prepare_clinic_patient_focus_overview_data(health_df_for_period_clinic_tabs, current_reporting_period_display_str)
        
        df_patient_load = patient_data.get("patient_load_by_key_condition_df")
        if isinstance(df_patient_load, pd.DataFrame) and not df_patient_load.empty:
            st.markdown("###### **Patient Load by Key Condition (Aggregated Weekly):**")
            st.plotly_chart(plot_bar_chart(df_patient_load, 'period_start_date', 'unique_patients_count', "Patient Load by Key Condition", 'condition', 'stack', True, "Week Starting", "Unique Patients Seen"), use_container_width=True)
        
        df_flagged_patients = patient_data.get("flagged_patients_for_review_df")
        if isinstance(df_flagged_patients, pd.DataFrame) and not df_flagged_patients.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top Priority):**")
            st.dataframe(df_flagged_patients.head(15), use_container_width=True, hide_index=True)
        elif isinstance(df_flagged_patients, pd.DataFrame):
            st.info("No patients currently flagged for clinical review in this period based on criteria.")
        
        for note in patient_data.get("processing_notes", []): st.caption(f"Note: {note}")
    else:
        st.info("No health data available in the selected period for patient focus analysis.")

with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring ({current_reporting_period_display_str})")
    if iot_df_for_period_clinic_tabs.empty:
        st.info("No environmental IoT data available for this specific period." if iot_data_source_is_available else "IoT environmental data source is generally unavailable. Detailed monitoring is not possible.")
    else:
        env_details = prepare_clinic_environmental_detail_data(iot_df_for_period_clinic_tabs, iot_data_source_is_available, current_reporting_period_display_str)
        
        alerts = env_details.get("current_environmental_alerts_list", [])
        actionable_alerts = [a for a in alerts if a.get("level") != "ACCEPTABLE"]
        if actionable_alerts:
            st.markdown("###### **Current Environmental Alerts (from Latest Readings in Period):**")
            for alert in actionable_alerts:
                render_traffic_light_indicator(message=alert.get('message', '...'), status_level=alert.get('level', 'UNKNOWN'), details_text=alert.get('alert_type', ''))
        elif alerts:
            st.success(f"✅ All monitored environmental parameters appear within acceptable limits.")

        co2_trend = env_details.get("hourly_avg_co2_trend")
        if isinstance(co2_trend, pd.Series) and not co2_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(co2_trend, "Hourly Avg. CO2 Levels (Clinic-wide)", "CO2 (ppm)", settings.ALERT_AMBIENT_CO2_HIGH_PPM, False, date_format_hover="%H:%M (%d-%b)"), use_container_width=True)
        
        df_latest_readings = env_details.get("latest_room_sensor_readings_df")
        if isinstance(df_latest_readings, pd.DataFrame) and not df_latest_readings.empty:
            st.markdown("###### **Latest Sensor Readings by Room (End of Period):**")
            st.dataframe(df_latest_readings, use_container_width=True, hide_index=True)
        
        for note in env_details.get("processing_notes", []): st.caption(f"Note: {note}")

logger.info(f"Clinic Operations & Management Console page loaded for period: {current_reporting_period_display_str}")
