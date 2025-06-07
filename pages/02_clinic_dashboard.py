# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Dict, Any, Tuple
import os
import sys

# --- Page Specific Logger ---
# Define logger early to be available for all parts of the script, including import error handling.
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
# This structure assumes the app is run from the 'sentinel_project_root' directory.
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart

    # Corrected relative import path for Streamlit pages
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e:
    st.error(
        f"Fatal Error: A required module could not be imported.\n"
        f"Details: {e}\n"
        f"This may be due to an incorrect project structure or missing dependencies. "
        f"Please ensure you run the app from the 'sentinel_project_root' directory and that all "
        f"requirements are installed."
    )
    logger.critical(f"Clinic Dashboard - Unrecoverable Import Error: {e}", exc_info=True)
    st.stop()


# --- Page Configuration and Title ---
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()


# --- Data Loading and Caching ---
@st.cache_data(
    ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS,
    show_spinner="Loading and processing clinic operational data...",
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True)}
)
def get_clinic_console_processed_data(
    start_date: date, end_date: date
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], bool]:
    """
    Loads, enriches, and filters data for the Clinic Console. This function is cached for performance.

    Returns:
        A tuple containing:
        - Full historical health DataFrame (AI enriched, for forecasting context).
        - Period-filtered health DataFrame (AI enriched, for tab displays).
        - Period-filtered IoT DataFrame.
        - Clinic summary KPIs for the selected period.
        - A boolean flag indicating if the IoT data source is available.
    """
    log_ctx = "ClinicConsoleDataLoad"
    logger.info(f"({log_ctx}) Executing data load for period: {start_date} to {end_date}")

    # 1. Load Raw Data
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadHealth")
    raw_iot_df = load_iot_clinic_environment_data(source_context=f"{log_ctx}/LoadIoT")

    is_iot_data_available = isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty

    # 2. Enrich Health Data with AI Models
    if isinstance(raw_health_df, pd.DataFrame) and not raw_health_df.empty:
        ai_enriched_health_df, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
    else:
        logger.warning(f"({log_ctx}) Raw health data is empty. AI enrichment skipped.")
        ai_enriched_health_df = pd.DataFrame()

    # 3. Filter Data for Selected Period
    health_df_period = pd.DataFrame()
    if not ai_enriched_health_df.empty and 'encounter_date' in ai_enriched_health_df.columns:
        ai_enriched_health_df['encounter_date'] = pd.to_datetime(ai_enriched_health_df['encounter_date'], errors='coerce')
        mask = ai_enriched_health_df['encounter_date'].dt.date.between(start_date, end_date)
        health_df_period = ai_enriched_health_df.loc[mask].copy()

    iot_df_period = pd.DataFrame()
    if is_iot_data_available and 'timestamp' in raw_iot_df.columns:
        raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')
        mask = raw_iot_df['timestamp'].dt.date.between(start_date, end_date)
        iot_df_period = raw_iot_df.loc[mask].copy()

    # 4. Calculate Summary KPIs for the Period
    kpis_for_period = {"test_summary_details": {}}  # Ensure default structure exists
    if not health_df_period.empty:
        try:
            kpis_for_period = get_clinic_summary_kpis(health_df_period, f"{log_ctx}/SummaryKPIs")
        except Exception as e_kpi:
            logger.error(f"({log_ctx}) Failed to calculate summary KPIs: {e_kpi}", exc_info=True)
    else:
        logger.info(f"({log_ctx}) No health data in period; using default KPIs.")

    return ai_enriched_health_df, health_df_period, iot_df_period, kpis_for_period, is_iot_data_available


# --- Sidebar Filters ---
st.sidebar.header("Console Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)

# Date Range Picker Logic
abs_min_date = date.today() - timedelta(days=365)
abs_max_date = date.today()
default_end = abs_max_date
default_start = max(abs_min_date, default_end - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1))

session_key = "clinic_date_range"
if session_key not in st.session_state:
    st.session_state[session_key] = (default_start, default_end)

selected_range = st.sidebar.date_input(
    "Select Date Range:",
    value=st.session_state[session_key],
    min_value=abs_min_date,
    max_value=abs_max_date,
)

# Unpack and validate the selected date range
start_date, end_date = (selected_range[0], selected_range[1]) if len(selected_range) == 2 else (default_start, default_end)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date. Adjusting end date.")
    end_date = start_date

MAX_DAYS = 90
if (end_date - start_date).days >= MAX_DAYS:
    st.sidebar.warning(f"Range limited to {MAX_DAYS} days for performance.")
    end_date = min(start_date + timedelta(days=MAX_DAYS - 1), abs_max_date)

# Update session state with the final, validated range for persistence across reruns
st.session_state[session_key] = (start_date, end_date)


# --- Main Data Loading and Initial Checks ---
try:
    (
        full_health_df,
        period_health_df,
        period_iot_df,
        period_kpis,
        iot_available
    ) = get_clinic_console_processed_data(start_date, end_date)
except Exception as e_load:
    logger.critical(f"Dashboard data loading function failed: {e_load}", exc_info=True)
    st.error(f"An error occurred while loading dashboard data. Please contact support.")
    st.stop()

if not iot_available:
    st.sidebar.warning("IoT environmental data is unavailable. Related metrics will be hidden.")

period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"**Displaying Clinic Console for:** `{period_str}`")


# --- Main Layout: KPIs and Tabs ---
st.header("🚀 Performance & Environment Snapshot")

# Section 1.1: Service & Disease KPIs
if period_kpis and period_kpis.get("test_summary_details"):
    main_kpis = structure_main_clinic_kpis(period_kpis, period_str)
    disease_kpis = structure_disease_specific_clinic_kpis(period_kpis, period_str)

    if main_kpis:
        st.markdown("##### Overall Service Performance")
        cols = st.columns(min(len(main_kpis), 4))
        for i, kpi in enumerate(main_kpis):
            with cols[i % 4]:
                render_kpi_card(**kpi)

    if disease_kpis:
        st.markdown("##### Key Disease & Supply Indicators")
        cols = st.columns(min(len(disease_kpis), 4))
        for i, kpi in enumerate(disease_kpis):
            with cols[i % 4]:
                render_kpi_card(**kpi)
else:
    st.warning("Core service KPIs could not be generated for this period.")

# Section 1.2: Environment Quick-Check KPIs
if iot_available:
    st.markdown("##### Clinic Environment Quick Check")
    env_kpis = get_clinic_environmental_summary_kpis(period_iot_df, "ClinicDash/EnvKPIs")
    if env_kpis and any(pd.notna(v) for k, v in env_kpis.items() if "avg" in k or "count" in k):
        cols = st.columns(4)
        with cols[0]:
            val = env_kpis.get('avg_co2_overall_ppm', np.nan)
            status = "HIGH_RISK" if pd.notna(val) and val > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM else ("MODERATE_CONCERN" if pd.notna(val) and val > settings.ALERT_AMBIENT_CO2_HIGH_PPM else "ACCEPTABLE")
            render_kpi_card("Avg. CO2", f"{val:.0f}" if pd.notna(val) else "N/A", "ppm", "💨", status, f"Target < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm.")
        with cols[1]:
            val = env_kpis.get('avg_pm25_overall_ugm3', np.nan)
            status = "HIGH_RISK" if pd.notna(val) and val > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 else ("MODERATE_CONCERN" if pd.notna(val) and val > settings.ALERT_AMBIENT_PM25_HIGH_UGM3 else "ACCEPTABLE")
            render_kpi_card("Avg. PM2.5", f"{val:.1f}" if pd.notna(val) else "N/A", "µg/m³", "🌫️", status, f"Target < {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³.")
        with cols[2]:
            val = env_kpis.get('avg_waiting_room_occupancy_overall_persons', np.nan)
            status = "MODERATE_CONCERN" if pd.notna(val) and val > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX else "ACCEPTABLE"
            render_kpi_card("Avg. Waiting Occupancy", f"{val:.1f}" if pd.notna(val) else "N/A", "persons", "👨‍👩‍👧‍👦", status, f"Target < {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons.")
        with cols[3]:
            val = env_kpis.get('rooms_noise_high_alert_latest_count', 0)
            status = "HIGH_RISK" if val > 1 else ("MODERATE_CONCERN" if val == 1 else "ACCEPTABLE")
            render_kpi_card("High Noise Alerts", str(val), "areas", "🔊", status, f"Areas with noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA.")
    else:
        st.info("No significant environmental data recorded for this period.")
st.divider()

# --- Section 2: Tabbed Interface for Deep Dives ---
st.header("🛠️ Operational Areas Deep Dive")
tabs = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi, tab_testing, tab_supply, tab_patient, tab_env = st.tabs(tabs)

# Tab 1: Epidemiology
with tab_epi:
    st.subheader(f"Local Epidemiological Intelligence")
    if period_health_df.empty:
        st.info("No health data available in this period for epidemiological analysis.")
    else:
        epi_data = calculate_clinic_epidemiological_data(period_health_df, period_str)
        df_symptoms = epi_data.get("symptom_trends_weekly_top_n_df")
        if isinstance(df_symptoms, pd.DataFrame) and not df_symptoms.empty:
            st.plotly_chart(plot_bar_chart(df_input=df_symptoms, x_col_name='week_start_date', y_col_name='count', chart_title="Weekly Symptom Frequency (Top Reported)", color_col_name='symptom', bar_mode_style='group'), use_container_width=True)
        
        rdt_name = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
        positivity_trend = epi_data.get("key_test_positivity_trends", {}).get(rdt_name)
        if isinstance(positivity_trend, pd.Series) and not positivity_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(data_series=positivity_trend, chart_title=f"Weekly {rdt_name} Positivity Rate", y_axis_label="Positivity %", target_ref_line_val=settings.TARGET_MALARIA_POSITIVITY_RATE, y_values_are_counts=False), use_container_width=True)
        
        for note in epi_data.get("calculation_notes", []): st.caption(f"ℹ️ {note}")

# Tab 2: Testing
with tab_testing:
    st.subheader(f"Testing & Diagnostics Performance")
    if period_health_df.empty:
        st.info("No health data available in this period for testing analysis.")
    else:
        insights = prepare_clinic_lab_testing_insights_data(period_health_df, period_kpis, period_str)
        
        df_summary = insights.get("all_critical_tests_summary_table_df")
        if isinstance(df_summary, pd.DataFrame) and not df_summary.empty:
            st.markdown("###### **Critical Tests Performance Summary**")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        df_overdue = insights.get("overdue_pending_tests_list_df")
        if isinstance(df_overdue, pd.DataFrame) and not df_overdue.empty:
            st.markdown("###### **Overdue Pending Tests (Top 15)**")
            st.dataframe(df_overdue.head(15), use_container_width=True, hide_index=True)
        elif isinstance(df_overdue, pd.DataFrame):
            st.success("✅ No tests are currently flagged as overdue.")
            
        for note in insights.get("processing_notes", []): st.caption(f"ℹ️ {note}")

# Tab 3: Supply Chain
with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status")
    use_ai = st.checkbox("Use Advanced AI Forecast (Simulated)", key="supply_ai_toggle")
    supply_data = prepare_clinic_supply_forecast_overview_data(full_health_df, period_str, use_ai)
    
    st.markdown(f"**Forecast Model Used:** `{supply_data.get('forecast_model_type_used', 'N/A')}`")
    overview = supply_data.get("forecast_items_overview_list", [])
    if overview:
        st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True, column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
    else:
        st.info("No supply forecast data could be generated.")
    for note in supply_data.get("data_processing_notes", []): st.caption(f"ℹ️ {note}")

# Tab 4: Patient Focus
with tab_patient:
    st.subheader(f"Patient Load & High-Interest Case Review")
    if period_health_df.empty:
        st.info("No health data available in this period for patient analysis.")
    else:
        patient_data = prepare_clinic_patient_focus_overview_data(period_health_df, period_str)
        
        df_load = patient_data.get("patient_load_by_key_condition_df")
        if isinstance(df_load, pd.DataFrame) and not df_load.empty:
            st.markdown("###### **Patient Load by Key Condition (Weekly)**")
            st.plotly_chart(plot_bar_chart(df_input=df_load, x_col_name='period_start_date', y_col_name='unique_patients_count', chart_title="Patient Load by Key Condition", color_col_name='condition', bar_mode_style='stack'), use_container_width=True)
            
        df_flagged = patient_data.get("flagged_patients_for_review_df")
        if isinstance(df_flagged, pd.DataFrame) and not df_flagged.empty:
            st.markdown("###### **Flagged Patients for Clinical Review (Top Priority)**")
            st.dataframe(df_flagged.head(15), use_container_width=True, hide_index=True)
        elif isinstance(df_flagged, pd.DataFrame):
            st.info("No patients met the criteria for clinical review in this period.")
        
        for note in patient_data.get("processing_notes", []): st.caption(f"ℹ️ {note}")

# Tab 5: Environment Details
with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring")
    if not iot_available or period_iot_df.empty:
        st.info("No environmental data available to display for this period.")
    else:
        env_details = prepare_clinic_environmental_detail_data(period_iot_df, iot_available, period_str)
        
        alerts = [a for a in env_details.get("current_environmental_alerts_list", []) if a.get("level") != "ACCEPTABLE"]
        if alerts:
            st.markdown("###### **Current Environmental Alerts**")
            for alert in alerts:
                render_traffic_light_indicator(message=alert.get('message', '...'), status_level=alert.get('level', 'UNKNOWN'), details_text=alert.get('alert_type', ''))
        else:
            st.success("✅ All monitored environmental parameters appear within acceptable limits.")

        co2_trend = env_details.get("hourly_avg_co2_trend")
        if isinstance(co2_trend, pd.Series) and not co2_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(data_series=co2_trend, chart_title="Hourly Avg. CO2 Levels (Clinic-wide)", y_axis_label="CO2 (ppm)", target_ref_line_val=settings.ALERT_AMBIENT_CO2_HIGH_PPM, date_format_hover="%H:%M (%d-%b)"), use_container_width=True)
        
        df_readings = env_details.get("latest_room_sensor_readings_df")
        if isinstance(df_readings, pd.DataFrame) and not df_readings.empty:
            st.markdown("###### **Latest Sensor Readings by Room**")
            st.dataframe(df_readings, use_container_width=True, hide_index=True)
            
        for note in env_details.get("processing_notes", []): st.caption(f"ℹ️ {note}")


logger.info(f"Clinic dashboard page render complete for period: {period_str}")
