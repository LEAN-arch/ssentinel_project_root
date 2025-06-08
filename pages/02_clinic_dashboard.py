# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import logging
from datetime import date, timedelta
from typing import Dict, Any, Tuple, List
import os
import sys

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart

    # Import all data preparation components
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e:
    st.error(f"Fatal Error: A required module could not be imported.\nDetails: {e}\nThis may be due to an incorrect project structure or missing dependencies. Please ensure you run the app from the 'sentinel_project_root' directory.")
    logger.critical(f"Clinic Dashboard - Unrecoverable Import Error: {e}", exc_info=True)
    st.stop()


# --- Page Configuration and Title ---
# st.set_page_config should be called only once, in the main app.py
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()


# --- Data Loading and Caching ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and processing clinic operational data...")
def get_clinic_console_processed_data(start_date: date, end_date: date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], bool]:
    """Loads, enriches, and filters all necessary data for the Clinic Console. This function is cached for performance."""
    log_ctx = "ClinicConsoleDataLoad"; logger.info(f"({log_ctx}) Executing data load for period: {start_date} to {end_date}")
    
    # 1. Load base data - loaders now handle cleaning and timezone normalization
    raw_health_df = load_health_records(source_context=log_ctx)
    raw_iot_df = load_iot_clinic_environment_data(source_context=log_ctx)
    iot_available = isinstance(raw_iot_df, pd.DataFrame) and not raw_iot_df.empty

    # 2. Enrich full historical data with AI models
    ai_enriched_health_df, _ = apply_ai_models(raw_health_df, source_context=f"{log_ctx}/AIEnrich")
    
    # 3. Filter data for the selected period
    health_df_period = pd.DataFrame()
    if not ai_enriched_health_df.empty:
        # The 'encounter_date' column is guaranteed to be datetime and tz-naive from the loader
        mask = ai_enriched_health_df['encounter_date'].dt.date.between(start_date, end_date)
        health_df_period = ai_enriched_health_df.loc[mask].copy()

    iot_df_period = pd.DataFrame()
    if iot_available:
        # The 'timestamp' column is guaranteed to be datetime and tz-naive
        mask = raw_iot_df['timestamp'].dt.date.between(start_date, end_date)
        iot_df_period = raw_iot_df.loc[mask].copy()

    # 4. Calculate summary KPIs on the period-filtered data
    kpis_for_period = get_clinic_summary_kpis(health_df_period) if not health_df_period.empty else {"test_summary_details": {}}
    
    return ai_enriched_health_df, health_df_period, iot_df_period, kpis_for_period, iot_available


# --- UI Rendering Helper Functions ---
def render_kpi_row(title: str, kpi_list: List[Dict[str, Any]]):
    """Renders a titled row of KPI cards."""
    if not kpi_list: return
    st.markdown(f"##### **{title}**"); cols = st.columns(min(len(kpi_list), 4))
    for i, kpi in enumerate(kpi_list):
        with cols[i % 4]: render_kpi_card(**kpi)

def display_processing_notes(notes: List[str]):
    """Displays a list of processing notes in an expander if any exist."""
    if notes:
        with st.expander("Show Processing Notes"):
            for note in notes: st.caption(f"ℹ️ {note}")

def _get_structured_env_kpis(env_kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Takes raw environmental KPIs and structures them for rendering using a declarative config."""
    kpi_defs = [
        {"key": "avg_co2_overall_ppm", "title": "Avg. CO2", "units": "ppm", "icon": "💨", "target": settings.ALERT_AMBIENT_CO2_HIGH_PPM, "logic": "lower_is_better"},
        {"key": "avg_pm25_overall_ugm3", "title": "Avg. PM2.5", "units": "µg/m³", "icon": "🌫️", "target": settings.ALERT_AMBIENT_PM25_HIGH_UGM3, "logic": "lower_is_better"},
        {"key": "avg_waiting_room_occupancy_overall_persons", "title": "Avg. Waiting Occupancy", "units": "persons", "icon": "👨‍👩‍👧‍👦", "target": settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX, "logic": "lower_is_better"},
        {"key": "rooms_noise_high_alert_latest_count", "title": "High Noise Alerts", "units": "areas", "icon": "🔊", "target": 0, "logic": "lower_is_better_count"},
    ]
    structured_kpis = []
    for kpi in kpi_defs:
        val = env_kpis.get(kpi['key'])
        status = "NO_DATA"
        if pd.notna(val):
            if kpi['logic'] == "lower_is_better": status = "GOOD_PERFORMANCE" if val <= kpi['target'] else "MODERATE_CONCERN" if val <= kpi['target'] * 1.5 else "HIGH_CONCERN"
            elif kpi['logic'] == "lower_is_better_count": status = "GOOD_PERFORMANCE" if val == kpi['target'] else "ACCEPTABLE" if val <= kpi['target'] + 2 else "HIGH_CONCERN"
        value_str = f"{val:.1f}" if pd.notna(val) and isinstance(val, float) and not float(val).is_integer() else str(int(val)) if pd.notna(val) else "N/A"
        structured_kpis.append({"title": kpi['title'], "value_str": value_str, "units": kpi['units'], "icon": kpi['icon'], "status_level": status, "help_text": f"Target: {'<' if 'better' in kpi['logic'] else '=='} {kpi['target']}{kpi['units']}"})
    return structured_kpis


# --- Sidebar and Main Data Loading ---
st.sidebar.header("Console Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH): st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
abs_min_date, abs_max_date = date.today() - timedelta(days=365), date.today()
default_start = max(abs_min_date, abs_max_date - timedelta(days=settings.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS - 1))
session_key = "clinic_date_range"
if session_key not in st.session_state: st.session_state[session_key] = (default_start, abs_max_date)
start_date, end_date = st.sidebar.date_input("Select Date Range:", value=st.session_state[session_key], min_value=abs_min_date, max_value=abs_max_date)
if isinstance(start_date, date) and isinstance(end_date, date) and start_date > end_date: end_date = start_date
MAX_DAYS = 90
if (end_date - start_date).days >= MAX_DAYS: end_date = min(start_date + timedelta(days=MAX_DAYS - 1), abs_max_date); st.sidebar.warning(f"Range limited to {MAX_DAYS} days.")
st.session_state[session_key] = (start_date, end_date)

try:
    full_health_df, period_health_df, period_iot_df, period_kpis, iot_available = get_clinic_console_processed_data(start_date, end_date)
except Exception as e:
    logger.critical(f"Dashboard data loading failed: {e}", exc_info=True); st.error(f"An error occurred while loading dashboard data."); st.stop()

if not iot_available: st.sidebar.warning("IoT environmental data is unavailable.")
period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"**Displaying Clinic Console for:** `{period_str}`")


# --- KPI Snapshot Section ---
st.header("🚀 Performance & Environment Snapshot")
if period_kpis and period_kpis.get("test_summary_details"):
    render_kpi_row("Overall Service Performance", structure_main_clinic_kpis(kpis_summary=period_kpis))
    render_kpi_row("Key Disease & Supply Indicators", structure_disease_specific_clinic_kpis(kpis_summary=period_kpis))
else:
    st.warning("Core service KPIs could not be generated. Health data may be missing for the selected period.")
if iot_available and not period_iot_df.empty:
    env_summary_kpis = get_clinic_environmental_summary_kpis(period_iot_df)
    render_kpi_row("Clinic Environment Quick Check", _get_structured_env_kpis(env_summary_kpis))
st.divider()


# --- Tabbed Deep Dive Section ---
st.header("🛠️ Operational Areas Deep Dive")
tab_titles = ["📈 Local Epidemiology", "🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_epi, tab_testing, tab_supply, tab_patient, tab_env = st.tabs(tab_titles)

with tab_epi:
    st.subheader(f"Local Epidemiological Intelligence")
    if period_health_df.empty:
        st.info("No health data available in this period for epidemiological analysis.")
    else:
        epi_data = calculate_clinic_epidemiological_data(filtered_health_df=period_health_df, reporting_period_context_str=period_str)
        if not epi_data or epi_data.get("symptom_trends_weekly_top_n_df", pd.DataFrame()).empty:
            st.info("No significant epidemiological trends to display for this period.")
        else:
            st.plotly_chart(plot_bar_chart(df_input=epi_data["symptom_trends_weekly_top_n_df"], x_col_name='week_start_date', y_col_name='count', chart_title="Weekly Symptom Frequency (Top Reported)", color_col_name='symptom', bar_mode_style='group'), use_container_width=True)
            rdt_name = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT")
            pos_trend = epi_data.get("key_test_positivity_trends", {}).get(rdt_name)
            if isinstance(pos_trend, pd.Series) and not pos_trend.empty:
                st.plotly_chart(plot_annotated_line_chart(data_series=pos_trend, chart_title=f"Weekly {rdt_name} Positivity Rate", y_axis_label="Positivity %", target_ref_line_val=settings.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
        display_processing_notes(epi_data.get("processing_notes", []))

with tab_testing:
    st.subheader(f"Testing & Diagnostics Performance")
    testing_data = prepare_clinic_lab_testing_insights_data(kpis_summary=period_kpis, health_df_period=period_health_df)
    summary_df, overdue_df = testing_data.get("all_critical_tests_summary_table_df"), testing_data.get("overdue_pending_tests_list_df")
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        st.markdown("###### **Critical Tests Performance Summary**"); st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if isinstance(overdue_df, pd.DataFrame) and not overdue_df.empty:
        st.markdown("###### **Overdue Pending Tests (Top 15)**"); st.dataframe(overdue_df.head(15), use_container_width=True, hide_index=True)
    elif isinstance(overdue_df, pd.DataFrame):
        st.success("✅ No tests are currently flagged as overdue.")
    display_processing_notes(testing_data.get("processing_notes", []))

with tab_patient:
    st.subheader(f"Patient Load & High-Interest Case Review")
    if period_health_df.empty:
        st.info("No health data available in this period for patient analysis.")
    else:
        patient_data = prepare_clinic_patient_focus_overview_data(filtered_health_df_for_clinic_period=period_health_df, reporting_period_context_str=period_str)
        load_df, flagged_df = patient_data.get("patient_load_by_key_condition_df"), patient_data.get("flagged_patients_for_review_df")
        if isinstance(load_df, pd.DataFrame) and not load_df.empty: st.markdown("###### **Patient Load by Key Condition (Weekly)**"); st.plotly_chart(plot_bar_chart(df_input=load_df, x_col_name='period_start_date', y_col_name='unique_patients_count', chart_title="Patient Load by Key Condition", color_col_name='condition', bar_mode_style='stack'), use_container_width=True)
        if isinstance(flagged_df, pd.DataFrame) and not flagged_df.empty: st.markdown("###### **Flagged Patients for Clinical Review (Top Priority)**"); st.dataframe(flagged_df.head(15), use_container_width=True, hide_index=True)
        elif isinstance(flagged_df, pd.DataFrame): st.info("No patients met the criteria for clinical review in this period.")
        display_processing_notes(patient_data.get("processing_notes", []))

with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring")
    if not iot_available or period_iot_df.empty:
        st.info("No environmental data available to display for this period.")
    else:
        env_details = prepare_clinic_environmental_detail_data(filtered_iot_df=period_iot_df, iot_data_source_is_generally_available=iot_available, reporting_period_context_str=period_str)
        actionable_alerts = [a for a in env_details.get("current_environmental_alerts_list", []) if a.get("level") != "ACCEPTABLE"]
        if actionable_alerts:
            st.markdown("###### **Current Environmental Alerts**"); [render_traffic_light_indicator(message=a.get('message', '...'), status_level=a.get('level', 'UNKNOWN'), details_text=a.get('alert_type', '')) for a in actionable_alerts]
        elif env_details.get("current_environmental_alerts_list"): st.success("✅ All monitored environmental parameters appear within acceptable limits.")
        co2_trend, readings_df = env_details.get("hourly_avg_co2_trend"), env_details.get("latest_room_sensor_readings_df")
        if isinstance(co2_trend, pd.Series) and not co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(data_series=co2_trend, chart_title="Hourly Avg. CO2 Levels (Clinic-wide)", y_axis_label="CO2 (ppm)", target_ref_line_val=settings.ALERT_AMBIENT_CO2_HIGH_PPM, date_format_hover="%H:%M (%d-%b)"), use_container_width=True)
        if isinstance(readings_df, pd.DataFrame) and not readings_df.empty: st.markdown("###### **Latest Sensor Readings by Room**"); st.dataframe(readings_df, use_container_width=True, hide_index=True)
        display_processing_notes(env_details.get("processing_notes", []))

with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status")
    use_ai = st.checkbox("Use Advanced AI Forecast (Simulated)", key="supply_ai_toggle", help="Toggles between a simple linear forecast and a more complex (simulated) AI model.")
    supply_data = prepare_clinic_supply_forecast_overview_data(historical_health_df=full_health_df, use_ai_supply_forecasting_model=use_ai)
    st.markdown(f"**Forecast Model Used:** `{supply_data.get('forecast_model_type_used', 'N/A')}`")
    overview = supply_data.get("forecast_items_overview_list", [])
    if overview:
        st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True, column_config={"estimated_stockout_date": st.column_config.TextColumn("Est. Stockout Date")})
    else:
        st.info("No supply forecast data could be generated for the selected model.")
    display_processing_notes(supply_data.get("processing_notes", []))

logger.info(f"Clinic dashboard page render complete for period: {period_str}")
