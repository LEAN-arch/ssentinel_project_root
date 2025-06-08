# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
import re
from datetime import date, timedelta
from typing import Dict, Any, Tuple, List, Optional

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
# These imports assume the project structure is sound and __init__.py files are correct.
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_clinic_environmental_summary_kpis, get_trend_data
    from analytics.orchestrator import apply_ai_models
    from analytics.supply_forecasting import forecast_supply_levels_advanced, generate_simple_supply_forecast
    from analytics.alerting import get_patient_alerts_for_clinic
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart
    
    # This import will now succeed because the circular dependency in the target file is removed.
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data

except ImportError as e:
    st.error(f"Fatal Error: A required module could not be imported.\nDetails: {e}\nThis may be due to an incorrect project structure. Ensure all files and '__init__.py' files are correctly placed.")
    logger.critical(f"Clinic Dashboard - Unrecoverable Import Error: {e}", exc_info=True)
    st.stop()


# --- Page Configuration and Title ---
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()


# --- Data Loading and Caching ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and processing all operational data...")
def get_dashboard_data_and_date_bounds() -> Tuple[pd.DataFrame, pd.DataFrame, bool, date, date]:
    """
    Loads all data and crucially, determines the available date range from the data itself.
    """
    log_ctx = "DashboardDataInit"; logger.info(f"({log_ctx}) Executing initial data load and bounds check.")
    
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    iot_available = isinstance(iot_df, pd.DataFrame) and not iot_df.empty

    min_date_in_data, max_date_in_data = date.today() - timedelta(days=365), date.today()
    if not health_df.empty and 'encounter_date' in health_df.columns:
        valid_dates = pd.to_datetime(health_df['encounter_date'], errors='coerce').dropna()
        if not valid_dates.empty:
            min_date_in_data = valid_dates.min().date()
            max_date_in_data = valid_dates.max().date()
    
    ai_enriched_health_df, _ = apply_ai_models(health_df, source_context=f"{log_ctx}/AIEnrich")
    
    return ai_enriched_health_df, iot_df, iot_available, min_date_in_data, max_date_in_data


# --- UI Rendering Helper Functions ---
def render_kpi_row(title: str, kpi_list: List[Dict[str, Any]]):
    if not kpi_list: return
    st.markdown(f"##### **{title}**"); cols = st.columns(min(len(kpi_list), 4))
    for i, kpi in enumerate(kpi_list):
        with cols[i % 4]: render_kpi_card(**kpi)

def display_processing_notes(notes: List[str]):
    if notes:
        with st.expander("Show Processing Notes"):
            for note in notes: st.caption(f"ℹ️ {note}")

# --- Main Application Logic ---
try:
    full_health_df, full_iot_df, iot_available, abs_min_date, abs_max_date = get_dashboard_data_and_date_bounds()
except Exception as e:
    logger.critical(f"Dashboard initial data loading failed: {e}", exc_info=True)
    st.error("A critical error occurred while loading initial data. The application cannot continue.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Console Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH): st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)

default_date_range_days = getattr(settings, 'WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
default_start = max(abs_min_date, abs_max_date - timedelta(days=default_date_range_days - 1))
default_end = abs_max_date

session_key = "clinic_date_range"
if session_key not in st.session_state: st.session_state[session_key] = (default_start, default_end)

selected_range = st.sidebar.date_input(
    "Select Date Range:", 
    value=st.session_state[session_key], 
    min_value=abs_min_date, 
    max_value=abs_max_date
)

start_date, end_date = (selected_range[0], selected_range[1]) if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2 else (default_start, default_end)
if start_date > end_date: end_date = start_date
MAX_DAYS = 90
if (end_date - start_date).days >= MAX_DAYS: end_date = min(start_date + timedelta(days=MAX_DAYS - 1), abs_max_date); st.sidebar.warning(f"Range limited to {MAX_DAYS} days.")
st.session_state[session_key] = (start_date, end_date)

# --- Process Data for Selected Period ---
period_health_df = full_health_df.loc[full_health_df['encounter_date'].dt.date.between(start_date, end_date)].copy() if not full_health_df.empty else pd.DataFrame()
period_iot_df = full_iot_df.loc[full_iot_df['timestamp'].dt.date.between(start_date, end_date)].copy() if iot_available and not full_iot_df.empty else pd.DataFrame()
period_kpis = get_clinic_summary_kpis(period_health_df) if not period_health_df.empty else {"test_summary_details": {}}

# --- Page Display ---
if not iot_available: st.sidebar.warning("IoT environmental data is unavailable.")
period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"**Displaying Clinic Console for:** `{period_str}`")

# --- KPI Snapshot Section ---
st.header("🚀 Performance & Environment Snapshot")
main_kpis = structure_main_clinic_kpis(kpis_summary=period_kpis)
disease_kpis = structure_disease_specific_clinic_kpis(kpis_summary=period_kpis)
render_kpi_row("Overall Service Performance", main_kpis)
render_kpi_row("Key Disease & Supply Indicators", disease_kpis)

if not main_kpis and not disease_kpis:
    if period_health_df.empty:
        st.info("No service performance data available for the selected period.")
    else:
        st.warning("Could not generate service performance KPIs. The underlying summary data might be incomplete.")

if iot_available and not period_iot_df.empty:
    env_summary_kpis = get_clinic_environmental_summary_kpis(period_iot_df)
    # The original file had a placeholder st.json here. I've restored the full KPI rendering.
    env_kpi_defs = [
        {"key": "avg_co2_overall_ppm", "title": "Avg. CO2", "units": "ppm", "icon": "💨", "target": settings.ALERT_AMBIENT_CO2_HIGH_PPM, "logic": "lower_is_better"},
        {"key": "avg_pm25_overall_ugm3", "title": "Avg. PM2.5", "units": "µg/m³", "icon": "🌫️", "target": settings.ALERT_AMBIENT_PM25_HIGH_UGM3, "logic": "lower_is_better"},
        {"key": "avg_waiting_room_occupancy_overall_persons", "title": "Avg. Waiting Occupancy", "units": "persons", "icon": "👨‍👩‍👧‍👦", "target": settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX, "logic": "lower_is_better"},
        {"key": "rooms_noise_high_alert_latest_count", "title": "High Noise Alerts", "units": "areas", "icon": "🔊", "target": 0, "logic": "lower_is_better_count"},
    ]
    env_kpi_cards = []
    for kpi in env_kpi_defs:
        val = env_summary_kpis.get(kpi['key'])
        status = "NO_DATA"
        if pd.notna(val):
            if kpi['logic'] == "lower_is_better": status = "GOOD_PERFORMANCE" if val <= kpi['target'] else "MODERATE_CONCERN" if val <= kpi['target'] * 1.5 else "HIGH_CONCERN"
            elif kpi['logic'] == "lower_is_better_count": status = "GOOD_PERFORMANCE" if val == kpi['target'] else "ACCEPTABLE" if val <= kpi['target'] + 2 else "HIGH_CONCERN"
        value_str = f"{val:.1f}" if pd.notna(val) and isinstance(val, float) and not float(val).is_integer() else str(int(val)) if pd.notna(val) else "N/A"
        env_kpi_cards.append({"title": kpi['title'], "value_str": value_str, "units": kpi['units'], "icon": kpi['icon'], "status_level": status, "help_text": f"Target: {'<' if 'better' in kpi['logic'] else '=='} {kpi['target']}{kpi['units']}"})
    render_kpi_row("Clinic Environment Quick Check", env_kpi_cards)
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
        st.dataframe(epi_data.get("symptom_trends_weekly_top_n_df"))
        display_processing_notes(epi_data.get("processing_notes", []))

with tab_testing:
    st.subheader(f"Testing & Diagnostics Performance")
    testing_data = prepare_clinic_lab_testing_insights_data(kpis_summary=period_kpis, health_df_period=period_health_df)
    st.dataframe(testing_data.get("all_critical_tests_summary_table_df"), hide_index=True)
    st.dataframe(testing_data.get("overdue_pending_tests_list_df"), hide_index=True)
    display_processing_notes(testing_data.get("processing_notes", []))

with tab_patient:
    st.subheader(f"Patient Load & High-Interest Case Review")
    patient_data = prepare_clinic_patient_focus_overview_data(filtered_health_df_for_clinic_period=period_health_df, reporting_period_context_str=period_str)
    st.dataframe(patient_data.get("flagged_patients_for_review_df"), hide_index=True)
    display_processing_notes(patient_data.get("processing_notes", []))

with tab_env:
    st.subheader(f"Facility Environment Detailed Monitoring")
    env_details = prepare_clinic_environmental_detail_data(filtered_iot_df=period_iot_df, iot_data_source_is_generally_available=iot_available, reporting_period_context_str=period_str)
    st.json(env_details.get("current_environmental_alerts_list"))
    display_processing_notes(env_details.get("processing_notes", []))

with tab_supply:
    st.subheader(f"Medical Supply Forecast & Status")
    use_ai = st.checkbox("Use Advanced AI Forecast", key="supply_ai_toggle")
    supply_data = prepare_clinic_supply_forecast_overview_data(historical_health_df=full_health_df, use_ai_supply_forecasting_model=use_ai)
    st.dataframe(pd.DataFrame(supply_data.get("forecast_items_overview_list", [])))
    display_processing_notes(supply_data.get("processing_notes", []))

logger.info(f"Clinic dashboard page render complete for period: {period_str}")
