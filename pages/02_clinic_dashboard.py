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
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart

    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash_import:
    import sys
    st.error(f"Clinic Dashboard Import Error: {e_clinic_dash_import}. Please ensure all dependencies are installed and the application is run from the project root.")
    st.stop()

logger = logging.getLogger(__name__)

# --- Configuration & Page Setup ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely retrieves a configuration setting or returns a default value."""
    return getattr(settings, attr_name, default_value)

def setup_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(
        page_title=f"Clinic Console - {_get_setting('APP_NAME', 'Sentinel')}",
        page_icon="🏥",
        layout=_get_setting('APP_LAYOUT', "wide")
    )

setup_page_config()

# --- Data Loading & Caching ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Loading and enriching health records...")
def load_and_prepare_health_data() -> pd.DataFrame:
    """Loads raw health records, applies AI models, and prepares dates. This is cached for performance."""
    log_ctx = "ClinicDash/LoadHealth"
    raw_df = load_health_records(source_context=log_ctx)
    if raw_df.empty: return pd.DataFrame()
    enriched_df, _ = apply_ai_models(raw_df, source_context=log_ctx)
    if not isinstance(enriched_df, pd.DataFrame) or 'encounter_date' not in enriched_df.columns:
        logger.warning(f"({log_ctx}) AI model application failed or returned invalid data. Using raw data.")
        enriched_df = raw_df
    
    if 'encounter_date' in enriched_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(enriched_df['encounter_date']):
            enriched_df['encounter_date'] = pd.to_datetime(enriched_df['encounter_date'], errors='coerce')
        return enriched_df.dropna(subset=['encounter_date'])
    return pd.DataFrame()

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Loading IoT environmental data...")
def load_and_prepare_iot_data() -> pd.DataFrame:
    """Loads and prepares IoT environmental data, cached for performance."""
    log_ctx = "ClinicDash/LoadIoT"
    raw_df = load_iot_clinic_environment_data(source_context=log_ctx)
    if raw_df.empty or 'timestamp' not in raw_df.columns: return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], errors='coerce')
    return raw_df.dropna(subset=['timestamp'])

# --- UI Components & Filters ---
def manage_date_range_filter(data_min_date: date, data_max_date: date) -> Tuple[date, date]:
    """Manages the Streamlit date range filter widget with data-aware defaults."""
    default_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    default_start = max(data_min_date, data_max_date - timedelta(days=default_days - 1))
    
    if 'clinic_date_range' not in st.session_state:
        st.session_state.clinic_date_range = (default_start, data_max_date)

    selected_range = st.sidebar.date_input(
        "Select Date Range for Clinic Review:",
        value=st.session_state.clinic_date_range,
        min_value=data_min_date, max_value=data_max_date
    )
    start_date, end_date = (selected_range[0], selected_range[1]) if len(selected_range) == 2 else st.session_state.clinic_date_range
    
    if start_date > end_date:
        end_date = start_date
        st.sidebar.warning("Start date cannot be after end date.")

    st.session_state.clinic_date_range = (start_date, end_date)
    return start_date, end_date

# --- Main Application ---
st.title("🏥 Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# 1. Load data and initialize state
try:
    full_health_df = load_and_prepare_health_data()
    full_iot_df = load_and_prepare_iot_data()
except Exception as e:
    logger.error(f"FATAL: Could not load initial data. {e}", exc_info=True)
    st.error(f"A critical error occurred while loading base data: {e}. The dashboard cannot proceed.")
    st.stop()

# 2. Sidebar and Data-Aware Filters
st.sidebar.image(str(Path(_get_setting('APP_LOGO_SMALL_PATH', ''))), width=230)
st.sidebar.header("Console Filters")

if not full_health_df.empty:
    min_date, max_date = full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()
    start_date, end_date = manage_date_range_filter(min_date, max_date)
else:
    st.sidebar.warning("Health data is empty. Cannot set date filters.")
    start_date, end_date = date.today() - timedelta(days=29), date.today()

current_period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"Displaying data for: **{current_period_str}**")

# 3. Filter data for the selected period
health_df_period = full_health_df[(full_health_df['encounter_date'].dt.date >= start_date) & (full_health_df['encounter_date'].dt.date <= end_date)]
iot_df_period = full_iot_df[(full_iot_df['timestamp'].dt.date >= start_date) & (full_iot_df['timestamp'].dt.date <= end_date)]

# 4. KPI Snapshot Section
st.header("🚀 Clinic Performance & Environment Snapshot")
try:
    if not health_df_period.empty:
        kpis = get_clinic_summary_kpis(health_df_period)
        main_kpis = structure_main_clinic_kpis(kpis, current_period_str)
        disease_kpis = structure_disease_specific_clinic_kpis(kpis, current_period_str)
        
        if main_kpis:
            st.markdown("##### **Overall Service Performance:**")
            cols = st.columns(min(len(main_kpis), 4))
            for i, kpi in enumerate(main_kpis):
                with cols[i]: render_kpi_card(**kpi)
        
        if disease_kpis:
            st.markdown("##### **Key Disease & Supply Indicators:**")
            cols = st.columns(min(len(disease_kpis), 4))
            for i, kpi in enumerate(disease_kpis):
                with cols[i]: render_kpi_card(**kpi)
    else:
        st.info("ℹ️ No health data in the selected period for service KPIs.")

    st.markdown("##### **Clinic Environment Quick Check:**")
    if not full_iot_df.empty and not iot_df_period.empty:
        env_kpis = get_clinic_environmental_summary_kpis(iot_df_period)
        cols = st.columns(4)
        co2, pm25, occup, noise = env_kpis.get('avg_co2_overall_ppm'), env_kpis.get('avg_pm25_overall_ugm3'), env_kpis.get('avg_waiting_room_occupancy_overall_persons'), env_kpis.get('rooms_noise_high_alert_latest_count', 0)
        with cols[0]: render_kpi_card(title="Avg. CO2", value_str=f"{co2:.0f}" if pd.notna(co2) else "N/A", units="ppm", icon="💨", status_level="MODERATE_CONCERN" if pd.notna(co2) and co2 > 1000 else "ACCEPTABLE")
        with cols[1]: render_kpi_card(title="Avg. PM2.5", value_str=f"{pm25:.1f}" if pd.notna(pm25) else "N/A", units="µg/m³", icon="🌫️", status_level="MODERATE_CONCERN" if pd.notna(pm25) and pm25 > 12 else "ACCEPTABLE")
        with cols[2]: render_kpi_card(title="Avg. Waiting Occupancy", value_str=f"{occup:.1f}" if pd.notna(occup) else "N/A", units="persons", icon="👨‍👩‍👧‍👦", status_level="MODERATE_CONCERN" if pd.notna(occup) and occup > 10 else "ACCEPTABLE")
        with cols[3]: render_kpi_card(title="High Noise Alerts", value_str=str(noise), units="areas", icon="🔊", status_level="HIGH_CONCERN" if noise > 0 else "ACCEPTABLE")
    elif not full_iot_df.empty:
        st.info("ℹ️ No environmental data in the selected period for environment snapshot.")
    else:
        st.info("🔌 IoT environmental data source is unavailable.")
except Exception as e:
    logger.error(f"Error rendering KPI snapshot section: {e}", exc_info=True)
    st.error("⚠️ An error occurred while rendering the KPI snapshot.")
st.divider()

# 5. Deep Dive Tabs
st.header("🛠️ Operational Areas Deep Dive")
tabs = st.tabs(["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"])

with tabs[0]:
    if health_df_period.empty: st.info("ℹ️ No health data in this period for epidemiological analysis.")
    else:
        try:
            epi_data = calculate_clinic_epidemiological_data(health_df_period, current_period_str)
            st.plotly_chart(plot_bar_chart(epi_data.get("symptom_trends_weekly_top_n_df"), 'week_start_date', 'count', "Weekly Symptom Frequency", 'symptom', 'group'), use_container_width=True)
        except Exception as e: st.error(f"⚠️ Could not generate epi insights: {e}")

with tabs[1]:
    if health_df_period.empty: st.info("ℹ️ No health data in this period for testing insights.")
    else:
        try:
            kpis = get_clinic_summary_kpis(health_df_period)
            insights = prepare_clinic_lab_testing_insights_data(health_df_period, kpis, current_period_str)
            st.dataframe(insights.get("all_critical_tests_summary_table_df"), use_container_width=True, hide_index=True)
            st.markdown("###### **Overdue Pending Tests:**")
            st.dataframe(insights.get("overdue_pending_tests_list_df"), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate testing insights: {e}")

with tabs[2]:
    if full_health_df.empty: st.info("ℹ️ No historical health data available for supply forecasting.")
    else:
        try:
            use_ai = st.checkbox("Use Advanced AI Forecast", key="supply_ai_toggle")
            forecast = prepare_clinic_supply_forecast_overview_data(full_health_df, current_period_str, use_ai)
            st.markdown(f"**Forecast Model Used:** `{forecast.get('forecast_model_type_used', 'N/A')}`")
            st.dataframe(pd.DataFrame(forecast.get("forecast_items_overview_list", [])), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate supply chain insights: {e}")

with tabs[3]:
    if health_df_period.empty: st.info("ℹ️ No health data in this period for patient focus analysis.")
    else:
        try:
            focus_data = prepare_clinic_patient_focus_overview_data(health_df_period, current_period_str)
            st.plotly_chart(plot_bar_chart(focus_data.get("patient_load_by_key_condition_df"), 'period_start_date', 'unique_patients_count', "Patient Load by Condition", 'condition', 'stack'), use_container_width=True)
            st.markdown("###### **Flagged Patients for Clinical Review:**")
            st.dataframe(focus_data.get("flagged_patients_for_review_df"), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate patient focus insights: {e}")

with tabs[4]:
    if not iot_available_flag: st.warning("🔌 IoT data source is unavailable.")
    elif iot_df_period.empty: st.info("ℹ️ No environmental data was recorded in this period.")
    else:
        try:
            env_data = prepare_clinic_environmental_detail_data(iot_df_period, current_period_str)
            st.markdown("###### **Current Environmental Alerts (Latest Readings):**")
            alerts = env_data.get("current_environmental_alerts_list", [])
            if not any(a['level'] != 'ACCEPTABLE' for a in alerts):
                st.success("✅ All monitored environmental parameters appear within acceptable limits.")
            else:
                for alert in alerts:
                    if alert.get("level") != "ACCEPTABLE": render_traffic_light_indicator(**alert)
            st.plotly_chart(plot_annotated_line_chart(env_data.get("hourly_avg_co2_trend"), "Hourly Avg. CO2 Levels", "CO2 (ppm)", target_ref_line_val=_get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1000)), use_container_width=True)
            st.markdown("###### **Latest Sensor Readings by Room:**")
            st.dataframe(env_data.get("latest_room_sensor_readings_df"), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate environmental details: {e}")

st.divider()
st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
