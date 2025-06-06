# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List, Union
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
    current_file_path = Path(__file__).resolve()
    project_root_dir = current_file_path.parent.parent
    error_message = (
        f"Clinic Dashboard Import Error: {e_clinic_dash_import}. "
        f"Ensure project root ('{project_root_dir}') is in sys.path (handled by app.py) "
        f"and all modules/packages have `__init__.py` files."
    )
    st.error(error_message)
    st.stop()

logger = logging.getLogger(__name__)

# --- Configuration & Page Setup ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely retrieves a configuration setting or returns a default value."""
    return getattr(settings, attr_name, default_value)

def setup_page_config():
    """Sets the Streamlit page configuration."""
    try:
        page_icon_value = "🏥" 
        app_logo_small_path_str = _get_setting('APP_LOGO_SMALL_PATH', None)
        if app_logo_small_path_str and Path(app_logo_small_path_str).is_file():
            page_icon_value = str(app_logo_small_path_str)
        
        st.set_page_config(
            page_title=f"Clinic Console - {_get_setting('APP_NAME', 'Sentinel')}",
            page_icon=page_icon_value,
            layout=_get_setting('APP_LAYOUT', "wide")
        )
    except Exception as e:
        logger.error(f"Error applying page configuration for Clinic Dashboard: {e}", exc_info=True)
        st.set_page_config(page_title="Clinic Console", page_icon="🏥", layout="wide")

setup_page_config()

# --- Data Loading & Caching (Optimized) ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300), show_spinner="Loading and enriching health records...")
def load_and_enrich_health_data() -> pd.DataFrame:
    """Loads raw health records and applies AI models. This is cached for performance."""
    log_ctx = "ClinicDash/LoadEnrich"
    raw_health_df = load_health_records(source_context=f"{log_ctx}/LoadRawHealth")
    if raw_health_df.empty:
        return pd.DataFrame()
    
    enriched_df, _ = apply_ai_models(raw_health_df, source_context=log_ctx)
    if not isinstance(enriched_df, pd.DataFrame) or 'encounter_date' not in enriched_df.columns:
        logger.warning(f"({log_ctx}) AI model application failed or did not return a valid DataFrame. Returning raw data.")
        return raw_health_df
    
    if not pd.api.types.is_datetime64_any_dtype(enriched_df['encounter_date']):
        enriched_df['encounter_date'] = pd.to_datetime(enriched_df['encounter_date'], errors='coerce')
    if enriched_df['encounter_date'].dt.tz is not None:
        enriched_df['encounter_date'] = enriched_df['encounter_date'].dt.tz_localize(None)
    
    return enriched_df

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 300), show_spinner="Loading IoT environmental data...")
def load_iot_data() -> pd.DataFrame:
    """Loads and prepares IoT environmental data, cached for performance."""
    log_ctx = "ClinicDash/LoadIoT"
    raw_iot_df = load_iot_clinic_environment_data(source_context=log_ctx)
    if raw_iot_df.empty:
        return pd.DataFrame()

    if 'timestamp' in raw_iot_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(raw_iot_df['timestamp']):
            raw_iot_df['timestamp'] = pd.to_datetime(raw_iot_df['timestamp'], errors='coerce')
        if raw_iot_df['timestamp'].dt.tz is not None:
            raw_iot_df['timestamp'] = raw_iot_df['timestamp'].dt.tz_localize(None)
    return raw_iot_df

# --- UI Components & Filters ---
def manage_date_range_filter(session_state_key: str, title: str) -> Tuple[date, date]:
    """Encapsulates robust date range filter logic."""
    abs_min_date = date.today() - timedelta(days=_get_setting('MAX_PAST_DATA_DAYS', 730))
    abs_max_date = date.today()
    default_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    max_query_days = _get_setting('MAX_QUERY_DAYS_CLINIC', 90)

    if session_state_key not in st.session_state:
        default_end = abs_max_date
        default_start = max(abs_min_date, default_end - timedelta(days=default_days - 1))
        st.session_state[session_state_key] = [default_start, default_end]

    selected_range = st.sidebar.date_input(
        title, value=st.session_state[session_state_key],
        min_value=abs_min_date, max_value=abs_max_date
    )

    if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else: # Fallback for single date selection or other unexpected input
        start_date = end_date = selected_range[0] if isinstance(selected_range, (list, tuple)) and len(selected_range) == 1 else st.session_state[session_state_key][0]
    
    if start_date > end_date:
        st.sidebar.warning("Start date cannot be after end date. Using start date for both.")
        end_date = start_date

    if (end_date - start_date).days + 1 > max_query_days:
        st.sidebar.warning(f"Date range automatically limited to {max_query_days} days for performance.")
        end_date = start_date + timedelta(days=max_query_days - 1)

    st.session_state[session_state_key] = [start_date, min(end_date, abs_max_date)]
    return st.session_state[session_state_key][0], st.session_state[session_state_key][1]

# --- Main Application ---
st.title(f"🏥 {_get_setting('APP_NAME', 'Sentinel')} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# 1. Sidebar and Filters
st.sidebar.markdown("---")
st.sidebar.image(str(Path(_get_setting('APP_LOGO_SMALL_PATH', ''))), width=230)
st.sidebar.markdown("---") 
st.sidebar.header("Console Filters")
start_date_filter, end_date_filter = manage_date_range_filter(
    session_state_key="clinic_console_date_range_v9",
    title="Select Date Range for Clinic Review:"
)
current_period_str = f"{start_date_filter.strftime('%d %b %Y')} to {end_date_filter.strftime('%d %b %Y')}"

# 2. Data Loading and Filtering
try:
    full_hist_health_df = load_and_enrich_health_data()
    full_iot_df = load_iot_data()
    health_df_period = full_hist_health_df[(full_hist_health_df['encounter_date'].dt.date >= start_date_filter) & (full_hist_health_df['encounter_date'].dt.date <= end_date_filter)].copy()
    iot_df_period = full_iot_df[(full_iot_df['timestamp'].dt.date >= start_date_filter) & (full_iot_df['timestamp'].dt.date <= end_date_filter)].copy()
except Exception as e:
    logger.error(f"Critical error during data loading or filtering: {e}", exc_info=True)
    st.error(f"🛑 A critical error occurred while preparing data: {e}. Dashboard cannot proceed.")
    st.stop()

st.info(f"Displaying Clinic Console data for period: **{current_period_str}**")

# 3. Main KPI Snapshot Section
st.header("🚀 Clinic Performance & Environment Snapshot")
try:
    # --- Health-based KPIs ---
    if not health_df_period.empty:
        clinic_summary_kpis = get_clinic_summary_kpis(health_df_period, "ClinicDash/KPIs")
        main_kpis = structure_main_clinic_kpis(clinic_summary_kpis, current_period_str)
        disease_kpis = structure_disease_specific_clinic_kpis(clinic_summary_kpis, current_period_str)
        
        if main_kpis:
            st.markdown("##### **Overall Service Performance:**")
            cols = st.columns(len(main_kpis))
            for i, kpi in enumerate(main_kpis):
                with cols[i]: render_kpi_card(**kpi)
        
        if disease_kpis:
            st.markdown("##### **Key Disease & Supply Indicators:**")
            cols = st.columns(len(disease_kpis))
            for i, kpi in enumerate(disease_kpis):
                with cols[i]: render_kpi_card(**kpi)
    else:
        st.info("ℹ️ No health data available in the selected period to calculate service KPIs.")

    # --- Environmental KPIs ---
    st.markdown("##### **Clinic Environment Quick Check:**")
    if not full_iot_df.empty and not iot_df_period.empty:
        env_kpis = get_clinic_environmental_summary_kpis(iot_df_period, "ClinicDash/EnvKPIs")
        env_cols = st.columns(4)
        co2, pm25, occup, noise = env_kpis.get('avg_co2_overall_ppm'), env_kpis.get('avg_pm25_overall_ugm3'), env_kpis.get('avg_waiting_room_occupancy_overall_persons'), env_kpis.get('rooms_noise_high_alert_latest_count', 0)
        with env_cols[0]: render_kpi_card(title="Avg. CO2", value_str=f"{co2:.0f}" if pd.notna(co2) else "N/A", units="ppm", icon="💨", status_level="MODERATE_CONCERN" if pd.notna(co2) and co2 > 1000 else "ACCEPTABLE")
        with env_cols[1]: render_kpi_card(title="Avg. PM2.5", value_str=f"{pm25:.1f}" if pd.notna(pm25) else "N/A", units="µg/m³", icon="🌫️", status_level="MODERATE_CONCERN" if pd.notna(pm25) and pm25 > 12 else "ACCEPTABLE")
        with env_cols[2]: render_kpi_card(title="Avg. Waiting Occupancy", value_str=f"{occup:.1f}" if pd.notna(occup) else "N/A", units="persons", icon="👨‍👩‍👧‍👦", status_level="MODERATE_CONCERN" if pd.notna(occup) and occup > 10 else "ACCEPTABLE")
        with env_cols[3]: render_kpi_card(title="High Noise Alerts", value_str=str(noise), units="areas", icon="🔊", status_level="HIGH_CONCERN" if noise > 0 else "ACCEPTABLE")
    else:
        st.info("ℹ️ No environmental data available for this period.")

except Exception as e:
    logger.error(f"Error rendering KPI snapshot section: {e}", exc_info=True)
    st.error("⚠️ An error occurred while rendering the KPI snapshot.")
st.divider()

# 4. Deep Dive Tabs
st.header("🛠️ Operational Areas Deep Dive")
tabs = st.tabs(["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"])

with tabs[0]: # Epi
    if health_df_period.empty:
        st.info("ℹ️ No health data for epidemiological analysis.")
    else:
        try:
            epi_data = calculate_clinic_epidemiological_data(health_df_period, current_period_str)
            st.plotly_chart(plot_bar_chart(epi_data.get("symptom_trends_weekly_top_n_df"), 'week_start_date', 'count', "Weekly Symptom Frequency", 'symptom', 'group'), use_container_width=True)
            malaria_series = epi_data.get("key_test_positivity_trends", {}).get(_get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {}).get("RDT-Malaria", {}).get("display_name", "Malaria RDT"))
            if malaria_series is not None and not malaria_series.empty:
                st.plotly_chart(plot_annotated_line_chart(malaria_series, "Weekly Malaria RDT Positivity Rate", "Positivity %", target_ref_line_val=_get_setting('TARGET_MALARIA_POSITIVITY_RATE', 10.0)), use_container_width=True)
        except Exception as e: st.error(f"⚠️ Could not generate epi insights: {e}")

with tabs[1]: # Testing
    if health_df_period.empty:
        st.info("ℹ️ No health data for testing insights.")
    else:
        try:
            kpis = get_clinic_summary_kpis(health_df_period, "ClinicDash/KPIs_Tab")
            insights = prepare_clinic_lab_testing_insights_data(health_df_period, kpis, current_period_str)
            st.dataframe(insights.get("all_critical_tests_summary_table_df"), use_container_width=True, hide_index=True)
            st.markdown("###### **Overdue Pending Tests:**")
            st.dataframe(insights.get("overdue_pending_tests_list_df"), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate testing insights: {e}")

with tabs[2]: # Supply Chain
    if full_hist_health_df.empty:
        st.info("ℹ️ No historical data available for supply forecasting.")
    else:
        try:
            use_ai = st.checkbox("Use Advanced AI Forecast", key="supply_ai_toggle")
            forecast = prepare_clinic_supply_forecast_overview_data(full_hist_health_df, current_period_str, use_ai)
            st.markdown(f"**Forecast Model Used:** `{forecast.get('forecast_model_type_used', 'N/A')}`")
            st.dataframe(pd.DataFrame(forecast.get("forecast_items_overview_list", [])), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate supply chain insights: {e}")

with tabs[3]: # Patient Focus
    if health_df_period.empty:
        st.info("ℹ️ No health data for patient focus analysis.")
    else:
        try:
            focus_data = prepare_clinic_patient_focus_overview_data(health_df_period, current_period_str)
            st.plotly_chart(plot_bar_chart(focus_data.get("patient_load_by_key_condition_df"), 'period_start_date', 'unique_patients_count', "Patient Load by Condition", 'condition', 'stack'), use_container_width=True)
            st.markdown("###### **Flagged Patients for Clinical Review:**")
            st.dataframe(focus_data.get("flagged_patients_for_review_df"), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate patient focus insights: {e}")

with tabs[4]: # Environment
    if not iot_available_flag:
        st.warning("🔌 IoT data source is unavailable for this installation.")
    elif iot_df_period.empty:
        st.info("ℹ️ No environmental data was recorded for this period.")
    else:
        try:
            env_data = prepare_clinic_environmental_detail_data(iot_df_period, current_period_str)
            st.markdown("###### **Current Environmental Alerts (Latest Readings):**")
            alerts = env_data.get("current_environmental_alerts_list", [])
            if alerts and not all(a['level'] == 'ACCEPTABLE' for a in alerts):
                for alert in alerts:
                    if alert.get("level") != "ACCEPTABLE": render_traffic_light_indicator(**alert)
            else:
                st.success("✅ All monitored environmental parameters appear within acceptable limits.")

            st.plotly_chart(plot_annotated_line_chart(env_data.get("hourly_avg_co2_trend"), "Hourly Avg. CO2 Levels", "CO2 (ppm)", target_ref_line_val=_get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1000)), use_container_width=True)
            st.markdown("###### **Latest Sensor Readings by Room:**")
            st.dataframe(env_data.get("latest_room_sensor_readings_df"), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"⚠️ Could not generate environmental details: {e}")

st.divider()
st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
