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
    st.error(f"Clinic Dashboard Import Error: {e_clinic_dash_import}. Please ensure all dependencies are installed and the application is run from the project root.")
    st.stop()

logger = logging.getLogger(__name__)

# --- Configuration & Page Setup ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

def setup_page_config():
    st.set_page_config(
        page_title=f"Clinic Console - {_get_setting('APP_NAME', 'Sentinel')}",
        page_icon="🏥",
        layout=_get_setting('APP_LAYOUT', "wide")
    )

setup_page_config()

# --- Data Loading & Caching (Optimized) ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Loading and enriching health records...")
def load_and_prepare_health_data() -> pd.DataFrame:
    log_ctx = "ClinicDash/LoadHealth"
    raw_df = load_health_records(source_context=log_ctx)
    if raw_df.empty: return pd.DataFrame()
    enriched_df, _ = apply_ai_models(raw_df, source_context=log_ctx)
    if not isinstance(enriched_df, pd.DataFrame) or 'encounter_date' not in enriched_df.columns:
        return raw_df
    if not pd.api.types.is_datetime64_any_dtype(enriched_df['encounter_date']):
        enriched_df['encounter_date'] = pd.to_datetime(enriched_df['encounter_date'], errors='coerce')
    return enriched_df.dropna(subset=['encounter_date'])

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Loading IoT environmental data...")
def load_and_prepare_iot_data() -> pd.DataFrame:
    log_ctx = "ClinicDash/LoadIoT"
    raw_df = load_iot_clinic_environment_data(source_context=log_ctx)
    if raw_df.empty or 'timestamp' not in raw_df.columns: return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], errors='coerce')
    return raw_df.dropna(subset=['timestamp'])

# --- Main Application ---
st.title("🏥 Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# 1. Load all data first to determine available date range
try:
    full_health_df = load_and_prepare_health_data()
    full_iot_df = load_and_prepare_iot_data()
except Exception as e:
    logger.error(f"FATAL: Could not load initial data. {e}", exc_info=True)
    st.error(f"A critical error occurred while loading base data: {e}. The dashboard cannot proceed.")
    st.stop()

# 2. Sidebar and Data-Aware Filters
st.sidebar.markdown("---")
st.sidebar.image(str(Path(_get_setting('APP_LOGO_SMALL_PATH', ''))), width=230)
st.sidebar.markdown("---") 
st.sidebar.header("Console Filters")

# CORRECTED: Make date filters data-aware by default.
if not full_health_df.empty:
    data_min_date = full_health_df['encounter_date'].min().date()
    data_max_date = full_health_df['encounter_date'].max().date()
    
    default_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    default_start = max(data_min_date, data_max_date - timedelta(days=default_days - 1))
    
    selected_range = st.sidebar.date_input(
        "Select Date Range for Clinic Review:",
        value=(default_start, data_max_date),
        min_value=data_min_date,
        max_value=data_max_date
    )
    start_date_filter, end_date_filter = (selected_range[0], selected_range[1]) if len(selected_range) == 2 else (default_start, data_max_date)
else:
    st.sidebar.warning("Health data is empty. Cannot set date filters.")
    start_date_filter, end_date_filter = date.today(), date.today()

current_period_str = f"{start_date_filter.strftime('%d %b %Y')} to {end_date_filter.strftime('%d %b %Y')}"
st.info(f"Displaying Clinic Console data for period: **{current_period_str}**")

# 3. Filter data for the selected period
health_df_period = full_health_df[(full_health_df['encounter_date'].dt.date >= start_date_filter) & (full_health_df['encounter_date'].dt.date <= end_date_filter)].copy()
iot_df_period = full_iot_df[(full_iot_df['timestamp'].dt.date >= start_date_filter) & (full_iot_df['timestamp'].dt.date <= end_date_filter)].copy()
iot_available_flag = not full_iot_df.empty

# 4. KPI Snapshot Section
st.header("🚀 Clinic Performance & Environment Snapshot")
try:
    if not health_df_period.empty:
        kpis = get_clinic_summary_kpis(health_df_period, "ClinicDash/KPIs")
        main_kpis = structure_main_clinic_kpis(kpis, current_period_str)
        disease_kpis = structure_disease_specific_clinic_kpis(kpis, current_period_str)
        
        if main_kpis:
            st.markdown("##### **Overall Service Performance:**")
            cols = st.columns(len(main_kpis))
            for i, kpi_data in enumerate(main_kpis):
                with cols[i]: render_kpi_card(**kpi_data)
        if disease_kpis:
            st.markdown("##### **Key Disease & Supply Indicators:**")
            cols = st.columns(len(disease_kpis))
            for i, kpi_data in enumerate(disease_kpis):
                with cols[i]: render_kpi_card(**kpi_data)
    else:
        st.info("ℹ️ No health data in the selected period for service KPIs.")

    st.markdown("##### **Clinic Environment Quick Check:**")
    if iot_df_period.empty:
        st.info("ℹ️ No environmental data in the selected period for environment snapshot.")
    else:
        env_kpis = get_clinic_environmental_summary_kpis(iot_df_period, "ClinicDash/EnvKPIs")
        if any(pd.notna(env_kpis.get(k)) for k in ['avg_co2_overall_ppm', 'avg_pm25_overall_ugm3']):
            cols = st.columns(4)
            # ... render env kpis ...
        else:
            st.info("ℹ️ Environmental data available, but summary metrics could not be calculated.")
except Exception as e:
    logger.error(f"Error rendering KPI snapshot section: {e}", exc_info=True)
    st.error("⚠️ An error occurred while rendering the KPI snapshot.")
st.divider()

# 5. Deep Dive Tabs
st.header("🛠️ Operational Areas Deep Dive")
tabs = st.tabs(["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"])

with tabs[0]: # Epi
    if health_df_period.empty:
        st.info("ℹ️ No health data in this period for epidemiological analysis.")
    else:
        # Render epi components
        pass # Placeholder for brevity

# ... similar robust checks for all other tabs ...

st.divider()
st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
