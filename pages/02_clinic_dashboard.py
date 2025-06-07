# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import logging
from datetime import date, timedelta
from typing import Dict, Any, Tuple
from pathlib import Path

# --- Sentinel System Imports (Absolute Imports from Project Root) ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis
    from data_processing.helpers import hash_dataframe_safe
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator, get_theme_color
    from visualization.plots import plot_bar_chart, plot_donut_chart

    from pages.clinic_components.env_details import prepare_clinic_environmental_detail_data
    from pages.clinic_components.kpi_structuring import structure_main_clinic_kpis, structure_disease_specific_clinic_kpis
    from pages.clinic_components.epi_data import calculate_clinic_epidemiological_data
    from pages.clinic_components.patient_focus import prepare_clinic_patient_focus_overview_data
    from pages.clinic_components.supply_forecast import prepare_clinic_supply_forecast_overview_data
    from pages.clinic_components.testing_insights import prepare_clinic_lab_testing_insights_data
except ImportError as e_clinic_dash_import:
    st.error(f"Clinic Dashboard Import Error: {e_clinic_dash_import}. Please ensure all dependencies are installed and the application is run from the project root.")
    st.stop()

logger = logging.getLogger(__name__)

# --- Constants & Configuration ---
APP_NAME = "Sentinel"
COL_ENCOUNTER_DATE = 'encounter_date'
COL_TIMESTAMP = 'timestamp'

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

# --- Data Loading & Caching ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_DATA_LOAD', 3600), show_spinner="Loading health records...")
def get_base_health_data() -> pd.DataFrame:
    return load_health_records()

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_AI_MODEL', 3600), show_spinner="Enriching records with AI models...")
def get_enriched_health_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    _ = hash_dataframe_safe(df)
    enriched_df, _ = apply_ai_models(df)
    if not isinstance(enriched_df, pd.DataFrame) or COL_ENCOUNTER_DATE not in enriched_df.columns: return pd.DataFrame()
    enriched_df[COL_ENCOUNTER_DATE] = pd.to_datetime(enriched_df[COL_ENCOUNTER_DATE], errors='coerce')
    return enriched_df.dropna(subset=[COL_ENCOUNTER_DATE])

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_DATA_LOAD', 3600), show_spinner="Loading IoT environmental data...")
def get_iot_data() -> pd.DataFrame:
    df = load_iot_clinic_environment_data()
    if df.empty or COL_TIMESTAMP not in df.columns: return pd.DataFrame()
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], errors='coerce')
    return df.dropna(subset=[COL_TIMESTAMP])

# --- Centralized Data Preparation ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Analyzing clinic data...")
def prepare_dashboard_data(_health_df_period: pd.DataFrame, _iot_df_period: pd.DataFrame, _full_health_df: pd.DataFrame, current_period_str: str, use_ai_supply_forecast: bool) -> Dict[str, Any]:
    results = {}
    if not _health_df_period.empty:
        kpis_summary = get_clinic_summary_kpis(_health_df_period)
        results['kpis_summary'] = kpis_summary
        results['epi_data'] = calculate_clinic_epidemiological_data(filtered_health_df=_health_df_period)
        results['testing_insights'] = prepare_clinic_lab_testing_insights_data(kpis_summary=kpis_summary, filtered_health_df=_health_df_period)
        results['patient_focus_data'] = prepare_clinic_patient_focus_overview_data(filtered_health_df=_health_df_period)
    if not _full_health_df.empty:
        results['supply_forecast_data'] = prepare_clinic_supply_forecast_overview_data(full_historical_health_df=_full_health_df, use_ai_supply_forecasting_model=use_ai_supply_forecast)
    if not _iot_df_period.empty:
        results['env_details'] = prepare_clinic_environmental_detail_data(filtered_iot_df=_iot_df_period, reporting_period_context_str=current_period_str)
    return results

# --- UI Rendering Functions ---
def render_sidebar(full_health_df: pd.DataFrame, full_iot_df: pd.DataFrame) -> Tuple[date, date, bool]:
    st.sidebar.image(str(Path(_get_setting('APP_LOGO_SMALL_PATH', 'assets/logo_small.png'))), width=230)
    st.sidebar.header("Console Filters")
    min_dates = [df[col].min().date() for df, col in [(full_health_df, COL_ENCOUNTER_DATE), (full_iot_df, COL_TIMESTAMP)] if not df.empty and pd.api.types.is_datetime64_any_dtype(df[col])]
    max_dates = [df[col].max().date() for df, col in [(full_health_df, COL_ENCOUNTER_DATE), (full_iot_df, COL_TIMESTAMP)] if not df.empty and pd.api.types.is_datetime64_any_dtype(df[col])]
    if not min_dates or not max_dates:
        st.sidebar.error("No data with valid dates available."); st.stop()
    data_min_date, data_max_date = min(min_dates), max(max_dates)
    default_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    default_start = max(data_min_date, data_max_date - timedelta(days=default_days - 1))
    selected_range = st.sidebar.date_input("Select Date Range:", value=(default_start, data_max_date), min_value=data_min_date, max_value=data_max_date)
    start_date, end_date = (selected_range[0], selected_range[1]) if len(selected_range) == 2 else (default_start, data_max_date)
    use_ai_supply = st.sidebar.checkbox("Use Advanced AI Forecast", value=True, key="supply_ai_toggle")
    return start_date, end_date, use_ai_supply

def render_kpi_snapshot(dashboard_data: Dict[str, Any]):
    st.header("🚀 Clinic Performance & Environment Snapshot")
    if kpis := dashboard_data.get('kpis_summary'):
        main_kpis = structure_main_clinic_kpis(kpis)
        if main_kpis:
            st.markdown("##### **Overall Service Performance:**")
            cols = st.columns(len(main_kpis)); [render_kpi_card(**kpi) for col, kpi in zip(cols, main_kpis) if col.button("", key=f"kpi_main_{kpi['title']}")]
        disease_kpis = structure_disease_specific_clinic_kpis(kpis)
        if disease_kpis:
            st.markdown("##### **Key Disease & Supply Indicators:**")
            cols = st.columns(len(disease_kpis)); [render_kpi_card(**kpi) for col, kpi in zip(cols, disease_kpis) if col.button("", key=f"kpi_disease_{kpi['title']}")]
    if env_details := dashboard_data.get('env_details'):
        st.markdown("##### **Clinic Environment Quick Check:**")
        if alerts := env_details.get('current_environmental_alerts_list'):
            cols = st.columns(len(alerts) or 1); [render_traffic_light_indicator(**alert) for col, alert in zip(cols, alerts) if col]
    st.divider()

def render_deep_dive_tabs(dashboard_data: Dict[str, Any]):
    st.header("🛠️ Operational Areas Deep Dive")
    tabs = st.tabs(["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"])
    with tabs[0]: # Epi
        if epi_data := dashboard_data.get('epi_data'):
            if isinstance(symptom_df := epi_data.get("symptom_trends_weekly_top_n_df"), pd.DataFrame) and not symptom_df.empty:
                st.plotly_chart(plot_bar_chart(symptom_df, x_col='week_start_date', y_col='count', title="Weekly Symptom Frequency", color_col='symptom', barmode='group'), use_container_width=True)
            else: st.info("ℹ️ No epidemiological data to display.")
    with tabs[1]: # Testing
        if insights := dashboard_data.get('testing_insights'):
            st.subheader("Laboratory and Testing Performance")
            col1, col2 = st.columns(2)
            if isinstance(tat_df := insights.get("avg_tat_by_test_df"), pd.DataFrame):
                with col1: st.plotly_chart(plot_bar_chart(tat_df, x_col='Average TAT (Days)', y_col='Test Type', title='Average Turnaround Time (TAT)', orientation='h'), use_container_width=True)
            if isinstance(rejection_df := insights.get("rejection_reasons_df"), pd.DataFrame):
                with col2: st.plotly_chart(plot_donut_chart(rejection_df, labels_col='Reason', values_col='Count', title='Top Sample Rejection Reasons'), use_container_width=True)
            if isinstance(overdue_df := insights.get("overdue_pending_tests_list_df"), pd.DataFrame) and not overdue_df.empty:
                st.dataframe(overdue_df, use_container_width=True, hide_index=True)
    with tabs[2]: # Supply
        if forecast_data := dashboard_data.get('supply_forecast_data'):
            st.subheader("Supply Chain Forecast & Status")
            if forecast_list := forecast_data.get("forecast_items_overview_list"):
                forecast_df = pd.DataFrame(forecast_list).dropna(subset=['days_of_supply_remaining'])
                if not forecast_df.empty:
                    color_map = {"Critical Low": get_theme_color("high", "risk"), "Warning Low": get_theme_color("moderate", "risk"), "Sufficient": get_theme_color("low", "risk")}
                    st.plotly_chart(plot_bar_chart(forecast_df.sort_values('days_of_supply_remaining', ascending=True), x_col='days_of_supply_remaining', y_col='item', title='Estimated Days of Supply', color_col='stock_status', orientation='h', color_discrete_map=color_map), use_container_width=True)
                    st.markdown(f"**Forecast Model:** `{forecast_data.get('forecast_model_type_used', 'N/A')}`")
    with tabs[3]: # Patient
        if focus_data := dashboard_data.get('patient_focus_data'):
            st.subheader("Patient Load and Clinical Focus")
            if isinstance(load_df := focus_data.get("patient_load_by_key_condition_df"), pd.DataFrame) and not load_df.empty:
                st.plotly_chart(plot_bar_chart(load_df, x_col='period_start_date', y_col='unique_patients_count', title="Patient Load by Condition", color_col='condition', barmode='stack'), use_container_width=True)
            if isinstance(flagged_df := focus_data.get("flagged_patients_for_review_df"), pd.DataFrame) and not flagged_df.empty:
                st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    with tabs[4]: # Environment
        if env_details := dashboard_data.get('env_details'):
            st.subheader("Facility Environment Monitoring")
            if co2_plot := env_details.get("co2_trend_plot"): st.plotly_chart(co2_plot, use_container_width=True)
            if isinstance(readings_df := env_details.get("latest_room_sensor_readings_df"), pd.DataFrame) and not readings_df.empty:
                st.dataframe(readings_df, use_container_width=True, hide_index=True)

def main():
    """Main function to orchestrate the dashboard page."""
    st.set_page_config(page_title=f"Clinic Console - {_get_setting('APP_NAME', APP_NAME)}", page_icon="🏥", layout=_get_setting('APP_LAYOUT', "wide"))
    st.title("🏥 Clinic Operations & Management Console")
    st.markdown("**Service Performance, Patient Care, Resource Management, and Facility Environment**")
    st.divider()

    base_health_df = get_base_health_data()
    full_health_df = get_enriched_health_data(base_health_df)
    full_iot_df = get_iot_data()
    
    start_date, end_date, use_ai_supply = render_sidebar(full_health_df, full_iot_df)
    
    health_df_period = full_health_df[(full_health_df[COL_ENCOUNTER_DATE].dt.date >= start_date) & (full_health_df[COL_ENCOUNTER_DATE].dt.date <= end_date)]
    iot_df_period = full_iot_df[(full_iot_df[COL_TIMESTAMP].dt.date >= start_date) & (full_iot_df[COL_TIMESTAMP].dt.date <= end_date)]
    
    current_period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
    st.info(f"Displaying data for: **{current_period_str}**")

    try:
        dashboard_data = prepare_dashboard_data(_health_df_period=health_df_period, _iot_df_period=iot_df_period, _full_health_df=full_health_df, current_period_str=current_period_str, use_ai_supply_forecast=use_ai_supply)
    except Exception as e:
        logger.error(f"Fatal error during central data preparation: {e}", exc_info=True)
        st.error("A critical error occurred while analyzing the data. Please check logs."); st.stop()
        
    render_kpi_snapshot(dashboard_data)
    render_deep_dive_tabs(dashboard_data)

    st.divider()
    st.caption(_get_setting('APP_FOOTER_TEXT', f"{APP_NAME} Health Co-Pilot."))

# FIX: The main function is now called unconditionally.
# This ensures the page renders when imported by Streamlit's multipage app runner.
main()
