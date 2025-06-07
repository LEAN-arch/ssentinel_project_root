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
    from visualization.ui_elements import render_kpi_card, render_traffic_light_indicator, get_theme_color
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart, plot_donut_chart

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

# --- Constants & Configuration ---
APP_NAME = "Sentinel"
COL_ENCOUNTER_DATE = 'encounter_date'
COL_TIMESTAMP = 'timestamp'

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely get a setting from the config file."""
    return getattr(settings, attr_name, default_value)

# --- Data Loading & Caching (Optimized) ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_DATA_LOAD', 3600), show_spinner="Loading health records...")
def get_base_health_data() -> pd.DataFrame:
    """Loads the base health records from the source."""
    return load_health_records()

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_AI_MODEL', 3600), show_spinner="Enriching records with AI models...")
def get_enriched_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies AI models to the health records and performs initial cleaning."""
    if df.empty:
        return pd.DataFrame()
    _ = hash_dataframe_safe(df)
    enriched_df, _ = apply_ai_models(df)
    if not isinstance(enriched_df, pd.DataFrame) or COL_ENCOUNTER_DATE not in enriched_df.columns:
        return pd.DataFrame()
    enriched_df[COL_ENCOUNTER_DATE] = pd.to_datetime(enriched_df[COL_ENCOUNTER_DATE], errors='coerce')
    return enriched_df.dropna(subset=[COL_ENCOUNTER_DATE])

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_DATA_LOAD', 3600), show_spinner="Loading IoT environmental data...")
def get_iot_data() -> pd.DataFrame:
    """Loads and prepares IoT environmental data."""
    df = load_iot_clinic_environment_data()
    if df.empty or COL_TIMESTAMP not in df.columns:
        return pd.DataFrame()
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], errors='coerce')
    return df.dropna(subset=[COL_TIMESTAMP])

# --- Centralized Data Preparation (Key Performance Optimization) ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Analyzing clinic data for selected period...")
def prepare_dashboard_data(
    _health_df_period: pd.DataFrame,
    _iot_df_period: pd.DataFrame,
    _full_health_df: pd.DataFrame,
    current_period_str: str,
    use_ai_supply_forecast: bool
) -> Dict[str, Any]:
    results = {}
    if not _health_df_period.empty:
        kpis_summary = get_clinic_summary_kpis(_health_df_period)
        results['kpis_summary'] = kpis_summary
        results['epi_data'] = calculate_clinic_epidemiological_data(filtered_health_df=_health_df_period)
        results['testing_insights'] = prepare_clinic_lab_testing_insights_data(kpis_summary=kpis_summary, filtered_health_df=_health_df_period)
        results['patient_focus_data'] = prepare_clinic_patient_focus_overview_data(filtered_health_df=_health_df_period)
    if not _full_health_df.empty:
        results['supply_forecast_data'] = prepare_clinic_supply_forecast_overview_data(
            full_historical_health_df=_full_health_df, current_period_context_str=current_period_str, use_ai_supply_forecasting_model=use_ai_supply_forecast
        )
    if not _iot_df_period.empty:
        results['env_kpis'] = get_clinic_environmental_summary_kpis(_iot_df_period)
        results['env_details'] = prepare_clinic_environmental_detail_data(filtered_iot_df=_iot_df_period, reporting_period_context_str=current_period_str)
    return results

# --- UI Rendering Functions (Modular & Clean) ---

def render_sidebar(full_health_df: pd.DataFrame, full_iot_df: pd.DataFrame) -> Tuple[date, date, bool]:
    st.sidebar.image(str(Path(_get_setting('APP_LOGO_SMALL_PATH', 'assets/logo_small.png'))), width=230)
    st.sidebar.header("Console Filters")
    min_dates = [df[col].min().date() for df, col in [(full_health_df, COL_ENCOUNTER_DATE), (full_iot_df, COL_TIMESTAMP)] if not df.empty]
    max_dates = [df[col].max().date() for df, col in [(full_health_df, COL_ENCOUNTER_DATE), (full_iot_df, COL_TIMESTAMP)] if not df.empty]
    if not min_dates or not max_dates:
        st.sidebar.error("No data available to set date range.")
        st.stop()
    data_min_date, data_max_date = min(min_dates), max(max_dates)
    default_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    default_start = max(data_min_date, data_max_date - timedelta(days=default_days - 1))
    selected_range = st.sidebar.date_input("Select Date Range:", value=(default_start, data_max_date), min_value=data_min_date, max_value=data_max_date, help="Select the period for clinical and operational review.")
    start_date, end_date = (selected_range[0], selected_range[1]) if len(selected_range) == 2 else (default_start, data_max_date)
    use_ai_supply = st.sidebar.checkbox("Use Advanced AI Forecast", value=True, key="supply_ai_toggle", help="Use a more complex model for supply chain forecasting.")
    return start_date, end_date, use_ai_supply

def render_kpi_snapshot(dashboard_data: Dict[str, Any]):
    """Renders the top-level KPI cards."""
    st.header("🚀 Clinic Performance & Environment Snapshot")
    kpis = dashboard_data.get('kpis_summary')
    if kpis:
        # === FIX APPLIED HERE ===
        # The TypeError indicates these functions now only take one argument (the kpis dictionary).
        # The context string is no longer passed.
        main_kpis = structure_main_clinic_kpis(kpis)
        disease_kpis = structure_disease_specific_clinic_kpis(kpis)
        
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

    env_kpis = dashboard_data.get('env_kpis')
    if env_kpis:
        st.markdown("##### **Clinic Environment Quick Check:**")
        cols = st.columns(4)
        with cols[0]: render_kpi_card(title="Avg. CO₂", value_str=f"{env_kpis.get('avg_co2_overall_ppm', 0):.0f}", units="ppm", icon="💨")
        with cols[1]: render_kpi_card(title="Avg. PM2.5", value_str=f"{env_kpis.get('avg_pm25_overall_ugm3', 0):.1f}", units="µg/m³", icon="🌫️")
        with cols[2]: render_kpi_card(title="Avg. Occupancy", value_str=f"{env_kpis.get('avg_waiting_room_occupancy_overall_persons', 0):.1f}", units="persons", icon="👨‍👩‍👧‍👦")
        with cols[3]: render_kpi_card(title="High Noise Alerts", value_str=str(env_kpis.get('rooms_noise_high_alert_latest_count', 0)), units="areas", icon="🔊")
    st.divider()

def render_deep_dive_tabs(dashboard_data: Dict[str, Any]):
    """Renders the main content tabs with robust checks for data existence."""
    st.header("🛠️ Operational Areas Deep Dive")
    tabs = st.tabs(["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"])
    
    with tabs[0]: # Local Epi
        epi_data = dashboard_data.get('epi_data')
        symptom_df = epi_data.get("symptom_trends_weekly_top_n_df") if epi_data else None
        if isinstance(symptom_df, pd.DataFrame) and not symptom_df.empty:
            fig = plot_bar_chart(symptom_df, x_col='week_start_date', y_col='count', title="Weekly Symptom Frequency", color_col='symptom', barmode='group', y_axis_title="Number of Symptom Reports")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No epidemiological data to display for the selected period.")

    with tabs[1]: # Testing
        st.subheader("Laboratory and Testing Performance")
        insights = dashboard_data.get('testing_insights')
        tat_df = insights.get("avg_tat_by_test_df") if insights else None
        rejection_df = insights.get("rejection_reasons_df") if insights else None
        overdue_df = insights.get("overdue_pending_tests_list_df") if insights else None

        if not any([isinstance(df, pd.DataFrame) and not df.empty for df in [tat_df, rejection_df, overdue_df]]):
             st.info("ℹ️ No testing data (turnaround, rejections, or overdue) available for this period.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if isinstance(tat_df, pd.DataFrame) and not tat_df.empty:
                    fig_tat = plot_bar_chart(tat_df, x_col='Average TAT (Days)', y_col='Test Type', title='Average Turnaround Time (TAT) by Test', orientation='h', y_axis_title="Test Type")
                    target_tat = _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)
                    fig_tat.add_vline(x=target_tat, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Target ({target_tat} days)")
                    st.plotly_chart(fig_tat, use_container_width=True)
                else:
                    st.write("No turnaround time data.")
            with col2:
                if isinstance(rejection_df, pd.DataFrame) and not rejection_df.empty:
                    fig_reject = plot_donut_chart(rejection_df, labels_col='Reason', values_col='Count', title='Top Sample Rejection Reasons')
                    st.plotly_chart(fig_reject, use_container_width=True)
                else:
                    st.write("No sample rejection data.")
            
            st.markdown("###### **Overdue Pending Tests:**")
            if isinstance(overdue_df, pd.DataFrame) and not overdue_df.empty:
                 st.dataframe(overdue_df, use_container_width=True, hide_index=True)
            else:
                st.write("No overdue tests found.")

    with tabs[2]: # Supply Chain
        st.subheader("Supply Chain Forecast & Status")
        forecast_data = dashboard_data.get('supply_forecast_data')
        forecast_list = forecast_data.get("forecast_items_overview_list", []) if forecast_data else []
        
        if forecast_list:
            forecast_df = pd.DataFrame(forecast_list)
            forecast_df['days_of_supply_remaining'] = pd.to_numeric(forecast_df['days_of_supply_remaining'], errors='coerce')
            forecast_df.sort_values('days_of_supply_remaining', ascending=True, inplace=True)
            color_map = {"Critical Low": get_theme_color("risk_high"), "Warning Low": get_theme_color("risk_moderate"), "Sufficient": get_theme_color("risk_low")}
            fig_supply = plot_bar_chart(forecast_df, x_col='days_of_supply_remaining', y_col='item', title='Estimated Days of Supply Remaining', color_col='stock_status', orientation='h', color_discrete_map=color_map, y_axis_title='Supply Item', x_axis_title='Days Remaining')
            st.plotly_chart(fig_supply, use_container_width=True)
            st.markdown(f"**Forecast Model Used:** `{forecast_data.get('forecast_model_type_used', 'N/A')}`")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No supply chain forecast data could be generated.")

    with tabs[3]: # Patient Focus
        focus_data = dashboard_data.get('patient_focus_data')
        patient_load_df = focus_data.get("patient_load_by_key_condition_df") if focus_data else None
        flagged_patients_df = focus_data.get("flagged_patients_for_review_df") if focus_data else None

        if isinstance(patient_load_df, pd.DataFrame) and not patient_load_df.empty:
            fig_load = plot_bar_chart(patient_load_df, x_col='period_start_date', y_col='unique_patients_count', title="Patient Load by Key Condition", color_col='condition', y_axis_title="Number of Unique Patients", barmode='stack')
            st.plotly_chart(fig_load, use_container_width=True)
        else:
            st.info("ℹ️ No patient load data available for this period.")
        
        st.markdown("###### **Flagged Patients for Clinical Review:**")
        if isinstance(flagged_patients_df, pd.DataFrame) and not flagged_patients_df.empty:
            st.dataframe(flagged_patients_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No patients were flagged for review in this period.")

    with tabs[4]: # Environment
        env_details = dashboard_data.get('env_details')
        if env_details:
            st.markdown("###### **Environmental Alerts (Latest Readings):**")
            alerts = [a for a in env_details.get("current_environmental_alerts_list", []) if a.get("level") != "ACCEPTABLE"]
            if not alerts:
                st.success("✅ All monitored environmental parameters are within acceptable limits.")
            else:
                for alert in alerts:
                    render_traffic_light_indicator(message=alert.get("message", "Alert"), status_level=alert.get("level", "NO_DATA"))
            
            co2_trend_df = env_details.get("hourly_avg_co2_trend")
            if isinstance(co2_trend_df, pd.DataFrame) and not co2_trend_df.empty:
                st.plotly_chart(plot_annotated_line_chart(co2_trend_df, "Hourly Avg. CO₂ Levels", "CO₂ (ppm)"), use_container_width=True)
            
            sensor_readings_df = env_details.get("latest_room_sensor_readings_df")
            if isinstance(sensor_readings_df, pd.DataFrame) and not sensor_readings_df.empty:
                st.markdown("###### **Latest Sensor Readings by Room:**")
                st.dataframe(sensor_readings_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No environmental data was recorded in this period.")

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
        st.error("A critical error occurred while analyzing the data. The dashboard cannot be displayed. Please check logs.")
        st.stop()
        
    try:
        # Pass only dashboard_data, as render_kpi_snapshot no longer needs the context string directly
        render_kpi_snapshot(dashboard_data)
        render_deep_dive_tabs(dashboard_data)
    except Exception as e:
        logger.error(f"Error rendering dashboard components: {e}", exc_info=True)
        st.error("An unexpected error occurred while displaying the dashboard visuals. Some components may be missing.")

    st.divider()
    st.caption(_get_setting('APP_FOOTER_TEXT', f"{APP_NAME} Health Co-Pilot."))

if __name__ == "__main__":
    main()
