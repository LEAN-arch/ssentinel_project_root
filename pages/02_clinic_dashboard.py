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
    from visualization.plots import plot_annotated_line_chart, plot_bar_chart, plot_donut_chart, create_empty_figure

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

st.set_page_config(
    page_title=f"Clinic Console - {_get_setting('APP_NAME', 'Sentinel')}",
    page_icon="🏥",
    layout=_get_setting('APP_LAYOUT', "wide")
)

# --- Data Loading & Caching ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Loading and enriching health records...")
def load_and_prepare_health_data() -> pd.DataFrame:
    df = load_health_records()
    if df.empty: return pd.DataFrame()
    enriched_df, _ = apply_ai_models(df)
    if not isinstance(enriched_df, pd.DataFrame) or 'encounter_date' not in enriched_df.columns:
        return pd.DataFrame()
    enriched_df['encounter_date'] = pd.to_datetime(enriched_df['encounter_date'], errors='coerce')
    return enriched_df.dropna(subset=['encounter_date'])

@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Loading IoT environmental data...")
def load_and_prepare_iot_data() -> pd.DataFrame:
    df = load_iot_clinic_environment_data()
    if df.empty or 'timestamp' not in df.columns: return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])

# --- UI Components & Filters ---
def manage_date_range_filter(data_min_date: date, data_max_date: date) -> Tuple[date, date]:
    default_days = _get_setting('WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
    default_start = max(data_min_date, data_max_date - timedelta(days=default_days - 1))
    
    selected_range = st.sidebar.date_input(
        "Select Date Range for Clinic Review:",
        value=(default_start, data_max_date),
        min_value=data_min_date, max_value=data_max_date
    )
    return (selected_range[0], selected_range[1]) if len(selected_range) == 2 else (default_start, data_max_date)

# --- Main Application ---
st.title("🏥 Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care, Resource Management, and Facility Environment**")
st.divider()

full_health_df = load_and_prepare_health_data()
full_iot_df = load_and_prepare_iot_data()

st.sidebar.image(str(Path(_get_setting('APP_LOGO_SMALL_PATH', ''))), width=230)
st.sidebar.header("Console Filters")

all_dates = []
if not full_health_df.empty and 'encounter_date' in full_health_df.columns:
    all_dates.extend([full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()])
if not full_iot_df.empty and 'timestamp' in full_iot_df.columns:
    all_dates.extend([full_iot_df['timestamp'].min().date(), full_iot_df['timestamp'].max().date()])

if all_dates:
    start_date, end_date = manage_date_range_filter(min(all_dates), max(all_dates))
else:
    st.sidebar.error("All data sources are empty. Cannot display dashboard.")
    st.stop()

health_df_period = full_health_df[(full_health_df['encounter_date'].dt.date >= start_date) & (full_health_df['encounter_date'].dt.date <= end_date)]
iot_df_period = full_iot_df[(full_iot_df['timestamp'].dt.date >= start_date) & (full_iot_df['timestamp'].dt.date <= end_date)]

current_period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"Displaying data for: **{current_period_str}**")

# --- KPI Snapshot & Deep Dive Tabs ---
st.header("🚀 Clinic Performance & Environment Snapshot")
try:
    if not health_df_period.empty:
        kpis = get_clinic_summary_kpis(health_df_period)
        main_kpis = structure_main_clinic_kpis(kpis_summary=kpis, reporting_period_context_str=current_period_str)
        disease_kpis = structure_disease_specific_clinic_kpis(kpis_summary=kpis, reporting_period_context_str=current_period_str)
        
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

    st.markdown("##### **Clinic Environment Quick Check:**")
    if not iot_df_period.empty:
        env_kpis = get_clinic_environmental_summary_kpis(iot_df_period)
        cols = st.columns(4)
        with cols[0]: render_kpi_card(title="Avg. CO₂", value_str=f"{env_kpis.get('avg_co2_overall_ppm', 0):.0f}", units="ppm", icon="💨")
        with cols[1]: render_kpi_card(title="Avg. PM2.5", value_str=f"{env_kpis.get('avg_pm25_overall_ugm3', 0):.1f}", units="µg/m³", icon="🌫️")
        with cols[2]: render_kpi_card(title="Avg. Occupancy", value_str=f"{env_kpis.get('avg_waiting_room_occupancy_overall_persons', 0):.1f}", units="persons", icon="👨‍👩‍👧‍👦")
        with cols[3]: render_kpi_card(title="High Noise Alerts", value_str=str(env_kpis.get('rooms_noise_high_alert_latest_count', 0)), units="areas", icon="🔊")
    else:
        st.info("ℹ️ No environmental data in the selected period.")
except Exception as e:
    logger.error(f"Error rendering KPI snapshot section: {e}", exc_info=True)
    # FIXED: Removed the invalid `exc_info=True` from the st.error call.
    st.error(f"⚠️ An error occurred while rendering the KPI snapshot: {e}")
st.divider()

# --- Deep Dive Tabs Section ---
st.header("🛠️ Operational Areas Deep Dive")
tabs = st.tabs(["📈 Local Epi", "🔬 Testing", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment"])

with tabs[0]: # Local Epi
    if health_df_period.empty:
        st.info("ℹ️ No health data in this period for epidemiological analysis.")
    else:
        try:
            epi_data = calculate_clinic_epidemiological_data(filtered_health_df=health_df_period)
            fig_epi = plot_bar_chart(epi_data.get("symptom_trends_weekly_top_n_df"), x_col='week_start_date', y_col='count', title="Weekly Symptom Frequency", color_col='symptom', barmode='group', y_values_are_counts=True, y_axis_title="Number of Symptom Reports")
            st.plotly_chart(fig_epi, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in 'Local Epi' tab: {e}", exc_info=True)
            st.error(f"⚠️ Could not generate epi insights: {e}")

with tabs[1]: # Testing
    st.subheader("Laboratory and Testing Performance")
    if health_df_period.empty:
        st.info("ℹ️ No health data in this period for testing insights.")
    else:
        try:
            kpis = get_clinic_summary_kpis(health_df_period)
            insights = prepare_clinic_lab_testing_insights_data(kpis_summary=kpis, filtered_health_df=health_df_period)
            
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                fig_tat = plot_bar_chart(
                    insights.get("avg_tat_by_test_df"), x_col='Average TAT (Days)', y_col='Test Type',
                    title='Average Turnaround Time (TAT) by Test', orientation='h', y_axis_title="Test Type"
                )
                target_tat_val = _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)
                fig_tat.add_vline(x=target_tat_val, line_width=2, line_dash="dash", line_color="red", annotation_text="Target TAT")
                st.plotly_chart(fig_tat, use_container_width=True)
            with plot_col2:
                fig_reject = plot_donut_chart(
                    insights.get("rejection_reasons_df"), labels_col='Reason', values_col='Count',
                    title='Top Sample Rejection Reasons'
                )
                st.plotly_chart(fig_reject, use_container_width=True)

            st.markdown("###### **Overdue Pending Tests:**")
            st.dataframe(insights.get("overdue_pending_tests_list_df"), use_container_width=True, hide_index=True)
        except Exception as e:
            logger.error(f"Error in 'Testing' tab: {e}", exc_info=True)
            st.error(f"⚠️ Could not generate testing insights: {e}")

with tabs[2]: # Supply Chain
    st.subheader("Supply Chain Forecast & Status")
    if full_health_df.empty:
        st.info("ℹ️ No historical data available for supply forecasting.")
    else:
        try:
            use_ai = st.checkbox("Use Advanced AI Forecast", key="supply_ai_toggle", help="Simulates a more complex forecast model.")
            forecast = prepare_clinic_supply_forecast_overview_data(full_historical_health_df=full_health_df, current_period_context_str=current_period_str, use_ai_supply_forecasting_model=use_ai)
            forecast_df = pd.DataFrame(forecast.get("forecast_items_overview_list", []))
            if not forecast_df.empty:
                forecast_df['Days of Supply Remaining'] = pd.to_numeric(forecast_df['days_of_supply_remaining'], errors='coerce')
                forecast_df.sort_values('Days of Supply Remaining', ascending=True, inplace=True)
                color_map = { "Critical Low": get_theme_color("risk_high"), "Warning Low": get_theme_color("risk_moderate"), "Sufficient": get_theme_color("risk_low") }
                fig_supply = plot_bar_chart(
                    forecast_df, x_col='Days of Supply Remaining', y_col='item',
                    title='Estimated Days of Supply Remaining', color_col='stock_status',
                    orientation='h', color_discrete_map=color_map, y_axis_title='Supply Item'
                )
                st.plotly_chart(fig_supply, use_container_width=True)
            st.markdown(f"**Forecast Model Used:** `{forecast.get('forecast_model_type_used', 'N/A')}`")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        except Exception as e:
            logger.error(f"Error in 'Supply Chain' tab: {e}", exc_info=True)
            st.error(f"⚠️ Could not generate supply chain insights: {e}")

with tabs[3]: # Patient Focus
    if health_df_period.empty:
        st.info("ℹ️ No health data for patient focus analysis.")
    else:
        try:
            focus_data = prepare_clinic_patient_focus_overview_data(filtered_health_df=health_df_period)
            fig_load = plot_bar_chart(focus_data.get("patient_load_by_key_condition_df"), x_col='period_start_date', y_col='unique_patients_count', title="Patient Load by Condition", color_col='condition', y_axis_title="Number of Unique Patients", y_values_are_counts=True, barmode='stack')
            st.plotly_chart(fig_load, use_container_width=True)
            st.markdown("###### **Flagged Patients for Clinical Review:**")
            st.dataframe(focus_data.get("flagged_patients_for_review_df"), use_container_width=True, hide_index=True)
        except Exception as e:
            logger.error(f"Error in 'Patient Focus' tab: {e}", exc_info=True)
            st.error(f"⚠️ Could not generate patient focus insights: {e}")

with tabs[4]: # Environment
    if iot_df_period.empty:
        st.info("ℹ️ No environmental data was recorded in this period.")
    else:
        try:
            env_data = prepare_clinic_environmental_detail_data(filtered_iot_df=iot_df_period, reporting_period_context_str=current_period_str)
            st.markdown("###### **Current Environmental Alerts (Latest Readings):**")
            alerts = env_data.get("current_environmental_alerts_list", [])
            if not any(a.get("level") != "ACCEPTABLE" for a in alerts):
                st.success("✅ All monitored environmental parameters appear within acceptable limits.")
            else:
                for alert in alerts: 
                    if alert.get("level") != "ACCEPTABLE":
                        render_traffic_light_indicator(message=alert.get("message", "Alert"), status_level=alert.get("level", "NO_DATA"))
            st.plotly_chart(plot_annotated_line_chart(env_data.get("hourly_avg_co2_trend"), "Hourly Avg. CO₂ Levels", "CO₂ (ppm)"), use_container_width=True)
        except Exception as e:
            logger.error(f"Error in 'Environment' tab: {e}", exc_info=True)
            st.error(f"⚠️ Could not generate environmental details: {e}")

st.divider()
st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
