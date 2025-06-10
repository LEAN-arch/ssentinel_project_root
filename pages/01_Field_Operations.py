# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - FIELD OPERATIONS DASHBOARD (V8 - AI FORECASTING INTEGRATION)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_forecast_chart, render_kpi_card)

# --- Page Setup ---
st.set_page_config(page_title="Field Operations", page_icon="🧑‍⚕️", layout="wide")
logger = logging.getLogger(__name__)


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    """
    Loads and caches the base health records, ensuring they are fully
    enriched with all required AI-generated scores.
    """
    raw_df = load_health_records()
    if raw_df.empty:
        return pd.DataFrame()
    enriched_df, _ = apply_ai_models(raw_df)
    return enriched_df

# --- Analytics & KPI Functions ---
def calculate_field_ops_kpis(df: pd.DataFrame) -> Dict:
    """
    Calculates a rich set of decision-grade KPIs for field operations.
    """
    if df.empty: return {}
    kpis = {}
    kpis['unique_patients_seen'] = df['patient_id'].nunique()
    
    # Malaria Screening Cascade
    malaria_symptomatic = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    screened_for_malaria = malaria_symptomatic[malaria_symptomatic['test_type'] == 'Malaria RDT']
    kpis['malaria_screening_rate'] = (len(screened_for_malaria) / len(malaria_symptomatic) * 100) if len(malaria_symptomatic) > 0 else 0
    
    # TB Linkage to Care
    positive_tb = df[(df['test_type'] == 'TB Screen') & (df['test_result'] == 'Positive')]
    linked_tb = positive_tb[positive_tb['referral_status'] == 'Completed']
    kpis['tb_linkage_rate'] = (len(linked_tb) / len(positive_tb) * 100) if len(positive_tb) > 0 else 0
    return kpis

@st.cache_data(ttl=3600, show_spinner="Generating disease forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    """
    Generates Prophet forecasts for multiple metrics.
    """
    if df.empty or len(df) < 10:
        return {}

    # 1. Forecast total daily encounters
    daily_encounters = df.set_index('encounter_date').resample('D').size().reset_index(name='count')
    encounters_fc = generate_prophet_forecast(daily_encounters.rename(columns={'encounter_date': 'ds', 'count': 'y'}))

    # 2. Forecast Malaria cases
    malaria_cases = df[df['diagnosis'] == 'Malaria'].set_index('encounter_date').resample('D').size().reset_index(name='count')
    malaria_fc = generate_prophet_forecast(malaria_cases.rename(columns={'encounter_date': 'ds', 'count': 'y'}))
    
    # 3. Forecast average daily risk
    avg_risk = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index()
    avg_risk_fc = generate_prophet_forecast(avg_risk.rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'}))

    return {
        "encounters": encounters_fc,
        "malaria": malaria_fc,
        "avg_risk": avg_risk_fc
    }

# --- UI Rendering ---
def display_alerts(df: pd.DataFrame):
    st.subheader("🚨 Priority Patient Alerts")
    alerts = generate_chw_alerts(patient_df=df)
    if not alerts:
        st.success("✅ No high-priority patient alerts for this selection.")
        return
    
    for alert in alerts:
        level, icon = ("CRITICAL", "🔴") if alert.get('alert_level') == 'CRITICAL' else (("WARNING", "🟠") if alert.get('alert_level') == 'WARNING' else ("INFO", "ℹ️"))
        with st.expander(f"{icon} {level}: {alert.get('reason')} for Pt. {alert.get('patient_id')}", expanded=(level == 'CRITICAL')):
            st.markdown(f"**Details:** {alert.get('details', 'N/A')}")
            st.markdown(f"**Priority Score:** {alert.get('priority', 0):.0f}")


# --- Main Page Execution ---
def main():
    st.title("🧑‍⚕️ Field Operations Command Center")
    st.markdown("An AI-powered dashboard for supervising field teams, monitoring program effectiveness, and forecasting health trends.")
    st.divider()

    full_df = get_data()

    if full_df.empty:
        st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_df['encounter_date'].min().date(), full_df['encounter_date'].max().date()
        zone_options = ["All Zones"] + sorted(full_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        
        # Date range for retrospective analysis
        st.markdown("---")
        st.subheader("Retrospective Analysis")
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=6)), max_date), min_value=min_date, max_value=max_date)
        
        # Controls for forecasting
        st.markdown("---")
        st.subheader("Predictive Analytics")
        forecast_days = st.slider("Days to Forecast Ahead:", min_value=7, max_value=90, value=30, step=7)


    # --- Filter Data ---
    analysis_df = full_df[full_df['encounter_date'].dt.date.between(start_date, end_date)]
    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]

    st.info(f"**Viewing Retrospective Data:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** {selected_zone}")

    # --- KPI Section ---
    kpis = calculate_field_ops_kpis(analysis_df)
    st.subheader(f"Performance Summary ({start_date:%d %b} to {end_date:%d %b})")
    cols = st.columns(3)
    cols[0].render_kpi_card("Patients Seen (Unique)", kpis.get('unique_patients_seen', 0), icon="👥")
    cols[1].render_kpi_card("Malaria Screening Rate", f"{kpis.get('malaria_screening_rate', 0):.1f}%", icon="🦟", help_text="% of febrile patients tested for Malaria.")
    cols[2].render_kpi_card("TB Linkage to Care Rate", f"{kpis.get('tb_linkage_rate', 0):.1f}%", icon="🫁", help_text="% of positive TB screens linked to care.")
    st.divider()
    
    # --- Main Layout: Alerts and Forecasting ---
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Pass the daily slice of data to the alerts function
        daily_df = full_df[full_df['encounter_date'].dt.date == end_date]
        if selected_zone != "All Zones": daily_df = daily_df[daily_df['zone_id'] == selected_zone]
        display_alerts(daily_df)

    with col2:
        st.header(f"🔮 AI-Powered Forecasts ({forecast_days} Days Ahead)")
        
        # Use the full historical dataset for forecasting for accuracy
        forecast_df = full_df if selected_zone == "All Zones" else full_df[full_df['zone_id'] == selected_zone]
        forecasts = generate_forecasts(forecast_df, forecast_days)

        if not forecasts:
            st.warning("Not enough historical data in the selected zone to generate reliable forecasts.")
        else:
            fc_type = st.selectbox("Select Forecast:", options=list(forecasts.keys()))
            
            if fc_type == "encounters":
                title, y_axis = "Daily Patient Encounters", "Encounters"
            elif fc_type == "malaria":
                title, y_axis = "Daily Malaria Cases", "Positive Cases"
            else: # avg_risk
                title, y_axis = "Daily Average Patient Risk", "Avg. Risk Score"

            fig = plot_forecast_chart(forecasts[fc_type], title=title, y_title=y_axis)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
