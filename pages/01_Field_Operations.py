# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - INTEGRATED FIELD COMMAND CENTER (V14 - FINAL)

import logging
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_forecast_chart, plot_line_chart)

# --- Page Setup ---
st.set_page_config(page_title="Field Command Center", page_icon="📡", layout="wide")
logger = logging.getLogger(__name__)


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, enriches, and caches all data for the dashboard."""
    raw_health_df = load_health_records()
    iot_df = load_iot_records()
    if raw_health_df.empty:
        return pd.DataFrame(), iot_df
    enriched_df, _ = apply_ai_models(raw_health_df)
    return enriched_df, iot_df

# --- Analytics & UI Components ---
def render_program_scorecard(df: pd.DataFrame, full_history_df: pd.DataFrame):
    st.subheader("📊 Program Scorecard")
    if df.empty:
        st.info("No activity in the selected period to generate a scorecard.")
        return

    # --- KPI Calculations ---
    symptomatic_malaria = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    tested_malaria = symptomatic_malaria[symptomatic_malaria['test_type'] == 'Malaria RDT']
    malaria_screening_rate = (len(tested_malaria) / len(symptomatic_malaria) * 100) if len(symptomatic_malaria) > 0 else 0

    symptomatic_tb = df[df['patient_reported_symptoms'].str.contains('cough', case=False, na=False)]
    tested_tb = symptomatic_tb[symptomatic_tb['test_type'] == 'TB Screen']
    positive_tb = tested_tb[tested_tb['test_result'] == 'Positive']
    linked_tb = positive_tb[positive_tb['referral_status'] == 'Completed']
    tb_linkage_rate = (len(linked_tb) / len(positive_tb) * 100) if len(positive_tb) > 0 else 100

    # Create DataFrame for display
    scorecard_df = pd.DataFrame([
        {'Metric': 'Patients Seen', 'Value': f"{df['patient_id'].nunique()}", 'Target': 'N/A', 'Status': 'Metric'},
        {'Metric': 'Malaria Screening Rate', 'Value': f"{malaria_screening_rate:.1f}%", 'Target': '> 90%', 'Status': 'Good' if malaria_screening_rate >= 90 else 'Alert'},
        {'Metric': 'TB Linkage to Care', 'Value': f"{tb_linkage_rate:.1f}%", 'Target': '> 85%', 'Status': 'Good' if tb_linkage_rate >= 85 else 'Alert'},
    ])
    
    def style_status(val):
        color = 'green' if val == 'Good' else 'red' if val == 'Alert' else 'gray'
        return f'color: {color}; font-weight: bold;'
        
    st.dataframe(
        scorecard_df.style.apply(lambda row: row.map(style_status), subset=['Status']),
        use_container_width=True, hide_index=True
    )


def display_alerts(df: pd.DataFrame):
    st.subheader("🚨 Daily Priority Alerts")
    alerts = generate_chw_alerts(patient_df=df)
    if not alerts:
        st.success("✅ No high-priority patient alerts for this selection."); return
    for alert in alerts:
        level, icon = ("CRITICAL", "🔴") if alert.get('alert_level') == 'CRITICAL' else (("WARNING", "🟠") if alert.get('alert_level') == 'WARNING' else ("INFO", "ℹ️"))
        with st.container(border=True):
            st.markdown(f"**{icon} {alert.get('reason')} for Pt. {alert.get('patient_id')}**")
            st.markdown(f"> {alert.get('details', 'N/A')} (Priority: {alert.get('priority', 0):.0f})")

def render_iot_wearable_tab(iot_df: pd.DataFrame, chw_id: str):
    st.subheader("🛰️ Environmental & Team Factors")
    if iot_df.empty:
        st.info("No IoT or wearable data available for this period."); return
    
    col1, col2 = st.columns(2)
    with col1:
        clinic_iot = iot_df.dropna(subset=['avg_co2_ppm'])
        co2_trend = clinic_iot.set_index('timestamp')['avg_co2_ppm'].resample('D').mean()
        fig_co2 = plot_line_chart(co2_trend, "Average Clinic CO₂ (Ventilation Proxy)", "CO₂ PPM")
        st.plotly_chart(fig_co2, use_container_width=True, config={'displayModeBar': False})
    with col2:
        wearable_iot = iot_df.dropna(subset=['chw_stress_score'])
        if chw_id != "All CHWs":
            wearable_iot = wearable_iot[wearable_iot['chw_id'] == chw_id]
        
        stress_trend = wearable_iot.set_index('timestamp')['chw_stress_score'].resample('D').mean()
        fig_stress = plot_line_chart(stress_trend, "Average Team Stress Index", "Stress Index (0-100)")
        st.plotly_chart(fig_stress, use_container_width=True, config={'displayModeBar': False})

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    if df.empty or len(df) < 10: return {}
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    return {
        "Patient Load": generate_prophet_forecast(encounters_hist, forecast_days=forecast_days),
        "Community Risk Index": generate_prophet_forecast(avg_risk_hist, forecast_days=forecast_days),
    }

# --- Main Page Execution ---
def main():
    st.title("📡 Field Operations Command Center")
    st.markdown("An integrated dashboard for supervising team activity, patient risk, and future trends.")
    
    health_df, iot_df = get_data()
    if health_df.empty: st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        zone_options = ["All Zones"] + sorted(health_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        
        chw_options = ["All CHWs"] + sorted(health_df['chw_id'].dropna().unique())
        selected_chw = st.selectbox("Filter by CHW:", options=chw_options)
        
        today = health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(today - timedelta(days=29), health_df['encounter_date'].min().date()), today), min_value=health_df['encounter_date'].min().date(), max_value=today)
        
        forecast_days = st.slider("Forecast Horizon (Days):", 7, 90, 14, 7)

    # --- Data Filtering ---
    analysis_df = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    forecast_source_df = health_df[(health_df['encounter_date'].dt.date <= end_date)]
    iot_filtered = iot_df[iot_df['timestamp'].dt.date.between(start_date, end_date)]

    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]
        forecast_source_df = forecast_source_df[forecast_source_df['zone_id'] == selected_zone]
        iot_filtered = iot_filtered[iot_filtered['zone_id'] == selected_zone]
    if selected_chw != "All CHWs":
        analysis_df = analysis_df[analysis_df['chw_id'] == selected_chw]
        forecast_source_df = forecast_source_df[forecast_source_df['chw_id'] == selected_chw]

    st.info(f"**Displaying Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** `{selected_zone}` | **CHW:** `{selected_chw}`")
    st.divider()

    # --- Main Layout ---
    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        render_program_scorecard(analysis_df, forecast_source_df)
        st.divider()
        display_alerts(analysis_df)
    with col2:
        tab1, tab2 = st.tabs(["**🔮 AI Forecasts**", "**🛰️ IoT & Wearables**"])
        with tab1:
            st.subheader(f"Predictive Analytics ({forecast_days} Days Ahead)")
            forecasts = generate_forecasts(forecast_source_df, forecast_days)
            if not forecasts:
                st.warning("Not enough historical data for the selected filters to generate reliable forecasts.")
            else:
                fc_type = st.selectbox("Select Forecast:", options=list(forecasts.keys()))
                y_axis = "Daily Encounters" if fc_type == "Patient Load" else "Avg. Community Risk"
                fig = plot_forecast_chart(forecasts[fc_type], title=f"{fc_type} Forecast", y_title=y_axis)
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            render_iot_wearable_tab(iot_filtered, selected_chw)

if __name__ == "__main__":
    main()
