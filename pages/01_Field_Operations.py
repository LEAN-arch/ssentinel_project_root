# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - INTEGRATED FIELD COMMAND CENTER (V12 - FINAL)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_forecast_chart, render_kpi_card,
                           plot_line_chart)

# --- Page Setup ---
st.set_page_config(page_title="Field Command Center", page_icon="📡", layout="wide")
logger = logging.getLogger(__name__)


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, enriches, and caches all data for the dashboard."""
    raw_health_df = load_health_records()
    iot_df = load_iot_records() # Load IoT data
    
    if raw_health_df.empty:
        return pd.DataFrame(), iot_df
        
    enriched_df, _ = apply_ai_models(raw_health_df)
    return enriched_df, iot_df

# --- Analytics & KPI Functions ---
def calculate_field_ops_kpis(df: pd.DataFrame) -> Dict:
    """Calculates a rich set of decision-grade KPIs for field operations."""
    if df.empty: return {}
    
    kpis = {
        'unique_patients_seen': df['patient_id'].nunique(),
        'high_priority_alerts': (df.get('ai_followup_priority_score', pd.Series(dtype=float)) >= 80).sum()
    }
    
    # Malaria Screening Rate
    symptomatic = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    screened = symptomatic[symptomatic['test_type'] == 'Malaria RDT']
    kpis['malaria_screening_rate'] = (len(screened) / len(symptomatic) * 100) if len(symptomatic) > 0 else 0
    
    return kpis

@st.cache_data(ttl=3600, show_spinner="Generating forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    """Generates Prophet forecasts for multiple health metrics."""
    if df.empty or len(df) < 10: return {}
    
    # Prep data for forecasting functions
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    
    return {
        "Patient Load": generate_prophet_forecast(encounters_hist, forecast_days=forecast_days),
        "Community Risk Index": generate_prophet_forecast(avg_risk_hist, forecast_days=forecast_days),
    }

# --- UI Rendering ---
def display_alerts(df: pd.DataFrame):
    st.subheader("🚨 Daily Priority Alerts")
    alerts = generate_chw_alerts(patient_df=df)
    if not alerts:
        st.success("✅ No high-priority patient alerts for this selection.")
        return
    
    for alert in alerts:
        level, icon = ("CRITICAL", "🔴") if alert.get('alert_level') == 'CRITICAL' else (("WARNING", "🟠") if alert.get('alert_level') == 'WARNING' else ("INFO", "ℹ️"))
        with st.container(border=True):
            st.markdown(f"**{icon} {alert.get('reason')} for Pt. {alert.get('patient_id')}**")
            st.markdown(f"> {alert.get('details', 'N/A')} (Priority: {alert.get('priority', 0):.0f})")
        st.markdown("---", unsafe_allow_html=True)

def render_iot_wearable_tab(iot_df: pd.DataFrame):
    st.subheader("🌿 Environmental & Wearable Factors")
    if iot_df.empty:
        st.info("No IoT or wearable data available for this period.")
        return

    col1, col2 = st.columns(2)
    with col1:
        # Assuming CO2 data comes from static clinics
        co2_trend = iot_df.set_index('timestamp')['avg_co2_ppm'].resample('D').mean()
        fig_co2 = plot_line_chart(co2_trend, "Average Clinic CO₂ Levels (Ventilation Proxy)", "CO₂ (PPM)")
        st.plotly_chart(fig_co2, use_container_width=True)
    with col2:
        # Assuming fatigue/stress data comes from CHW wearables
        # For this demo, we'll simulate it from noise data
        fatigue_trend = iot_df.set_index('timestamp')['avg_noise_db'].resample('D').mean()
        fig_fatigue = plot_line_chart(fatigue_trend, "Team Stress/Fatigue Index (Wearable Proxy)", "Stress Index")
        st.plotly_chart(fig_fatigue, use_container_width=True)

# --- Main Page Execution ---
def main():
    st.title("📡 Field Operations Command Center")
    st.markdown("An integrated dashboard for supervising team activity, patient risk, environmental factors, and future trends.")
    st.divider()

    health_df, iot_df = get_data()

    if health_df.empty:
        st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        zone_options = ["All Zones"] + sorted(health_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        
        today = health_df['encounter_date'].max().date()
        view_date = st.date_input("Select Date for Daily Alerts:", value=today, min_value=health_df['encounter_date'].min().date(), max_value=today)
        
        st.markdown("---")
        forecast_days = st.slider("Days to Forecast Ahead:", min_value=7, max_value=90, value=14, step=7)

    # --- Filter Data ---
    daily_df = health_df[health_df['encounter_date'].dt.date == view_date]
    if selected_zone != "All Zones":
        daily_df = daily_df[daily_df['zone_id'] == selected_zone]
    
    # The full historical data for the selected zone is used for forecasting
    forecast_source_df = health_df if selected_zone == "All Zones" else health_df[health_df['zone_id'] == selected_zone]

    st.info(f"**Displaying Daily Alerts For:** `{view_date:%A, %d %b %Y}` | **Zone:** {selected_zone}")

    # --- Top-Line KPIs ---
    kpis = calculate_field_ops_kpis(daily_df)
    cols = st.columns(3)
    with cols[0]:
        render_kpi_card("Patients Seen Today", kpis.get('unique_patients_seen', 0), icon="👥")
    with cols[1]:
        render_kpi_card("High Priority Alerts", kpis.get('high_priority_alerts', 0), icon="🚨", status_level="HIGH_CONCERN" if kpis.get('high_priority_alerts', 0) > 0 else "GOOD_PERFORMANCE")
    with cols[2]:
        render_kpi_card("Malaria Screening Rate", f"{kpis.get('malaria_screening_rate', 0):.1f}%", icon="🦟", status_level="MODERATE_CONCERN" if kpis.get('malaria_screening_rate', 100) < 80 else "GOOD_PERFORMANCE")
    st.divider()

    # --- Main Content Area with Tabs ---
    tab1, tab2, tab3 = st.tabs(["**🚨 Daily Alerts**", "**🔮 AI Forecasts**", "**🛰️ IoT & Wearables**"])

    with tab1:
        display_alerts(daily_df)

    with tab2:
        st.subheader(f"Predictive Analytics ({forecast_days} Days Ahead)")
        forecasts = generate_forecasts(forecast_source_df, forecast_days)

        if not forecasts:
            st.warning("Not enough historical data in the selected zone to generate reliable forecasts.")
        else:
            fc_type = st.selectbox("Select Forecast:", options=list(forecasts.keys()))
            
            if fc_type == "Patient Load": y_axis = "Daily Encounters"
            else: y_axis = "Avg. Community Risk"

            fig = plot_forecast_chart(forecasts[fc_type], title=f"{fc_type} Forecast", y_title=y_axis)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        iot_filtered = iot_df[iot_df['timestamp'].dt.date == view_date]
        if selected_zone != "All Zones":
            iot_filtered = iot_filtered[iot_filtered['zone_id'] == selected_zone]
        render_iot_wearable_tab(iot_filtered)

if __name__ == "__main__":
    main()
