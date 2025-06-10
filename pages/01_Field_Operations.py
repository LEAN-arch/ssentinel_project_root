# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - INTEGRATED FIELD COMMAND CENTER (V18 - FINAL)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from data_processing.cached import get_cached_trend
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
@st.cache_data
def get_kpis_for_period(df: pd.DataFrame) -> Dict:
    if df.empty: return {}
    kpis = {'Patients Seen': df['patient_id'].nunique()}
    symptomatic_malaria = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    tested_malaria = symptomatic_malaria[symptomatic_malaria['test_type'] == 'Malaria RDT']
    kpis['Malaria Screening Rate'] = (len(tested_malaria) / len(symptomatic_malaria) * 100) if len(symptomatic_malaria) > 0 else 0
    positive_tb = df[(df.get('test_type') == 'TB Screen') & (df.get('test_result') == 'Positive')]
    linked_tb = positive_tb[positive_tb.get('referral_status') == 'Completed']
    kpis['TB Linkage to Care'] = (len(linked_tb) / len(positive_tb) * 100) if len(positive_tb) > 0 else 100
    return kpis

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

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    if df.empty or len(df) < 10: return {}
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    return {"Patient Load": generate_prophet_forecast(encounters_hist, forecast_days=forecast_days), "Community Risk Index": generate_prophet_forecast(avg_risk_hist, forecast_days=forecast_days)}

def render_iot_wearable_tab(iot_df: pd.DataFrame, chw_id_filter: str):
    """Renders the IoT and Wearable data visualization tab."""
    st.header("🛰️ Environmental & Team Factors")
    if iot_df.empty:
        st.info("No IoT or wearable data available for this period."); return
    
    col1, col2 = st.columns(2)
    with col1:
        clinic_iot = iot_df.dropna(subset=['avg_co2_ppm'])
        if not clinic_iot.empty:
            co2_trend = clinic_iot.set_index('timestamp')['avg_co2_ppm'].resample('D').mean()
            fig_co2 = plot_line_chart(co2_trend, "Average Clinic CO₂ (Ventilation Proxy)", "CO₂ PPM")
            st.plotly_chart(fig_co2, use_container_width=True)
        else:
            st.info("No clinic environmental sensor data for this period.")
    with col2:
        wearable_iot = iot_df.dropna(subset=['chw_stress_score'])
        if chw_id_filter != "All CHWs":
            wearable_iot = wearable_iot[wearable_iot['chw_id'] == chw_id_filter]
        
        if not wearable_iot.empty:
            stress_trend = wearable_iot.set_index('timestamp')['chw_stress_score'].resample('D').mean()
            fig_stress = plot_line_chart(stress_trend, f"Average Stress Index for {chw_id_filter}", "Stress Index (0-100)")
            st.plotly_chart(fig_stress, use_container_width=True)
        else:
            st.info(f"No wearable data available for {chw_id_filter} in this period.")


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
    iot_filtered = iot_df[iot_df['timestamp'].dt.date.between(start_date, end_date)] if not iot_df.empty else pd.DataFrame()

    # Apply CHW and Zone filters to all relevant dataframes
    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]
        forecast_source_df = forecast_source_df[forecast_source_df['zone_id'] == selected_zone]
        iot_filtered = iot_filtered[iot_filtered['zone_id'] == selected_zone]
    if selected_chw != "All CHWs":
        analysis_df = analysis_df[analysis_df['chw_id'] == selected_chw]
        forecast_source_df = forecast_source_df[forecast_source_df['chw_id'] == selected_chw]
        # SME FIX: The CHW filter must also be applied to the IoT/Wearable data.
        iot_filtered = iot_filtered[iot_filtered['chw_id'] == selected_chw]

    st.info(f"**Displaying Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** `{selected_zone}` | **CHW:** `{selected_chw}`")
    st.divider()

    # --- Situation Report Header ---
    st.header("Situation Report")
    kpis_current = get_kpis_for_period(analysis_df)
    prev_start = start_date - (end_date - start_date + timedelta(days=1))
    kpis_previous = get_kpis_for_period(health_df[health_df['encounter_date'].dt.date.between(prev_start, start_date - timedelta(days=1))])
    
    sit_rep_cols = st.columns(3)
    with sit_rep_cols[0]: st.metric("Patients Seen", f"{kpis_current.get('Patients Seen', 0):,}", f"{kpis_current.get('Patients Seen', 0) - kpis_previous.get('Patients Seen', 0):+d} vs. prior period")
    with sit_rep_cols[1]: st.metric("Malaria Screening Rate", f"{kpis_current.get('Malaria Screening Rate', 0):.1f}%", f"{kpis_current.get('Malaria Screening Rate', 0) - kpis_previous.get('Malaria Screening Rate', 0):+.1f} pts")
    with sit_rep_cols[2]: st.metric("TB Linkage to Care", f"{kpis_current.get('TB Linkage to Care', 0):.1f}%", f"{kpis_current.get('TB Linkage to Care', 0) - kpis_previous.get('TB Linkage to Care', 0):+.1f} pts")
    st.divider()

    # --- Main Content Area ---
    tab1, tab2, tab3 = st.tabs(["**🚨 Daily Alerts**", "**🔮 AI Forecasts**", "**🛰️ IoT & Wearables**"])

    with tab1:
        daily_alerts_df = analysis_df[analysis_df['encounter_date'].dt.date == end_date]
        display_alerts(daily_alerts_df)
    with tab2:
        st.subheader(f"Predictive Analytics ({forecast_days} Days Ahead)")
        forecasts = generate_forecasts(forecast_source_df, forecast_days)
        if not forecasts:
            st.warning("Not enough historical data for the selected filters to generate reliable forecasts.")
        else:
            fc_col1, fc_col2 = st.columns(2)
            with fc_col1: st.plotly_chart(plot_forecast_chart(forecasts['Patient Load'], title="Forecasted Patient Load", y_title="Daily Encounters"), use_container_width=True)
            with fc_col2: st.plotly_chart(plot_forecast_chart(forecasts['Community Risk Index'], title="Forecasted Community Risk", y_title="Average AI Risk Score"), use_container_width=True)
    with tab3:
        render_iot_wearable_tab(iot_filtered, selected_chw)

if __name__ == "__main__":
    main()
