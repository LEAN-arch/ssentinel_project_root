# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - FIELD OPERATIONS DASHBOARD (V10 - DEFINITIVE INDENTATION FIX)

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
    
    malaria_symptomatic = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    screened_for_malaria = malaria_symptomatic[malaria_symptomatic['test_type'] == 'Malaria RDT']
    kpis['malaria_screening_rate'] = (len(screened_for_malaria) / len(malaria_symptomatic) * 100) if len(malaria_symptomatic) > 0 else 0
    
    positive_tb = df[(df.get('test_type') == 'TB Screen') & (df.get('test_result') == 'Positive')]
    linked_tb = positive_tb[positive_tb.get('referral_status') == 'Completed']
    kpis['tb_linkage_rate'] = (len(linked_tb) / len(positive_tb) * 100) if len(positive_tb) > 0 else 0
    return kpis

@st.cache_data(ttl=3600, show_spinner="Generating disease forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    """
    Generates Prophet forecasts for multiple metrics.
    """
    if df.empty or len(df) < 10: return {}
    daily_encounters = df.set_index('encounter_date').resample('D').size().reset_index(name='count')
    encounters_fc = generate_prophet_forecast(daily_encounters.rename(columns={'encounter_date': 'ds', 'count': 'y'}))
    malaria_cases = df[df['diagnosis'] == 'Malaria'].set_index('encounter_date').resample('D').size().reset_index(name='count')
    malaria_fc = generate_prophet_forecast(malaria_cases.rename(columns={'encounter_date': 'ds', 'count': 'y'}))
    avg_risk = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index()
    avg_risk_fc = generate_prophet_forecast(avg_risk.rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'}))
    return {"encounters": encounters_fc, "malaria": malaria_fc, "avg_risk": avg_risk_fc}

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

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_df['encounter_date'].min().date(), full_df['encounter_date'].max().date()
        zone_options = ["All Zones"] + sorted(full_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        st.markdown("---")
        st.subheader("Retrospective Analysis")
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=6)), max_date), min_value=min_date, max_value=max_date)
        st.markdown("---")
        st.subheader("Predictive Analytics")
        forecast_days = st.slider("Days to Forecast Ahead:", min_value=7, max_value=90, value=30, step=7)

    analysis_df = full_df[full_df['encounter_date'].dt.date.between(start_date, end_date)]
    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]

    st.info(f"**Viewing Retrospective Data:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** {selected_zone}")

    kpis = calculate_field_ops_kpis(analysis_df)
    st.subheader(f"Performance Summary ({start_date:%d %b} to {end_date:%d %b})")
    
    # SME FIX: The indentation of this entire block is corrected.
    cols = st.columns(3)
    with cols[0]:
        render_kpi_card("Patients Seen (Unique)", kpis.get('unique_patients_seen', 0), icon="👥", help_text="Total unique patients with an encounter in the selected period.")
    with cols[1]:
        render_kpi_card("Malaria Screening Rate", f"{kpis.get('malaria_screening_rate', 0):.1f}%", icon="🦟", status_level="MODERATE_CONCERN" if kpis.get('malaria_screening_rate', 100) < 80 else "GOOD_PERFORMANCE", help_text="% of febrile patients tested for Malaria.")
    with cols[2]:
        render_kpi_card("TB Linkage to Care", f"{kpis.get('tb_linkage_rate', 0):.1f}%", icon="🫁", status_level="HIGH_CONCERN" if kpis.get('tb_linkage_rate', 100) < 75 else "GOOD_PERFORMANCE", help_text="% of positive TB screens linked to care.")
    
    st.divider()
    
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        daily_df_for_alerts = full_df[full_df['encounter_date'].dt.date == end_date]
        if selected_zone != "All Zones":
            daily_df_for_alerts = daily_df_for_alerts[daily_df_for_alerts['zone_id'] == selected_zone]
        display_alerts(daily_df_for_alerts)
    with col2:
        st.header(f"🔮 AI-Powered Forecasts ({forecast_days} Days Ahead)")
        forecast_df_source = full_df if selected_zone == "All Zones" else full_df[full_df['zone_id'] == selected_zone]
        forecasts = generate_forecasts(forecast_df_source, forecast_days)

        if not forecasts:
            st.warning("Not enough historical data in the selected zone to generate reliable forecasts.")
        else:
            fc_type = st.selectbox("Select Forecast:", options=list(forecasts.keys()), format_func=lambda x: x.replace('_', ' ').title())
            if fc_type == "encounters":
                title, y_axis = "Daily Patient Encounters", "Encounters"
            elif fc_type == "malaria":
                title, y_axis = "Daily Malaria Cases", "Positive Cases"
            else:
                title, y_axis = "Daily Average Patient Risk", "Avg. Risk Score"
            
            # Check if forecast data for the selected type is available and not empty
            if forecasts.get(fc_type) is not None and not forecasts[fc_type].empty:
                fig = plot_forecast_chart(forecasts[fc_type], title=title, y_title=y_axis)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No forecast data available for '{title}'. There may be insufficient historical data for this specific metric.")


if __name__ == "__main__":
    main()
