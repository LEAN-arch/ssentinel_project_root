# sentinel_project_root/pages/02_Clinic_Dashboard.py
# SME PLATINUM STANDARD - CLINIC DASHBOARD (V7 - FINAL FIX)

import logging
from datetime import date, timedelta
import pandas as pd
import streamlit as st

from analytics import apply_ai_models, generate_kpi_analysis_table
from config import settings
from data_processing import load_health_records, load_iot_records
from data_processing.cached import get_cached_environmental_kpis, get_cached_trend
from visualization import create_empty_figure, plot_bar_chart, plot_line_chart, render_traffic_light_indicator

st.set_page_config(page_title="Clinic Dashboard", page_icon="🏥", layout="wide")
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600, show_spinner="Loading operational data...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_health_df = load_health_records()
    health_df, _ = apply_ai_models(raw_health_df)
    iot_df = load_iot_records()
    return health_df, iot_df

def render_epidemiology_tab(df: pd.DataFrame):
    st.subheader("Top 5 Diagnoses by Volume")
    if df.empty:
        st.info("No encounter data for this period.")
        return
    top_diagnoses = df['diagnosis'].value_counts().nlargest(5).reset_index()
    top_diagnoses.columns = ['diagnosis', 'count']
    fig = plot_bar_chart(top_diagnoses, x_col='diagnosis', y_col='count', title="Top 5 Diagnoses")
    st.plotly_chart(fig, use_container_width=True)

def render_environment_tab(df: pd.DataFrame):
    st.subheader("Facility Environmental Quality")
    if df.empty:
        st.info("No environmental data for this period.")
        return
    env_kpis = get_cached_environmental_kpis(df)
    render_traffic_light_indicator("Average CO₂ Levels", "HIGH_RISK" if env_kpis.get('avg_co2_ppm', 0) > 1500 else "MODERATE_CONCERN" if env_kpis.get('avg_co2_ppm', 0) > 1000 else "ACCEPTABLE", f"{env_kpis.get('avg_co2_ppm', 0):.0f} PPM")
    render_traffic_light_indicator("Rooms with High Noise", "HIGH_RISK" if env_kpis.get('rooms_with_high_noise_count', 0) > 0 else "ACCEPTABLE", f"{env_kpis.get('rooms_with_high_noise_count', 0)} rooms")
    co2_trend = get_cached_trend(df=df, value_col='avg_co2_ppm', date_col='timestamp', freq='h', agg_func='mean')
    st.plotly_chart(plot_line_chart(co2_trend, title="Hourly Average CO₂ Trend", y_title="CO₂ (PPM)"), use_container_width=True)

def main():
    st.title("🏥 Clinic Operations & Management Console")
    st.markdown("Monitor service efficiency, care quality, resource management, and facility safety.")
    st.divider()

    full_health_df, full_iot_df = get_data()
    if full_health_df.empty:
        st.error("No health data available. Dashboard cannot be rendered.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=29)), max_date), min_value=min_date, max_value=max_date)

    period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if not full_iot_df.empty else pd.DataFrame()

    st.info(f"**Displaying Clinic Console for:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}`")

    st.header("🚀 Performance Snapshot with Trend Analysis")
    if not period_health_df.empty:
        kpi_analysis_df = generate_kpi_analysis_table(full_health_df, start_date, end_date)
        st.dataframe(kpi_analysis_df, hide_index=True, use_container_width=True, column_config={"Current": st.column_config.NumberColumn(format="%.1f"), "Previous": st.column_config.NumberColumn(format="%.1f"), "Trend (90d)": st.column_config.ImageColumn(label="90-Day Trend")})
    else:
        st.info("No encounter data available to generate KPI analysis for this period.")
    st.divider()

    st.header("🛠️ Operational Areas Deep Dive")
    tabs = st.tabs(["📈 Epidemiology", "🌿 Environment"])
    with tabs[0]: render_epidemiology_tab(period_health_df)
    with tabs[1]: render_environment_tab(period_iot_df)

if __name__ == "__main__":
    main()
