# sentinel_project_root/pages/02_Clinic_Dashboard.py
# SME PLATINUM STANDARD - INTEGRATED CLINIC COMMAND CENTER (V15 - FINAL FIX)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_kpi_analysis_table, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from data_processing.cached import get_cached_environmental_kpis
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_forecast_chart, plot_line_chart,
                           render_traffic_light_indicator)

# --- Page Setup ---
st.set_page_config(page_title="Clinic Command Center", page_icon="🏥", layout="wide")
logger = logging.getLogger(__name__)


# --- Disease Program Definitions ---
PROGRAM_DEFINITIONS = {
    "Tuberculosis": {"icon": "🫁", "symptom": "cough", "test": "TB Screen"},
    "Malaria": {"icon": "🦟", "symptom": "fever", "test": "Malaria RDT"},
    "HIV": {"icon": "🎗️", "symptom": "fatigue", "test": "HIV Test"},
    "Anemia": {"icon": "🩸", "symptom": "fatigue", "test": "CBC"},
}


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and enriches all data required for the clinic dashboard."""
    raw_health_df = load_health_records()
    iot_df = load_iot_records()
    if raw_health_df.empty:
        return pd.DataFrame(), iot_df
    health_df, _ = apply_ai_models(raw_health_df)
    return health_df, iot_df

# --- UI Rendering Components for Tabs ---

def render_program_analysis_tab(df: pd.DataFrame, program_config: Dict):
    """Renders a comprehensive analysis tab for a specific disease program."""
    program_name = program_config['name']
    st.header(f"{program_config['icon']} {program_name} Program Analysis")
    st.markdown(f"Analyze the screening-to-treatment cascade for **{program_name}** to identify bottlenecks and improve patient outcomes.")
    
    symptomatic = df[df['patient_reported_symptoms'].str.contains(program_config['symptom'], case=False, na=False)]
    tested = symptomatic[symptomatic['test_type'] == program_config['test']]
    positive = tested[tested['test_result'] == 'Positive']
    linked = positive[positive['referral_status'] == 'Completed']
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Screening Funnel Metrics")
        st.metric("Symptomatic/At-Risk Cohort", f"{len(sympttest']]
    positive = tested[tested['test_result'] == 'Positive']
    linked = positive[positive['referral_status'] == 'Completed']
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Screening Funnel Metrics")
        st.metric("Symptomatic/At-Risk Cohort", f"{len(symptomatic):,}")
        st.metric("Patients Tested", f"{len(tested):,}")
        st.metric("Positive Cases Detected", f"{len(positive):,}")
        st.metric("Successfully Linked to Care", f"{len(linked):,}")

    with col2:
        funnel_data = pd.DataFrame([
            dict(stage="Symptomatic/At-Risk", count=len(symptomatic)),
            dict(stage="Tested", count=len(tested)),
            dict(stage="Positive", count=len(positive)),
            dict(stage="Linked to Care", count=len(linked)),
        ])
        if funnel_data['count'].sum() > 0:
            fig = px.funnel(funnel_data, x='count', y='stage', title=f"Screening & Linkage Funnel: {program_name}")
            fig.update_yaxes(categoryorder="array", categoryarray=["Symptomatic/At-Risk", "Tested", "Positive", "Linked to Care"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No activity recorded for the {program_name} screening program in this period.")

def render_demographics_tab(df: pd.DataFrame):
    """Renders the patient demographics analysis tab."""
    st.header("🧑‍🤝‍🧑 Patient Demographics")
    if df.empty:
        st.info("No patient data for demographic analysis."); return

    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    
    # SME FIX: Explicitly sanitize the 'gender' column before any operations.
    # This prevents errors if the column contains NaNs after filtering.
    df_unique['gender'] = df_unique['gender'].fillna('Unknown').astype(str)
    
    age_bins = [0, 5, 15, 25, 50, 150]
    age_labels = ['0-4', '5-14', '15-24', '25-49', '50+']
    df_unique['age_group'] = pd.cut(df_unique['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

    demo_counts = df_unique.groupby(['age_group', 'gender'], observed=False).size().reset_index(name='count')
    
    if not demo_counts.empty:
        fig = plot_bar_chart(
            demo_counts, x_col='age_group', y_col='count', color='gender',
            barmode='group', title="Patient Encounters by Age and Gender",
            x_title="Age Group", y_title="Number of Unique Patients",
            category_orders={'age_group': age_labels}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No demographic data to display for the selected filters.")

def render_forecasting_tab(df: pd.DataFrame):
    """Renders the AI-powered forecasting tab."""
    st.header("🔮 AI-Powered Forecasts")
    st.info("These forecasts use historical data to predict future trends, helping with resource and staff planning.")
    
    forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7, key="clinic_forecast_days")
    
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    
    encounter_fc = generate_prophet_forecast(encounters_hist, forecast_days=forecast_days)
    risk_fc = generate_prophet_forecast(avg_risk_hist, forecast_days=forecast_days)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_forecast_chart(encounter_fc, "Forecasted Daily Patient Load", "Patient Encounters"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_forecast_chart(risk_fc, "Forecasted Community Risk Index", "Average Patient Risk Score"), use_container_width=True)

def render_environment_tab(iot_df: pd.DataFrame):
    omatic):,}")
        st.metric("Patients Tested", f"{len(tested):,}")
        st.metric("Positive Cases Detected", f"{len(positive):,}")
        st.metric("Successfully Linked to Care", f"{len(linked):,}")

    with col2:
        funnel_data = pd.DataFrame([
            dict(stage="Symptomatic/At-Risk", count=len(symptomatic)),
            dict(stage="Tested", count=len(tested)),
            dict(stage="Positive", count=len(positive)),
            dict(stage="Linked to Care", count=len(linked)),
        ])
        if funnel_data['count'].sum() > 0:
            fig = px.funnel(funnel_data, x='count', y='stage', title=f"Screening & Linkage Funnel: {program_name}")
            fig.update_yaxes(categoryorder="array", categoryarray=["Symptomatic/At-Risk", "Tested", "Positive", "Linked to Care"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No activity recorded for the {program_name} screening program in this period.")

def render_demographics_tab(df: pd.DataFrame):
    """Renders the patient demographics analysis tab."""
    st.header("🧑‍🤝‍🧑 Patient Demographics")
    if df.empty:
        st.info("No patient data for demographic analysis."); return

    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    
    # SME FIX: Explicitly sanitize the 'gender' column before any operations.
    # This prevents errors if the column contains NaNs after filtering.
    df_unique['gender'] = df_unique['gender'].fillna('Unknown').astype(str)
    
    age_bins = [0, 5, 15, 25, 50, 150]
    age_labels = ['0-4', '5-14', '15-24', '25-49', '50+']
    df_unique['age_group'] = pd.cut(df_unique['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

    demo_counts = df_unique.groupby(['age_group', 'gender'], observed=False).size().reset_index(name='count')
    
    if not demo_counts.empty:
        fig = plot_bar_chart(
            demo_counts, x_col='age_group', y_col='count', color='gender',
            barmode='group', title="Patient Encounters by Age and Gender",
            x_title="Age Group", y_title="Number of Unique Patients",
            category_orders={'age_group': age_labels}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No demographic data to display for the selected filters.")

def render_forecasting_tab(df: pd.DataFrame):
    # ... [This function is correct and remains unchanged] ...
    st.header("🔮 AI-Powered Forecasts")
    st.info("These forecasts use historical data to predict future trends, helping with resource and staff planning.")
    forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7, key="clinic_forecast_days")
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    encounter_fc = generate_prophet_forecast(encounters_hist, forecast_days=forecast_days)
    risk_fc = generate_prophet_forecast(avg_risk_hist, forecast_days=forecast_days)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_forecast_chart(encounter_fc, "Forecasted Daily Patient Load", "Patient Encounters"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_forecast_chart(risk_fc, "Forecasted Community Risk Index", "Average Patient Risk Score"), use_container_width=True)

def render_environment_tab(iot_df: pd.DataFrame):
    # ... [This function is correct and remains unchanged] ...
    st.header("🌿 Facility Environmental Safety")
    if iot_df.empty:
        st.info("No environmental data available for this period."); return
    env_kpis = get_cached_environmental_kpis(iot_df)
    render_traffic_light_indicator("Average CO₂ Levels", "HIGH_RISK" if env_kpis.get('avg_co2_ppm', 0) > 1500 else "MODERATE_CONCERN" if env_kpis.get('avg_co2_ppm', 0) > 1000 else "ACCEPTABLE", f"{env_kpis.get('avg_co2_ppm', 0):.0f} PPM")
    render_traffic_light_indicator("Rooms with High Noise", "HIGH_RISK" if env_kpis.get('rooms_with_high_noise_count', 0) > 0 else "ACCEPTABLE", f"{env_kpis.get('rooms_with_high_noise_count', 0)} rooms")
    from data_processing.cached import get_cached_trend
    co2_trend = get_cached_trend(df=iot_df, value_col='avg_co2_ppm', date_col='timestamp', freq='h', agg_func='mean')
    st.plotly_chart(plot_line_chart(co2_trend, "Hourly Average CO₂ Trend", y_title="CO₂ (PPM)"), use_container_width=True)

# --- Main Page Execution ---
def main():
    st.title("🏥 Clinic Command Center")
    st.markdown("A strategic console for managing clinical services, program performance, and facility operations.")
    st.divider()

    full_health_df, full_iot_df = get_data()
    if full_health_df.empty:
        st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range for Analysis:", value=(max(min_date, max_date - timedelta(days=29)), max_date), min_value=min_date, max_value=max_date)

    period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if not full_iot_df.empty else pd.DataFrame()

    st.info(f"**Displaying Clinic Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}`")

    tab_keys = ["🚀 Snapshot"] + list(PROGRAM_DEFINITIONS.keys()) + ["Demographics", "Forecasting", "Environment"]
    tab_icons = [""] + [p['icon'] for p in PROGRAM_DEFINITIONS.values()] + ["🧑‍🤝‍🧑", "🔮", "🌿"]
    tabs = st.tabs([f"{icon} {key}" for icon, key in zip(tab_icons, tab_keys)])

    with tabs[0]:
        st.header("Operational Performance Snapshot")
        st.markdown("Period-over-period analysis of key testing and supply chain metrics.")
        kpi_analysis_df = generate_kpi_analysis_table(full_health_df, start_date, end_date)
        st.dataframe(kpi_analysis_df, hide_index=True, use_container_width=True, column_config={"Current": st.column_config.NumberColumn(format="%.1f"), "Previous": st.column_config.NumberColumn(format="%.1f"), "Trend (90d)": st.column_config.ImageColumn(label="90-Day Trend")})

    for i, (program_name, config) in enumerate(PROGRAM_DEFINITIONS.items()):
        with tabs[i + 1]:
            config['name'] = program_name
            render_program_analysis_tab(period_health_df, config)
            
    with tabs[-3]:
        render_demographics_tab(period_health_df)
    with tabs[-2]:
        render_forecasting_tab(full_health_df)
    with tabs[-1]:
        render_environment_tab(period_iot_df)

if __name__ == "__main__":
    main()
```"""Renders the environmental monitoring tab."""
    st.header("🌿 Facility Environmental Safety")
    if iot_df.empty:
        st.info("No environmental data available for this period."); return
        
    env_kpis = get_cached_environmental_kpis(iot_df)
    render_traffic_light_indicator("Average CO₂ Levels", "HIGH_RISK" if env_kpis.get('avg_co2_ppm', 0) > 1500 else "MODERATE_CONCERN" if env_kpis.get('avg_co2_ppm', 0) > 1000 else "ACCEPTABLE", f"{env_kpis.get('avg_co2_ppm', 0):.0f} PPM")
    render_traffic_light_indicator("Rooms with High Noise", "HIGH_RISK" if env_kpis.get('rooms_with_high_noise_count', 0) > 0 else "ACCEPTABLE", f"{env_kpis.get('rooms_with_high_noise_count', 0)} rooms")
    
    from data_processing.cached import get_cached_trend
    co2_trend = get_cached_trend(df=iot_df, value_col='avg_co2_ppm', date_col='timestamp', freq='h', agg_func='mean')
    st.plotly_chart(plot_line_chart(co2_trend, "Hourly Average CO₂ Trend", y_title="CO₂ (PPM)"), use_container_width=True)


# --- Main Page Execution ---
def main():
    st.title("🏥 Clinic Command Center")
    st.markdown("A strategic console for managing clinical services, program performance, and facility operations.")
    st.divider()

    full_health_df, full_iot_df = get_data()
    if full_health_df.empty:
        st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range for Analysis:", value=(max(min_date, max_date - timedelta(days=29)), max_date), min_value=min_date, max_value=max_date)

    period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if not full_iot_df.empty else pd.DataFrame()

    st.info(f"**Displaying Clinic Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}`")

    tab_keys = ["🚀 Snapshot"] + list(PROGRAM_DEFINITIONS.keys()) + ["🧑‍🤝‍🧑 Demographics", "🔮 Forecasting", "🌿 Environment"]
    tab_icons = ["🚀"] + [p['icon'] for p in PROGRAM_DEFINITIONS.values()] + ["🧑‍🤝‍🧑", "🔮", "🌿"]
    tabs = st.tabs([f"{icon} {key}" for icon, key in zip(tab_icons, tab_keys)])

    with tabs[0]:
        st.header("Operational Performance Snapshot")
        st.markdown("Period-over-period analysis of key testing and supply chain metrics.")
        kpi_analysis_df = generate_kpi_analysis_table(full_health_df, start_date, end_date)
        st.dataframe(kpi_analysis_df, hide_index=True, use_container_width=True, column_config={"Current": st.column_config.NumberColumn(format="%.1f"), "Previous": st.column_config.NumberColumn(format="%.1f"), "Trend (90d)": st.column_config.ImageColumn(label="90-Day Trend")})

    for i, (program_name, config) in enumerate(PROGRAM_DEFINITIONS.items()):
        with tabs[i + 1]:
            config['name'] = program_name
            render_program_analysis_tab(period_health_df, config)
            
    with tabs[-3]:
        render_demographics_tab(period_health_df)
    with tabs[-2]:
        render_forecasting_tab(full_health_df)
    with tabs[-1]:
        render_environment_tab(period_iot_df)

if __name__ == "__main__":
    main()
