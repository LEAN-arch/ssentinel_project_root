# sentinel_project_root/pages/02_clinic_dashboard.py
# Clinic Operations & Management Console for Sentinel Health Co-Pilot.
# Definitive, functional, and self-contained version with advanced visualizations.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Dict, Any, Tuple, List, Optional
import os
import plotly.graph_objects as go
import io

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis, get_trend_data
    from analytics.orchestrator import apply_ai_models
    from analytics.supply_forecasting import generate_simple_supply_forecast
    from analytics.alerting import get_patient_alerts_for_clinic
    from visualization.plots import plot_bar_chart, plot_donut_chart, plot_annotated_line_chart
except ImportError as e:
    st.error(f"Fatal Error: A required module could not be imported.\nDetails: {e}\nThis may be due to an incorrect project structure or dependencies.")
    st.stop()


# --- Self-Contained Data Science & Visualization Logic ---

def create_sparkline_bytes(data: pd.Series, color: str) -> Optional[bytes]:
    """Creates a sparkline and returns it as PNG bytes to embed in a DataFrame."""
    if data is None or data.empty:
        return None
        
    fig = go.Figure(go.Scatter(x=data.index, y=data, mode='lines', line=dict(color=color, width=2.5)))
    fig.update_layout(
        width=150, height=50,
        margin=dict(l=0, r=0, t=5, b=5),
        xaxis=dict(visible=False, showticklabels=False),
        yaxis=dict(visible=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception as e:
        logger.warning(f"Could not generate sparkline image. Error: {e}")
        return None

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_kpi_analysis_table(full_df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Performs a period-over-period KPI analysis and generates sparklines."""
    current_period_df = full_df[full_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_days = (end_date - start_date).days + 1
    prev_start_date = start_date - timedelta(days=period_days)
    previous_period_df = full_df[full_df['encounter_date'].dt.date.between(prev_start_date, start_date - timedelta(days=1))]
    
    kpi_current = get_clinic_summary_kpis(current_period_df) if not current_period_df.empty else {}
    kpi_previous = get_clinic_summary_kpis(previous_period_df) if not previous_period_df.empty else {}
    
    kpi_defs = {
        "Avg. Test TAT (Days)": ("overall_avg_test_turnaround_conclusive_days", 'lower_is_better'),
        "Sample Rejection Rate (%)": ("sample_rejection_rate_perc", 'lower_is_better'),
        "Pending Critical Tests": ("total_pending_critical_tests_patients", 'lower_is_better'),
        "Key Drug Stockouts": ("key_drug_stockouts_count", 'lower_is_better'),
    }
    
    analysis_data = []
    trend_start_date = end_date - timedelta(days=90)
    trend_df = full_df[full_df['encounter_date'].dt.date.between(trend_start_date, end_date)]
    
    for name, (key, trend_logic) in kpi_defs.items():
        current_val = kpi_current.get(key)
        prev_val = kpi_previous.get(key)
        
        trend_series = get_trend_data(trend_df, value_col=key, period='W-MON') if not trend_df.empty and key in trend_df.columns else pd.Series()
        
        change_str = "N/A"
        delta_color = "gray"
        if pd.notna(current_val) and pd.notna(prev_val) and prev_val > 0:
            change = ((current_val - prev_val) / prev_val) * 100
            change_str = f"{change:+.1f}%"
            if trend_logic == 'lower_is_better': delta_color = "red" if change > 0 else "green"
            else: delta_color = "green" if change > 0 else "red"
        
        analysis_data.append({
            "Metric": name,
            "Current Period": f"{current_val:.1f}" if isinstance(current_val, (float, np.floating)) and pd.notna(current_val) else str(current_val if pd.notna(current_val) else 'N/A'),
            "Previous Period": f"{prev_val:.1f}" if isinstance(prev_val, (float, np.floating)) and pd.notna(prev_val) else str(prev_val if pd.notna(prev_val) else 'N/A'),
            "Change": f'<span style="color: {delta_color};">{change_str}</span>',
            "90-Day Trend": create_sparkline_bytes(trend_series, "#007BFF")
        })
        
    return pd.DataFrame(analysis_data)


# --- Page Title & Setup ---
st.title(f"🏥 {settings.APP_NAME} - Clinic Operations & Management Console")
st.markdown("**Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring**")
st.divider()

# --- Data Loading ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and processing all operational data...")
def get_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, bool, date, date]:
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    iot_available = isinstance(iot_df, pd.DataFrame) and not iot_df.empty
    min_date, max_date = date.today() - timedelta(days=365), date.today()
    if not health_df.empty and 'encounter_date' in health_df.columns:
        valid_dates = health_df['encounter_date'].dropna()
        if not valid_dates.empty: min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    ai_enriched_health_df, _ = apply_ai_models(health_df)
    return ai_enriched_health_df, iot_df, iot_available, min_date, max_date

# --- Main App Logic ---
full_health_df, full_iot_df, iot_available, abs_min_date, abs_max_date = get_dashboard_data()

# --- Sidebar ---
st.sidebar.header("Console Filters")
if os.path.exists(settings.APP_LOGO_SMALL_PATH): st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)
default_date_range_days = getattr(settings, 'WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
default_start = max(abs_min_date, abs_max_date - timedelta(days=default_date_range_days - 1))
session_key = "clinic_date_range"
if session_key not in st.session_state: st.session_state[session_key] = (default_start, abs_max_date)
start_date, end_date = st.sidebar.date_input("Select Date Range:", value=st.session_state[session_key], min_value=abs_min_date, max_value=abs_max_date)
if start_date > end_date: end_date = start_date
st.session_state[session_key] = (start_date, end_date)

# --- Filter Data for Display ---
period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if iot_available and not full_iot_df.empty else pd.DataFrame()

period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
st.info(f"**Displaying Clinic Console for:** `{period_str}`")

# --- KPI Section ---
st.header("🚀 Performance Snapshot with Trend Analysis")
if not period_health_df.empty:
    kpi_analysis_df = get_kpi_analysis_table(full_health_df, start_date, end_date)
    # Convert the 'Change' column to be rendered as HTML
    st.write(kpi_analysis_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("No data available for this period to generate KPI analysis.")
st.divider()

# --- Tabbed Section ---
st.header("🛠️ Operational Areas Deep Dive")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Epidemiology", "🔬 Testing", "💊 Supply Chain", "🧍 Patients", "🌿 Environment"])

with tab1:
    st.subheader("Local Epidemiological Intelligence")
    if period_health_df.empty: st.info("No data for epidemiological analysis.")
    else:
        st.markdown("###### **Weekly Symptom Trends (Top 5)**")
        symptoms_df = period_health_df[['encounter_date', 'patient_reported_symptoms']].dropna()
        symptoms_df = symptoms_df.assign(symptom=symptoms_df['patient_reported_symptoms'].str.split(r'[;,|]')).explode('symptom')
        symptoms_df['symptom'] = symptoms_df['symptom'].str.strip().str.title()
        top_5_symptoms = symptoms_df['symptom'].value_counts().nlargest(5).index
        symptom_trend_data = symptoms_df[symptoms_df['symptom'].isin(top_5_symptoms)]
        symptom_weekly = symptom_trend_data.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'symptom']).size().reset_index(name='count')
        if not symptom_weekly.empty:
            fig = plot_bar_chart(symptom_weekly, x_col='encounter_date', y_col='count', color='symptom', title='Weekly Encounters for Top 5 Symptoms', x_axis_title='Week', y_axis_title='Number of Encounters')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View Symptom Data Table"): st.dataframe(symptom_weekly, hide_index=True, use_container_width=True)
        else: st.info("No significant symptom data to plot for this period.")

with tab2:
    st.subheader("Testing & Diagnostics Performance")
    if period_health_df.empty: st.info("No data for testing analysis.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###### **Average Turnaround Time (TAT) vs. Target**")
            tat_df = period_health_df.groupby('test_type')['test_turnaround_days'].mean().dropna().sort_values().reset_index()
            tat_df.columns = ['Test Type', 'Avg. TAT (Days)']
            if not tat_df.empty:
                tat_df['On Target'] = tat_df['Avg. TAT (Days)'] <= settings.TARGET_TEST_TURNAROUND_DAYS
                fig = go.Figure()
                fig.add_trace(go.Bar(x=tat_df['Avg. TAT (Days)'], y=tat_df['Test Type'], orientation='h', marker_color=np.where(tat_df['On Target'], '#27AE60', '#D32F2F')))
                fig.add_vline(x=settings.TARGET_TEST_TURNAROUND_DAYS, line_width=2, line_dash="dash", line_color="black", annotation_text="Target TAT")
                fig.update_layout(title_text="<b>Average Turnaround Time (TAT) by Test</b>", yaxis={'categoryorder':'total ascending'}, xaxis_title="Average Days")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("###### **Sample Rejection Reasons**")
            rejection_df = period_health_df[period_health_df['sample_status'].str.lower() == 'rejected by lab']['rejection_reason'].value_counts().nlargest(5).reset_index()
            rejection_df.columns = ['Reason', 'Count']
            if not rejection_df.empty:
                st.plotly_chart(plot_donut_chart(rejection_df, labels_col='Reason', values_col='Count', title="Top 5 Sample Rejection Reasons"), use_container_width=True)
            else: st.info("No sample rejections in this period.")

with tab3:
    st.subheader("Medical Supply Forecast")
    forecastable_items = sorted([item for item in full_health_df['item'].dropna().unique() if any(sub in item for sub in getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', []))])
    if not forecastable_items: st.info("No forecastable supply items found in the data.")
    else:
        with st.spinner("Generating forecasts for all critical items..."):
            forecast_df = generate_simple_supply_forecast(full_health_df, item_filter=forecastable_items)
        if not forecast_df.empty:
            fig = go.Figure()
            for item in forecastable_items:
                item_data = forecast_df[forecast_df['item'] == item]
                fig.add_trace(go.Scatter(x=item_data['forecast_date'], y=item_data['forecasted_days_of_supply'], mode='lines+markers', name=item))
            fig.add_hrect(y0=0, y1=settings.CRITICAL_SUPPLY_DAYS_REMAINING, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Critical")
            fig.add_hrect(y0=settings.CRITICAL_SUPPLY_DAYS_REMAINING, y1=settings.LOW_SUPPLY_DAYS_REMAINING, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Warning")
            fig.update_layout(title_text="<b>Forecasted Days of Supply for Critical Items</b>", xaxis_title="Date", yaxis_title="Days of Supply Remaining", legend_title="Item")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Could not generate supply forecast.")

with tab4:
    st.subheader("Patient Risk & Demographics")
    if period_health_df.empty: st.info("No data for patient analysis.")
    else:
        st.markdown("###### **Patient Risk Quadrant (Age vs. AI Risk)**")
        risk_df = period_health_df[['patient_id', 'age', 'ai_risk_score']].dropna().drop_duplicates('patient_id')
        if not risk_df.empty:
            risk_df['Risk Category'] = pd.cut(risk_df['ai_risk_score'], bins=[0, 60, 75, 101], labels=['Low-Moderate', 'High', 'Very High'], right=False)
            fig = plot_bar_chart(risk_df, x_col='age', y_col='ai_risk_score', color='Risk Category', title="Patient Risk Score vs. Age", x_axis_title="Patient Age", y_axis_title="AI Risk Score", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("###### **Flagged Patients for Clinical Review**")
        flagged_patients = get_patient_alerts_for_clinic(health_df_period=period_health_df)
        if not flagged_patients.empty:
            st.dataframe(flagged_patients[['patient_id', 'age', 'gender', 'condition', 'ai_risk_score', 'Alert Reason']].head(15), use_container_width=True, hide_index=True)
        else: st.success("✅ No patients currently flagged for review.")

with tab5:
    st.subheader("Facility Environment Monitoring")
    if period_iot_df.empty: st.info("No environmental data for this period.")
    else:
        st.markdown("###### **Hourly Average CO2 Levels**")
        co2_trend = get_trend_data(period_iot_df, 'avg_co2_ppm', date_col='timestamp', period='H')
        if not co2_trend.empty:
            fig = plot_annotated_line_chart(co2_trend, "", y_axis_title="CO2 (ppm)")
            fig.add_hrect(y0=settings.ALERT_AMBIENT_CO2_HIGH_PPM, y1=settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM, fillcolor="orange", opacity=0.2, line_width=0, annotation_text="High")
            fig.add_hrect(y0=settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM, y1=co2_trend.max()*1.1 if not co2_trend.empty else 3000, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Very High")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No CO2 trend data to display.")
        st.markdown("###### **Latest Environmental Readings by Room**")
        latest_readings = period_iot_df.sort_values('timestamp', ascending=False).drop_duplicates('room_name', keep='first')
        st.dataframe(latest_readings[['room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_noise_db']], use_container_width=True, hide_index=True)
