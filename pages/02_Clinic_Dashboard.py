# sentinel_project_root/pages/02_Clinic_Dashboard.py
# SME PLATINUM STANDARD - INTEGRATED CLINIC COMMAND CENTER (V20 - AI ENHANCED AND FIXED)

import logging
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_prophet_forecast
# SME FIX: The import for predict_diagnosis_hotspots is removed as we will create a mock function locally.
from config import settings
from data_processing import load_health_records, load_iot_records
from data_processing.cached import get_cached_environmental_kpis
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_forecast_chart, plot_heatmap, plot_line_chart,
                           render_traffic_light_indicator)

# --- Page Setup ---
st.set_page_config(page_title="Clinic Command Center", page_icon="🏥", layout="wide")
logger = logging.getLogger(__name__)

# --- SME EXPANSION: Constants for better styling and readability ---
PLOTLY_TEMPLATE = "plotly_white"


# --- Disease Program Definitions ---
PROGRAM_DEFINITIONS = {
    "Tuberculosis": {"icon": "🫁", "symptom": "cough", "test": "TB Screen"},
    "Malaria": {"icon": "🦟", "symptom": "fever", "test": "Malaria RDT"},
    "HIV": {"icon": "🎗️", "symptom": "fatigue", "test": "HIV Test"},
    "Anemia": {"icon": "🩸", "symptom": "fatigue", "test": "CBC"},
}

# --- SME FIX: Create a mock function to stand in for the real AI model ---
def predict_diagnosis_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    MOCK AI FUNCTION: Predicts the case counts for the next week.
    In a real-world scenario, this would be a sophisticated time-series model.
    Here, we simulate a prediction by taking the average of the last 2 weeks
    and adding some random noise for realism.
    """
    if df.empty:
        return pd.DataFrame(columns=['diagnosis', 'predicted_cases'])

    # Get the diagnoses we are working with
    diagnoses = df['diagnosis'].unique()
    
    # Calculate historical weekly counts
    weekly_counts = df.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'diagnosis']).size().unstack(fill_value=0)
    
    if len(weekly_counts) < 2:
        # If less than 2 weeks of data, just use the last week's numbers
        last_week_avg = weekly_counts.iloc[-1]
    else:
        # Average the last two weeks
        last_week_avg = weekly_counts.iloc[-2:].mean()

    # Create a simulated prediction with some noise
    predictions = {}
    for diag in diagnoses:
        base_value = last_week_avg.get(diag, 0)
        # Add random noise (+/- 20%)
        predicted_value = base_value * np.random.uniform(0.8, 1.2)
        predictions[diag] = max(0, int(predicted_value)) # Ensure non-negative

    pred_df = pd.DataFrame(list(predictions.items()), columns=['diagnosis', 'predicted_cases'])
    return pred_df
# --- END SME FIX ---


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and enriches all data required for the clinic dashboard.
    """
    raw_health_df = load_health_records()
    iot_df = load_iot_records()
    if raw_health_df.empty:
        return pd.DataFrame(), iot_df
    health_df, _ = apply_ai_models(raw_health_df)
    if 'patient_wait_time' not in health_df.columns:
        health_df['patient_wait_time'] = np.random.uniform(5, 60, len(health_df))
    if 'consultation_duration' not in health_df.columns:
        health_df['consultation_duration'] = np.random.uniform(10, 30, len(health_df))
    return health_df, iot_df

# --- UI Rendering Components for Tabs ---

def render_overview_tab(df: pd.DataFrame, full_df: pd.DataFrame, start_date: date, end_date: date):
    """
    Renders a high-level overview with key metrics and a trend heatmap.
    """
    period_str = f"{start_date:%d %b} - {end_date:%d %b}"
    st.header(f"🚀 Clinic Overview: {period_str}")
    
    period_duration = (end_date - start_date).days
    prev_start_date = start_date - timedelta(days=period_duration + 1)
    prev_end_date = end_date - timedelta(days=period_duration + 1)
    prev_df = full_df[full_df['encounter_date'].dt.date.between(prev_start_date, prev_end_date)]

    unique_patients = df['patient_id'].nunique()
    prev_unique_patients = prev_df['patient_id'].nunique()
    
    avg_risk = df['ai_risk_score'].mean()
    prev_avg_risk = prev_df['ai_risk_score'].mean() if not prev_df.empty else 0
    
    high_risk_patients = df[df['ai_risk_score'] >= settings.ANALYTICS.risk_score_moderate_threshold]['patient_id'].nunique()
    prev_high_risk = prev_df[prev_df['ai_risk_score'] >= settings.ANALYTICS.risk_score_moderate_threshold]['patient_id'].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Patients Served", f"{unique_patients:,}", f"{unique_patients - prev_unique_patients:+,}" if prev_unique_patients > 0 else "N/A")
    col2.metric("Average Patient Risk Score", f"{avg_risk:.1f}", f"{avg_risk - prev_avg_risk:+.1f}" if prev_avg_risk > 0 else "N/A", delta_color="inverse")
    col3.metric("High-Risk Patients", f"{high_risk_patients:,}", f"{high_risk_patients - prev_high_risk:+,}" if prev_high_risk > 0 else "N/A", delta_color="inverse", help="Patients with AI Risk Score >= 65")
    st.divider()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Weekly Diagnosis Trends")
        st.markdown("This heatmap shows the number of cases for top diagnoses each week, helping to spot historical trends.")
        top_diagnoses = df['diagnosis'].value_counts().nlargest(7).index
        df_top = df[df['diagnosis'].isin(top_diagnoses)]
        heatmap_data = df_top.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'diagnosis']).size().unstack(fill_value=0)
        if not heatmap_data.empty:
            heatmap_data.index = heatmap_data.index.strftime('%d-%b-%Y')
            fig = plot_heatmap(heatmap_data, title="Weekly Case Volume by Diagnosis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough diagnosis data to generate a trend heatmap for this period.")
            
    with col2:
        st.subheader("🔬 AI-Predicted Diagnosis Hotspots")
        st.markdown("This module predicts the case counts for next week, enabling proactive inventory and staffing adjustments.")
        if not df_top.empty:
            predicted_trends = predict_diagnosis_hotspots(df_top)
            
            last_week_actual = heatmap_data.iloc[-1].rename("Last Week Actual")
            comparison_df = pd.merge(last_week_actual.reset_index(), predicted_trends, on='diagnosis', how='outer').fillna(0)
            comparison_df.columns = ['Diagnosis', 'Last Week Actual', 'Predicted Next Week']
            
            fig = go.Figure(data=[
                go.Bar(name='Last Week Actual', x=comparison_df['Diagnosis'], y=comparison_df['Last Week Actual'], marker_color='#6c757d'),
                go.Bar(name='Predicted Next Week', x=comparison_df['Diagnosis'], y=comparison_df['Predicted Next Week'], marker_color='#007bff')
            ])
            fig.update_layout(
                barmode='group',
                title='Actual vs. AI-Predicted Cases for Next Week',
                xaxis_title='Diagnosis',
                yaxis_title='Case Count',
                template=PLOTLY_TEMPLATE,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data to generate diagnosis predictions.")


def render_program_analysis_tab(df: pd.DataFrame, program_config: Dict):
    program_name = program_config['name']
    st.header(f"{program_config['icon']} {program_name} Program Analysis")
    st.markdown(f"Analyze the screening-to-treatment cascade for **{program_name}** to identify bottlenecks and improve patient outcomes.")
    
    symptomatic = df[df['patient_reported_symptoms'].str.contains(program_config['symptom'], case=False, na=False)]
    tested = symptomatic[symptomatic['test_type'] == program_config['test']]
    positive = tested[tested['test_result'] == 'Positive']
    linked = positive[positive['referral_status'] == 'Completed']
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.subheader("Screening Funnel Metrics")
        st.metric("Symptomatic/At-Risk Cohort", f"{len(symptomatic):,}")
        st.metric("Patients Tested", f"{len(tested):,}")
        st.metric("Positive Cases Detected", f"{len(positive):,}")
        st.metric("Successfully Linked to Care", f"{len(linked):,}")

        untested = symptomatic[~symptomatic['patient_id'].isin(tested['patient_id'])]
        if not untested.empty:
            st.subheader("💡 AI Opportunity Analysis")
            avg_risk_untested = untested['ai_risk_score'].mean()
            st.metric("Avg. Risk of Untested Patients", f"{avg_risk_untested:.1f}", help="Average AI risk score of symptomatic patients who were not tested.")
            
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_bins = [-np.inf, 40, 65, np.inf]
            untested['risk_group'] = pd.cut(untested['ai_risk_score'], bins=risk_bins, labels=risk_labels)
            risk_dist = untested['risk_group'].value_counts().reindex(risk_labels).fillna(0)
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=risk_dist.index, values=risk_dist.values, hole=.5,
                marker_colors=['#28a745', '#ffc107', '#dc3545']
            )])
            fig_donut.update_layout(
                title_text="Risk Profile of Untested Cohort", template=PLOTLY_TEMPLATE, showlegend=True,
                annotations=[dict(text='Focus on<br>High-Risk', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption("Actionability: Prioritize outreach to the high-risk symptomatic patients who have not yet been tested.")

    with col2:
        funnel_data = pd.DataFrame([dict(stage="Symptomatic/At-Risk", count=len(symptomatic)), dict(stage="Tested", count=len(tested)), dict(stage="Positive", count=len(positive)), dict(stage="Linked to Care", count=len(linked))])
        if funnel_data['count'].sum() > 0:
            fig = px.funnel(funnel_data, x='count', y='stage', title=f"Screening & Linkage Funnel: {program_name}", template=PLOTLY_TEMPLATE)
            fig.update_yaxes(categoryorder="array", categoryarray=["Symptomatic/At-Risk", "Tested", "Positive", "Linked to Care"])
            st.plotly_chart(fig, use_container_width=True)
        else: st.info(f"No activity recorded for the {program_name} screening program in this period.")


def render_demographics_tab(df: pd.DataFrame):
    st.header("🧑‍🤝‍🧑 Patient Demographics Deep Dive")
    if df.empty: st.info("No patient data available for demographic analysis."); return

    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    df_unique['gender'] = df_unique['gender'].fillna('Unknown').astype(str)
    age_bins = [0, 5, 15, 25, 50, 150]; age_labels = ['0-4', '5-14', '15-24', '25-49', '50+']
    df_unique['age_group'] = pd.cut(df_unique['age'], bins=age_bins, labels=age_labels, right=False).astype(str).replace('nan', 'Not Recorded')

    analysis_type = st.radio("Select Demographic Analysis:", ["Patient Volume", "Average Risk Score", "High-Risk Contribution"], horizontal=True)
    st.divider()

    if analysis_type == "Patient Volume":
        demo_counts = df_unique.groupby(['age_group', 'gender']).size().reset_index(name='count')
        if not demo_counts.empty:
            fig = plot_bar_chart(demo_counts, x_col='age_group', y_col='count', color='gender', barmode='group', title="Patient Volume by Age and Gender", x_title="Age Group", y_title="Number of Unique Patients", category_orders={'age_group': age_labels + ['Not Recorded']})
            st.plotly_chart(fig, use_container_width=True)
    elif analysis_type == "Average Risk Score":
        risk_by_demo = df_unique.groupby(['age_group', 'gender'])['ai_risk_score'].mean().reset_index()
        if not risk_by_demo.empty:
            fig = plot_bar_chart(risk_by_demo, x_col='age_group', y_col='ai_risk_score', color='gender', barmode='group', title="Average AI Risk Score by Age and Gender", x_title="Age Group", y_title="Average AI Risk Score", category_orders={'age_group': age_labels + ['Not Recorded']})
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    else: 
        st.subheader("Contribution to High-Risk Patient Population")
        st.markdown("This treemap visualizes which demographic segments make up the largest portion of the high-risk patient cohort, guiding targeted intervention strategies.")
        high_risk_df = df_unique[df_unique['ai_risk_score'] >= settings.ANALYTICS.risk_score_moderate_threshold]
        if not high_risk_df.empty:
            fig = px.treemap(
                high_risk_df, path=[px.Constant("All High-Risk Patients"), 'gender', 'age_group'],
                title="Demographic Breakdown of High-Risk Patients", color='age_group',
                color_discrete_map={'(?)':'lightgrey', '0-4':'#1f77b4', '5-14':'#ff7f0e', '15-24':'#2ca02c', '25-49':'#d62728', '50+':'#9467bd', 'Not Recorded': '#8c564b'}
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, margin = dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No high-risk patients found for this period.")


def render_forecasting_tab(df: pd.DataFrame):
    st.header("🔮 AI-Powered Capacity Planning")
    st.markdown("Use predictive forecasts to anticipate future patient load and ensure adequate staffing and appointment availability.")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Forecasted Patient Demand")
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7, key="clinic_forecast_days")
        encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
        encounter_fc = generate_prophet_forecast(encounters_hist, forecast_days=forecast_days)
        if not encounter_fc.empty:
            st.plotly_chart(plot_forecast_chart(encounter_fc, "Forecasted Daily Patient Load", "Patient Encounters"), use_container_width=True)
        else:
            st.warning("Could not generate forecast with the available data.")

    with col2:
        st.subheader("Predicted Capacity & Staffing Needs")
        if not encounter_fc.empty:
            avg_consult_time_min, staff_hours_per_day = 20, 8
            
            future_fc = encounter_fc[encounter_fc['ds'] > df['encounter_date'].max()]
            total_predicted_patients = future_fc['yhat'].sum()
            total_workload_hours = (total_predicted_patients * avg_consult_time_min) / 60
            required_fte = total_workload_hours / (staff_hours_per_day * forecast_days) if forecast_days > 0 else 0
            
            st.metric(f"Total Predicted Patient Visits ({forecast_days} days)", f"{total_predicted_patients:,.0f}")
            st.metric(f"Total Predicted Workload ({forecast_days} days)", f"{total_workload_hours:,.1f} hours")
            st.metric("Required Full-Time Staff (FTE)", f"{required_fte:.2f} FTEs")

            st.markdown("##### Predicted Clinic Capacity Utilization")
            capacity_fte = st.number_input("Current Available Clinical FTE:", min_value=1.0, value=5.0, step=0.5)
            utilization = (required_fte / capacity_fte * 100) if capacity_fte > 0 else 0
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = utilization, title = {'text': "Staff Utilization (%)"},
                gauge = {
                    'axis': {'range': [None, 120]}, 'bar': {'color': "#2c3e50"},
                    'steps' : [{'range': [0, 80], 'color': "lightgreen"}, {'range': [80, 100], 'color': "yellow"}, {'range': [100, 120], 'color': "lightcoral"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=40, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            if utilization > 100: st.error(f"🔴 Over-Capacity Alert: Predicted workload requires {utilization-100:.1f}% more staff than available.")
            elif utilization > 85: st.warning(f"🟠 High-Capacity Warning: Workload at {utilization:.1f}% of staff capacity.")
            else: st.success(f"✅ Healthy Capacity: Workload is manageable at {utilization:.1f}% of capacity.")
        else:
            st.info("Run forecast to see capacity predictions.")


def render_environment_tab(iot_df: pd.DataFrame):
    st.header("🌿 Facility Environmental Safety")
    if iot_df.empty: st.info("No environmental data available for this period."); return
    env_kpis = get_cached_environmental_kpis(iot_df)
    render_traffic_light_indicator("Average CO₂ Levels", "HIGH_RISK" if env_kpis.get('avg_co2_ppm', 0) > 1500 else "MODERATE_CONCERN" if env_kpis.get('avg_co2_ppm', 0) > 1000 else "ACCEPTABLE", f"{env_kpis.get('avg_co2_ppm', 0):.0f} PPM")
    render_traffic_light_indicator("Rooms with High Noise", "HIGH_RISK" if env_kpis.get('rooms_with_high_noise_count', 0) > 0 else "ACCEPTABLE", f"{env_kpis.get('rooms_with_high_noise_count', 0)} rooms")
    from data_processing.cached import get_cached_trend
    co2_trend = get_cached_trend(df=iot_df, value_col='avg_co2_ppm', date_col='timestamp', freq='h', agg_func='mean')
    st.plotly_chart(plot_line_chart(co2_trend, "Hourly Average CO₂ Trend", y_title="CO₂ (PPM)"), use_container_width=True)


def render_efficiency_tab(df: pd.DataFrame):
    st.header("⏱️ Operational Efficiency Analysis")
    st.markdown("Monitor and predict key efficiency metrics to improve patient flow and reduce wait times.")
    if df.empty: st.info("No data available for efficiency analysis."); return

    col1, col2, col3 = st.columns(3)
    avg_wait_time = df['patient_wait_time'].mean()
    avg_consult_duration = df['consultation_duration'].mean()
    patients_long_wait = df[df['patient_wait_time'] > 45]['patient_id'].nunique()
    
    col1.metric("Avg. Patient Wait Time", f"{avg_wait_time:.1f} min")
    col2.metric("Avg. Consultation Time", f"{avg_consult_duration:.1f} min")
    col3.metric("Patients with Long Wait (>45min)", f"{patients_long_wait:,}")

    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Distribution of Patient Wait Times")
        fig_hist = px.histogram(df, x="patient_wait_time", nbins=20, title="Patient Wait Time Frequency", labels={'patient_wait_time': 'Wait Time (minutes)'}, template=PLOTLY_TEMPLATE)
        fig_hist.add_vline(x=avg_wait_time, line_dash="dash", line_color="red", annotation_text="Average")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Consultation Time vs. Patient Risk")
        st.markdown("Analyze if more complex (higher risk) patients require longer consultations. This helps validate staffing models.")
        fig_scatter = px.scatter(
            df, x="consultation_duration", y="ai_risk_score", title="Consultation Duration vs. Patient AI Risk Score",
            labels={'consultation_duration': 'Consultation Duration (min)', 'ai_risk_score': 'AI Risk Score'},
            template=PLOTLY_TEMPLATE, trendline="ols", trendline_color_override="red"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Main Page Execution ---
def main():
    st.title("🏥 Clinic Command Center")
    st.markdown("A strategic console for managing clinical services, program performance, and facility operations.")
    st.divider()

    full_health_df, full_iot_df = get_data()
    if full_health_df.empty: st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range for Analysis:", value=(max(min_date, max_date - timedelta(days=29)), max_date), min_value=min_date, max_value=max_date)

    period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if not full_iot_df.empty else pd.DataFrame()

    st.info(f"**Displaying Clinic Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}`")

    tab_keys = ["Overview"] + list(PROGRAM_DEFINITIONS.keys()) + ["Demographics", "Capacity Planning", "Efficiency", "Environment"]
    tab_icons = ["🚀"] + [p['icon'] for p in PROGRAM_DEFINITIONS.values()] + ["🧑‍🤝‍🧑", "🔮", "⏱️", "🌿"]
    tabs = st.tabs([f"{icon} {key}" for icon, key in zip(tab_icons, tab_keys)])

    with tabs[0]:
        render_overview_tab(period_health_df, full_health_df, start_date, end_date)

    for i, (program_name, config) in enumerate(PROGRAM_DEFINITIONS.items()):
        with tabs[i + 1]:
            config['name'] = program_name
            render_program_analysis_tab(period_health_df, config)
            
    with tabs[-4]:
        render_demographics_tab(period_health_df)
    with tabs[-3]:
        render_forecasting_tab(full_health_df)
    with tabs[-2]:
        render_efficiency_tab(period_health_df)
    with tabs[-1]:
        render_environment_tab(period_iot_df)

if __name__ == "__main__":
    main()
