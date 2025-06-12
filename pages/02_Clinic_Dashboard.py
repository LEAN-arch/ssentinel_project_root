# sentinel_project_root/pages/02_Clinic_Dashboard.py
# SME PLATINUM STANDARD - INTEGRATED CLINIC COMMAND CENTER (V23 - POPULATION HEALTH INTELLIGENCE)

import logging
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Core Sentinel Imports ---
# Assumed to exist and work as described
from analytics import apply_ai_models, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import create_empty_figure

# --- Page Setup ---
st.set_page_config(page_title="Clinic Command Center", page_icon="🏥", layout="wide")
logging.getLogger("prophet").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- SME VISUALIZATION & KPI UPGRADE: Constants ---
PLOTLY_TEMPLATE = "plotly_white"
GENDER_COLORS = {"Female": "#E1396C", "Male": "#1f77b4", "Unknown": "#7f7f7f"}
RISK_COLORS = {'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'}

PROGRAM_DEFINITIONS = {
    "Tuberculosis": {"icon": "🫁", "symptom": "cough", "test": "TB Screen"},
    "Malaria": {"icon": "🦟", "symptom": "fever", "test": "Malaria RDT"},
    "HIV": {"icon": "🎗️", "symptom": "fatigue", "test": "HIV Test"},
    "Anemia": {"icon": "🩸", "symptom": "fatigue", "test": "CBC"},
}

# --- SME RESILIENCE & ACTIONABILITY UPGRADE: Improved Mock Functions ---
def predict_diagnosis_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=['diagnosis', 'predicted_cases', 'resource_needed'])
    diagnoses = df['diagnosis'].unique()
    weekly_counts = df.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'diagnosis']).size().unstack(fill_value=0)
    last_week_avg = weekly_counts.iloc[-1] if len(weekly_counts) >= 1 else weekly_counts.mean()
    resource_map = {"Malaria": "Malaria RDTs", "Tuberculosis": "TB Test Kits", "Anemia": "CBC Vials", "HIV": "HIV Test Kits", "Default": "General Supplies"}
    predictions = [{'diagnosis': diag, 'predicted_cases': max(0, int(last_week_avg.get(diag, 0) * np.random.uniform(0.8, 1.3))), 'resource_needed': resource_map.get(diag, resource_map["Default"])} for diag in diagnoses]
    return pd.DataFrame(predictions)

def generate_moving_average_forecast(df: pd.DataFrame, days_to_forecast: int, window: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    last_known_date = df['ds'].max()
    moving_avg = df['y'].rolling(window=window, min_periods=1).mean().iloc[-1]
    future_dates = pd.to_datetime([last_known_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)])
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': moving_avg})
    forecast_df['yhat_lower'], forecast_df['yhat_upper'] = moving_avg, moving_avg
    return forecast_df

@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    health_df, iot_df = load_health_records(), load_iot_records()
    if health_df.empty: return pd.DataFrame(), iot_df
    if 'ai_risk_score' not in health_df.columns:
        logger.warning("Column 'ai_risk_score' not found. Generating dummy data.")
        st.session_state['using_dummy_risk'] = True
        health_df['ai_risk_score'] = np.random.uniform(0, 100, len(health_df))
    else: st.session_state['using_dummy_risk'] = False
    if 'patient_wait_time' not in health_df.columns:
        logger.warning("Column 'patient_wait_time' not found. Generating dummy data.")
        st.session_state['using_dummy_efficiency'] = True
        health_df['patient_wait_time'] = np.random.uniform(5, 60, len(health_df))
        health_df['consultation_duration'] = np.random.uniform(10, 30, len(health_df))
    else: st.session_state['using_dummy_efficiency'] = False
    health_df, _ = apply_ai_models(health_df)
    return health_df, iot_df

# --- SME UX UPGRADE: Custom Component for Better Visuals ---
def _render_custom_indicator(title: str, value: str, state: str, help_text: str):
    color_map = {"HIGH_RISK": "#dc3545", "MODERATE_CONCERN": "#ffc107", "ACCEPTABLE": "#28a745"}
    border_color = color_map.get(state, "#6c757d")
    st.markdown(f"""<div style="border: 1px solid #e1e4e8; border-left: 5px solid {border_color}; border-radius: 5px; padding: 10px; margin-bottom: 10px;"><div style="font-size: 0.9em; color: #586069;">{title}</div><div style="font-size: 1.5em; font-weight: bold; color: {border_color};">{value}</div></div>""", unsafe_allow_html=True)

# --- UI Rendering Components for Tabs ---
# ... (render_overview_tab, render_program_analysis_tab are unchanged) ...
def render_overview_tab(df: pd.DataFrame, full_df: pd.DataFrame, start_date: date, end_date: date):
    st.header("🚀 Clinic Overview")
    with st.container(border=True):
        st.subheader("Clinic at a Glance")
        period_duration = max(1, (end_date - start_date).days)
        prev_start_date, prev_end_date = start_date - timedelta(days=period_duration), start_date - timedelta(days=1)
        prev_df = full_df[full_df['encounter_date'].dt.date.between(prev_start_date, prev_end_date)]
        unique_patients, prev_unique_patients = df['patient_id'].nunique(), prev_df['patient_id'].nunique() if not prev_df.empty else 0
        avg_risk, prev_avg_risk = df['ai_risk_score'].mean(), prev_df['ai_risk_score'].mean() if not prev_df.empty else 0
        high_risk_patients = df[df['ai_risk_score'] >= 65]['patient_id'].nunique()
        prev_high_risk = prev_df[prev_df['ai_risk_score'] >= 65]['patient_id'].nunique() if not prev_df.empty else 0
        avg_wait_time = df['patient_wait_time'].mean() if 'patient_wait_time' in df.columns else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Unique Patients", f"{unique_patients:,}", f"{unique_patients - prev_unique_patients:+,}" if prev_unique_patients > 0 else "N/A", help=f"Compared to {prev_start_date:%d %b} - {prev_end_date:%d %b}")
        col2.metric("High-Risk Patients (>65)", f"{high_risk_patients:,}", f"{high_risk_patients - prev_high_risk:+,}" if prev_high_risk > 0 else "N/A", delta_color="inverse")
        col3.metric("Avg. Patient Risk Score", f"{avg_risk:.1f}", f"{avg_risk - prev_avg_risk:+.1f}" if prev_avg_risk > 0 else "N/A", delta_color="inverse")
        col4.metric("Avg. Patient Wait Time", f"{avg_wait_time:.1f} min")
    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Diagnosis Heatmap")
        st.markdown("Monitor weekly case volumes for common diagnoses.")
        top_diagnoses = df['diagnosis'].value_counts().nlargest(7).index
        df_top = df[df['diagnosis'].isin(top_diagnoses)]
        if not df_top.empty:
            heatmap_data = df_top.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'diagnosis']).size().unstack(fill_value=0)
            if not heatmap_data.empty:
                heatmap_data.index = heatmap_data.index.strftime('%d-%b-%Y')
                fig = px.imshow(heatmap_data.T, text_auto=True, aspect="auto", color_continuous_scale=px.colors.sequential.Blues, labels=dict(x="Week Start Date", y="Diagnosis", color="Cases"), title="<b>Weekly Case Volume by Diagnosis</b>")
                fig.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Not enough weekly data to generate a heatmap.")
        else: st.info("No diagnosis data available for this period.")
    with col2:
        st.subheader("🔬 AI-Predicted Resource Hotspots")
        st.markdown("Anticipate next week's caseload to guide inventory and staff planning.")
        if not df_top.empty:
            predicted_trends = predict_diagnosis_hotspots(df_top)
            fig = px.bar(predicted_trends, x='diagnosis', y='predicted_cases', color='resource_needed', text='predicted_cases', title="<b>Predicted Cases & Resource Needs for Next Week</b>", labels={'predicted_cases': 'Predicted Case Count', 'diagnosis': 'Diagnosis', 'resource_needed': 'Key Resource'})
            fig.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, yaxis_title='Case Count', xaxis_title=None, showlegend=True, legend_title_text='Key Resource')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📝 Show Recommended Actions"):
                for _, row in predicted_trends.nlargest(3, 'predicted_cases').iterrows():
                    st.markdown(f"- **Prepare for {row['predicted_cases']} `{row['diagnosis']}` cases.** Key resource: `{row['resource_needed']}`.")
                st.markdown("- Review staffing schedules to align with predicted patient load.")
        else: st.info("Insufficient data to generate diagnosis predictions.")

def render_program_analysis_tab(df: pd.DataFrame, program_config: Dict):
    program_name = program_config['name']
    st.header(f"{program_config['icon']} {program_name} Program Analysis")
    st.markdown(f"Analyze the screening-to-treatment cascade for **{program_name}** to identify bottlenecks.")
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
        st.divider()
        screening_rate = (len(tested) / len(symptomatic) * 100) if len(symptomatic) > 0 else 0
        linkage_rate = (len(linked) / len(positive) * 100) if len(positive) > 0 else 100
        st.progress(int(screening_rate), text=f"Screening Rate: {screening_rate:.1f}%")
        st.progress(int(linkage_rate), text=f"Linkage to Care Rate: {linkage_rate:.1f}%")
    with col2:
        st.subheader("💡 AI Opportunity Analysis")
        untested = symptomatic[~symptomatic['patient_id'].isin(tested['patient_id'])]
        if not untested.empty:
            risk_labels, risk_bins = ['Low Risk', 'Medium Risk', 'High Risk'], [-np.inf, 40, 65, np.inf]
            untested['risk_group'] = pd.cut(untested['ai_risk_score'], bins=risk_bins, labels=risk_labels)
            risk_dist = untested['risk_group'].value_counts().reindex(risk_labels).fillna(0)
            fig_donut = go.Figure(data=[go.Pie(labels=risk_dist.index, values=risk_dist.values, hole=.6, marker_colors=[RISK_COLORS[label] for label in risk_dist.index], hoverinfo="label+percent", textinfo='value', textfont_size=16)])
            fig_donut.update_layout(title_text="<b>Who Are We Missing?</b><br><sup>Risk Profile of Untested Cohort</sup>", template=PLOTLY_TEMPLATE, showlegend=True, title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), annotations=[dict(text=f'{int(risk_dist.get("High Risk", 0))}<br>High-Risk', x=0.5, y=0.5, font_size=16, showarrow=False)])
            st.plotly_chart(fig_donut, use_container_width=True, key=f"donut_chart_{program_name}")
            st.caption("Actionability: Prioritize outreach to high-risk symptomatic patients who have not yet been tested.")
        else: st.success("✅ Excellent! All symptomatic patients in this cohort have been tested.")

def render_demographics_tab(df: pd.DataFrame):
    st.header("🧑‍🤝‍🧑 Population Health Intelligence")
    st.markdown("Analyze demographic segments to identify high-risk groups and their specific clinical needs, guiding targeted interventions.")
    if df.empty:
        st.info("No patient data available for demographic analysis.")
        return

    # --- Data Preparation ---
    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    df_unique['gender'] = df_unique['gender'].fillna('Unknown').astype(str)
    age_bins, age_labels = [0, 5, 15, 25, 50, 150], ['0-4', '5-14', '15-24', '25-49', '50+']
    df_unique['age_group'] = pd.cut(df_unique['age'], bins=age_bins, labels=age_labels, right=False).astype(str).replace('nan', 'Not Recorded')
    
    # --- SME UPGRADE: High-Impact "At a Glance" Metrics ---
    gender_dist = df_unique['gender'].value_counts(normalize=True).mul(100)
    col1, col2, col3 = st.columns(3)
    col1.metric("Median Patient Age", f"{df_unique['age'].median():.1f} years")
    col2.metric("Female Patients", f"{gender_dist.get('Female', 0):.1f}%")
    col3.metric("Male Patients", f"{gender_dist.get('Male', 0):.1f}%")
    st.divider()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Comparative Breakdown")
        # Kept original bar charts for foundational analysis
        demo_counts = df_unique.groupby(['age_group', 'gender']).size().reset_index(name='count')
        fig_vol = px.bar(demo_counts, x='age_group', y='count', color='gender', barmode='group', title="<b>Patient Volume by Age & Gender</b>", category_orders={'age_group': age_labels + ['Not Recorded']}, color_discrete_map=GENDER_COLORS)
        fig_vol.update_layout(template=PLOTLY_TEMPLATE, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_vol, use_container_width=True)
        
        risk_by_demo = df_unique.groupby(['age_group', 'gender'])['ai_risk_score'].mean().reset_index()
        fig_risk = px.bar(risk_by_demo, x='age_group', y='ai_risk_score', color='gender', barmode='group', title="<b>Average AI Risk Score by Age & Gender</b>", category_orders={'age_group': age_labels + ['Not Recorded']}, color_discrete_map=GENDER_COLORS)
        fig_risk.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="Avg. Risk Score", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_risk.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        st.subheader("🎯 Actionable Insight Engine")
        
        # --- SME UPGRADE: Risk/Volume Quadrant Analysis ---
        demo_agg = df_unique.groupby(['age_group', 'gender']).agg(
            patient_volume=('patient_id', 'count'),
            avg_risk_score=('ai_risk_score', 'mean'),
            high_risk_count=('ai_risk_score', lambda x: (x >= 65).sum())
        ).reset_index()
        demo_agg['segment'] = demo_agg['gender'] + ', ' + demo_agg['age_group']

        if not demo_agg.empty:
            fig_bubble = px.scatter(
                demo_agg, x='patient_volume', y='avg_risk_score', size='high_risk_count', color='gender',
                hover_name='segment', size_max=60, color_discrete_map=GENDER_COLORS,
                title='<b>Risk/Volume Quadrant Analysis</b>',
                labels={'patient_volume': 'Patient Volume (Count)', 'avg_risk_score': 'Average Risk Score'}
            )
            # Add quadrant lines
            avg_vol = demo_agg['patient_volume'].mean()
            avg_risk = demo_agg['avg_risk_score'].mean()
            fig_bubble.add_vline(x=avg_vol, line_dash="dash", line_color="grey")
            fig_bubble.add_hline(y=avg_risk, line_dash="dash", line_color="grey")
            fig_bubble.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.caption("Actionability: Focus on segments in the top-right quadrant (High Volume, High Risk). Bubble size indicates total high-risk patient impact.")

            # --- SME UPGRADE: Automated Drill-Down Analysis ---
            st.subheader("Clinical Deep Dive on Critical Segment")
            critical_segment = demo_agg.loc[demo_agg['high_risk_count'].idxmax()]
            critical_age = critical_segment['age_group']
            critical_gender = critical_segment['gender']
            
            st.info(f"Most critical segment identified: **{critical_gender}, {critical_age}** (based on highest number of high-risk patients).")

            critical_patients_df = df_unique[(df_unique['age_group'] == critical_age) & (df_unique['gender'] == critical_gender) & (df_unique['ai_risk_score'] >= 65)]
            if not critical_patients_df.empty:
                diagnoses_in_critical_segment = df[df['patient_id'].isin(critical_patients_df['patient_id'])]['diagnosis'].value_counts().nlargest(5)
                
                fig_drill = px.bar(diagnoses_in_critical_segment, y=diagnoses_in_critical_segment.index, x=diagnoses_in_critical_segment.values, orientation='h', title=f"<b>Top Diagnoses for High-Risk {critical_gender}, {critical_age}</b>", labels={'y': 'Diagnosis', 'x': 'Number of Cases'})
                fig_drill.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_drill, use_container_width=True)
                st.caption("Actionability: Use this to guide targeted screening and resource allocation for your most vulnerable group.")
        else:
            st.info("Not enough aggregated data to perform quadrant analysis.")

# ... (render_forecasting_tab, render_environment_tab, render_efficiency_tab, and main function are unchanged) ...
def render_forecasting_tab(df: pd.DataFrame):
    st.header("🔮 AI-Powered Capacity Planning")
    st.markdown("Use predictive forecasts to anticipate future patient load and ensure adequate staffing and appointment availability.")
    
    forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7, key="clinic_forecast_days")
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    
    final_forecast_df, model_used = pd.DataFrame(), "None"
    if len(encounters_hist) > 1 and encounters_hist['y'].std() > 0:
        prophet_fc = generate_prophet_forecast(encounters_hist, forecast_days=forecast_days)
        if 'yhat' in prophet_fc.columns:
            final_forecast_df, model_used = prophet_fc, "Primary (Prophet AI)"
        else:
            final_forecast_df, model_used = generate_moving_average_forecast(encounters_hist, forecast_days, 7), "Fallback (7-Day Avg)"
    
    col1, col2 = st.columns([1.5, 1], gap="large")
    with col1:
        st.subheader("Forecasted Patient Demand")
        if not final_forecast_df.empty:
            plot_data = pd.merge(encounters_hist, final_forecast_df, on='ds', how='outer')
            fig = px.line(plot_data, x='ds', y=['y', 'yhat'], title=f"<b>Forecasted Daily Patient Load ({model_used})</b>")
            fig.update_traces(selector=dict(name='y'), name='Historical', line=dict(color='grey'), showlegend=True)
            fig.update_traces(selector=dict(name='yhat'), name='Forecast', line=dict(color='#007bff', width=3), showlegend=True)
            fig.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, yaxis_title="Patient Encounters", xaxis_title="Date", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Could not generate forecast. Data may be too sparse or lack variation.")

    with col2:
        st.subheader("Capacity & Staffing Scorecard")
        if not final_forecast_df.empty:
            avg_consult_time_min, staff_hours_per_day = 20, 8
            future_fc = final_forecast_df[final_forecast_df['ds'] > df['encounter_date'].max()]
            total_predicted_patients, total_workload_hours = future_fc['yhat'].sum(), (future_fc['yhat'].sum() * avg_consult_time_min) / 60
            required_fte = total_workload_hours / (staff_hours_per_day * forecast_days) if forecast_days > 0 else 0
            capacity_fte = st.number_input("Current Available Clinical FTE:", min_value=1.0, value=5.0, step=0.5, key="capacity_fte")
            utilization = (required_fte / capacity_fte * 100) if capacity_fte > 0 else 0
            surplus_deficit = capacity_fte - required_fte
            with st.container(border=True):
                st.metric(f"Predicted Visits ({forecast_days} days)", f"{total_predicted_patients:,.0f}")
                st.metric(f"Required Full-Time Staff (FTE)", f"{required_fte:.2f} FTEs")
                st.metric("Staffing Surplus / Deficit", f"{surplus_deficit:+.2f} FTEs", delta_color="normal")
            st.markdown("##### **Predicted Clinic Capacity Utilization**")
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=utilization, title={'text': "Staff Utilization (%)"}, gauge={'axis': {'range': [None, 120]}, 'bar': {'color': "#2c3e50"}, 'steps': [{'range': [0, 85], 'color': "#28a745"}, {'range': [85, 100], 'color': "#ffc107"}, {'range': [100, 120], 'color': "#dc3545"}]}))
            fig_gauge.update_layout(height=200, margin=dict(t=30, b=30, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            if utilization > 100: st.error(f"🔴 **Over-Capacity Alert:** Predicted workload requires **{utilization-100:.1f}% more staff**.")
            elif utilization > 85: st.warning(f"🟠 **High-Capacity Warning:** Workload at {utilization:.1f}% of capacity.")
            else: st.success(f"✅ **Healthy Capacity:** Workload is manageable at {utilization:.1f}% of capacity.")
        else: st.info("Run forecast to see capacity predictions.")

def render_environment_tab(iot_df: pd.DataFrame):
    st.header("🌿 Facility Environmental Safety")
    if iot_df.empty: st.info("No environmental sensor data available for this period.", icon="📡"); return
    st.subheader("Real-Time Environmental Indicators")
    avg_co2 = iot_df['avg_co2_ppm'].mean()
    high_noise_rooms = iot_df[iot_df['avg_noise_db'] > 70]['room_id'].nunique()
    co2_state = "HIGH_RISK" if avg_co2 > 1500 else "MODERATE_CONCERN" if avg_co2 > 1000 else "ACCEPTABLE"
    noise_state = "HIGH_RISK" if high_noise_rooms > 0 else "ACCEPTABLE"
    col1, col2 = st.columns(2)
    with col1: _render_custom_indicator("Average CO₂ Levels", f"{avg_co2:.0f} PPM", co2_state, "CO₂ levels are a proxy for ventilation quality. High levels increase airborne transmission risk.")
    with col2: _render_custom_indicator("Rooms with High Noise (>70dB)", f"{high_noise_rooms} rooms", noise_state, "High noise levels can impact patient comfort and staff communication.")
    st.divider()
    st.subheader("Hourly CO₂ Trend (Ventilation Proxy)")
    co2_trend = iot_df.set_index('timestamp').resample('h')['avg_co2_ppm'].mean().dropna()
    fig = px.line(co2_trend, title="<b>Hourly Average CO₂ Trend</b>", labels={'value': 'CO₂ (PPM)', 'timestamp': 'Time'})
    fig.add_hline(y=1000, line_dash="dot", line_color="orange", annotation_text="High Risk Threshold")
    fig.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_efficiency_tab(df: pd.DataFrame):
    st.header("⏱️ Operational Efficiency Analysis")
    st.markdown("Monitor and predict key efficiency metrics to improve patient flow and reduce wait times.")
    if df.empty: st.info("No data available for efficiency analysis."); return
    avg_wait, avg_consult = df['patient_wait_time'].mean(), df['consultation_duration'].mean()
    long_wait_count = df[df['patient_wait_time'] > 45]['patient_id'].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Patient Wait Time", f"{avg_wait:.1f} min")
    col2.metric("Avg. Consultation Time", f"{avg_consult:.1f} min")
    col3.metric("Patients with Long Wait (>45min)", f"{long_wait_count:,}")
    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Wait Time Distribution")
        fig_hist = px.histogram(df, x="patient_wait_time", nbins=20, title="<b>Distribution of Patient Wait Times</b>", labels={'patient_wait_time': 'Wait Time (minutes)'}, template=PLOTLY_TEMPLATE, marginal="box")
        fig_hist.update_traces(marker_color='#007bff', opacity=0.7).add_vline(x=avg_wait, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_wait:.1f} min")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        st.subheader("When are wait times longest?")
        df['hour_of_day'] = df['encounter_date'].dt.hour
        wait_by_hour = df.groupby('hour_of_day')['patient_wait_time'].mean().reset_index()
        fig_line = px.line(wait_by_hour, x='hour_of_day', y='patient_wait_time', title='<b>Average Wait Time by Hour of Day</b>', markers=True, labels={'hour_of_day': 'Hour of Day (24h)', 'patient_wait_time': 'Average Wait Time (min)'})
        fig_line.update_layout(template=PLOTLY_TEMPLATE, title_x=0.5)
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("Actionability: Consider reallocating staff to the peak hours identified above to reduce wait times.")

# --- Main Page Execution ---
def main():
    st.title("🏥 Clinic Command Center")
    st.markdown("A strategic console for managing clinical services, program performance, and facility operations.")
    full_health_df, full_iot_df = get_data()
    if full_health_df.empty: st.error("CRITICAL: No health data available. Dashboard cannot be rendered."); st.stop()
    if st.session_state.get('using_dummy_risk', False): st.warning("⚠️ **Risk Demo Mode:** `ai_risk_score` was not found and has been simulated.", icon="🤖")
    if st.session_state.get('using_dummy_efficiency', False): st.warning("⚠️ **Efficiency Demo Mode:** Wait/consultation times were not found and have been simulated.", icon="⏱️")
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = full_health_df['encounter_date'].min().date(), full_health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=29)), max_date), min_value=min_date, max_value=max_date, key="date_range")
    period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)] if not full_iot_df.empty else pd.DataFrame()
    st.info(f"**Displaying Clinic Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}`")
    st.divider()
    TABS_CONFIG = { "Overview": {"icon": "🚀", "func": render_overview_tab, "args": [period_health_df, full_health_df, start_date, end_date]}, "Demographics": {"icon": "🧑‍🤝‍🧑", "func": render_demographics_tab, "args": [period_health_df]}, "Efficiency": {"icon": "⏱️", "func": render_efficiency_tab, "args": [period_health_df]}, "Capacity Planning": {"icon": "🔮", "func": render_forecasting_tab, "args": [full_health_df]}, "Environment": {"icon": "🌿", "func": render_environment_tab, "args": [period_iot_df]} }
    program_tabs = {name: {"icon": conf['icon'], "func": render_program_analysis_tab, "args": [period_health_df, {**conf, 'name': name}]} for name, conf in PROGRAM_DEFINITIONS.items()}
    all_tabs_config = list(TABS_CONFIG.items())
    all_tabs_config.insert(1, *program_tabs.items())
    tab_titles = [f"{conf['icon']} {name}" for name, conf in all_tabs_config]
    tabs = st.tabs(tab_titles)
    for i, (name, conf) in enumerate(all_tabs_config):
        with tabs[i]:
            conf["func"](*conf["args"])

if __name__ == "__main__":
    main()
