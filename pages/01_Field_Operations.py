# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - INTEGRATED FIELD COMMAND CENTER (V13 - FINAL DESIGN)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

#today,days_task_overdue,referral_status,medication_adherence_self_report,chronic_condition_flag,encounter_type
EID001,PID001,2023-11-01T10:00:00Z,Zone-A,34,Female,Malaria,Malaria RDT,Positive,"fever; headache",Paracetamol,15000,120,1,Completed,NA,98,38.5,0,0,Completed,Good,0,CHW_HOME_VISIT
EID002,PID002,2023-11-01T11:30:00Z,Zone-B,25,Male,Pneumonia,TB Screen,Pending,"cough; fever; difficulty breathing",Amoxicillin,1500,80,,Pending,NA,94,39.1,0,0,Pending,Fair,0,CLINIC_INTAKE
EID003,PID003,2023-11-02T09:00:00Z,Zone-A,45,Male,Hypertension,HIV Test,Negative,"dizziness",Gloves,50,5,1,Completed,NA,99 --- Core Sentinel Imports ---
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
def render_screening_cascade(df: pd.DataFrame, program_name: str, icon: str, symptom_keyword: str, test_name: str):
    st.subheader(f"{icon} {program_name} Screening Cascade")
    if df.empty:
        st.info(f"No data to analyze for the {program_name} program."); return
    symptomatic = df[df['patient_reported_symptoms'].str.contains(symptom_keyword, case=False, na=False)]
    tested = symptomatic[symptomatic['test_type'] == test_name]
    positive = tested[tested['test_result'] == 'Positive']
    linked = positive[positive['referral_status'] == 'Completed']
    funnel_data = pd.DataFrame([
        dict(stage="Symptomatic/At-Risk", count=len(symptomatic,37.0,0,2,Completed,Good,1,ROUTINE_CHECKUP
EID004,PID001,2023-11-05T14:00:00Z,Zone-A,34,Female,Malaria,Malaria RDT,Negative,"fever",Paracetamol,14400,125,1,Completed,NA,97,37.5,0,0,Completed,Good,0,FOLLOW_UP
EID005,PID004,2023-11-06T10:00:00Z,Zone-C,5,Female,Anemia,CBC,Positive,"fatigue; weakness",ORS,5000,40,2,Completed,NA,96,36.8,0,0,Pending,Good,1,CHW_HOME_VISIT
EID006,PID005,2023-11-08T11:00:00Z,Zone-B,60,Male,Malaria,Malaria RDT,Positive,"fever; chills",Metformin,8000,60,1,Completed,NA,92,39.8,0,5,Completed,Poor,1,CHW_ALERT_RESPONSE
EID007,PID002,2023-11-09T16:00:00Z,Zone-B,25,Male,Pneumonia,TB Screen,Positive,"cough; fever",Amoxicillin,1200,85,2,Completed,NA,96,37.8,0,0,Pending,Good,0,FOLLOW_UP
EID008,PID010,2023-11-12T09:30:00Z,Zone-A,68,Male,Tuberculosis,TB Screen,Positive,"prolonged cough; fever; night sweats",Paracetamol,13900,118,3,Completed,NA,93,38.2,0,0,Completed,Good,1,CHW_HOME_VISIT
EID009,PID011,2023-11-15T13:00:00Z,Zone-C,2,Male,Diarrhea,Stool Test,Positive,"dehydration",ORS,4750,42,3,Completed,NA,95,38.2,1,0,Completed,Good,0,CHW_ALERT_RESPONSE
EID010,PID012,2023-11-18T10:00:00Z,Zone-B,33,Female,HIV,HIV Test,Positive,"fever; fatigue",Metformin,7400,65,1,Completed,NA,99,37.0,0,0,Completed,Good)),
        dict(stage="Tested", count=len(tested)),
        dict(stage="Positive", count=len(positive)),
        dict(stage="Linked to Care", count=len(linked)),
    ])
    if funnel_data['count'].sum() == 0:
        st.info(f"No activity recorded for the {program_name} screening program in this period."); return
    fig = px.funnel(funnel_data, x='count', y='stage', title=f"{program_name} Program Funnel")
    fig.update_yaxes(categoryorder="array", categoryarray=["Symptomatic/At-Risk", "Tested", "Positive", "Linked to Care"])
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600, show_spinner="Generating forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    if df.empty or len(df) < 10: return {}
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    return {
        "Patient Load": generate_prophet_forecast(encounters_hist, forecast_days=forecast_days),
        "Community Risk Index": generate_prophet_forecast(avg_risk_hist, forecast_days=forecast_days),
    }

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

def render_iot_wearable_tab(iot_df: pd.DataFrame, chw_id: Optional[str]):
    st.subheader("🛰️ Environmental & Team Factors")
    if,0,CLINIC_INTAKE
EID011,PID013,2023-11-20T11:45:00Z,Zone-A,41,Male,Tuberculosis,TB Screen,Positive,"cough; fever",Amoxicillin,850,90,4,Completed,NA,93,38.1,0,3,Completed,Fair,1,FOLLOW_UP
EID012,PID014,2023-11-25T09:00:00Z,Zone-A,22,Female,Anemia,CBC,Pending,"fatigue",Paracetamol,13200,122,,Pending,NA,98,36.9,0,0,Pending,Good,0,ROUTINE_CHECKUP
