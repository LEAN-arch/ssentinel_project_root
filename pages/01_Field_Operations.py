# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - FIELD COMMAND CENTER (V13 - PROGRAM SCORECARD)

import logging
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import create_empty_figure, plot_forecast_chart

# --- Page Setup ---
st.set_page_config(page_title="Field Command Center", page_icon="📡", layout="wide")
logger =Of course. I will now generate the definitive CSV data files required to fully populate the re logging.getLogger(__name__)


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, enriches, and caches all data for the dashboard."""
    raw_health_df = load_health_records-engineered **Field Operations Command Center**. I will also provide the final, updated version of the dashboard page (`01_Field_()
    iot_df = load_iot_records()
    if raw_health_df.empty:
        return pd.DataFrame(), iot_df
    enriched_df, _ = apply_ai_models(raw_health_df)
    return enriched_df, iot_df

# --- Analytics & KPI Functions ---
defOperations.py`) that replaces the basic KPI cards with a more integrated and visually informative "Situation Report" section.

This get_program_scorecard_data(df: pd.DataFrame, full_history_df: pd.DataFrame) -> List[Dict]:
    """
    Calculates a list of decision-grade KPIs for the Program Scorecard.
    Each KPI is a dictionary containing its value, target, and status.
    """
    if df.empty:
 new design directly addresses your request by:
1.  **Eliminating KPI Cards:** The card-based layout is replaced with a more professional and data-dense "Situation Report" header.
2.  **Integrating Key Metrics:** Core daily metrics are now presented as part of a cohesive summary, providing immediate context.
3.  **Providing Rich, Population-Specific Data:** The CSV files are expanded with new columns (`chw_stress_score`, `encounter_type`, `        return []

    scorecard = []
    
    # --- Screening & Diagnosis Program KPIs ---
    # Malaria
    symptomatic_malaria = df[df['patient_reported_symptoms'].str.contains('fever', case=False, na=False)]
    tested_malaria = symptomatic_malaria[symptomatic_referral_status`) and more diverse data to power all the advanced visualizations on the dashboard, including specific screening programs and demographic analysis.

---

### 1. Definitive `data_sources/health_records_expanded.csv`

This file now contains a richer dataset specifically designed to populate all modules of the Field Operations dashboard, including data for Malaria andmalaria['test_type'] == 'Malaria RDT']
    positive_malaria = tested_malaria[tested_malaria['test_result'] == 'Positive']
    
    # TB
    symptomatic_tb = df[df['patient_reported_symptoms'].str.contains('cough', case=False, na=False)]
    tested_tb = symptomatic_tb[symptomatic_tb['test_type'] == 'TB TB screening cascades, encounter types, and CHW stress scores.

```csv
encounter_id,patient_id,chw_id,encounter_date,zone_id,age,gender,diagnosis,test_type,test_result,patient_reported_symptoms,item,item_stock_agg_zone,consumption_rate_per_day,test_turnaround_days,sample_status,rejection_reason,min_spo2_pct,vital_signs_temperature Screen']
    positive_tb = tested_tb[tested_tb['test_result'] == 'Positive']
    linked_tb = positive_tb[positive_tb['referral_status'] == 'Completed']
    
    # HIV
    tested_hiv = df[df['test_type'] == 'HIV Test']
    positive_hiv = tested_hiv[tested_hiv['test_result'] == 'Positive']

    # --- Trend Calculation ---
    def get_weekly_trend(key_col: str, filter_cond=None):
        _celsius,fall_detected_today,days_task_overdue,referral_status,medication_adherence_self_report,chronic_condition_flag,chw_stress_score,encounter_type
EIDsubset = full_history_df[filter_cond] if filter_cond is not None else full_history_df
        trend = subset.set_index('encounter_date')[key_col].resample('W').nunique()
        return trend.iloc[-1] if len(trend) > 1 else 0, trend.iloc[-2] if len(trend) > 1 else 0

    # --- Build Scorecard ---
    # Metric001,PID001,CHW001,2023-11-20T10:00:00Z,Zone-A,34,Female,Malaria,Malaria RDT,Positive,"fever;headache",Paracetamol,15000,120,1,Completed,NA,98,38.5,0,0,Completed,Good,0,22,Routine Visit
EID002,PID002,CHW002,2023-11-20T11:30:00Z,Zone-B,25,Male,Pneumonia,TB Screen,Negative,"cough;fever;difficulty breathing",Amoxicillin,1500, Value, Target, Status (Good, Warning, Alert), Trend
    
    # Patient Load
    current_patients, prev_patients = get_weekly_trend('patient_id')
    scorecard.append({
        'Metric': 'Patients Seen (Weekly)', 'Value': f"{current_patients}", 'Target': '> 50',
        'Status': 'Good' if current_patients > 50 else 'Warning', 'Trend': current_patients - prev_patients
    })
    
    # Malaria Screening
    screening_rate = (len(tested_malaria) / len(symptomatic_malaria) * 100) if len,80,2,Completed,NA,94,39.1,0,0,Completed,Fair,0,35,Alert Response
EID003,PID003,CHW001,2023-11-20T09:00:00Z,Zone-A,45,Male,Hypertension,NA,NA,"dizziness",Lisinopril,50,5,NA,NA,NA,99,37.0,0,2,Pending,Good,1,25,Medication Delivery
EID004,PID001,CHW001,2023-11-19T14:00:00Z,Zone-A,34,Female,Malaria,Malaria RDT,Negative,"fever",Paracetamol,14400(symptomatic_malaria) > 0 else 0
    scorecard.append({
        'Metric': 'Malaria Screening Rate', 'Value': f"{screening_rate:.1f}%", 'Target': '> 90%',
        'Status': 'Good' if screening_rate >= 90 else 'Alert'
    })
    
    # TB Linkage
    linkage_rate = (len(linked_tb) / len(positive_tb) * 100) if len(positive_tb) > 0 else 100
    scorecard.append({
        'Metric': 'TB Linkage to Care', 'Value': f"{linkage_rate:.1f}%", 'Target': '> 85%',
        'Status': 'Good' if linkage_rate >= 85 else 'Alert'
    })

    # HIV Positivity
    positivity_rate = (len(positive_hiv) / len(tested_hiv) * 10,125,1,Completed,NA,97,37.5,0,0,Completed,Good,0,28,Follow-up
EID005,PID004,CHW003,2023-11-19T10:00:00Z,Zone-C,5,Female,Diarrhea,Stool Test,Pending,"diarrhea;vomiting;sunken eyes",ORS,5000,40,NA,Pending,NA,96,38.8,0,0,Pending,Good,0,45,Routine Visit
EID006,PID005,CHW002,2023-11-18T11:00:00Z,Zone-B,60,Male,Diabetes,HIV Test,Negative,"fatigue;frequent urination",Metformin,8000,60,0.2,Completed,NA0) if len(tested_hiv) > 0 else 0
    scorecard.append({
        'Metric': 'HIV Positivity Rate', 'Value': f"{positivity_rate:.1f}%", 'Target': '< 5%',
        'Status': 'Good' if positivity_rate < 5 else 'Warning'
    })

    return scorecard

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def generate_forecasts(df: pd.DataFrame, forecast_days: int) -> Dict[str, pd.DataFrame]:
    if df.empty or len(df) < 10:,92,36.8,0,5,Completed,Poor,1,65,Follow-up
EID007,PID002,CHW002,2023-11-17T16:00:00Z,Zone-B,25,Male,Pneumonia,TB Screen,Negative,"cough;fever",Amoxicillin,1200,85,2,Completed,NA,96,37.8,0,0,Completed,Good,0,33,Alert Response
EID008,PID006,CHW001,2023-11-16T09:30:00Z,Zone-A,28,Female,URI,NA,NA,"sore throat;cough",Ibuprofen,13900,118,NA,NA,NA,98,37.2,0,0,Completed,Good,0,21,Routine Visit
EID009,PID007,CHW003,20 return {}
    encounters_hist = df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
    avg_risk_hist = df.set_index('encounter_date')['ai_risk_score'].resample('D').mean().reset_index().rename(columns={'encounter_date': 'ds', 'ai_risk_score': 'y'})
    return {"Patient Load": generate_prophet_forecast(encounters_hist, forecast_days), "Community Risk Index": generate_prophet_forecast(avg_risk_hist, forecast_days)}

# --- UI Rendering ---
def render_program_scorecard(scorecard_data: List[Dict]):
    """Renders a rich, data-driven scorecard using st.dataframe."""
    st.subheader("📊 Program Scorecard")
    if not scorecard_data:
        st.info("Not enough data in the selected period to generate a scorecard."); return
    
    df = pd.DataFrame(scorecard_data)
    
    def get_color(status):
        if status == 'Good': return 'background-color: #d4edda; color: #155724' # Green
        if status == 'Warning': return 'background-color: #fff3cd; color: #856404' # Yellow
        if status == 'Alert': return 'background-color: #f8d7da; color: #721c24' # Red23-11-15T13:00:00Z,Zone-C,2,Male,Anemia,CBC,Positive,"fatigue;pale skin",Iron Tablets,4750,42,3,Completed,NA,95,37.2,1,0,Pending,Good,0,48,Routine Visit
EID010,PID008,CHW002,2023-11-14T10:00:00Z,Zone-B,33,Female,Malaria,Malaria RDT,Positive,fever,AL,7400,65,1,Completed,NA,99,39.0,0,0,Completed,Good,0,39,Alert Response
EID011,PID009,CHW001,2023-11-20T11:45:00Z,Zone-A,41,Male,Tuberculosis,TB Screen,Positive,"prolonged cough;fever;fatigue",Rifampicin,850,90,1,Completed,NA,93,38.1,0,3,Pending,Fair,1,55,Alert Response
EID012,PID010,CHW004,2023-11-20T08:00:00Z,Zone-A,22,Female,HIV,HIV Test,Positive,"fever;fat
        return ''

    st.dataframe(
        df.style.applymap(get_color, subset=['Status']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn(width="large"),
            "Trend": st.column_config.BarChartColumn("Weekly Trend", y_min=-20, y_max=20),
        }
    )

# --- Main Page Execution ---
def main():
    st.title("📡 Field Operations Command Center")
    st.markdown("An integrated dashboard for supervising team activity, patient risk, environmental factors, and future trends.")
    st.divider()

    health_df, iot_df = get_data()

igue",Tenofovir,200,10,0.2,Completed,NA,98,38.2,0,0,Pending,Good,0,75,Initial Assessment
EID013,PID010,CHW004,2023-11-20T08:05:00Z,Zone-A,22,Female,HIV,NA,NA,"",NA,NA,NA,NA,NA,NA,98,38.2,0,0,Completed,Good,0,76,Referral
EID014,PID011,CHW002,2023-11-20T14:00:00Z,Zone-B,55,Male,Malaria,Malaria RDT,Positive,"fever",AL,7000,60,1,Completed,NA,89,39.5,0,0,Completed,Good,1,41,Alert Response
EID015,PID009,CHW001,2023-11-19T14:30:00Z,Zone-A,    if health_df.empty: st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        zone_options = ["All Zones"] + sorted(health_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        
        today = health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(today - timedelta(days=29), health_df['encounter_date'].min().date()), today), min_value=health_df['encounter_date'].min().date(), max_value=today)
        
        st.markdown("---")
        forecast_days = st.slider("Days to Forecast Ahead:", min_value=7, max_value=90, value=21, step=7)

    # --- Filter Data ---
    analysis_df = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    full_history_df = health_df.copy() # Keep full history for trend context
    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]
        full_history_df = full_history_df[full_history_df['zone_id'] == selected_zone]

    st.info(f"**Viewing Data:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** {selected_zone}")

    # --- Main Layout ---
    scorecard_data = get_program_scorecard_data(analysis_df, full_history_df)
    render_program_scorecard(scorecard_data)
    st.divider()

    tab1, tab2 = st.tabs(["**🔮 AI-Powered Forecasts**", "**🛰️ Environmental & Wearable Factors**"])

    with tab1:
        st.subheader(f"Predictive Analytics ({forecast_days} Days Ahead)")
        forecasts = generate_forecasts(41,Male,Tuberculosis,TB Screen,Positive,"cough",Rifampicin,500,95,3,Completed,NA,94,37.9,0,0,Completed,Poor,1,58,Follow-up
