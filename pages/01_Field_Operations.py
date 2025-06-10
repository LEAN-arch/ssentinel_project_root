# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - INTEGRATED FIELD COMMAND CENTER (V22 - FINAL)

import logging
from datetime import date, timedelta
from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records
from visualization import create_empty_figure, plot_bar_chart, plot_forecast_chart

# --- Page Setup ---
st.set_page_config(page_title="Field Command Center", page_icon="📡", layout="wide")
logger = logging.getLogger(__name__)


# --- Disease Program Definitions ---
PROGRAM_DEFINITIONS = {
    "Tuberculosis": {"icon": "🫁", "symptom": "cough", "test": "TB Screen"},
    "Malaria": {"icon": "🦟", "symptom": "fever", "test": "Malaria RDT"},
    "HIV & STIs": {"icon": "🎗️", "symptom": "fatigue", "test": "HIV Test"},
    "Anemia & NTDs": {"icon": "🩸", "symptom": "fatigue|weakness", "test": "CBC"},
}


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    """Loads and enriches data for the dashboard."""
    raw_df = load_health_records()
    if raw_df.empty:
        return pd.DataFrame()
    enriched_df, _ = apply_ai_models(raw_df)
    return enriched_df

# --- Analytics & UI Components ---
def render_program_cascade(df: pd.DataFrame, config: Dict):
    """Renders a visual funnel and KPIs for a specific screening program."""
    st.subheader(f"{config['icon']} {config['name']} Screening Cascade")
    
    symptomatic = df[df['patient_reported_symptoms'].str.contains(config['symptom'], case=False, na=False)]
    tested = symptomatic[symptomatic['test_type'] == config['test']]
    positive = tested[tested['test_result'] == 'Positive']
    linked = positive[positive['referral_status'] == 'Completed']
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.metric("Symptomatic/At-Risk Cohort", f"{len(symptomatic):,}")
        st.metric("Patients Tested", f"{len(tested):,}")
        st.metric("Positive Cases", f"{len(positive):,}")
        st.metric("Linked to Care", f"{len(linked):,}")

        screening_rate = (len(tested) / len(symptomatic) * 100) if len(symptomatic) > 0 else 0
        linkage_rate = (len(linked) / len(positive) * 100) if len(positive) > 0 else 100
        st.progress(int(screening_rate), text=f"Screening Rate: {screening_rate:.1f}%")
        st.progress(int(linkage_rate), text=f"Linkage to Care Rate: {linkage_rate:.1f}%")

    with col2:
        funnel_data = pd.DataFrame([
            dict(stage="Symptomatic/At-Risk", count=len(symptomatic)),
            dict(stage="Tested", count=len(tested)),
            dict(stage="Positive", count=len(positive)),
            dict(stage="Linked to Care", count=len(linked)),
        ])
        if funnel_data['count'].sum() > 0:
            fig = px.funnel(funnel_data, x='count', y='stage', title=f"Program Funnel: {configOf course. I will now perform a definitive re-engineering of the **Field Operations Command Center** to align with your request. This transformation focuses on making the dashboard a powerful, actionable tool for a field supervisor overseeing multiple critical disease programs.

### Definitive Re-engineering Plan:

1.  **Eliminate the "Situation Report":** The generic `st.metric` KPIs will be removed. Actionable, program-specific metrics will be integrated directly into new, dedicated visualization components.
2.  **Introduce Program-Centric Screening Cascades:** The core of the new dashboard will be a series of **Screening Cascade Funnels**. These are highly effective visualizations that instantly show a supervisor where patients are being lost in the "find, test, treat" pathway for each key disease. I will implement cascades for Tuberculosis, Malaria, and HIV as examples.
3.  **Add Syndromic Surveillance:** A new component will analyze and display the top patient-reported symptoms. This is a crucial tool for early outbreak detection and for understanding the current health landscape of the community.
4.  **Integrate Demographic & Workload Analysis:** A new section will visualize the demographic reach of the team (age/gender) and the breakdown of their activities (e.g., home visits vs. alert responses), providing insight into both population coverage and team workload.
5.  **Retain Core Functionality:** The critical "Priority Alerts" and "AI Forecasts" modules will be preserved and integrated into the new, more intuitive layout.

This comprehensive redesign transforms the page from a simple report into a true command center, providing supervisors with the precise, actionable, and predictive intelligence needed to manage scalable diagnostic and screening programs effectively.

---

### **Definitively Re-engineered `pages/01_Field_Operations.py`**

This is the final, complete replacement for the Field Operations dashboard.

```python
# sentinel_project_root/pages/01_Field_Operations.py
# SME PLATINUM STANDARD - FIELD COMMAND CENTER (V22 - PROGRAM-CENTRIC REDESIGN)

import logging
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Core Sentinel Imports ---
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_donut_chart, plot_forecast_chart)

# --- Page Setup ---
st.set_page_config(page_title="Field Command Center", page_icon="📡", layout="wide")
logger = logging.getLogger(__name__)


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, enriches, and caches all data for the dashboard."""
    raw_health_df = load_health_records['name']}")
            fig.update_yaxes(categoryorder="array", categoryarray=["Symptomatic/At-Risk", "Tested", "Positive", "Linked to Care"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No activity recorded for the {config['name']} program in this period.")


# --- Main Page Execution ---
def main():
    st.title("📡 Field Operations Command Center")
    st.markdown("An actionable dashboard for supervising team activity and managing public health screening programs.")
    
    health_df = get_data()
    if health_df.empty: st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        zone_options = ["All Zones"] + sorted(health_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
        
        chw_options = ["All CHWs"] + sorted(health_df['chw_id'].dropna().unique())
        selected_chw = st.selectbox("Filter by CHW:", options=chw_options)
        
        today = health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(today - timedelta(days=29), health_df['encounter_date'].min().date()), today), min_value=health_df['encounter_date'].min().date(), max_value=today)

    # --- Data Filtering ---
    analysis_df = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]
    if selected_chw != "All CHWs":
        analysis_df = analysis_df[analysis_df['chw_id'] == selected_chw]

    st.info(f"**Displaying Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** `{selected_zone}` | **CHW:** `{selected_chw}`")
    st.divider()

    # --- Main Layout ---
    st.header("Program Performance Analysis")
    st.markdown("Use the tabs below to monitor the performance of each key public health screening program.")
    
    program_tabs = st.tabs([f"{p['icon']} {name}" for name, p in PROGRAM_DEFINITIONS.items()])
    
    for i, (program_name, config) in enumerate(PROGRAM_DEFINITIONS.items()):
        with program_tabs[i]:
            render_program_cascade(analysis_df, {**config, "name": program_name})

    st.divider()

    st.header("AI-Powered Decision Support")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🚨 Priority Patient Alerts")
        alerts = generate_chw_alerts(patient_df=analysis_df)
        if not alerts:
            st.success("✅ No high-priority patient alerts for this selection.")
        for alert in alerts:
            level, icon = ("CRITICAL", "🔴") if alert.get('alert_level') == 'CRITICAL' else (("WARNING", "🟠") if alert.get('alert_level') == 'WARNING' else ("INFO", "ℹ️"))
            with st.container(border=True):
                st.markdown(f"**{icon} {alert.get('reason')} for Pt. {alert.get('patient_id')}**")
                st.markdown(f"> {alert.get('details', 'N/A')} (Priority: {alert.get('priority', 0):.0f})")

    with col2:
        st.subheader("🔮 Patient Load Forecast")
        forecast_days = st.slider("Forecast Horizon (Days):", 7, 30, 14, 7)
        
        forecast_source_df = health_df[health_df['encounter_date'].dt.date <= end_date]
        if selected_zone != "All Zones":
            forecast_source_df = forecast_source_df[forecast_source_df['zone_id'] == selected_zone]
        if selected_chw != "All CHWs":
            forecast_source_df = forecast_source_df[forecast_source_df['chw_id'] == selected_chw]
            
        if len(forecast_source_df) < 10:
            st.warning("Not enough historical data for the selected filters to generate a forecast.")
        else:
            encounters_hist = forecast_source_df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
            forecast = generate_prophet_forecast(encounters_hist, forecast_days)
            fig = plot_forecast_chart(forecast, title="Forecasted Daily Patient Encounters", y_title="Patient Encounters")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
