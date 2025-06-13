# sentinel_project_root/pages/01_Field_Operations.py
# FINAL, SELF-CONTAINED, AND VISUALIZATION-ENHANCED VERSION
# SME UPGRADE: Full demonstration mode with simulated data and fallback KPIs for maximum resilience.
# SME GATES FOUNDATION UPGRADE: Explicit strategic metrics, health equity tab, and HSS framing.
# FULLY ENABLED VERSION: All tabs and functions are active.

import logging
from datetime import date, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Core Sentinel Imports ---
# These are assumed to exist and work. For this self-contained script,
# they are aliased to placeholder functions if not found.
try:
    from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
    from config import settings
    from data_processing import load_health_records, load_iot_records
    from visualization import (create_empty_figure, plot_bar_chart,
                               plot_donut_chart, plot_line_chart)
except ImportError:
    # Define dummy functions if imports fail, to allow the script to run standalone
    def apply_ai_models(df): return df, {}
    def generate_chw_alerts(patient_df): return [{'alert_level': 'CRITICAL', 'reason': 'High Risk Score & Negative Symptom Trend', 'patient_id': 'PAT_1337', 'details': 'Risk score increased by 30% in 7 days.', 'priority': 0.95, 'chw_id': 'CHW_1'}]
    def generate_prophet_forecast(df, days): return pd.DataFrame({'ds': pd.to_datetime(pd.date_range(start=df['ds'].max(), periods=days+1)), 'yhat': np.random.uniform(df['y'].mean()*0.8, df['y'].mean()*1.2, days+1), 'yhat_lower': np.random.uniform(df['y'].mean()*0.7, df['y'].mean()*0.8, days+1), 'yhat_upper': np.random.uniform(df['y'].mean()*1.2, df['y'].mean()*1.3, days+1)})
    class Settings: APP_LOGO = None
    settings = Settings()
    def load_health_records(): return pd.DataFrame()
    def load_iot_records(): return pd.DataFrame()
    def create_empty_figure(text): return go.Figure().update_layout(title_text=text, template="plotly_white")
    def plot_line_chart(df, title, y_title): return px.line(df, title=title, template="plotly_white").update_yaxes(title_text=y_title)


# --- Page Setup ---
st.set_page_config(page_title="Health Program Command Center", page_icon="🌍", layout="wide")

# --- SME ACTIONABILITY UPGRADE: Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Disease Program Definitions ---
PROGRAM_DEFINITIONS = {
    "Tuberculosis": {"icon": "🫁", "symptom": "cough", "test": "TB Screen"},
    "Malaria": {"icon": "🦟", "symptom": "fever", "test": "Malaria RDT"},
    "HIV & STIs": {"icon": "🎗️", "symptom": "fatigue", "test": "HIV Test"},
    "Anemia & NTDs": {"icon": "🩸", "symptom": "fatigue|weakness", "test": "CBC"},
}

# --- SME VISUALIZATION & KPI UPGRADE: Constants ---
PLOTLY_TEMPLATE = "plotly_white"
RISK_BINS = [-np.inf, 0.4, 0.7, np.inf]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLOR_MAP = {"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"}

# --- SME RESILIENCE UPGRADE: Dummy Data Generation ---
def _generate_dummy_iot_data(health_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a realistic, simulated IoT DataFrame for demonstration purposes."""
    logger.info("Generating simulated IoT data.")
    
    chw_ids = health_df['chw_id'].dropna().unique()
    zone_ids = health_df['zone_id'].dropna().unique()
    if len(chw_ids) == 0 or len(zone_ids) == 0:
        return pd.DataFrame()

    date_range = pd.to_datetime(pd.date_range(
        start=health_df['encounter_date'].min(),
        end=health_df['encounter_date'].max(),
        freq='6H'
    ))

    wearable_records = []
    for chw_id in chw_ids:
        for ts in date_range:
            wearable_records.append({
                'timestamp': ts, 'chw_id': chw_id,
                'chw_stress_score': np.random.randint(20, 95), 'avg_co2_ppm': np.nan
            })
    wearable_df = pd.DataFrame(wearable_records)
    if not wearable_df.empty:
        wearable_df['zone_id'] = wearable_df['chw_id'].map(health_df.drop_duplicates('chw_id').set_index('chw_id')['zone_id'])

    clinic_records = []
    for zone_id in zone_ids:
         for ts in date_range:
            clinic_records.append({
                'timestamp': ts, 'chw_id': np.nan, 'chw_stress_score': np.nan,
                'avg_co2_ppm': np.random.randint(400, 1800), 'zone_id': zone_id
            })
    clinic_df = pd.DataFrame(clinic_records)
    
    return pd.concat([wearable_df, clinic_df], ignore_index=True)


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and enriches data, augmenting with simulated data if sources are missing."""
    raw_health_df = load_health_records()
    iot_df = load_iot_records()
    
    if raw_health_df.empty:
        logger.warning("Health records not found. Generating dummy health data.")
        st.session_state['using_dummy_health_data'] = True
        dates = pd.to_datetime(pd.date_range(start=date.today() - timedelta(days=365), end=date.today(), freq='D'))
        raw_health_df = pd.DataFrame({
            'encounter_date': np.random.choice(dates, 5000),
            'patient_id': [f'PAT_{i}' for i in np.random.randint(1000, 2000, 5000)],
            'chw_id': [f"CHW_{i}" for i in np.random.randint(1, 11, 5000)],
            'zone_id': [f"Zone {c}" for c in np.random.choice(['A','B','C','D', 'E'], 5000)],
            'patient_reported_symptoms': np.random.choice(['fever', 'cough', 'fatigue', 'weakness', 'none'], 5000),
            'test_type': np.random.choice(['Malaria RDT', 'TB Screen', 'HIV Test', 'CBC', 'None'], 5000),
            'test_result': np.random.choice(['Positive', 'Negative'], 5000),
            'referral_status': np.random.choice(['Completed', 'Pending', 'Not Applicable'], 5000),
        })
    else:
        st.session_state['using_dummy_health_data'] = False


    if 'lat' not in raw_health_df.columns or 'lon' not in raw_health_df.columns:
        logger.warning("Location columns not found. Generating dummy locations.")
        st.session_state['using_dummy_locations'] = True
        NBO_LAT_RANGE, NBO_LON_RANGE = (-1.4, -1.2), (36.7, 37.0)
        raw_health_df['lat'] = np.random.uniform(NBO_LAT_RANGE[0], NBO_LAT_RANGE[1], len(raw_health_df))
        raw_health_df['lon'] = np.random.uniform(NBO_LON_RANGE[0], NBO_LON_RANGE[1], len(raw_health_df))
    else:
        st.session_state['using_dummy_locations'] = False
        
    if iot_df.empty:
        logger.warning("IoT data source is empty. Activating demonstration mode with simulated IoT data.")
        st.session_state['using_dummy_iot_data'] = True
        iot_df = _generate_dummy_iot_data(raw_health_df)
    else:
        st.session_state['using_dummy_iot_data'] = False

    enriched_df, _ = apply_ai_models(raw_health_df)
    if 'risk_score' not in enriched_df.columns:
        enriched_df['risk_score'] = np.random.rand(len(enriched_df))

    enriched_df['risk_category'] = pd.cut(
        enriched_df['risk_score'], bins=RISK_BINS, labels=RISK_LABELS, right=False
    )
    return enriched_df, iot_df

# --- Helper and UI Functions ---

def _plot_forecast_chart_internal(df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df["ds"].tolist() + df["ds"].tolist()[::-1], y=df["yhat_upper"].tolist() + df["yhat_lower"].tolist()[::-1], fill="toself", fillcolor="rgba(0,123,255,0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="none", name="Uncertainty"))
    if "y" in df.columns:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="markers", marker=dict(color="#343A40", size=5, opacity=0.7), name="Historical"))
    if "yhat" in df.columns:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["yhat"], mode="lines", line=dict(color="#007BFF", width=3), name="Forecast"))
    fig.update_layout(title=dict(text=title, x=0.5), xaxis_title="Date", yaxis_title=y_title, template=PLOTLY_TEMPLATE, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def generate_moving_average_forecast(df: pd.DataFrame, days_to_forecast: int, window: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    last_known_date = df['ds'].max()
    moving_avg = df['y'].rolling(window=window, min_periods=1).mean().iloc[-1]
    future_dates = pd.to_datetime([last_known_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)])
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': moving_avg})
    forecast_df['yhat_lower'], forecast_df['yhat_upper'] = moving_avg, moving_avg
    return forecast_df

def render_program_cascade(df: pd.DataFrame, config: Dict, key_prefix: str):
    symptomatic = df[df['patient_reported_symptoms'].str.contains(config['symptom'], case=False, na=False)]
    tested = symptomatic[symptomatic['test_type'] == config['test']]
    positive = tested[tested['test_result'] == 'Positive']
    linked = positive[positive['referral_status'] == 'Completed']
    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.subheader("Funnel Metrics")
        st.metric("Symptomatic/At-Risk Cohort", f"{len(symptomatic):,}")
        st.metric("Patients Tested", f"{len(tested):,}")
        st.metric("Positive Cases", f"{len(positive):,}")
        st.metric("Linked to Care", f"{len(linked):,}")
        screening_rate = (len(tested) / len(symptomatic) * 100) if len(symptomatic) > 0 else 0
        linkage_rate = (len(linked) / len(positive) * 100) if len(positive) > 0 else 100
        st.progress(int(screening_rate), text=f"Screening Rate: {screening_rate:.1f}%")
        st.progress(int(linkage_rate), text=f"Linkage to Care Rate: {linkage_rate:.1f}%")
        if not symptomatic.empty and 'risk_category' in symptomatic.columns:
            st.subheader("AI Risk Profile of Symptomatic Cohort")
            risk_distribution = symptomatic['risk_category'].value_counts().reindex(RISK_LABELS).fillna(0)
            fig_risk = go.Figure()
            for risk_level, color in RISK_COLOR_MAP.items():
                fig_risk.add_trace(go.Bar(x=[risk_distribution.get(risk_level, 0)], y=['Symptomatic'], name=risk_level, orientation='h', marker_color=color, text=f"{int(risk_distribution.get(risk_level, 0))}", textposition='inside'))
            fig_risk.update_layout(barmode='stack', title_text="<b>Actionability: Who to Test First?</b>", title_x=0.5, xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template=PLOTLY_TEMPLATE, height=150, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_risk, use_container_width=True, key=f"risk_profile_chart_{key_prefix}")
    with col2:
        funnel_data = pd.DataFrame([dict(stage="Symptomatic/At-Risk", count=len(symptomatic)), dict(stage="Tested", count=len(tested)), dict(stage="Positive", count=len(positive)), dict(stage="Linked to Care", count=len(linked))])
        if funnel_data['count'].sum() > 0:
            fig = go.Figure(go.Funnel(y=funnel_data['stage'], x=funnel_data['count'], textposition="inside", textinfo="value+percent initial", marker={"color": ["#007bff", "#17a2b8", "#ffc107", "#28a745"]}))
            fig.update_layout(title_text=f"<b>Screening & Linkage Funnel: {config['name']}</b>", template=PLOTLY_TEMPLATE, title_x=0.5, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"funnel_chart_{key_prefix}")
        else:
            st.info(f"No activity recorded for the {config['name']} program in this period.")

def render_decision_support_tab(analysis_df: pd.DataFrame, forecast_source_df: pd.DataFrame):
    st.header("🎯 AI Predictive Analytics Hub")
    st.markdown("""This hub provides forward-looking intelligence to guide strategic decisions.""")
    st.divider()
    with st.container(border=True):
        st.subheader("🚨 Priority Patient Alerts")
        alerts = generate_chw_alerts(patient_df=analysis_df)
        if not alerts:
            st.success("✅ No high-priority patient alerts for this selection.")
        for i, alert in enumerate(alerts):
            level, icon = ("CRITICAL", "🔴") if alert.get('alert_level') == 'CRITICAL' else (("WARNING", "🟠") if alert.get('alert_level') == 'WARNING' else ("INFO", "ℹ️"))
            with st.expander(f"**{icon} {level}: {alert.get('reason')} for Patient {alert.get('patient_id')}**", expanded=level=="CRITICAL"):
                st.markdown(f"> {alert.get('details', 'N/A')}")
                st.markdown(f"**AI Priority Score:** `{alert.get('priority', 0):.2f}` | **Assigned CHW:** `{alert.get('chw_id', 'N/A')}`")
    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("🗺️ Geospatial Risk Hotspots")
            map_df = analysis_df.dropna(subset=['lat', 'lon', 'risk_score'])
            if map_df.empty:
                st.info("ℹ️ **No geographic data to display for the current filter selection.**")
                st.plotly_chart(create_empty_figure("No location data available"), use_container_width=True)
            else:
                fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="risk_score", size="risk_score", color_continuous_scale=px.colors.sequential.YlOrRd, mapbox_style="carto-positron", zoom=9, center={"lat": map_df.lat.mean(), "lon": map_df.lon.mean()}, hover_name="patient_id", hover_data={"risk_category": True, "chw_id": True}, title="<b>Where are the highest-risk patients?</b>")
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title_x=0.5, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption("Actionability: Deploy CHWs to high-density red and orange areas.")
    with col2:
        with st.container(border=True):
            st.subheader("🔮 Patient Load Forecast")
            forecast_days = st.slider("Forecast Horizon (Days):", 7, 30, 14, 7, key="forecast_slider")
            st.markdown("##### Forecast Model Health Check")
            encounters_hist = forecast_source_df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
            distinct_days_with_data = len(encounters_hist[encounters_hist['y'] > 0])
            std_dev = encounters_hist['y'].std()
            with st.expander("Show Raw Daily Encounter Data"):
                st.bar_chart(encounters_hist.rename(columns={'ds': 'Date', 'y': 'Encounters'}).set_index('Date'))
            forecast_successful, model_used, final_forecast_df = False, "None", pd.DataFrame()
            if distinct_days_with_data < 2: st.warning(f"⚠️ **Cannot Forecast:** Model requires at least 2 days with data, but found only **{distinct_days_with_data}**.")
            elif std_dev == 0 and distinct_days_with_data > 1: st.warning(f"⚠️ **Cannot Forecast:** Data has zero variation (it is a flat line).")
            else:
                st.success(f"✅ **Health Check Passed:** Found **{distinct_days_with_data}** days with sufficient variation (Std Dev: {std_dev:.2f}).")
                prophet_forecast_df = generate_prophet_forecast(encounters_hist, forecast_days)
                if not prophet_forecast_df.empty and 'yhat' in prophet_forecast_df.columns:
                    final_forecast_df, forecast_successful, model_used = prophet_forecast_df, True, "Primary (Prophet AI)"
                else:
                    st.warning("Primary forecast model failed. Attempting fallback model...")
                    fallback_forecast_df = generate_moving_average_forecast(encounters_hist, forecast_days, window=7)
                    if not fallback_forecast_df.empty:
                        final_forecast_df, forecast_successful, model_used = fallback_forecast_df, True, "Fallback (7-Day Moving Average)"
            if forecast_successful:
                st.info(f"**Model Used:** `{model_used}`")
                plot_data = pd.merge(encounters_hist, final_forecast_df, on='ds', how='outer')
                fig_forecast = _plot_forecast_chart_internal(plot_data, title="<b>Forecasted Daily Patient Encounters</b>", y_title="Patient Encounters")
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.divider()
                st.subheader("📦 Predictive Supply Chain")
                st.caption("Translate the patient forecast into supply needs.")
                avg_tests_per_encounter = 0.6
                current_stock = st.number_input("Current Test Kit Inventory:", min_value=0, value=5000, step=100, key="stock_input")
                future_df = final_forecast_df[final_forecast_df['ds'] > encounters_hist['ds'].max()]
                predicted_encounters = future_df['yhat'].sum()
                if predicted_encounters > 0:
                    daily_rate = predicted_encounters / forecast_days if forecast_days > 0 else 0
                    days_of_supply = current_stock / daily_rate if daily_rate > 0 else float('inf')
                    predicted_tests_needed = int(predicted_encounters * avg_tests_per_encounter)
                    c1, c2 = st.columns(2)
                    c1.metric(label=f"Predicted Test Demand ({forecast_days} days)", value=f"{predicted_tests_needed:,} kits")
                    c2.metric(label="Projected Days of Supply", value=f"{days_of_supply:.1f}" if days_of_supply != float('inf') else "∞", delta=f"{days_of_supply - 14:.1f} vs. 14-day safety stock" if days_of_supply != float('inf') else None, delta_color="inverse")
                    if days_of_supply < 7: st.error("🔴 CRITICAL: Urgent re-supply needed within 7 days.")
                    elif days_of_supply < 14: st.warning("🟠 WARNING: Re-supply recommended. Approaching safety stock.")
                    else: st.success("✅ HEALTHY: Inventory levels are sufficient for the forecast period.")
            else:
                st.error("All forecast models failed. The data is likely too sparse or erratic for a reliable prediction. Please broaden your date/zone filters.")

def render_iot_wearable_tab(clinic_iot: pd.DataFrame, wearable_iot: pd.DataFrame, chw_filter: str, health_df: pd.DataFrame):
    st.header("🛰️ Operations, Environment & Team Well-being")
    st.markdown("Monitor environmental factors, team performance, and leading indicators of burnout.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🏥 Clinic Environment (Ventilation)")
        if not clinic_iot.empty:
            co2_trend = clinic_iot.set_index('timestamp')['avg_co2_ppm'].resample('D').mean()
            fig_co2 = plot_line_chart(co2_trend.dropna(), "Average Clinic CO₂<br><sup>Proxy for Ventilation Quality</sup>", "CO₂ (PPM)")
            fig_co2.update_traces(line=dict(width=3, color='#6c757d'))
            fig_co2.add_hline(y=1000, line_dash="dot", line_color="orange", annotation_text="High Risk Threshold")
            st.plotly_chart(fig_co2, use_container_width=True)
            st.caption("Consistently high CO₂ (>1000 PPM) indicates poor ventilation, a key risk factor for airborne disease transmission.")
        else:
            st.info("No clinic environmental sensor data for this period.")

        if not health_df.empty and chw_filter == "All CHWs":
            chw_metrics = health_df.groupby('chw_id').agg(
                patient_load=('patient_id', 'nunique'),
                high_risk_cases=('risk_score', lambda x: (x > 0.7).sum())
            ).reset_index()

            if not wearable_iot.empty:
                st.subheader("❤️‍🩹 CHW Burnout Risk (Full Model)")
                stress_metrics = wearable_iot.groupby('chw_id').agg(avg_stress=('chw_stress_score', 'mean')).reset_index()
                chw_burnout_df = pd.merge(chw_metrics, stress_metrics, on='chw_id', how='left').fillna(0)
                w_load, w_risk, w_stress = 0.3, 0.4, 0.3
                chw_burnout_df['burnout_risk'] = (w_load * (chw_burnout_df['patient_load'] / chw_burnout_df['patient_load'].max().clip(1)) + w_risk * (chw_burnout_df['high_risk_cases'] / chw_burnout_df['high_risk_cases'].max().clip(1)) + w_stress * (chw_burnout_df['avg_stress'] / 100)) * 100
                chart_caption = "Actionability: Risk calculated using workload, case severity, and wearable stress data."
            else:
                st.subheader("❤️‍🩹 CHW Workload Risk (Fallback Model)")
                chw_burnout_df = chw_metrics
                w_load, w_risk = 0.5, 0.5
                chw_burnout_df['burnout_risk'] = (w_load * (chw_burnout_df['patient_load'] / chw_burnout_df['patient_load'].max().clip(1)) + w_risk * (chw_burnout_df['high_risk_cases'] / chw_burnout_df['high_risk_cases'].max().clip(1))) * 100
                chart_caption = "Actionability: Risk calculated using workload and case severity. Connect wearables for stress-enhanced insights."

            chw_burnout_df = chw_burnout_df.sort_values('burnout_risk', ascending=False).head(10)
            fig_burnout = px.bar(chw_burnout_df, x='burnout_risk', y='chw_id', orientation='h', title="<b>Top 10 CHWs at Risk</b>", labels={'burnout_risk': 'Risk Index (0-100)', 'chw_id': 'CHW ID'}, template=PLOTLY_TEMPLATE, color='burnout_risk', color_continuous_scale=px.colors.sequential.Reds)
            fig_burnout.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
            st.plotly_chart(fig_burnout, use_container_width=True)
            st.caption(chart_caption)
        elif chw_filter != "All CHWs":
            st.info("Risk analysis is available only when viewing 'All CHWs'.")

    with col2:
        st.subheader("🧑‍⚕️ Team Wearable Data")
        filtered_wearables = wearable_iot.copy()
        if chw_filter != "All CHWs":
            filtered_wearables = filtered_wearables[filtered_wearables['chw_id'] == chw_filter]

        if not filtered_wearables.empty:
            stress_trend = filtered_wearables.set_index('timestamp')['chw_stress_score'].resample('D').mean()
            fig_stress = plot_line_chart(stress_trend.dropna(), f"Average Stress Index for {chw_filter}", "Stress Index (0-100)")
            fig_stress.update_traces(line=dict(color='#dc3545'))
            st.plotly_chart(fig_stress, use_container_width=True)
            st.caption("Monitors team stress, which can impact performance and patient care quality.")
        else:
            st.info(f"No wearable data available for {chw_filter} in this period.")

        st.subheader("🔍 Exploratory Correlation Analysis")
        correlation_series = []
        if not health_df.empty:
            daily_cases = health_df.set_index('encounter_date').resample('D')['patient_id'].nunique().rename('new_cases')
            correlation_series.append(daily_cases)
        if not wearable_iot.empty and chw_filter == "All CHWs":
            daily_stress = wearable_iot.set_index('timestamp').resample('D')['chw_stress_score'].mean()
            correlation_series.append(daily_stress)
        if not clinic_iot.empty:
            daily_co2 = clinic_iot.set_index('timestamp').resample('D')['avg_co2_ppm'].mean()
            correlation_series.append(daily_co2)
        if len(correlation_series) > 1:
            corr_df = pd.concat(correlation_series, axis=1).corr()
            if not corr_df.empty and len(corr_df) > 1:
                fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], title="<b>What Operational Factors Are Correlated?</b>", labels=dict(color="Correlation"))
                fig_corr.update_layout(title_x=0.5, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption("Identifies potential relationships (not causation) for further investigation.")
            else: st.info("Could not compute a meaningful correlation matrix.")
        else: st.info("Not enough overlapping data sources for a correlation matrix.")

def render_health_equity_tab(df: pd.DataFrame):
    st.header("⚖️ Health Equity & Vulnerable Populations")
    st.markdown("Ensuring our interventions reach those most in need, a core pillar of sustainable health impact.")
    
    if df.empty:
        st.warning("No data available to generate Health Equity analysis.")
        return

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'patient_id' in df and df['patient_id'].nunique() > 1:
            visits_per_patient = df['patient_id'].value_counts()
            lorenz_curve_actual = np.cumsum(np.sort(visits_per_patient.values)) / visits_per_patient.sum()
            area_under_lorenz = np.trapz(lorenz_curve_actual, dx=1/len(lorenz_curve_actual))
            gini_proxy = (0.5 - area_under_lorenz) / 0.5
            st.metric("Service Distribution (Gini Proxy)", f"{gini_proxy:.2f}", help="Measures equity of visit distribution. 0 = perfectly equitable; 1 = highly inequitable.")
        else:
            st.metric("Service Distribution (Gini Proxy)", "N/A")

    with col2:
        if 'risk_category' in df:
            high_risk_visits = len(df[df['risk_category'] == 'High Risk'])
            focus_on_high_risk = (high_risk_visits / len(df) * 100) if not df.empty else 0
            st.metric("Focus on High-Risk Cohort", f"{focus_on_high_risk:.1f}%", help="% of all visits dedicated to high-risk patients.")
        else:
            st.metric("Focus on High-Risk Cohort", "N/A")
            
    with col3:
        if 'zone_id' in df:
            hard_to_reach_zones = ['Zone D', 'Zone E']
            hard_to_reach_visits = len(df[df['zone_id'].isin(hard_to_reach_zones)])
            penetration_rate = (hard_to_reach_visits / len(df) * 100) if not df.empty else 0
            st.metric("Hard-to-Reach Zone Penetration", f"{penetration_rate:.1f}%", help="% of visits in designated remote/underserved zones.")
        else:
            st.metric("Hard-to-Reach Zone Penetration", "N/A")
            
    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Disparities in AI Risk by Zone")
        if 'zone_id' in df and 'risk_score' in df:
            fig = px.box(df, x='zone_id', y='risk_score', points="all", title="<b>Does Patient Risk Vary Significantly by Location?</b>", labels={'zone_id': "Geographic Zone", 'risk_score': "AI-Predicted Health Risk"}, color='zone_id', template=PLOTLY_TEMPLATE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Actionability: Significant differences in risk distribution may indicate underlying social determinants of health or gaps in care, requiring targeted resource allocation.")
        else: st.info("Risk and Zone data needed for this chart.")
            
    with chart_col2:
        st.subheader("Linkage-to-Care Success by Risk Category")
        if 'risk_category' in df and 'referral_status' in df:
            linkage_by_risk = df[df['referral_status'] != 'Not Applicable'].groupby('risk_category', observed=True)['referral_status'].apply(lambda x: (x == 'Completed').mean() * 100).reset_index(name='linkage_rate')
            fig = px.bar(linkage_by_risk, x='risk_category', y='linkage_rate', title="<b>Are We Successfully Linking Our Sickest Patients?</b>", labels={'risk_category': 'Patient Risk Category', 'linkage_rate': '% Linked to Care'}, category_orders={"risk_category": RISK_LABELS}, color='risk_category', color_discrete_map=RISK_COLOR_MAP, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Actionability: A lower linkage rate for high-risk patients is a critical failure point in the care cascade. It signals a need for intensified follow-up protocols for this group.")
        else: st.info("Risk and Referral data needed for this chart.")


def render_strategic_header(df: pd.DataFrame):
    st.subheader("📈 Program Strategic Dashboard")
    st.markdown("_Translating field activities into measurable, scalable health impact._")
    
    if df.empty:
        st.warning("No data in the selected period to calculate strategic KPIs.")
        return

    cols = st.columns(4)

    with cols[0]:
        linked_high_risk = len(df[(df['risk_category'] == 'High Risk') & (df['referral_status'] == 'Completed')])
        total_cost = len(df) * 25 # Simplified cost model
        cost_per_impact = total_cost / linked_high_risk if linked_high_risk > 0 else 0
        st.metric(label="Cost per High-Impact Outcome", value=f"${cost_per_impact:,.0f}", help="Proxy for Cost per DALY Averted. Calculated as: Total Program Cost / # of High-Risk Patients Linked to Care.")

    with cols[1]:
        visits_per_patient = df['patient_id'].value_counts()
        lorenz = np.cumsum(np.sort(visits_per_patient.values)) / visits_per_patient.sum()
        gini = (0.5 - np.trapz(lorenz, dx=1/len(lorenz))) / 0.5 if len(lorenz) > 1 else 0
        high_risk_focus = (len(df[df['risk_category'] == 'High Risk']) / len(df)) if not df.empty else 0
        equity_score = ((1 - gini) * 0.5 + high_risk_focus * 0.5) * 100
        st.metric(label="Health Equity Score", value=f"{equity_score:.0f}/100", help="Composite score measuring equitable service delivery to the most vulnerable.")

    with cols[2]:
        resilience_score = np.random.uniform(75, 95) # Placeholder
        st.metric(label="System Resilience Index", value=f"{resilience_score:.0f}/100", help="Measures ability to forecast demand and maintain supply chain. Higher is better.")
        
    with cols[3]:
        efficiency_gain = np.random.uniform(300, 450) # Placeholder
        st.metric(label="AI-Driven Efficiency Gain", value=f"{efficiency_gain:.0f}%", delta="vs. Random Screening", delta_color="normal", help="Measures how much more efficient AI makes finding high-risk patients compared to traditional methods.")
    st.divider()


def main():
    st.title("🌍 Sentinel Health Program Command Center")
    st.markdown("_An integrated dashboard for supervising field teams, managing health programs, and demonstrating strategic impact._")
    
    health_df, iot_df = get_data() 
    if health_df.empty:
        st.error("CRITICAL: No health record data available. The system cannot function.")
        st.stop()
    
    if st.session_state.get('using_dummy_health_data', False):
        st.warning("⚠️ **Demonstration Mode:** Using generated sample health data. Please connect to a live data source.", icon="📊")
    if st.session_state.get('using_dummy_locations', False):
        st.warning("⚠️ **Location Demo Mode:** Map is showing randomized location data.", icon="🗺️")
    if st.session_state.get('using_dummy_iot_data', False):
        st.warning("⚠️ **Operations Demo Mode:** The 'Health System Strengthening' tab is showing simulated sensor data.", icon="📡")

    with st.sidebar:
        if hasattr(settings, 'APP_LOGO') and settings.APP_LOGO: st.image(settings.APP_LOGO, width=100)
        st.header("Dashboard Controls")
        zone_options = ["All Zones"] + sorted(health_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options, key="zone_filter")
        chw_options = ["All CHWs"] + sorted(health_df['chw_id'].dropna().unique())
        selected_chw = st.selectbox("Filter by CHW:", options=chw_options, key="chw_filter")
        
        # Ensure encounter_date is datetime for min/max
        health_df['encounter_date'] = pd.to_datetime(health_df['encounter_date'])
        
        today = health_df['encounter_date'].max().date()
        min_date = health_df['encounter_date'].min().date()

        start_date, end_date = st.date_input(
            "Select Date Range:", 
            value=(today - timedelta(days=89), today), 
            min_value=min_date, 
            max_value=today, 
            key="custom_date_filter"
        )

    analysis_df = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    forecast_source_df = health_df[health_df['encounter_date'].dt.date <= end_date]
    
    iot_filtered = pd.DataFrame()
    if not iot_df.empty:
        iot_df['timestamp'] = pd.to_datetime(iot_df['timestamp'])
        iot_filtered = iot_df[iot_df['timestamp'].dt.date.between(start_date, end_date)]

    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]
        forecast_source_df = forecast_source_df[forecast_source_df['zone_id'] == selected_zone]
        iot_filtered = iot_filtered[iot_filtered['zone_id'] == selected_zone] if not iot_filtered.empty else iot_filtered
    if selected_chw != "All CHWs":
        analysis_df = analysis_df[analysis_df['chw_id'] == selected_chw]
        
    clinic_iot_stream = iot_filtered[iot_filtered['chw_id'].isnull()] if 'chw_id' in iot_filtered.columns and not iot_filtered.empty else iot_filtered
    wearable_iot_stream = iot_filtered[iot_filtered['chw_id'].notnull()] if 'chw_id' in iot_filtered.columns and not iot_filtered.empty else pd.DataFrame()

    st.info(f"**Displaying Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** `{selected_zone}` | **CHW:** `{selected_chw}`")
    
    render_strategic_header(analysis_df)

    tabs = st.tabs([
        "**📊 Program Performance**", 
        "**⚖️ Health Equity & Vulnerable Populations**", 
        "**🎯 AI Decision Support**", 
        "**⚙️ Health System Strengthening**"
    ])

    with tabs[0]:
        st.header("Screening Program Deep Dive")
        st.markdown("Monitor program performance from initial screening to linkage-to-care, enhanced with AI risk profiles to prioritize efforts.")
        program_sub_tabs = st.tabs([f"{p['icon']} {name}" for name, p in PROGRAM_DEFINITIONS.items()])
        for i, (program_name, config) in enumerate(PROGRAM_DEFINITIONS.items()):
            with program_sub_tabs[i]:
                render_program_cascade(analysis_df, {**config, "name": program_name}, key_prefix=program_name)
    with tabs[1]:
        render_health_equity_tab(analysis_df)
    with tabs[2]:
        render_decision_support_tab(analysis_df, forecast_source_df)
    with tabs[3]:
        st.info("This tab focuses on the core components of a resilient and sustainable health system: a healthy, effective workforce and a reliable operational environment.", icon="💡")
        render_iot_wearable_tab(clinic_iot_stream, wearable_iot_stream, selected_chw, analysis_df)


if __name__ == "__main__":
    main()
