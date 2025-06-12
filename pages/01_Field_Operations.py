# sentinel_project_root/pages/01_Field_Operations.py
# FINAL, SELF-CONTAINED, AND VISUALIZATION-ENHANCED VERSION
# SME UPGRADE: Added intelligent data augmentation for missing location data.

import logging
from datetime import date, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Core Sentinel Imports ---
# These are assumed to exist and work as described in the original file.
from analytics import apply_ai_models, generate_chw_alerts, generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_donut_chart, plot_line_chart)

# --- Page Setup ---
st.set_page_config(page_title="Field Command Center", page_icon="📡", layout="wide")

# --- SME ACTIONABILITY UPGRADE: Logging Configuration ---
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

# --- SME VISUALIZATION & KPI UPGRADE: Constants for professional, consistent styling and logic ---
PLOTLY_TEMPLATE = "plotly_white"
RISK_BINS = [-np.inf, 0.4, 0.7, np.inf]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLOR_MAP = {"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"}


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and enriches data.
    SME UPGRADE: Intelligently augments missing location data for demonstration
    and robustness, while flagging the issue.
    """
    raw_health_df = load_health_records()
    iot_df = load_iot_records()
    if raw_health_df.empty:
        return pd.DataFrame(), iot_df

    # --- SME ROBUSTNESS UPGRADE: Handle missing location data ---
    if 'lat' not in raw_health_df.columns or 'lon' not in raw_health_df.columns:
        logger.warning("Location columns ('lat', 'lon') not found in source data. Generating dummy locations for demonstration.")
        st.session_state['using_dummy_locations'] = True # Flag for the UI
        
        # Generate plausible coordinates (e.g., around Nairobi, Kenya)
        NBO_LAT_RANGE = (-1.4, -1.2)
        NBO_LON_RANGE = (36.7, 37.0)
        num_records = len(raw_health_df)
        
        raw_health_df['lat'] = np.random.uniform(NBO_LAT_RANGE[0], NBO_LAT_RANGE[1], num_records)
        raw_health_df['lon'] = np.random.uniform(NBO_LON_RANGE[0], NBO_LON_RANGE[1], num_records)
    else:
        st.session_state['using_dummy_locations'] = False

    enriched_df, _ = apply_ai_models(raw_health_df)
    if 'risk_score' in enriched_df.columns:
        enriched_df['risk_category'] = pd.cut(
            enriched_df['risk_score'], bins=RISK_BINS, labels=RISK_LABELS, right=False
        )
    else:
        enriched_df['risk_score'] = 0.1
        enriched_df['risk_category'] = "Low Risk"
    return enriched_df, iot_df

# --- SME FUNCTIONALITY UPGRADE: Internalized plotting & robust forecasting logic ---
# ... (This section remains unchanged from the previous version) ...
def _plot_forecast_chart_internal(df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["ds"].tolist() + df["ds"].tolist()[::-1], y=df["yhat_upper"].tolist() + df["yhat_lower"].tolist()[::-1],
            fill="toself", fillcolor="rgba(0,123,255,0.2)", line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="none", name="Uncertainty"
        ))
    if "y" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["y"], mode="markers", marker=dict(color="#343A40", size=5, opacity=0.7), name="Historical"
        ))
    if "yhat" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["yhat"], mode="lines", line=dict(color="#007BFF", width=3), name="Forecast"
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5), xaxis_title="Date", yaxis_title=y_title, template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def generate_moving_average_forecast(df: pd.DataFrame, days_to_forecast: int, window: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    last_known_date = df['ds'].max()
    moving_avg = df['y'].rolling(window=window, min_periods=1).mean().iloc[-1]
    future_dates = pd.to_datetime([last_known_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)])
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': moving_avg})
    forecast_df['yhat_lower'] = moving_avg
    forecast_df['yhat_upper'] = moving_avg
    return forecast_df

# --- UI Rendering Components ---
# ... (render_program_cascade remains unchanged) ...
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
                fig_risk.add_trace(go.Bar(
                    x=[risk_distribution.get(risk_level, 0)], y=['Symptomatic'], name=risk_level, orientation='h', marker_color=color, text=f"{int(risk_distribution.get(risk_level, 0))}", textposition='inside'
                ))
            fig_risk.update_layout(
                barmode='stack', title_text="<b>Actionability: Who to Test First?</b>", title_x=0.5, xaxis=dict(visible=False), yaxis=dict(visible=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template=PLOTLY_TEMPLATE, height=150, margin=dict(t=50, b=10, l=10, r=10)
            )
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
    st.markdown("""This hub provides forward-looking intelligence to guide strategic decisions, from individual patient prioritization to resource planning.""")
    st.divider()

    with st.container(border=True):
        st.subheader("🚨 Priority Patient Alerts")
        alerts = generate_chw_alerts(patient_df=analysis_df)
        if not alerts:
            st.success("✅ No high-priority patient alerts for this selection.")
        for i, alert in enumerate(alerts):
            level, icon = ("CRITICAL", "🔴") if alert.get('alert_level') == 'CRITICAL' else \
                          (("WARNING", "🟠") if alert.get('alert_level') == 'WARNING' else ("INFO", "ℹ️"))
            with st.expander(f"**{icon} {level}: {alert.get('reason')} for Patient {alert.get('patient_id')}**", expanded=level=="CRITICAL"):
                st.markdown(f"> {alert.get('details', 'N/A')}")
                st.markdown(f"**AI Priority Score:** `{alert.get('priority', 0):.2f}` | **Assigned CHW:** `{alert.get('chw_id', 'N/A')}`")

    st.divider()
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("🗺️ Geospatial Risk Hotspots")
            
            # --- SME DIAGNOSTIC UPGRADE: This logic now handles the case where data exists but is null for the filter ---
            map_df = analysis_df.dropna(subset=['lat', 'lon', 'risk_score'])
            if map_df.empty:
                st.info("ℹ️ **No geographic data to display for the current filter selection.**")
                # Add an empty map figure to maintain layout consistency
                st.plotly_chart(create_empty_figure("No location data available"), use_container_width=True)
            else:
                fig_map = px.scatter_mapbox(
                    map_df, lat="lat", lon="lon", color="risk_score",
                    size="risk_score",
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    mapbox_style="carto-positron", zoom=9, # Zoom out slightly for generated data
                    center={"lat": map_df.lat.mean(), "lon": map_df.lon.mean()},
                    hover_name="patient_id", hover_data={"risk_category": True, "chw_id": True},
                    title="<b>Where are the highest-risk patients?</b>"
                )
                fig_map.update_layout(
                    margin={"r":0,"t":40,"l":0,"b":0}, title_x=0.5,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption("Actionability: Deploy CHWs to high-density red and orange areas.")
    
    with col2:
        with st.container(border=True):
            st.subheader("🔮 Patient Load Forecast")
            # ... (Rest of the forecast logic remains unchanged) ...
            forecast_days = st.slider("Forecast Horizon (Days):", 7, 30, 14, 7, key="forecast_slider")
            st.markdown("##### Forecast Model Health Check")
            encounters_hist = forecast_source_df.set_index('encounter_date').resample('D').size().reset_index(name='count').rename(columns={'encounter_date': 'ds', 'count': 'y'})
            distinct_days_with_data = len(encounters_hist[encounters_hist['y'] > 0])
            std_dev = encounters_hist['y'].std()
            with st.expander("Show Raw Daily Encounter Data"):
                st.bar_chart(encounters_hist.rename(columns={'ds': 'Date', 'y': 'Encounters'}).set_index('Date'))
            forecast_successful, model_used, final_forecast_df = False, "None", pd.DataFrame()
            if distinct_days_with_data < 2:
                st.warning(f"⚠️ **Cannot Forecast:** Model requires at least 2 days with data, but found only **{distinct_days_with_data}**.")
            elif std_dev == 0 and distinct_days_with_data > 1:
                st.warning(f"⚠️ **Cannot Forecast:** Data has zero variation (it is a flat line).")
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


# --- The rest of the file (render_iot_wearable_tab, main) remains largely unchanged ---
# ... (render_iot_wearable_tab is unchanged) ...
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
        st.subheader("❤️‍🩹 CHW Well-being & Burnout Risk")
        if not health_df.empty and chw_filter == "All CHWs":
            chw_metrics = health_df.groupby('chw_id').agg(patient_load=('patient_id', 'nunique'), high_risk_cases=('risk_score', lambda x: (x > 0.7).sum())).reset_index()
            if not wearable_iot.empty:
                stress_metrics = wearable_iot.groupby('chw_id').agg(avg_stress=('chw_stress_score', 'mean')).reset_index()
                chw_burnout_df = pd.merge(chw_metrics, stress_metrics, on='chw_id', how='inner')
                w_load, w_risk, w_stress = 0.3, 0.4, 0.3
                if not chw_burnout_df.empty:
                    chw_burnout_df['burnout_risk'] = (w_load * chw_burnout_df['patient_load'] / chw_burnout_df['patient_load'].max().clip(1) + w_risk * chw_burnout_df['high_risk_cases'] / chw_burnout_df['high_risk_cases'].max().clip(1) + w_stress * chw_burnout_df['avg_stress'] / 100) * 100
                    chw_burnout_df = chw_burnout_df.sort_values('burnout_risk', ascending=False).head(10)
                    fig_burnout = px.bar(chw_burnout_df, x='burnout_risk', y='chw_id', orientation='h', title="<b>Top 10 CHWs at Risk of Burnout</b>", labels={'burnout_risk': 'Burnout Risk Index (0-100)', 'chw_id': 'CHW ID'}, template=PLOTLY_TEMPLATE, color='burnout_risk', color_continuous_scale=px.colors.sequential.Reds)
                    fig_burnout.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
                    st.plotly_chart(fig_burnout, use_container_width=True)
                    st.caption("Actionability: Proactively support CHWs with the highest risk index through workload adjustments, check-ins, or additional resources.")
                else: st.info("Not enough overlapping health and wearable data to calculate CHW burnout risk.")
            else: st.info("No CHW wearable data available for burnout calculation.")
        elif chw_filter != "All CHWs": st.info("Burnout risk analysis is available only when viewing 'All CHWs'.")
        else: st.info("Not enough data to calculate CHW burnout risk.")
    with col2:
        st.subheader("🧑‍⚕️ Team Wearable Data")
        if chw_filter != "All CHWs":
            wearable_iot = wearable_iot[wearable_iot['chw_id'] == chw_filter]
        if not wearable_iot.empty:
            stress_trend = wearable_iot.set_index('timestamp')['chw_stress_score'].resample('D').mean()
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
                st.caption("Identifies potential relationships (not causation) for further investigation or hypothesis generation.")
            else: st.info("Could not compute a meaningful correlation matrix.")
        else: st.info("Not enough overlapping data sources for a correlation matrix.")


def main():
    st.title("📡 Field Operations Command Center")
    st.markdown("_An integrated dashboard for supervising team activity, managing screening programs, and forecasting operational needs._")
    health_df, iot_df = get_data()
    if health_df.empty: st.error("CRITICAL: No health record data available. The system cannot function."); st.stop()

    # --- SME UX UPGRADE: Display a persistent warning if using dummy data ---
    if st.session_state.get('using_dummy_locations', False):
        st.warning(
            "⚠️ **Demonstration Mode:** The map is displaying randomized location data because "
            "the original `lat` and `lon` columns were not found in your source file. "
            "Please fix the upstream data pipeline to see real patient locations.",
            icon="🗺️"
        )
    
    with st.sidebar:
        if hasattr(settings, 'APP_LOGO') and settings.APP_LOGO:
            st.image(settings.APP_LOGO, width=100)
        st.header("Dashboard Controls")
        zone_options = ["All Zones"] + sorted(health_df['zone_id'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options, key="zone_filter")
        chw_options = ["All CHWs"] + sorted(health_df['chw_id'].dropna().unique())
        selected_chw = st.selectbox("Filter by CHW:", options=chw_options, key="chw_filter")
        today = health_df['encounter_date'].max().date()
        date_range_options = ["Last 30 Days", "Last 90 Days", "Year to Date", "Custom"]
        selected_range = st.selectbox("Select Date Range:", options=date_range_options, index=0, key="date_range_filter")
        if selected_range == "Custom":
            start_date, end_date = st.date_input("Select Date Range:", value=(today - timedelta(days=29), today), min_value=health_df['encounter_date'].min().date(), max_value=today, key="custom_date_filter")
        elif selected_range == "Last 30 Days":
            start_date, end_date = today - timedelta(days=29), today
        elif selected_range == "Last 90 Days":
            start_date, end_date = today - timedelta(days=89), today
        else:
            start_date, end_date = date(today.year, 1, 1), today

    # --- Data Filtering Logic ---
    analysis_df = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    forecast_source_df = health_df[health_df['encounter_date'].dt.date <= end_date]
    iot_filtered = iot_df[iot_df['timestamp'].dt.date.between(start_date, end_date)] if not iot_df.empty else pd.DataFrame()

    if selected_zone != "All Zones":
        analysis_df = analysis_df[analysis_df['zone_id'] == selected_zone]
        forecast_source_df = forecast_source_df[forecast_source_df['zone_id'] == selected_zone]
        iot_filtered = iot_filtered[iot_filtered['zone_id'] == selected_zone] if not iot_filtered.empty else iot_filtered
    if selected_chw != "All CHWs":
        analysis_df = analysis_df[analysis_df['chw_id'] == selected_chw]
        forecast_source_df = forecast_source_df[forecast_source_df['chw_id'] == selected_chw]

    clinic_iot_stream = iot_filtered[iot_filtered['chw_id'].isnull()] if 'chw_id' in iot_filtered.columns and not iot_filtered.empty else iot_filtered
    wearable_iot_stream = iot_filtered[iot_filtered['chw_id'].notnull()] if 'chw_id' in iot_filtered.columns and not iot_filtered.empty else pd.DataFrame()

    st.info(f"**Displaying Data For:** `{start_date:%d %b %Y}` to `{end_date:%d %b %Y}` | **Zone:** `{selected_zone}` | **CHW:** `{selected_chw}`")
    st.divider()

    # --- Main Tab Layout ---
    program_tab_list = [f"{p['icon']} {name}" for name, p in PROGRAM_DEFINITIONS.items()]
    tabs = st.tabs(["**📊 Program Performance**", "**🎯 AI Decision Support**", "**🛰️ Operations & Well-being**"])

    with tabs[0]:
        st.header("Screening Program Deep Dive")
        st.markdown("Monitor program performance from initial screening to linkage-to-care, enhanced with AI risk profiles to prioritize efforts.")
        program_sub_tabs = st.tabs(program_tab_list)
        for i, (program_name, config) in enumerate(PROGRAM_DEFINITIONS.items()):
            with program_sub_tabs[i]:
                render_program_cascade(analysis_df, {**config, "name": program_name}, key_prefix=program_name)
    with tabs[1]:
        render_decision_support_tab(analysis_df, forecast_source_df)
    with tabs[2]:
        render_iot_wearable_tab(clinic_iot_stream, wearable_iot_stream, selected_chw, analysis_df)

if __name__ == "__main__":
    main()
