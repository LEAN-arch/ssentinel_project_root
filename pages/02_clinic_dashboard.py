# sentinel_project_root/pages/02_clinic_dashboard.py
# SME-EVALUATED AND OPTIMIZED VERSION (V4)

"""
Streamlit dashboard page for Clinic Operations and Management.

Provides a comprehensive console for monitoring:
- Key Performance Indicators (KPIs) with period-over-period analysis.
- Local epidemiological trends based on patient-reported symptoms.
- Laboratory and testing performance, including turnaround times and rejection rates.
- Medical supply chain forecasting for critical items.
- Patient risk stratification using AI-driven scores.
- Real-time environmental monitoring of clinic facilities.

SME Revisions (V4):
- Performance: Added caching to the supply forecasting to prevent re-computation on every UI interaction.
- Maintainability: Refactored UI rendering logic for each tab into separate functions for clarity.
- Robustness: Added commentary on centralizing business logic (e.g., 'is_rejected' column) into data_processing layers.
- Minor robustness improvements in utility functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Tuple, Optional, Dict, Union, List
import os
import plotly.graph_objects as go

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data
    from data_processing.aggregation import get_clinic_summary_kpis
    from analytics.orchestrator import apply_ai_models
    from analytics.supply_forecasting import generate_simple_supply_forecast
    from analytics.alerting import get_patient_alerts_for_clinic
    from visualization.plots import plot_bar_chart, plot_donut_chart, plot_annotated_line_chart
except ImportError as e:
    st.error(
        "Fatal Error: A required module could not be imported.\n"
        f"Details: {e}\n\n"
        "This is likely due to an incorrect project structure or missing dependencies. "
        "Ensure you are running Streamlit from the `sentinel_project_root` directory and have installed all required packages."
    )
    st.stop()


# --- Self-Contained Data Science & Visualization Logic ---

def get_trend_data(
    df: pd.DataFrame,
    value_col: str,
    period: str,
    date_col: str = 'encounter_date',
    agg_func: Union[str, callable] = 'mean'
) -> pd.Series:
    """
    Calculates a trend series by resampling time-series data. This version is
    robust and accepts different aggregation functions.
    """
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=float)

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, value_col])

    if df_copy.empty:
        return pd.Series(dtype=float)

    try:
        trend_series = df_copy.set_index(date_col)[value_col].resample(period).agg(agg_func)
        return trend_series
    except Exception as e:
        logger.error(f"Error during resampling in get_trend_data: {e}")
        return pd.Series(dtype=float)

def create_sparkline_bytes(data: pd.Series, color: str) -> Optional[bytes]:
    """Creates a compact sparkline chart and returns it as PNG bytes."""
    # <<< SME REVISION >>> Added isna().all() check for robustness.
    if data is None or data.empty or data.isna().all():
        return None
    fig = go.Figure(go.Scatter(
        x=data.index, y=data, mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)"
    ))
    fig.update_layout(
        width=150, height=50, margin=dict(l=0, r=0, t=5, b=5),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception as e:
        logger.warning(f"Could not generate sparkline image. Is 'kaleido' installed? Error: {e}")
        return None

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_kpi_analysis_table(full_df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Performs a period-over-period KPI analysis, calculates change, and generates trend sparklines."""
    if full_df.empty:
        return pd.DataFrame()

    current_period_df = full_df[full_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_days = max((end_date - start_date).days + 1, 1)
    prev_start_date = start_date - timedelta(days=period_days)
    prev_end_date = start_date - timedelta(days=1)
    previous_period_df = full_df[full_df['encounter_date'].dt.date.between(prev_start_date, prev_end_date)]

    kpi_current = get_clinic_summary_kpis(current_period_df)
    kpi_previous = get_clinic_summary_kpis(previous_period_df)

    kpi_to_trend_source = {
        "overall_avg_test_turnaround_conclusive_days": ("test_turnaround_conclusive_days", "mean"),
        "sample_rejection_rate_perc": ("is_rejected", "mean"),
        "total_pending_critical_tests_patients": ("is_critical_and_pending", "sum"),
        "key_drug_stockouts_count": ("is_stockout", "sum")
    }
    
    trend_source_df = full_df.copy()
    # <<< SME REVISION >>> This on-the-fly column creation is fragile.
    # RECOMMENDATION: This logic should be moved to an upstream data processing module
    # (e.g., in `data_processing.loaders` or a new transformation script). The dashboard
    # should consume data that already has these boolean flags pre-calculated.
    # This prevents silent failures if source data values change (e.g., "rejected by lab" -> "Rejected").
    if 'sample_status' in trend_source_df.columns:
        trend_source_df['is_rejected'] = (trend_source_df['sample_status'].str.lower() == 'rejected by lab').astype(int)
    else:
        trend_source_df['is_rejected'] = 0
    if 'is_critical_and_pending' not in trend_source_df.columns: trend_source_df['is_critical_and_pending'] = 0
    if 'is_stockout' not in trend_source_df.columns: trend_source_df['is_stockout'] = 0

    kpi_defs = {
        "Avg. Test TAT (Days)": "overall_avg_test_turnaround_conclusive_days",
        "Sample Rejection (%)": "sample_rejection_rate_perc",
        "Pending Critical Tests": "total_pending_critical_tests_patients",
        "Key Drug Stockouts": "key_drug_stockouts_count"
    }

    analysis_data = []
    trend_start_date = end_date - timedelta(days=90)
    trend_df_subset = trend_source_df[trend_source_df['encounter_date'].dt.date.between(trend_start_date, end_date)]

    for name, key in kpi_defs.items():
        current_val, prev_val = kpi_current.get(key), kpi_previous.get(key)
        
        change_str = "N/A"
        if pd.notna(current_val) and pd.notna(prev_val) and prev_val != 0:
            change = ((current_val - prev_val) / prev_val) * 100
            change_str = f"{change:+.1f}%"
        elif pd.notna(current_val) and pd.notna(prev_val) and prev_val == 0 and current_val > 0:
            change_str = "∞"
            
        trend_series = pd.Series(dtype=float)
        if key in kpi_to_trend_source and not trend_df_subset.empty:
            source_col, agg_method = kpi_to_trend_source[key]
            if source_col in trend_df_subset.columns:
                trend_series = get_trend_data(trend_df_subset, value_col=source_col, period='W-MON', agg_func=agg_method)
                if key == "sample_rejection_rate_perc":
                    trend_series *= 100

        analysis_data.append({
            "Metric": name, "Current Period": current_val, "Previous Period": prev_val,
            "Change": change_str, "90-Day Trend": create_sparkline_bytes(trend_series, "#007BFF")
        })
        
    return pd.DataFrame(analysis_data)


# --- Data Loading and Caching ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading and processing all operational data...")
def get_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, bool, date, date]:
    """Loads, processes, and enriches all data required for the dashboard."""
    # <<< SME REVISION (Observation) >>> The apply_ai_models call is inside this cached function.
    # For very large datasets or slow models, consider running AI enrichment as an offline
    # batch process. The dashboard would then load pre-enriched data, improving load times.
    try:
        health_df = load_health_records()
        iot_df = load_iot_clinic_environment_data()
    except Exception as e:
        logger.error(f"Failed to load source data: {e}")
        st.error(f"Error: Could not load source data files. Please check data sources. Details: {e}")
        return pd.DataFrame(), pd.DataFrame(), False, date.today() - timedelta(days=30), date.today()

    iot_available = isinstance(iot_df, pd.DataFrame) and not iot_df.empty
    
    min_date, max_date = date.today() - timedelta(days=365), date.today()
    if not health_df.empty and 'encounter_date' in health_df.columns:
        valid_dates = health_df['encounter_date'].dropna()
        if not valid_dates.empty:
            min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
            
    ai_enriched_health_df, _ = apply_ai_models(health_df)
    return ai_enriched_health_df, iot_df, iot_available, min_date, max_date

# <<< SME REVISION (Performance) >>> Cache the expensive forecast generation.
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def get_cached_supply_forecast(data: pd.DataFrame, items: List[str]) -> pd.DataFrame:
    """Cached wrapper for the supply forecasting function."""
    if not items:
        return pd.DataFrame()
    logger.info(f"Cache miss: Generating supply forecast for {len(items)} items.")
    # The list `items` must be converted to a hashable type (tuple) for caching.
    return generate_simple_supply_forecast(data, item_filter=tuple(items))


# --- UI Rendering Functions for Each Tab ---
# <<< SME REVISION (Maintainability) >>> Encapsulate tab logic into functions.

def render_epidemiology_tab(data: pd.DataFrame):
    st.subheader("Local Epidemiological Intelligence")
    if data.empty:
        st.info("No data for epidemiological analysis in this period.")
        return

    st.markdown("###### **Weekly Symptom Trends (Top 5)**")
    symptoms_df = data[['encounter_date', 'patient_reported_symptoms']].dropna()
    symptoms_df = symptoms_df.assign(symptom=symptoms_df['patient_reported_symptoms'].str.split(r'[;,|]')).explode('symptom')
    symptoms_df['symptom'] = symptoms_df['symptom'].str.strip().str.title()
    top_5_symptoms = symptoms_df['symptom'].value_counts().nlargest(5).index
    symptom_trend_data = symptoms_df[symptoms_df['symptom'].isin(top_5_symptoms)]
    symptom_weekly = symptom_trend_data.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'symptom']).size().reset_index(name='count')
    
    if not symptom_weekly.empty:
        fig = plot_bar_chart(
            symptom_weekly, x_col='encounter_date', y_col='count', color='symptom',
            title='Weekly Encounters for Top 5 Symptoms', x_axis_title='Week', y_axis_title='Number of Encounters',
            y_values_are_counts=True
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Symptom Data Table"):
            st.dataframe(symptom_weekly, hide_index=True, use_container_width=True)
    else:
        st.info("No significant symptom data to plot for this period.")

def render_testing_tab(data: pd.DataFrame):
    st.subheader("Testing & Diagnostics Performance")
    if data.empty:
        st.info("No data for testing analysis in this period.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###### **Average Turnaround Time (TAT) vs. Target**")
        tat_df = data.groupby('test_type')['test_turnaround_days'].mean().dropna().sort_values().reset_index()
        tat_df.columns = ['Test Type', 'Avg. TAT (Days)']
        if not tat_df.empty:
            tat_df['On Target'] = tat_df['Avg. TAT (Days)'] <= settings.TARGET_TEST_TURNAROUND_DAYS
            fig = go.Figure(go.Bar(
                x=tat_df['Avg. TAT (Days)'], y=tat_df['Test Type'], orientation='h',
                marker_color=np.where(tat_df['On Target'], '#27AE60', '#D32F2F')
            ))
            fig.add_vline(x=settings.TARGET_TEST_TURNAROUND_DAYS, line_width=2, line_dash="dash", line_color="black", annotation_text="Target TAT")
            fig.update_layout(title_text="<b>Average Turnaround Time (TAT) by Test</b>", yaxis={'categoryorder':'total ascending'}, xaxis_title="Average Days")
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.markdown("###### **Sample Rejection Reasons**")
        rejection_df = data[data['sample_status'].str.lower() == 'rejected by lab']['rejection_reason'].value_counts().nlargest(5).reset_index()
        rejection_df.columns = ['Reason', 'Count']
        if not rejection_df.empty:
            fig = plot_donut_chart(rejection_df, labels_col='Reason', values_col='Count', title="Top 5 Sample Rejection Reasons")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sample rejections recorded in this period.")

def render_supply_chain_tab(data: pd.DataFrame):
    st.subheader("Medical Supply Forecast")
    forecastable_items = sorted([
        item for item in data['item'].dropna().unique() 
        if any(sub in item.lower() for sub in getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', []))
    ])
    
    if not forecastable_items:
        st.info("No forecastable supply items found in the dataset.")
        return

    with st.spinner("Analyzing current stock levels..."):
        latest_stock = data.loc[data.groupby('item')['encounter_date'].idxmax()].copy()
        latest_stock['days_of_supply'] = latest_stock['item_stock_agg_zone'] / latest_stock['consumption_rate_per_day'].clip(lower=0.001)
        short_supply_items = latest_stock[latest_stock['days_of_supply'] < settings.LOW_SUPPLY_DAYS_REMAINING]['item'].tolist()

    selected_items = st.multiselect(
        "Select items to forecast (items in short supply are pre-selected):",
        options=forecastable_items,
        default=short_supply_items
    )
    if selected_items:
        with st.spinner(f"Generating forecasts for {len(selected_items)} selected item(s)..."):
            # <<< SME REVISION >>> Use the new cached function for performance.
            forecast_df = get_cached_supply_forecast(data, selected_items)
        
        if not forecast_df.empty:
            fig = go.Figure()
            for item in selected_items:
                item_data = forecast_df[forecast_df['item'] == item]
                fig.add_trace(go.Scatter(x=item_data['forecast_date'], y=item_data['forecasted_days_of_supply'], mode='lines+markers', name=item))
            fig.add_hrect(y0=0, y1=settings.CRITICAL_SUPPLY_DAYS_REMAINING, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Critical")
            fig.add_hrect(y0=settings.CRITICAL_SUPPLY_DAYS_REMAINING, y1=settings.LOW_SUPPLY_DAYS_REMAINING, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Warning")
            fig.update_layout(title_text="<b>Forecasted Days of Supply for Critical Items</b>", xaxis_title="Date", yaxis_title="Days of Supply Remaining", legend_title="Item", yaxis_tickformat='d')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate supply forecast for the selected items.")

def render_patients_tab(data: pd.DataFrame):
    st.subheader("Patient Risk & Demographics")
    if data.empty:
        st.info("No data for patient analysis in this period.")
        return

    st.markdown("###### **Patient Risk Distribution (Age vs. AI Risk Score)**")
    risk_df = data[['patient_id', 'age', 'ai_risk_score']].dropna().drop_duplicates('patient_id')
    if not risk_df.empty:
        risk_df['Risk Category'] = pd.cut(risk_df['ai_risk_score'], bins=[0, 60, 80, 101], labels=['Low-Moderate', 'High', 'Very High'], right=False)
        fig = go.Figure()
        for category, color in zip(['Low-Moderate', 'High', 'Very High'], ['#27AE60', '#F2C94C', '#D32F2F']):
            cat_df = risk_df[risk_df['Risk Category'] == category]
            fig.add_trace(go.Scatter(x=cat_df['age'], y=cat_df['ai_risk_score'], mode='markers', name=category, marker=dict(color=color, size=7, opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))))
        fig.update_layout(title_text="<b>Patient Risk Score vs. Age</b>", xaxis_title="Patient Age", yaxis_title="AI Risk Score", legend_title="Risk Category")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("###### **Flagged Patients for Clinical Review**")
    flagged_patients = get_patient_alerts_for_clinic(health_df_period=data)
    if not flagged_patients.empty:
        display_cols = ['patient_id', 'age', 'gender', 'condition', 'ai_risk_score', 'Alert Reason']
        st.dataframe(flagged_patients[display_cols].head(20), use_container_width=True, hide_index=True, column_config={"ai_risk_score": st.column_config.ProgressColumn("Risk Score", format="%d", min_value=0, max_value=100)})
    else:
        st.success("✅ No patients currently flagged for review in this period.")

def render_environment_tab(data: pd.DataFrame):
    st.subheader("Facility Environment Monitoring")
    if data.empty:
        st.info("No environmental data available for this period.")
        return

    st.markdown("###### **Hourly Average CO2 Levels**")
    co2_trend = get_trend_data(data, 'avg_co2_ppm', date_col='timestamp', period='h', agg_func='mean')
    if not co2_trend.empty:
        fig = plot_annotated_line_chart(co2_trend, "Hourly Average CO2", y_axis_title="CO2 (ppm)")
        fig.add_hrect(y0=settings.ALERT_AMBIENT_CO2_HIGH_PPM, y1=settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM, fillcolor="orange", opacity=0.2, line_width=0, annotation_text="High")
        very_high_max = max(co2_trend.max() * 1.1, settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM * 1.5)
        fig.add_hrect(y0=settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM, y1=very_high_max, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Very High")
        fig.update_yaxes(tickformat='d')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CO2 trend data to display for this period.")
    
    st.markdown("###### **Latest Environmental Readings by Room**")
    latest_readings = data.sort_values('timestamp', ascending=False).drop_duplicates('room_name', keep='first')
    display_cols = ['room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_noise_db']
    st.dataframe(latest_readings[display_cols], use_container_width=True, hide_index=True, column_config={"timestamp": st.column_config.DatetimeColumn("Last Reading", format="YYYY-MM-DD HH:mm"), "avg_temp_celsius": st.column_config.NumberColumn("Temp (°C)")})


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Clinic Dashboard", page_icon="🏥", layout="wide")
    st.title("🏥 Clinic Operations & Management Console")
    st.markdown("##### Service Performance, Patient Care Quality, Resource Management, and Facility Environment Monitoring")
    st.divider()

    # --- Load Data ---
    full_health_df, full_iot_df, iot_available, abs_min_date, abs_max_date = get_dashboard_data()

    # --- Sidebar Filters ---
    st.sidebar.header("Console Filters")
    if os.path.exists(settings.APP_LOGO_SMALL_PATH):
        st.sidebar.image(settings.APP_LOGO_SMALL_PATH, width=120)

    if "clinic_date_range" not in st.session_state:
        default_days = getattr(settings, 'WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND', 30)
        default_start = max(abs_min_date, abs_max_date - timedelta(days=default_days - 1))
        st.session_state.clinic_date_range = (default_start, abs_max_date)

    start_date, end_date = st.sidebar.date_input(
        "Select Date Range:", value=st.session_state.clinic_date_range,
        min_value=abs_min_date, max_value=abs_max_date, help="Select the time period for the dashboard analysis."
    )
    if start_date > end_date:
        st.sidebar.warning("Start date cannot be after end date. Adjusting end date.")
        end_date = start_date
    st.session_state.clinic_date_range = (start_date, end_date)

    # --- Filter Data for Display based on Sidebar Selection ---
    if full_health_df.empty:
        st.warning("No health data available to display. Please check the data source.")
        st.stop()

    period_health_df = full_health_df[full_health_df['encounter_date'].dt.date.between(start_date, end_date)]
    period_iot_df = pd.DataFrame()
    if iot_available and not full_iot_df.empty:
        period_iot_df = full_iot_df[full_iot_df['timestamp'].dt.date.between(start_date, end_date)]

    period_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
    st.info(f"**Displaying Clinic Console for:** `{period_str}`")

    # --- KPI Section ---
    st.header("🚀 Performance Snapshot with Trend Analysis")
    if not period_health_df.empty:
        kpi_analysis_df = get_kpi_analysis_table(full_health_df, start_date, end_date)
        st.dataframe(
            kpi_analysis_df, hide_index=True, use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn(width="large"),
                "Current Period": st.column_config.NumberColumn(format="%.2f"),
                "Previous Period": st.column_config.NumberColumn(format="%.2f"),
                "Change": st.column_config.TextColumn(help="Change vs. previous equivalent period"),
                "90-Day Trend": st.column_config.ImageColumn(width="medium", help="Weekly trend over the last 90 days")
            }
        )
    else:
        st.info("No encounter data available for this period to generate KPI analysis.")
    st.divider()

    # --- Tabbed Section for Detailed Analysis ---
    st.header("🛠️ Operational Areas Deep Dive")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Epidemiology", "🔬 Testing", "💊 Supply Chain", "🧍 Patients", "🌿 Environment"])

    with tab1:
        render_epidemiology_tab(period_health_df)
    with tab2:
        render_testing_tab(period_health_df)
    with tab3:
        # Pass the full dataset for forecasting, not just the period data
        render_supply_chain_tab(full_health_df)
    with tab4:
        render_patients_tab(period_health_df)
    with tab5:
        render_environment_tab(period_iot_df)

if __name__ == "__main__":
    main()
