# sentinel_project_root/pages/04_Population_Analytics.py
# SME PLATINUM STANDARD - POPULATION HEALTH ANALYTICS (V3 - DEFINITIVE FIX)

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import apply_ai_models
from config import settings
from data_processing import (convert_to_numeric, get_cached_trend,
                             load_health_records, load_zone_data,
                             enrich_zone_data_with_aggregates)
from visualization import (plot_bar_chart, plot_choropleth_map,
                           plot_donut_chart, plot_line_chart,
                           render_custom_kpi)

st.set_page_config(page_title="Population Analytics", page_icon="📊", layout="wide")
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600, show_spinner="Loading population datasets...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and caches the primary health and zone data, ensuring all
    AI models and enrichments are applied in the correct order.
    """
    # SME FIX: The health_df must be enriched with AI scores *before* being
    # used to enrich the zone data to prevent KeyErrors.
    raw_health_df = load_health_records()
    if raw_health_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    health_df, _ = apply_ai_models(raw_health_df)
    
    zone_df = load_zone_data()
    enriched_zone_df = enrich_zone_data_with_aggregates(zone_df, health_df)
    
    return health_df, enriched_zone_df

# ... [The rest of the file is correct and remains unchanged] ...
@st.cache_data
def get_risk_stratification(df: pd.DataFrame) -> dict:
    if df.empty or 'ai_risk_score' not in df.columns: return {'pyramid_data': pd.DataFrame()}
    def assign_tier(score):
        if score >= settings.ANALYTICS.risk_score_moderate_threshold: return 'High Risk'
        if score >= settings.ANALYTICS.risk_score_low_threshold: return 'Moderate Risk'
        return 'Low Risk'
    df_copy = df.copy().drop_duplicates('patient_id')
    df_copy['risk_tier'] = convert_to_numeric(df_copy['ai_risk_score']).apply(assign_tier)
    pyramid_data = df_copy['risk_tier'].value_counts().reset_index()
    pyramid_data.columns = ['risk_tier', 'patient_count']
    return {'pyramid_data': pyramid_data}
def main():
    st.title("📊 Population Health Analytics")
    st.markdown("Explore demographic distributions, epidemiological patterns, and geospatial risk.")
    st.divider()
    health_df, zone_df = get_data()
    if health_df.empty or zone_df.empty:
        st.error("Could not load necessary data. Dashboard cannot be rendered.")
        st.stop()
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = health_df['encounter_date'].min().date(), health_df['encounter_date'].max().date()
        start_date, end_date = st.date_input("Select Date Range:", value=(max(min_date, max_date - timedelta(days=89)), max_date), min_value=min_date, max_value=max_date)
        zone_options = ["All Zones"] + sorted(zone_df['zone_name'].dropna().unique())
        selected_zone = st.selectbox("Filter by Zone:", options=zone_options)
    df_filtered = health_df[health_df['encounter_date'].dt.date.between(start_date, end_date)]
    if selected_zone != "All Zones":
        zone_id = zone_df.loc[zone_df['zone_name'] == selected_zone, 'zone_id'].iloc[0]
        df_filtered = df_filtered[df_filtered['zone_id'] == zone_id]
    st.subheader("Population Snapshot")
    cols = st.columns(4)
    unique_patients = df_filtered['patient_id'].nunique()
    with cols[0]: render_custom_kpi("Unique Patients", unique_patients, "In selected period")
    with cols[1]: render_custom_kpi("Avg. Risk Score", df_filtered.get('ai_risk_score', pd.Series(dtype=float)).mean(), "0-100 scale")
    high_risk_count = (df_filtered.get('ai_risk_score', pd.Series(dtype=float)) >= settings.ANALYTICS.risk_score_moderate_threshold).sum()
    with cols[2]: render_custom_kpi("High-Risk Patients", high_risk_count, f"Score >= {settings.ANALYTICS.risk_score_moderate_threshold}", highlight_status='high-risk')
    with cols[3]: render_custom_kpi("Median Patient Age", df_filtered.get('age', pd.Series(dtype=float)).median(), "Years")
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Epidemiology", "🚨 Risk Stratification", "🗺️ Geospatial", "🧑‍🤝‍🧑 Demographics"])
    with tab1:
        st.header("Epidemiological Overview")
        trend = get_cached_trend(df_filtered, value_col='encounter_id', date_col='encounter_date', freq='W', agg_func='count')
        fig = plot_line_chart(trend, "Weekly Encounters Trend", "Total Encounters")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.header("Population Risk Stratification")
        risk_data = get_risk_stratification(df_filtered)
        pyramid_data = risk_data.get('pyramid_data')
        if not pyramid_data.empty:
            fig = px.funnel(pyramid_data, x='patient_count', y='risk_tier', title="Risk Pyramid")
            fig.update_yaxes(categoryorder="array", categoryarray=['High Risk', 'Moderate Risk', 'Low Risk'])
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.header("Geospatial Analysis")
        if 'geometry' not in zone_df.columns: st.warning("Geospatial data is unavailable.")
        else:
            geojson_data = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"zone_id": row['zone_id']}, "geometry": row['geometry']} for _, row in zone_df.iterrows() if pd.notna(row.get('geometry'))]}
            map_metric = st.selectbox("Select Map Metric:", ["Average AI Risk Score", "Prevalence per 1,000 People"])
            color_metric = 'avg_risk_score' if "Risk" in map_metric else 'prevalence_per_1000_pop'
            fig = plot_choropleth_map(zone_df, geojson_data, color_metric, "Zonal Health Metrics", hover_name='zone_name', color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.header("Demographic Insights")
        df_unique = df_filtered.drop_duplicates(subset=['patient_id'])
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_unique, x='age', nbins=20, title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_donut_chart(df_unique, 'gender', 'patient_id', "Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
