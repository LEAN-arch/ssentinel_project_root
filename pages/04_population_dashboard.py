# sentinel_project_root/pages/04_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List
import plotly.express as px

# --- Sentinel Project Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric
    from visualization.plots import create_empty_figure, plot_annotated_line_chart, plot_bar_chart
except ImportError as e:
    import sys
    project_root_dir = Path(__file__).resolve().parent.parent
    st.error(f"Import Error: {e}. Ensure '{project_root_dir}' is in sys.path and restart the app.")
    st.stop()

# --- Logging and Constants ---
logger = logging.getLogger(__name__)

class C:
    PAGE_TITLE = "Population Analytics"; PAGE_ICON = "🌍"; TIME_AGG_PERIOD = 'W-MON'
    TOP_N_CONDITIONS = 10; SS_DATE_RANGE = "pop_dashboard_date_range_v3"
    SS_CONDITIONS = "pop_dashboard_conditions_v3"; SS_ZONE = "pop_dashboard_zone_v3"

# --- Helper & Analytics Functions ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

@st.cache_data
def get_condition_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes conditions by frequency and risk."""
    if df.empty or 'condition' not in df.columns or 'ai_risk_score' not in df.columns:
        return pd.DataFrame(columns=['condition', 'count', 'avg_risk_score'])
    
    # --- DEFINITIVE FIX ---
    # Ensure ai_risk_score is numeric before aggregation
    df_copy = df.copy()
    df_copy['ai_risk_score'] = convert_to_numeric(df_copy['ai_risk_score'])
    
    # Aggregate, then fill na for avg_risk_score to keep all conditions
    agg_df = df_copy.groupby('condition').agg(
        count=('patient_id', 'size'),
        avg_risk_score=('ai_risk_score', 'mean')
    ).reset_index()
    agg_df['avg_risk_score'] = agg_df['avg_risk_score'].fillna(0)
    return agg_df

@st.cache_data
def get_risk_stratification_data(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or 'patient_id' not in df.columns or 'ai_risk_score' not in df.columns:
        return {'pyramid_data': pd.DataFrame(), 'trend_data': pd.DataFrame()}
    risk_low, risk_mod = _get_setting('RISK_SCORE_LOW_THRESHOLD', 40), _get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60)
    df_unique_patients = df.sort_values('encounter_date').drop_duplicates(subset='patient_id', keep='last')
    def assign_tier(score):
        if score >= risk_mod: return 'High Risk'
        if score >= risk_low: return 'Moderate Risk'
        return 'Low Risk'
    df_unique_patients['risk_tier'] = convert_to_numeric(df_unique_patients['ai_risk_score']).apply(assign_tier)
    pyramid_data = df_unique_patients['risk_tier'].value_counts().reset_index(); pyramid_data.columns = ['risk_tier', 'patient_count']
    df['risk_tier'] = convert_to_numeric(df['ai_risk_score']).apply(assign_tier)
    trend_data = df.groupby([pd.Grouper(key='encounter_date', freq=C.TIME_AGG_PERIOD), 'risk_tier'])['patient_id'].nunique().reset_index()
    return {'pyramid_data': pyramid_data, 'trend_data': trend_data}

# --- Page Setup & Data Loading ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), hash_funcs={pd.DataFrame: hash_dataframe_safe}, show_spinner="Loading population analytics dataset...")
def get_population_analytics_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    raw_health_df = load_health_records()
    if not isinstance(raw_health_df, pd.DataFrame) or raw_health_df.empty: return None, None
    enriched_health_df, _ = apply_ai_models(raw_health_df.copy())
    zone_attributes_df = load_zone_data()
    return enriched_health_df, zone_attributes_df

def initialize_session_state(health_df: pd.DataFrame, zone_df: Optional[pd.DataFrame]):
    """Centralizes initialization of all session state filter values."""
    if 'pop_data_initialized' in st.session_state: return
    
    min_data_date, max_data_date = date.today() - timedelta(days=365), date.today()
    if 'encounter_date' in health_df.columns and not health_df['encounter_date'].isna().all():
        valid_dates = health_df['encounter_date'].dropna()
        if not valid_dates.empty:
            min_calc, max_calc = valid_dates.min().date(), valid_dates.max().date()
            if min_calc <= max_calc: min_data_date, max_data_date = min_calc, max_calc
    
    default_start = max(min_data_date, max_data_date - timedelta(days=90))
    st.session_state[C.SS_DATE_RANGE] = (default_start, max_data_date)
    st.session_state['min_data_date'], st.session_state['max_data_date'] = min_data_date, max_data_date
    
    st.session_state['all_conditions'] = sorted(list(health_df['condition'].dropna().astype(str).unique()))
    st.session_state[C.SS_CONDITIONS] = []
    
    zone_options, zone_map = ["All Zones/Regions"], {}
    if zone_df is not None and not zone_df.empty:
        valid_zones = zone_df.dropna(subset=['name', 'zone_id'])
        if not valid_zones.empty:
            zone_map = valid_zones.set_index('name')['zone_id'].to_dict()
            zone_options.extend(sorted(list(zone_map.keys())))
    st.session_state['zone_options'], st.session_state['zone_name_id_map'] = zone_options, zone_map
    st.session_state[C.SS_ZONE] = "All Zones/Regions"
    st.session_state['pop_data_initialized'] = True

# --- Main Application Logic ---
def run_dashboard():
    st.set_page_config(page_title=f"{C.PAGE_TITLE} - {_get_setting('APP_NAME', 'Sentinel')}", page_icon=C.PAGE_ICON, layout="wide")
    st.title(f"🌍 {_get_setting('APP_NAME', 'Sentinel')} - Population Health Analytics Console")
    st.markdown("Strategic exploration of demographic distributions, epidemiological patterns, and clinical trends.")
    st.divider()

    health_df_main, zone_attr_main = get_population_analytics_datasets()
    if health_df_main is None or health_df_main.empty:
        st.error("🚨 Critical Data Failure: Could not load health dataset."); st.stop()
    
    initialize_session_state(health_df_main, zone_attr_main)

    with st.sidebar:
        st.header("🔎 Analytics Filters")
        start_date, end_date = st.date_input("Select Date Range:", value=st.session_state[C.SS_DATE_RANGE], min_value=st.session_state['min_data_date'], max_value=st.session_state['max_data_date'])
        st.session_state[C.SS_DATE_RANGE] = (start_date, end_date)
        st.selectbox("Filter by Zone/Region:", options=st.session_state['zone_options'], key=C.SS_ZONE)
        st.multiselect("Filter by Condition(s):", options=st.session_state['all_conditions'], key=C.SS_CONDITIONS)

    df_filtered = health_df_main[health_df_main['encounter_date'].dt.date.between(start_date, end_date)]
    if st.session_state[C.SS_CONDITIONS]: df_filtered = df_filtered[df_filtered['condition'].isin(st.session_state[C.SS_CONDITIONS])]
    
    total_population = 0
    if st.session_state[C.SS_ZONE] != "All Zones/Regions":
        zone_id = st.session_state['zone_name_id_map'].get(st.session_state[C.SS_ZONE])
        if zone_id and zone_attr_main is not None:
            df_filtered = df_filtered[df_filtered['zone_id'].astype(str) == str(zone_id)]
            total_population = zone_attr_main.loc[zone_attr_main['zone_id'] == str(zone_id), 'population'].sum()
    elif zone_attr_main is not None: total_population = zone_attr_main['population'].sum()

    if df_filtered.empty: st.info("ℹ️ No data available for the selected filters."); st.stop()

    st.subheader("Strategic Population Health Indicators")
    kpi_cols = st.columns(4)
    unique_patients = df_filtered['patient_id'].nunique()
    kpi_cols[0].metric("Unique Patients Affected", f"{unique_patients:,}")
    prevalence = (unique_patients / total_population * 1000) if total_population > 0 else 0
    kpi_cols[1].metric("Prevalence per 1,000 Pop.", f"{prevalence:.1f}")
    high_risk_patients = df_filtered[df_filtered['ai_risk_score'] >= settings.RISK_SCORE_MODERATE_THRESHOLD]['patient_id'].nunique()
    kpi_cols[2].metric("High-Risk Patient Cohort", f"{high_risk_patients:,}", f"{high_risk_patients/unique_patients:.1%}" if unique_patients > 0 else "0.0%")
    cond_analytics = get_condition_analytics(df_filtered)
    top_risk_condition = cond_analytics.sort_values('avg_risk_score', ascending=False).iloc[0]['condition'] if not cond_analytics.empty else "N/A"
    kpi_cols[3].metric("Top Condition by Avg. Risk", top_risk_condition)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Epidemiological Overview", "🚨 Population Risk Stratification", "🗺️ Geospatial Analysis", "🧑‍🤝‍🧑 Demographic Insights"])

    with tab1:
        st.header("Epidemiological Overview")
        st.subheader("Encounter Trends")
        df_trend = df_filtered.set_index('encounter_date').resample(C.TIME_AGG_PERIOD).size()
        st.plotly_chart(plot_annotated_line_chart(df_trend, "Weekly Encounters Trend", "Encounters"), use_container_width=True)
        st.subheader("Top Conditions by Volume & Severity")
        col1, col2 = st.columns(2)
        with col1:
            top_by_count = cond_analytics.sort_values('count', ascending=False).head(C.TOP_N_CONDITIONS)
            if not top_by_count.empty:
                st.plotly_chart(plot_bar_chart(top_by_count, x_col='count', y_col='condition', orientation='h', title="Most Frequent Conditions", x_axis_title='Number of Encounters', y_values_are_counts=True), use_container_width=True)
            else:
                st.info("No condition data to display for volume analysis.")
        with col2:
            top_by_risk = cond_analytics.sort_values('avg_risk_score', ascending=False).head(C.TOP_N_CONDITIONS)
            if not top_by_risk.empty:
                st.plotly_chart(plot_bar_chart(top_by_risk, x_col='avg_risk_score', y_col='condition', orientation='h', title="Highest-Risk Conditions", x_axis_title='Average AI Risk Score', range_x=[0,100]), use_container_width=True)
            else:
                st.info("No condition data to display for severity analysis.")
    
    with tab2:
        st.header("Population Risk Stratification")
        risk_data = get_risk_stratification_data(df_filtered)
        pyramid_data, trend_data = risk_data.get('pyramid_data'), risk_data.get('trend_data')
        col1, col2 = st.columns([1, 2])
        if not pyramid_data.empty:
            with col1:
                fig = px.funnel(pyramid_data, x='patient_count', y='risk_tier', title="Risk Pyramid")
                fig.update_yaxes(categoryorder="array", categoryarray=['High Risk', 'Moderate Risk', 'Low Risk'])
                st.plotly_chart(fig, use_container_width=True)
        if not trend_data.empty:
            with col2:
                fig_trend = px.area(trend_data, x='encounter_date', y='patient_id', color='risk_tier', title="Risk Tier Trends (Weekly)", labels={'patient_id': 'Unique Patients'}, category_orders={"risk_tier": ["Low Risk", "Moderate Risk", "High Risk"]})
                st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.header("Geospatial Analysis")
        if zone_attr_main is not None and not zone_attr_main.empty and 'geometry_obj' in zone_attr_main.columns:
            geo_agg = df_filtered.groupby('zone_id').agg(avg_risk_score=('ai_risk_score', 'mean'), unique_patients=('patient_id', 'nunique')).reset_index()
            map_df = pd.merge(zone_attr_main, geo_agg, on='zone_id', how='left').fillna(0)
            map_df['prevalence_per_1000'] = (map_df['unique_patients'] / map_df['population'] * 1000).where(map_df['population'] > 0, 0)
            geojson_data = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": row['geometry_obj'], "id": str(row['zone_id']), "properties": {"name": row['name'], "avg_risk_score": row['avg_risk_score'], "prevalence_per_1000": row['prevalence_per_1000']}} for _, row in map_df.iterrows()]}
            map_metric = st.selectbox("Select Map Metric:", ["Prevalence per 1,000", "Average AI Risk Score"])
            color_metric = 'prevalence_per_1000' if map_metric == "Prevalence per 1,000" else 'avg_risk_score'
            fig = px.choropleth_mapbox(map_df, geojson=geojson_data, locations="zone_id", color=color_metric, mapbox_style="carto-positron", zoom=8, center={"lat": -1.28, "lon": 36.81}, opacity=0.6, hover_name="name", hover_data={"avg_risk_score": ":.2f", "prevalence_per_1000": ":.2f"}, labels={color_metric: map_metric})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geospatial data is unavailable. Cannot render map.")
    
    with tab4:
        st.header("Demographic Insights")
        df_unique = df_filtered.drop_duplicates(subset=['patient_id'])
        col1, col2 = st.columns(2)
        with col1:
            if not df_unique['age'].dropna().empty:
                st.plotly_chart(px.histogram(df_unique, x='age', nbins=20, title="Age Distribution"), use_container_width=True)
        with col2:
            if not df_unique['gender'].dropna().empty:
                st.plotly_chart(px.pie(df_unique, names='gender', title="Gender Distribution"), use_container_width=True)
        st.subheader("Risk by Demographics")
        if not df_unique.empty:
             df_unique['age_band'] = pd.cut(df_unique['age'], bins=[0, 18, 40, 60, 120], labels=['0-18', '19-40', '41-60', '60+'])
             risk_by_demo = df_unique.groupby(['age_band', 'gender'])['ai_risk_score'].mean().reset_index()
             if not risk_by_demo.empty:
                fig = plot_bar_chart(risk_by_demo, x_col='age_band', y_col='ai_risk_score', color='gender', barmode='group', title="Average AI Risk Score by Age and Gender", y_axis_title='Avg. AI Risk Score')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_dashboard()
