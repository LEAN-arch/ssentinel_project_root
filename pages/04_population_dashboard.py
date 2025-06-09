# sentinel_project_root/pages/04_population_dashboard.py
# SME PLATINUM STANDARD (V6 - FINAL INTEGRATED VERSION)
# This definitive version is architecturally refactored for performance and
# maintainability, and is fully integrated with the Pydantic-based settings
# module and other refactored components.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Optional, Any, Tuple, Dict, List
import plotly.express as px
from pydantic import BaseModel, field_validator

# --- Sentinel Project Imports ---
try:
    # <<< SME INTEGRATION >>> Use the Pydantic settings object.
    from config import settings
    from data_processing import load_health_records, load_zone_data, hash_dataframe_safe, convert_to_numeric
    from visualization.plots import create_empty_figure, plot_annotated_line_chart, plot_bar_chart
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure project structure and `__init__.py` files are correct.")
    st.stop()

# --- Page Constants & State Management ---
logger = logging.getLogger(__name__)

class PopDashboardState(BaseModel):
    """Manages the interactive state of the Population Dashboard using a validated model."""
    start_date: date
    end_date: date
    selected_zone: str = "All Zones"
    selected_diagnoses: List[str] = []

    @field_validator('start_date', 'end_date', mode='before')
    def parse_date(cls, v):
        return v if isinstance(v, date) else date.fromisoformat(v)

    @property
    def is_filtered_by_zone(self) -> bool:
        return self.selected_zone != "All Zones"

# --- Data Loading and Caching ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading core datasets...", hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_main_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and caches the primary health and zone data."""
    health_df = load_health_records()
    zone_df = load_zone_data()
    return health_df, zone_df

# --- Analytics Functions ---
@st.cache_data
def get_diagnosis_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes diagnoses by frequency and average risk score."""
    if df.empty or 'diagnosis' not in df.columns or 'ai_risk_score' not in df.columns:
        return pd.DataFrame()
    agg_df = df.groupby('diagnosis').agg(
        count=('patient_id', 'size'),
        avg_risk_score=('ai_risk_score', 'mean')
    ).reset_index()
    return agg_df.fillna({'avg_risk_score': 0})

@st.cache_data
def get_risk_stratification_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Segments the population into risk tiers and calculates trends."""
    if df.empty or 'patient_id' not in df.columns or 'ai_risk_score' not in df.columns:
        return {'pyramid_data': pd.DataFrame(), 'trend_data': pd.DataFrame()}

    def assign_tier(score):
        if score >= settings.RISK_SCORE_MODERATE_THRESHOLD: return 'High Risk'
        if score >= settings.RISK_SCORE_LOW_THRESHOLD: return 'Moderate Risk'
        return 'Low Risk'
    
    df_copy = df.copy()
    df_copy['risk_tier'] = convert_to_numeric(df_copy['ai_risk_score']).apply(assign_tier)
    
    pyramid_data = df_copy.drop_duplicates('patient_id')['risk_tier'].value_counts().reset_index()
    pyramid_data.columns = ['risk_tier', 'patient_count']
    
    trend_data = df_copy.groupby([pd.Grouper(key='encounter_date', freq='W-MON'), 'risk_tier'])['patient_id'].nunique().reset_index()
    return {'pyramid_data': pyramid_data, 'trend_data': trend_data}

# --- Modular UI Rendering Functions ---
def render_sidebar(health_df: pd.DataFrame, zone_df: pd.DataFrame) -> PopDashboardState:
    """Renders sidebar filters and returns the current state as a Pydantic model."""
    st.sidebar.header("🔎 Analytics Filters")
    min_date, max_date = health_df['encounter_date'].min().date(), health_df['encounter_date'].max().date()
    default_start = max(min_date, max_date - timedelta(days=90))
    
    s_state = st.session_state.get('pop_dashboard_state', {})
    start_val = date.fromisoformat(s_state.get('start_date')) if s_state.get('start_date') else default_start
    end_val = date.fromisoformat(s_state.get('end_date')) if s_state.get('end_date') else max_date

    start_date, end_date = st.sidebar.date_input("Select Date Range:", value=[start_val, end_val], min_value=min_date, max_value=max_date)
    
    zone_options = ["All Zones"] + sorted(zone_df['zone_name'].dropna().unique())
    selected_zone = st.sidebar.selectbox("Filter by Zone/Region:", zone_options, index=zone_options.index(s_state.get('selected_zone', "All Zones")))
    
    all_diagnoses = sorted(health_df['diagnosis'].dropna().unique())
    selected_diagnoses = st.sidebar.multiselect("Filter by Diagnosis:", all_diagnoses, default=s_state.get('selected_diagnoses', []))
    
    current_state = PopDashboardState(start_date=start_date, end_date=end_date, selected_zone=selected_zone, selected_diagnoses=selected_diagnoses)
    st.session_state['pop_dashboard_state'] = current_state.model_dump(mode='json')
    return current_state

def render_kpis(df: pd.DataFrame, zone_df: pd.DataFrame, state: PopDashboardState):
    """Renders the main KPI metrics at the top of the page."""
    # ... (code identical to previous "platinum" review, already robust)
    st.subheader("Strategic Population Health Indicators")
    cols = st.columns(4)
    unique_patients = df['patient_id'].nunique()
    cols[0].metric("Unique Patients in Cohort", f"{unique_patients:,}")

    total_pop = zone_df.loc[zone_df['zone_name'] == state.selected_zone, 'population'].sum() if state.is_filtered_by_zone else zone_df['population'].sum()
    prevalence = (unique_patients / total_pop * 1000) if total_pop > 0 else 0
    cols[1].metric("Prevalence per 1,000", f"{prevalence:.1f}")

    high_risk_patients = df[df['ai_risk_score'] >= settings.RISK_SCORE_MODERATE_THRESHOLD]['patient_id'].nunique()
    cols[2].metric("High-Risk Cohort Pct.", f"{high_risk_patients/unique_patients:.1%}" if unique_patients > 0 else "0.0%")

    diag_analytics = get_diagnosis_analytics(df)
    top_risk_diag = diag_analytics.nlargest(1, 'avg_risk_score')['diagnosis'].iloc[0] if not diag_analytics.empty else "N/A"
    cols[3].metric("Top Diagnosis by Avg. Risk", top_risk_diag)

def render_epi_overview(df: pd.DataFrame, diag_analytics: pd.DataFrame):
    # ... (code identical to previous "platinum" review, already robust)
    st.header("Epidemiological Overview")
    trend = df.set_index('encounter_date').resample('W-MON').size()
    st.plotly_chart(plot_annotated_line_chart(trend, "Weekly Encounters Trend", "Encounters"), use_container_width=True)
    
    st.subheader("Top Diagnoses by Volume & Severity")
    col1, col2 = st.columns(2)
    with col1:
        top_by_count = diag_analytics.nlargest(10, 'count')
        st.plotly_chart(plot_bar_chart(top_by_count, x_col='count', y_col='diagnosis', orientation='h', title="Most Frequent Diagnoses"), use_container_width=True)
    with col2:
        top_by_risk = diag_analytics.nlargest(10, 'avg_risk_score')
        st.plotly_chart(plot_bar_chart(top_by_risk, x_col='avg_risk_score', y_col='diagnosis', orientation='h', title="Highest-Risk Diagnoses", range_x=[0, 100]), use_container_width=True)

def render_risk_stratification(df: pd.DataFrame):
    st.header("Population Risk Stratification")
    risk_data = get_risk_stratification_data(df)
    pyramid_data, trend_data = risk_data.get('pyramid_data'), risk_data.get('trend_data')
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if not pyramid_data.empty:
            fig = px.funnel(pyramid_data, x='patient_count', y='risk_tier', title="Risk Pyramid")
            fig.update_yaxes(categoryorder="array", categoryarray=['High Risk', 'Moderate Risk', 'Low Risk'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.altair_chart(create_empty_figure("Risk Pyramid", "No data to display."), use_container_width=True)

    with col2:
        if not trend_data.empty:
            fig = px.area(trend_data, x='encounter_date', y='patient_id', color='risk_tier', title="Risk Tier Trends (Weekly)", labels={'patient_id': 'Unique Patients'}, category_orders={"risk_tier": ["Low Risk", "Moderate Risk", "High Risk"]})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.altair_chart(create_empty_figure("Risk Tier Trends", "No data to display."), use_container_width=True)

def render_geospatial_analysis(df: pd.DataFrame, zone_df: pd.DataFrame):
    st.header("Geospatial Analysis")
    if 'geometry_obj' not in zone_df.columns:
        st.warning("Geospatial data is unavailable. Cannot render map."); return
        
    geo_agg = df.groupby('zone_id').agg(avg_risk_score=('ai_risk_score', 'mean'), unique_patients=('patient_id', 'nunique')).reset_index()
    map_df = pd.merge(zone_df, geo_agg, on='zone_id', how='left').fillna(0)
    map_df['prevalence_per_1000'] = np.where(map_df['population'] > 0, (map_df['unique_patients'] / map_df['population']) * 1000, 0)
    
    geojson_data = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": row['geometry_obj'], "id": str(row['zone_id']), 
         "properties": {"zone_name": row['zone_name'], "avg_risk_score": row['avg_risk_score'], "prevalence_per_1000": row['prevalence_per_1000']}}
        for _, row in map_df.iterrows() if pd.notna(row.get('geometry_obj'))
    ]}
    
    map_metric = st.selectbox("Select Map Metric:", ["Prevalence per 1,000", "Average AI Risk Score"])
    color_metric = 'prevalence_per_1000' if map_metric == "Prevalence per 1,000" else 'avg_risk_score'
    
    fig = px.choropleth_mapbox(map_df, geojson=geojson_data, locations="zone_id", color=color_metric,
                              mapbox_style=settings.MAPBOX_STYLE_WEB, zoom=settings.MAP_DEFAULT_ZOOM_LEVEL, 
                              center={"lat": settings.MAP_DEFAULT_CENTER_LAT, "lon": settings.MAP_DEFAULT_CENTER_LON},
                              opacity=0.6, hover_name="zone_name", 
                              hover_data={"zone_name":True, "avg_risk_score": ":.2f", "prevalence_per_1000": ":.2f"},
                              labels={color_metric: map_metric})
    st.plotly_chart(fig, use_container_width=True)

def render_demographics(df: pd.DataFrame):
    st.header("Demographic Insights")
    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df_unique, x='age', nbins=20, title="Age Distribution"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(df_unique, names='gender', title="Gender Distribution"), use_container_width=True)
    
    st.subheader("Risk by Demographics")
    df_unique['age_band'] = pd.cut(df_unique['age'], bins=[0, 18, 40, 60, 120], labels=['0-18', '19-40', '41-60', '60+'])
    risk_by_demo = df_unique.groupby(['age_band', 'gender'], observed=False)['ai_risk_score'].mean().reset_index()
    if not risk_by_demo.empty:
        fig = plot_bar_chart(risk_by_demo, x_col='age_band', y_col='ai_risk_score', color='gender', barmode='group', title="Average AI Risk Score by Age and Gender", y_axis_title='Avg. AI Risk Score')
        st.plotly_chart(fig, use_container_width=True)

# --- Main Application Execution ---
def main():
    st.set_page_config(page_title="Population Analytics", page_icon="🌍", layout="wide")
    st.title(f"🌍 {settings.APP_NAME} - Population Health Analytics")
    st.markdown("Strategic exploration of demographic distributions, epidemiological patterns, and clinical trends.")

    health_df_main, zone_df_main = load_main_datasets()
    if health_df_main.empty:
        st.error("🚨 Critical Data Failure: Could not load health dataset."); st.stop()

    state = render_sidebar(health_df_main, zone_df_main)
    
    df_filtered = health_df_main[
        (health_df_main['encounter_date'].dt.date.between(state.start_date, state.end_date)) &
        (health_df_main['zone_id'] == zone_df_main.set_index('zone_name').at[state.selected_zone, 'zone_id'] if state.is_filtered_by_zone else True) &
        (health_df_main['diagnosis'].isin(state.selected_diagnoses) if state.selected_diagnoses else True)
    ]

    if df_filtered.empty:
        st.info("ℹ️ No data available for the selected filters."); st.stop()

    render_kpis(df_filtered, zone_df_main, state)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Epidemiological Overview", "🚨 Risk Stratification", "🗺️ Geospatial Analysis", "🧑‍🤝‍🧑 Demographics"])
    with tab1:
        render_epi_overview(df_filtered, get_diagnosis_analytics(df_filtered))
    with tab2:
        render_risk_stratification(df_filtered)
    with tab3:
        render_geospatial_analysis(df_filtered, zone_df_main)
    with tab4:
        render_demographics(df_filtered)
        
if __name__ == "__main__":
    main()
