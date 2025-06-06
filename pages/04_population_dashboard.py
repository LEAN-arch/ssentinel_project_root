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
import plotly.graph_objects as go

# --- Sentinel Project Imports ---
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric
    from visualization.plots import create_empty_figure, plot_annotated_line_chart
except ImportError as e:
    import sys
    project_root_dir = Path(__file__).resolve().parent.parent
    st.error(f"Import Error: {e}. Ensure '{project_root_dir}' is in sys.path and restart the app.")
    st.stop()

# --- Logging and Constants ---
logger = logging.getLogger(__name__)

class C:
    """Centralized constants for maintainability."""
    PAGE_TITLE = "Population Analytics"
    PAGE_ICON = "🌍"
    TIME_AGG_PERIOD = 'W-MON'
    TOP_N_CONDITIONS = 10
    SS_DATE_RANGE = "pop_dashboard_date_range_v3"
    SS_CONDITIONS = "pop_dashboard_conditions_v3"
    SS_ZONE = "pop_dashboard_zone_v3"

# --- Helper & Analytics Functions ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

@st.cache_data
def get_condition_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes conditions by frequency and risk."""
    if df.empty or 'condition' not in df.columns or 'ai_risk_score' not in df.columns:
        return pd.DataFrame(columns=['condition', 'count', 'avg_risk_score'])
    
    df_copy = df.copy()
    df_copy['ai_risk_score'] = convert_to_numeric(df_copy['ai_risk_score'])
    agg_df = df_copy.groupby('condition').agg(
        count=('patient_id', 'size'),
        avg_risk_score=('ai_risk_score', 'mean')
    ).reset_index().dropna(subset=['avg_risk_score'])
    return agg_df

@st.cache_data
def get_risk_stratification_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Segments the population into risk tiers and calculates trends."""
    if df.empty or 'patient_id' not in df.columns or 'ai_risk_score' not in df.columns:
        return {'pyramid_data': pd.DataFrame(), 'trend_data': pd.DataFrame()}

    risk_low = _get_setting('RISK_SCORE_LOW_THRESHOLD', 40)
    risk_mod = _get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60)
    
    df_unique_patients = df.sort_values('encounter_date').drop_duplicates(subset='patient_id', keep='last')
    
    def assign_tier(score):
        if score >= risk_mod: return 'High Risk'
        if score >= risk_low: return 'Moderate Risk'
        return 'Low Risk'

    df_unique_patients['risk_tier'] = convert_to_numeric(df_unique_patients['ai_risk_score']).apply(assign_tier)
    
    pyramid_data = df_unique_patients['risk_tier'].value_counts().reset_index()
    pyramid_data.columns = ['risk_tier', 'patient_count']
    
    # Trend of risk tiers
    df['risk_tier'] = convert_to_numeric(df['ai_risk_score']).apply(assign_tier)
    trend_data = df.groupby([pd.Grouper(key='encounter_date', freq=C.TIME_AGG_PERIOD), 'risk_tier'])['patient_id'].nunique().reset_index()
    
    return {'pyramid_data': pyramid_data, 'trend_data': trend_data}

# --- Page Setup & Data Loading ---
def setup_page_config():
    """Sets the Streamlit page configuration."""
    try:
        icon_to_use = C.PAGE_ICON
        page_icon_path_str = _get_setting('APP_LOGO_SMALL_PATH', None)
        if page_icon_path_str:
            page_icon_path = Path(page_icon_path_str)
            if page_icon_path.is_file(): icon_to_use = str(page_icon_path)
        
        st.set_page_config(
            page_title=f"{C.PAGE_TITLE} - {_get_setting('APP_NAME', 'Sentinel App')}",
            page_icon=icon_to_use, layout="wide"
        )
    except Exception as e:
        logger.error(f"Error applying page configuration: {e}", exc_info=True)
        st.set_page_config(page_title=C.PAGE_TITLE, page_icon=C.PAGE_ICON, layout="wide")

@st.cache_data(
    ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600),
    hash_funcs={pd.DataFrame: hash_dataframe_safe},
    show_spinner="Loading and preparing population analytics dataset..."
)
def get_population_analytics_datasets(log_ctx: str = "PopAnalytics/LoadData") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads, enriches, and validates the primary health and zone datasets."""
    logger.info(f"({log_ctx}) Initiating data load.")
    
    enriched_health_df = None
    try:
        raw_health_df = load_health_records(source_context=f"{log_ctx}/HealthRecs")
        if not isinstance(raw_health_df, pd.DataFrame) or raw_health_df.empty: return None, None
        
        enriched_health_df, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
        if not isinstance(enriched_health_df, pd.DataFrame): enriched_health_df = raw_health_df
    except Exception as e:
        logger.error(f"({log_ctx}) CRITICAL FAILURE during health data loading: {e}", exc_info=True)
        return None, None

    zone_attributes_df = load_zone_data(source_context=f"{log_ctx}/ZoneData")
    return enriched_health_df, zone_attributes_df

def initialize_session_state(health_df: pd.DataFrame, zone_df: Optional[pd.DataFrame]):
    """Centralizes initialization of all session state filter values."""
    min_fallback, max_fallback = date.today() - timedelta(days=3*365), date.today()
    min_data_date, max_data_date = min_fallback, max_fallback
    
    if 'encounter_date' in health_df.columns and not health_df['encounter_date'].isna().all():
        valid_dates = health_df['encounter_date'].dropna()
        if not valid_dates.empty:
            min_calc, max_calc = valid_dates.min().date(), valid_dates.max().date()
            if min_calc <= max_calc: min_data_date, max_data_date = min_calc, max_calc

    if C.SS_DATE_RANGE not in st.session_state: st.session_state[C.SS_DATE_RANGE] = [min_data_date, max_data_date]
    st.session_state['min_data_date'], st.session_state['max_data_date'] = min_data_date, max_data_date

    all_conditions = sorted(list(health_df['condition'].dropna().astype(str).unique()))
    st.session_state['all_conditions'] = all_conditions
    if C.SS_CONDITIONS not in st.session_state: st.session_state[C.SS_CONDITIONS] = []

    zone_options = ["All Zones/Regions"]
    zone_map = {}
    if zone_df is not None and not zone_df.empty and 'name' in zone_df.columns and 'zone_id' in zone_df.columns:
        valid_zones = zone_df.dropna(subset=['name', 'zone_id'])
        if not valid_zones.empty:
            zone_map = valid_zones.set_index('name')['zone_id'].to_dict()
            zone_options.extend(sorted(list(zone_map.keys())))
    st.session_state['zone_options'], st.session_state['zone_name_id_map'] = zone_options, zone_map
    if C.SS_ZONE not in st.session_state: st.session_state[C.SS_ZONE] = "All Zones/Regions"

# --- Main Application Logic ---
def run_dashboard():
    setup_page_config()
    st.title(f"🌍 {_get_setting('APP_NAME', 'Sentinel')} - Population Health Analytics Console")
    st.markdown("Strategic exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
    st.divider()

    health_df_main, zone_attr_main = get_population_analytics_datasets()

    if health_df_main is None or health_df_main.empty:
        st.error(f"🚨 **Critical Data Failure:** The primary health dataset is empty or could not be loaded. Please check logs and data sources at `{_get_setting('HEALTH_RECORDS_CSV_PATH', 'N/A')}`.")
        st.stop()
    
    health_df_main['encounter_date'] = pd.to_datetime(health_df_main['encounter_date'], errors='coerce').dt.tz_localize(None)
    initialize_session_state(health_df_main, zone_attr_main)

    # --- Sidebar Filters ---
    with st.sidebar:
        logo_path_str = _get_setting('APP_LOGO_SMALL_PATH', None)
        if logo_path_str and Path(logo_path_str).is_file(): st.image(logo_path_str, width=230)
        st.header("🔎 Analytics Filters")
        st.date_input("Select Date Range:", value=st.session_state[C.SS_DATE_RANGE], min_value=st.session_state['min_data_date'], max_value=st.session_state['max_data_date'], key=C.SS_DATE_RANGE)
        st.selectbox("Filter by Zone/Region:", options=st.session_state['zone_options'], key=C.SS_ZONE)
        st.multiselect("Filter by Condition(s):", options=st.session_state['all_conditions'], help="Select conditions to analyze across all tabs.", key=C.SS_CONDITIONS)

    # --- Apply Filters & Calculate KPIs ---
    df_filtered = health_df_main[health_df_main['encounter_date'].between(pd.to_datetime(st.session_state[C.SS_DATE_RANGE][0]), pd.to_datetime(st.session_state[C.SS_DATE_RANGE][1]), inclusive='both')]
    
    if st.session_state[C.SS_CONDITIONS]:
        df_filtered = df_filtered[df_filtered['condition'].isin(st.session_state[C.SS_CONDITIONS])]
    
    total_population = 0
    if st.session_state[C.SS_ZONE] != "All Zones/Regions":
        zone_id = st.session_state['zone_name_id_map'].get(st.session_state[C.SS_ZONE])
        if zone_id and zone_attr_main is not None:
            df_filtered = df_filtered[df_filtered['zone_id'].astype(str) == str(zone_id)]
            total_population = zone_attr_main.loc[zone_attr_main['zone_id'] == zone_id, 'population'].sum()
    elif zone_attr_main is not None:
        total_population = zone_attr_main['population'].sum()

    if df_filtered.empty:
        st.info("ℹ️ No data available for the selected filters.")
        st.stop()

    # --- Strategic KPI Display ---
    st.subheader("Strategic Population Health Indicators")
    kpi_cols = st.columns(4)
    unique_patients = df_filtered['patient_id'].nunique()
    kpi_cols[0].metric("Unique Patients Affected", f"{unique_patients:,}")
    prevalence = (unique_patients / total_population * 1000) if total_population > 0 else 0
    kpi_cols[1].metric("Prevalence per 1,000 Pop.", f"{prevalence:.1f}")

    risk_high_threshold = _get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60)
    high_risk_patients = df_filtered[df_filtered['ai_risk_score'] >= risk_high_threshold]['patient_id'].nunique()
    kpi_cols[2].metric("High-Risk Patient Cohort", f"{high_risk_patients:,}", f"{high_risk_patients/unique_patients:.1%}" if unique_patients > 0 else "0.0%")
    
    cond_analytics = get_condition_analytics(df_filtered)
    top_risk_condition = cond_analytics.sort_values('avg_risk_score', ascending=False).iloc[0]['condition'] if not cond_analytics.empty else "N/A"
    kpi_cols[3].metric("Top Condition by Avg. Risk", top_risk_condition)
    st.divider()

    # --- Main Analysis Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Epidemiological Overview", "🚨 Population Risk Stratification", "🗺️ Geospatial Analysis", "🧑‍🤝‍🧑 Demographic Insights"])

    with tab1:
        st.header("Epidemiological Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Conditions by Volume")
            top_by_count = cond_analytics.sort_values('count', ascending=False).head(C.TOP_N_CONDITIONS)
            fig_count = px.bar(top_by_count, x='count', y='condition', orientation='h', title="Most Frequent Conditions", labels={'y': '', 'x': 'Number of Encounters'})
            fig_count.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_count, use_container_width=True)
        with col2:
            st.subheader("Top Conditions by Severity")
            top_by_risk = cond_analytics.sort_values('avg_risk_score', ascending=False).head(C.TOP_N_CONDITIONS)
            fig_risk = px.bar(top_by_risk, x='avg_risk_score', y='condition', orientation='h', title="Highest-Risk Conditions", labels={'y': '', 'x': 'Average AI Risk Score'}, range_x=[0,100])
            fig_risk.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_risk, use_container_width=True)

        st.subheader("Encounter Trends")
        df_trend = df_filtered.set_index('encounter_date').resample(C.TIME_AGG_PERIOD).size()
        st.plotly_chart(plot_annotated_line_chart(df_trend, "Weekly Encounters Trend (All Selected)", "Encounters", y_values_are_counts=True), use_container_width=True)

    with tab2:
        st.header("Population Risk Stratification")
        risk_data = get_risk_stratification_data(df_filtered)
        pyramid_data = risk_data.get('pyramid_data')
        trend_data = risk_data.get('trend_data')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Risk Pyramid")
            if not pyramid_data.empty:
                fig = px.funnel(pyramid_data, x='patient_count', y='risk_tier', title="Patient Distribution by Risk Tier")
                fig.update_yaxes(categoryorder="array", categoryarray=['High Risk', 'Moderate Risk', 'Low Risk'])
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Risk Tier Trends")
            if not trend_data.empty:
                fig_trend = px.area(trend_data, x='encounter_date', y='patient_id', color='risk_tier', title="Trend of Patients in Each Risk Tier (Weekly)", labels={'patient_id': 'Unique Patients'}, category_orders={"risk_tier": ["Low Risk", "Moderate Risk", "High Risk"]})
                st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.header("Geospatial Analysis")
        if zone_attr_main is not None and not zone_attr_main.empty and 'geometry_obj' in zone_attr_main.columns:
            st.info("This map shows metrics aggregated by zone for the filtered data.")
            geo_agg = df_filtered.groupby('zone_id').agg(
                avg_risk_score=('ai_risk_score', 'mean'),
                unique_patients=('patient_id', 'nunique')
            ).reset_index()
            map_df = pd.merge(zone_attr_main, geo_agg, on='zone_id', how='left').fillna(0)
            
            map_df['prevalence_per_1000'] = (map_df['unique_patients'] / map_df['population'] * 1000).where(map_df['population'] > 0, 0)
            
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": row['geometry_obj'], "id": row['zone_id'], 
                     "properties": {"name": row['name'], "avg_risk_score": row['avg_risk_score'], "prevalence_per_1000": row['prevalence_per_1000']}}
                    for _, row in map_df.iterrows()
                ]
            }

            map_metric = st.selectbox("Select Map Metric:", ["Prevalence per 1,000", "Average AI Risk Score"])
            color_metric = 'prevalence_per_1000' if map_metric == "Prevalence per 1,000" else 'avg_risk_score'
            
            fig = px.choropleth_mapbox(map_df, geojson=geojson_data, locations="zone_id", color=color_metric,
                                       mapbox_style="carto-positron", zoom=8, center={"lat": -1.28, "lon": 36.81},
                                       opacity=0.6, hover_name="name",
                                       hover_data={"avg_risk_score": ":.2f", "prevalence_per_1000": ":.2f"},
                                       labels={color_metric: map_metric})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geospatial data is unavailable. Cannot render map.")

    with tab4:
        st.header("Demographic Insights")
        df_unique = df_filtered.drop_duplicates(subset=['patient_id'])
        col1, col2 = st.columns(2)
        if not df_unique['age'].dropna().empty:
            col1.plotly_chart(px.histogram(df_unique, x='age', nbins=20, title="Age Distribution (Unique Patients)"), use_container_width=True)
        if not df_unique['gender'].dropna().empty:
            counts = df_unique['gender'].value_counts()
            col2.plotly_chart(px.pie(counts, values=counts.values, names=counts.index, title="Gender Distribution (Unique Patients)"), use_container_width=True)
        
        st.subheader("Risk by Demographics")
        if not df_unique.empty:
             df_unique['age_band'] = pd.cut(df_unique['age'], bins=[0, 18, 40, 60, 120], labels=['0-18', '19-40', '41-60', '60+'])
             risk_by_demo = df_unique.groupby(['age_band', 'gender'])['ai_risk_score'].mean().reset_index()
             if not risk_by_demo.empty:
                fig = px.bar(risk_by_demo, x='age_band', y='ai_risk_score', color='gender', barmode='group', title="Average AI Risk Score by Age and Gender", labels={'ai_risk_score': 'Avg. AI Risk Score'})
                st.plotly_chart(fig, use_container_width=True)
        
    st.divider()
    st.caption(_get_setting('APP_FOOTER_TEXT', "© Sentinel Health"))
    logger.info(f"Dashboard rendered. Rows: {df_filtered.shape[0]}.")

if __name__ == "__main__":
    run_dashboard()
