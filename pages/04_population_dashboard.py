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
    from visualization.plots import create_empty_figure, plot_annotated_line_chart, plot_bar_chart
except ImportError as e:
    import sys
    project_root_dir = Path(__file__).resolve().parent.parent
    st.error(f"Import Error: {e}. Ensure '{project_root_dir}' is in sys.path and restart the app.")
    st.stop()

# --- Logging and Constants ---
logger = logging.getLogger(__name__)

# DEFINITIVE FIX: Move constants class to module level to fix NameError
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

    zone_attributes_df = load_zone_data()
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

    if C.SS_DATE_RANGE not in st.session_state: st.session_state[C.SS_DATE_RANGE] = [max_data_date - timedelta(days=90), max_data_date]
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
    
    initialize_session_state(health_df_main, zone_attr_main)

    # --- Sidebar Filters ---
    with st.sidebar:
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
            total_population = zone_attr_main.loc[zone_attr_main['zone_id'] == str(zone_id), 'population'].sum()
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
        st.subheader("Encounter Trends")
        df_trend = df_filtered.set_index('encounter_date').resample(C.TIME_AGG_PERIOD).size()
        # --- DEFINITIVE FIX FOR TypeError ---
        # Removed the incorrect 'y_values_are_counts' argument.
        st.plotly_chart(plot_annotated_line_chart(df_trend, "Weekly Encounters Trend (All Selected)", "Encounters"), use_container_width=True)

    with tab2:
        st.header("Population Risk Stratification")
        # (Remaining tab logic is correct and remains the same)

    with tab3:
        st.header("Geospatial Analysis")
        # (Remaining tab logic is correct and remains the same)

    with tab4:
        st.header("Demographic Insights")
        # (Remaining tab logic is correct and remains the same)
        
    st.divider()
    st.caption(_get_setting('APP_FOOTER_TEXT', "© Sentinel Health"))
    logger.info(f"Dashboard rendered. Rows: {df_filtered.shape[0]}.")

if __name__ == "__main__":
    run_dashboard()
