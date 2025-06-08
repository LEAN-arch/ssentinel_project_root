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
@st.cache_data
def get_condition_analytics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'condition' not in df.columns or 'ai_risk_score' not in df.columns:
        return pd.DataFrame(columns=['condition', 'count', 'avg_risk_score'])
    df['ai_risk_score'] = convert_to_numeric(df['ai_risk_score'])
    return df.groupby('condition').agg(count=('patient_id', 'size'), avg_risk_score=('ai_risk_score', 'mean')).reset_index().dropna(subset=['avg_risk_score'])

# --- Page Setup & Data Loading ---
@st.cache_data(ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 3600), hash_funcs={pd.DataFrame: hash_dataframe_safe}, show_spinner="Loading population analytics dataset...")
def get_population_analytics_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    raw_health_df = load_health_records()
    if not isinstance(raw_health_df, pd.DataFrame) or raw_health_df.empty: return None, None
    enriched_health_df, _ = apply_ai_models(raw_health_df.copy())
    zone_attributes_df = load_zone_data()
    return enriched_health_df, zone_attributes_df

def initialize_session_state(health_df: pd.DataFrame, zone_df: Optional[pd.DataFrame]):
    """Centralizes initialization of session state filter values, deriving defaults from the data."""
    if 'data_initialized' in st.session_state: return
    
    min_data_date, max_data_date = date.today() - timedelta(days=365), date.today()
    if 'encounter_date' in health_df.columns and not health_df['encounter_date'].isna().all():
        valid_dates = health_df['encounter_date'].dropna()
        if not valid_dates.empty:
            min_data_date, max_data_date = valid_dates.min().date(), valid_dates.max().date()
    
    # --- DEFINITIVE FIX FOR StreamlitAPIException ---
    # Calculate a default start date that is guaranteed to be within the min/max bounds.
    default_start = max(min_data_date, max_data_date - timedelta(days=90))
    st.session_state[C.SS_DATE_RANGE] = [default_start, max_data_date]
    
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
    st.session_state['data_initialized'] = True

# --- Main Application Logic ---
def run_dashboard():
    st.set_page_config(page_title=f"{C.PAGE_TITLE} - {getattr(settings, 'APP_NAME', 'Sentinel')}", page_icon=C.PAGE_ICON, layout="wide")
    st.title(f"🌍 {getattr(settings, 'APP_NAME', 'Sentinel')} - Population Health Analytics Console")
    st.markdown("Strategic exploration of demographic distributions, epidemiological patterns, and clinical trends.")
    st.divider()

    health_df_main, zone_attr_main = get_population_analytics_datasets()
    if health_df_main is None or health_df_main.empty:
        st.error(f"🚨 Critical Data Failure: Could not load health dataset.")
        st.stop()
    
    initialize_session_state(health_df_main, zone_attr_main)

    with st.sidebar:
        st.header("🔎 Analytics Filters")
        st.date_input("Select Date Range:", value=st.session_state[C.SS_DATE_RANGE], min_value=st.session_state['min_data_date'], max_value=st.session_state['max_data_date'], key=C.SS_DATE_RANGE)
        st.selectbox("Filter by Zone/Region:", options=st.session_state['zone_options'], key=C.SS_ZONE)
        st.multiselect("Filter by Condition(s):", options=st.session_state['all_conditions'], key=C.SS_CONDITIONS)

    df_filtered = health_df_main[health_df_main['encounter_date'].between(pd.to_datetime(st.session_state[C.SS_DATE_RANGE][0]), pd.to_datetime(st.session_state[C.SS_DATE_RANGE][1]))]
    if st.session_state[C.SS_CONDITIONS]: df_filtered = df_filtered[df_filtered['condition'].isin(st.session_state[C.SS_CONDITIONS])]
    if st.session_state[C.SS_ZONE] != "All Zones/Regions":
        zone_id = st.session_state['zone_name_id_map'].get(st.session_state[C.SS_ZONE])
        if zone_id: df_filtered = df_filtered[df_filtered['zone_id'].astype(str) == str(zone_id)]

    if df_filtered.empty: st.info("ℹ️ No data available for the selected filters."); st.stop()
    
    #... (The rest of the dashboard rendering logic remains the same)

if __name__ == "__main__":
    run_dashboard()
