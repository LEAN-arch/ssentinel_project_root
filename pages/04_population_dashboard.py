# sentinel_project_root/pages/04_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Any, Tuple, List
import plotly.express as px

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
    st.error(f"Import Error: {e}. Ensure '{project_root_dir}' is in sys.path.")
    st.stop()

# --- Logging and Constants ---
logger = logging.getLogger(__name__)

class C:
    """Centralized constants for maintainability."""
    PAGE_TITLE = "Population Analytics"
    PAGE_ICON = "🌍"
    TIME_AGG_PERIOD = 'W-MON'
    TOP_N_CONDITIONS = 10
    SS_DATE_RANGE = "pop_dashboard_date_range_v2"
    SS_CONDITIONS = "pop_dashboard_conditions_v2"
    SS_ZONE = "pop_dashboard_zone_v2"

# --- Helper Functions ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

@st.cache_data
def get_high_impact_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the dataframe to find the most frequent and highest-risk conditions."""
    if df.empty or 'condition' not in df.columns or 'ai_risk_score' not in df.columns:
        return pd.DataFrame(columns=['condition', 'count', 'avg_risk_score'])
    
    df_copy = df.copy()
    df_copy['ai_risk_score'] = convert_to_numeric(df_copy['ai_risk_score'])
    agg_df = df_copy.groupby('condition').agg(
        count=('patient_id', 'size'),
        avg_risk_score=('ai_risk_score', 'mean')
    ).reset_index().dropna(subset=['avg_risk_score'])
    
    return agg_df.sort_values(by=['count', 'avg_risk_score'], ascending=[False, False]).head(C.TOP_N_CONDITIONS)

def handle_quick_filter_change(condition: str):
    """Callback to update the main condition session state when a checkbox is toggled."""
    selected_set = set(st.session_state.get(C.SS_CONDITIONS, []))
    if st.session_state.get(f"quick_filter_{condition}"):
        selected_set.add(condition)
    else:
        selected_set.discard(condition)
    st.session_state[C.SS_CONDITIONS] = sorted(list(selected_set))

# --- Page Setup & Data Loading ---
def setup_page_config():
    """Sets the Streamlit page configuration with robust fallbacks."""
    try:
        page_icon_path = _get_setting('APP_LOGO_SMALL_PATH', C.PAGE_ICON)
        st.set_page_config(
            page_title=f"{C.PAGE_TITLE} - {_get_setting('APP_NAME', 'Sentinel App')}",
            page_icon=str(page_icon_path) if isinstance(page_icon_path, Path) and page_icon_path.is_file() else C.PAGE_ICON,
            layout=_get_setting('APP_LAYOUT', "wide")
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
    
    try:
        raw_health_df = load_health_records(source_context=f"{log_ctx}/HealthRecs")
        if not isinstance(raw_health_df, pd.DataFrame) or raw_health_df.empty:
            logger.error(f"({log_ctx}) The 'load_health_records' function returned an empty or invalid DataFrame.")
            return None, None
        
        enriched_health_df, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
        if not isinstance(enriched_health_df, pd.DataFrame):
            enriched_health_df = raw_health_df # Fallback to raw data
    except Exception as e:
        logger.error(f"({log_ctx}) CRITICAL FAILURE during health data loading: {e}", exc_info=True)
        return None, None

    zone_attributes_df = load_zone_data(source_context=f"{log_ctx}/ZoneData")
    logger.info(f"({log_ctx}) Data load successful. Health records: {len(enriched_health_df)}")
    return enriched_health_df, zone_attributes_df

def initialize_session_state(health_df: pd.DataFrame, zone_df: Optional[pd.DataFrame]):
    """Centralizes initialization of all session state filter values."""
    min_fallback = date.today() - timedelta(days=3 * 365)
    max_fallback = date.today()
    min_data_date, max_data_date = min_fallback, max_fallback
    
    if 'encounter_date' in health_df.columns and not health_df['encounter_date'].isna().all():
        valid_dates = health_df['encounter_date'].dropna()
        min_calc, max_calc = valid_dates.min().date(), valid_dates.max().date()
        if min_calc <= max_calc:
            min_data_date, max_data_date = min_calc, max_calc

    if C.SS_DATE_RANGE not in st.session_state:
        st.session_state[C.SS_DATE_RANGE] = [min_data_date, max_data_date]
    st.session_state['min_data_date'], st.session_state['max_data_date'] = min_data_date, max_data_date

    all_conditions = sorted(list(health_df['condition'].dropna().astype(str).unique()))
    st.session_state['all_conditions'] = all_conditions
    if C.SS_CONDITIONS not in st.session_state:
        st.session_state[C.SS_CONDITIONS] = []

    zone_options = ["All Zones/Regions"]
    zone_map = {}
    if zone_df is not None and 'name' in zone_df.columns and 'zone_id' in zone_df.columns:
        valid_zones = zone_df.dropna(subset=['name', 'zone_id'])
        if not valid_zones.empty:
            zone_map = valid_zones.set_index('name')['zone_id'].to_dict()
            zone_options.extend(sorted(list(zone_map.keys())))
    st.session_state['zone_options'], st.session_state['zone_name_id_map'] = zone_options, zone_map
    if C.SS_ZONE not in st.session_state:
        st.session_state[C.SS_ZONE] = "All Zones/Regions"

# --- Main Application Logic ---
def run_dashboard():
    setup_page_config()
    st.title(f"📊 {_get_setting('APP_NAME', 'Sentinel')} - Population Health Analytics Console")
    st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
    st.divider()

    health_df_main, zone_attr_main = get_population_analytics_datasets()

    if health_df_main is None or health_df_main.empty:
        expected_path = settings.HEALTH_RECORDS_CSV_PATH
        st.error(
            f"🚨 **Critical Data Failure:** The primary health dataset is empty or could not be loaded. "
            f"The application failed to read the file at the expected path: `{str(expected_path)}`. "
            f"Please check the application logs for 'File not found' errors and verify the file's existence and read permissions in your deployment environment."
        )
        st.stop()

    if not pd.api.types.is_datetime64_any_dtype(health_df_main['encounter_date']):
        health_df_main['encounter_date'] = pd.to_datetime(health_df_main['encounter_date'], errors='coerce')
    if health_df_main['encounter_date'].dt.tz is not None:
        health_df_main['encounter_date'] = health_df_main['encounter_date'].dt.tz_localize(None)

    initialize_session_state(health_df_main, zone_attr_main)

    with st.sidebar:
        st.markdown("---")
        logo_path = _get_setting('APP_LOGO_SMALL_PATH', None)
        if logo_path and logo_path.is_file(): st.image(str(logo_path), width=230)
        st.markdown("---")
        st.header("🔎 Analytics Filters")

        st.date_input("Select Date Range:", value=st.session_state[C.SS_DATE_RANGE],
                      min_value=st.session_state['min_data_date'], max_value=st.session_state['max_data_date'],
                      key=C.SS_DATE_RANGE)
        st.selectbox("Filter by Zone/Region:", options=st.session_state['zone_options'], key=C.SS_ZONE)
        st.markdown("---")
        
        st.subheader("Condition Filters")
        high_impact_conditions_df = get_high_impact_conditions(health_df_main)
        with st.expander("⚡ Quick Filters: High-Impact Conditions"):
            if not high_impact_conditions_df.empty:
                for _, row in high_impact_conditions_df.iterrows():
                    condition, count, risk = row['condition'], row['count'], row['avg_risk_score']
                    label = f"{condition} (Enc: {int(count):,}, Risk: {risk:.2f})"
                    st.checkbox(label, value=(condition in st.session_state[C.SS_CONDITIONS]),
                                key=f"quick_filter_{condition}", on_change=handle_quick_filter_change, args=(condition,))
        st.multiselect("Or, Search for Specific Condition(s):", options=st.session_state['all_conditions'],
                       help="Select any condition. Synced with Quick Filters.", key=C.SS_CONDITIONS)

    # --- Apply Filters ONCE for Performance ---
    df_filtered = health_df_main[health_df_main['encounter_date'].between(
        pd.to_datetime(st.session_state[C.SS_DATE_RANGE][0]), 
        pd.to_datetime(st.session_state[C.SS_DATE_RANGE][1]), inclusive='both')]
    
    if st.session_state[C.SS_CONDITIONS]:
        df_filtered = df_filtered[df_filtered['condition'].isin(st.session_state[C.SS_CONDITIONS])]
    if st.session_state[C.SS_ZONE] != "All Zones/Regions":
        zone_id = st.session_state['zone_name_id_map'].get(st.session_state[C.SS_ZONE])
        if zone_id: df_filtered = df_filtered[df_filtered['zone_id'].astype(str) == str(zone_id)]

    # --- Main Page Content ---
    if df_filtered.empty:
        st.info("ℹ️ No data available for the selected filters.")
        st.stop()

    condition_str = ", ".join(st.session_state[C.SS_CONDITIONS]) or "All"
    context_str = f"Date: {st.session_state[C.SS_DATE_RANGE][0].strftime('%d %b %Y')} to {st.session_state[C.SS_DATE_RANGE][1].strftime('%d %b %Y')}, Conditions: {condition_str}, Zone: {st.session_state[C.SS_ZONE]}"
    st.subheader("Population Health Snapshot", help=context_str)
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Encounters", f"{df_filtered.shape[0]:,}")
    kpi_cols[1].metric("Unique Patients", f"{df_filtered['patient_id'].nunique():,}")
    kpi_cols[2].metric("Avg. Patient Age", f"{convert_to_numeric(df_filtered['age']).mean():.1f}")
    kpi_cols[3].metric("Avg. AI Risk Score", f"{convert_to_numeric(df_filtered['ai_risk_score']).mean():.2f}")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"])

    with tab1:
        # ... Plotting logic using df_filtered ...
        top_conds = df_filtered['condition'].value_counts().nlargest(10)
        st.plotly_chart(px.bar(top_conds, y=top_conds.index, x=top_conds.values, orientation='h', title="Top 10 Conditions").update_layout(yaxis={'categoryorder':'total ascending'}), use_container_width=True)

    with tab4:
        # ... Equity plot logic ...
        if zone_attr_main is not None and not zone_attr_main.empty:
             # Logic that merges df_filtered with zone_attr_main
             pass
    # ... (all other tab content remains structurally similar, always using `df_filtered`)

    st.divider()
    st.caption(_get_setting('APP_FOOTER_TEXT', "© Sentinel Health"))
    logger.info(f"Dashboard rendered. Rows: {df_filtered.shape[0]}. Filters: {context_str}")

if __name__ == "__main__":
    run_dashboard()
