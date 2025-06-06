# sentinel_project_root/pages/04_population_dashboard.py
# Population Health Analytics & Research Console for Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Any, Tuple, List

# Removed unused imports: re, html
# Renamed plot_bar_chart to its source (px.bar) for clarity
import plotly.express as px

# --- Sentinel Project Imports ---
# Encapsulated import block for robust error handling and clear dependency management.
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_zone_data
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe, convert_to_numeric
    from visualization.plots import create_empty_figure, plot_annotated_line_chart
except ImportError as e_pop_dash_import:
    import sys
    current_file_path = Path(__file__).resolve()
    project_root_dir = current_file_path.parent.parent
    error_message = (
        f"Population Dashboard Import Error: {e_pop_dash_import}. "
        f"Ensure project root ('{project_root_dir}') is in sys.path and all modules are correct. "
        f"Current sys.path: {sys.path}"
    )
    try:
        st.error(error_message)
        st.stop()
    except NameError:
        print(error_message, file=sys.stderr)
        raise SystemExit(1) from e_pop_dash_import

# --- Logging and Constants ---
logger = logging.getLogger(__name__)

class C:
    """Centralized constants for maintainability."""
    # Page Config
    PAGE_TITLE = "Population Analytics"
    PAGE_ICON = "🌍"
    # Data Schemas
    HEALTH_BASE_SCHEMA = ['patient_id', 'encounter_date', 'condition', 'age', 'gender', 'zone_id', 
                          'ai_risk_score', 'ai_followup_priority_score']
    SDOH_BASE_SCHEMA = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min', 
                        'predominant_hazard_type', 'primary_livelihood', 'water_source_main', 'area_sqkm', 
                        'num_clinics', 'num_chws']
    # Analysis & UI
    TIME_AGG_PERIOD = 'W-MON'  # SOLVES NameError bug
    TOP_N_CONDITIONS = 10
    # Session State Keys
    SS_DATE_RANGE = "pop_dashboard_date_range_v2"
    SS_CONDITIONS = "pop_dashboard_conditions_v2"
    SS_ZONE = "pop_dashboard_zone_v2"

# --- Helper Functions ---
def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely retrieve a value from the settings object with a fallback."""
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
    checkbox_key = f"quick_filter_{condition}"
    selected_set = set(st.session_state.get(C.SS_CONDITIONS, []))
    
    if st.session_state.get(checkbox_key):
        selected_set.add(condition)
    else:
        selected_set.discard(condition)
        
    st.session_state[C.SS_CONDITIONS] = sorted(list(selected_set))

# --- Page Setup & Data Loading ---
def setup_page_config():
    """Sets the Streamlit page configuration with robust fallbacks."""
    try:
        page_icon_value = C.PAGE_ICON
        if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_FAVICON_PATH'):
            favicon_path = Path(_get_setting('PROJECT_ROOT_DIR', '.')) / _get_setting('APP_FAVICON_PATH', 'assets/favicon.ico')
            if favicon_path.is_file():
                page_icon_value = str(favicon_path)
            else:
                logger.warning(f"Favicon not found: {favicon_path}")
        st.set_page_config(
            page_title=f"{C.PAGE_TITLE} - {_get_setting('APP_NAME', 'Sentinel App')}",
            page_icon=page_icon_value,
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
            logger.error(f"({log_ctx}) Health records are empty or invalid. Aborting.")
            return None, None
        
        enriched_health_df, _ = apply_ai_models(raw_health_df.copy(), source_context=f"{log_ctx}/AIEnrich")
        if not isinstance(enriched_health_df, pd.DataFrame):
            logger.warning(f"({log_ctx}) AI model app did not return a DataFrame. Reverting.")
            enriched_health_df = raw_health_df
        
        for col in C.HEALTH_BASE_SCHEMA:
            if col not in enriched_health_df.columns:
                enriched_health_df[col] = np.nan
    except Exception as e:
        logger.error(f"({log_ctx}) CRITICAL FAILURE during health data loading: {e}", exc_info=True)
        return None, None

    try:
        zone_data_full = load_zone_data(source_context=f"{log_ctx}/ZoneData")
        if not isinstance(zone_data_full, pd.DataFrame) or zone_data_full.empty:
            zone_attributes_df = pd.DataFrame(columns=C.SDOH_BASE_SCHEMA)
        else:
            cols_to_keep = [col for col in C.SDOH_BASE_SCHEMA if col in zone_data_full.columns]
            zone_attributes_df = zone_data_full[cols_to_keep].copy()
            for col in C.SDOH_BASE_SCHEMA:
                if col not in zone_attributes_df.columns:
                    zone_attributes_df[col] = np.nan
    except Exception as e:
        logger.error(f"({log_ctx}) Failure during zone data loading. Error: {e}", exc_info=True)
        zone_attributes_df = pd.DataFrame(columns=C.SDOH_BASE_SCHEMA)

    logger.info(f"({log_ctx}) Data load successful. Health: {len(enriched_health_df)}, Zones: {len(zone_attributes_df)}.")
    return enriched_health_df, zone_attributes_df

def initialize_session_state(health_df: pd.DataFrame, zone_df: pd.DataFrame):
    """Centralizes initialization of all session state filter values."""
    min_fallback = date.today() - timedelta(days=3 * 365)
    max_fallback = date.today()
    min_data_date, max_data_date = min_fallback, max_fallback
    
    if 'encounter_date' in health_df.columns and not health_df['encounter_date'].isna().all():
        valid_dates = health_df['encounter_date'].dropna()
        min_calc, max_calc = valid_dates.min().date(), valid_dates.max().date()
        if min_calc <= max_calc:
            min_data_date, max_data_date = min_calc, max_calc

    st.session_state['min_data_date'] = min_data_date
    st.session_state['max_data_date'] = max_data_date
    if C.SS_DATE_RANGE not in st.session_state:
        st.session_state[C.SS_DATE_RANGE] = [min_data_date, max_data_date]

    all_conditions = sorted(list(health_df['condition'].dropna().astype(str).unique()))
    st.session_state['all_conditions'] = all_conditions
    if C.SS_CONDITIONS not in st.session_state:
        st.session_state[C.SS_CONDITIONS] = []
    else: # Prune stale selections
        st.session_state[C.SS_CONDITIONS] = [c for c in st.session_state[C.SS_CONDITIONS] if c in all_conditions]

    zone_options = ["All Zones/Regions"]
    zone_map = {}
    if 'name' in zone_df.columns and 'zone_id' in zone_df.columns:
        valid_zones = zone_df.dropna(subset=['name', 'zone_id'])
        if not valid_zones.empty:
            zone_map = valid_zones.set_index('name')['zone_id'].to_dict()
            zone_options.extend(sorted(list(zone_map.keys())))
    st.session_state['zone_options'] = zone_options
    st.session_state['zone_name_id_map'] = zone_map
    if C.SS_ZONE not in st.session_state:
        st.session_state[C.SS_ZONE] = "All Zones/Regions"

# --- Main Application Logic ---
def run_dashboard():
    setup_page_config()
    st.title(f"📊 {_get_setting('APP_NAME', 'Sentinel Health Co-Pilot')} - Population Health Analytics Console")
    st.markdown("In-depth exploration of demographic distributions, epidemiological patterns, clinical trends, and health system factors using aggregated population-level data.")
    st.divider()

    health_df_main, zone_attr_main = get_population_analytics_datasets()

    if health_df_main is None or health_df_main.empty:
        data_dir = _get_setting('DATA_DIR', "data_sources/")
        health_file = Path(_get_setting('HEALTH_RECORDS_CSV_PATH', "health_records_expanded.csv")).name
        st.error(f"🚨 **Critical Data Failure:** The primary health dataset is empty or could not be loaded. Please ensure `{health_file}` is valid in `{str(Path(data_dir).resolve())}`.")
        st.stop()

    if not pd.api.types.is_datetime64_any_dtype(health_df_main['encounter_date']):
        health_df_main['encounter_date'] = pd.to_datetime(health_df_main['encounter_date'], errors='coerce')
    if health_df_main['encounter_date'].dt.tz is not None:
        health_df_main['encounter_date'] = health_df_main['encounter_date'].dt.tz_localize(None)

    with st.sidebar:
        st.markdown("---")
        try:
            logo_path = Path(_get_setting('PROJECT_ROOT_DIR', '.')) / _get_setting('APP_LOGO_SMALL_PATH', 'assets/logo_placeholder.png')
            if logo_path.is_file(): st.image(str(logo_path.resolve()), width=230)
        except Exception as e: logger.error(f"Sidebar logo error: {e}", exc_info=True)
        st.markdown("---")
        st.header("🔎 Analytics Filters")

        initialize_session_state(health_df_main, zone_attr_main)

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
            else:
                st.caption("No data to determine high-impact conditions.")
        
        st.multiselect("Or, Search for Specific Condition(s):", options=st.session_state['all_conditions'],
                       help="Select any condition. Selections are synced with Quick Filters.", key=C.SS_CONDITIONS)

    # --- Apply Filters ONCE for Performance ---
    df_filtered = health_df_main.copy()
    start_date, end_date = st.session_state[C.SS_DATE_RANGE]
    df_filtered = df_filtered[df_filtered['encounter_date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date), inclusive='both')]
    
    selected_conditions = st.session_state[C.SS_CONDITIONS]
    if selected_conditions:
        df_filtered = df_filtered[df_filtered['condition'].isin(selected_conditions)]

    selected_zone = st.session_state[C.SS_ZONE]
    if selected_zone != "All Zones/Regions":
        zone_map = st.session_state['zone_name_id_map']
        if selected_zone in zone_map:
            df_filtered = df_filtered[df_filtered['zone_id'].astype(str) == str(zone_map[selected_zone])]

    # --- Main Page Content ---
    condition_str = ", ".join(selected_conditions) if selected_conditions else "All Conditions"
    context_str = f"({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}, Cond: {condition_str}, Zone: {selected_zone})"
    st.subheader("Population Health Snapshot", help=f"Currently showing data for: {context_str}")

    if df_filtered.empty:
        st.info("ℹ️ No data available for the selected filters.")
        st.stop()
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Encounters", f"{df_filtered.shape[0]:,}")
    kpi_cols[1].metric("Unique Patients", f"{df_filtered['patient_id'].nunique():,}")
    kpi_cols[2].metric("Avg. Patient Age", f"{convert_to_numeric(df_filtered['age']).mean():.1f}")
    kpi_cols[3].metric("Avg. AI Risk Score", f"{convert_to_numeric(df_filtered['ai_risk_score']).mean():.2f}")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Epi Overview", "🧑‍🤝‍🧑 Demographics & SDOH", "🔬 Clinical Insights", "⚙️ Systems & Equity"])

    with tab1:
        st.header("Epidemiological Overview")
        top_conds = df_filtered['condition'].value_counts().nlargest(10)
        st.plotly_chart(px.bar(top_conds, y=top_conds.index, x=top_conds.values, orientation='h', title="Top 10 Conditions in Filtered Data", labels={'y': 'Condition', 'x': 'Encounters'}).update_layout(yaxis={'categoryorder':'total ascending'}), use_container_width=True)
        
        df_trend = df_filtered.set_index('encounter_date').resample(C.TIME_AGG_PERIOD).size()
        if not df_trend.empty:
            st.plotly_chart(plot_annotated_line_chart(df_trend, "Weekly Encounters Trend", "Encounters", True), use_container_width=True)
        
        if selected_conditions:
            st.markdown("#### Trend of Selected Condition(s)")
            cond_trend_df = df_filtered.groupby([pd.Grouper(key='encounter_date', freq=C.TIME_AGG_PERIOD), 'condition']).size().reset_index(name='count')
            st.plotly_chart(px.line(cond_trend_df, x='encounter_date', y='count', color='condition', markers=True, title=f"Weekly Trends for: {condition_str}", labels={'encounter_date': 'Week', 'count': 'Encounters'}), use_container_width=True)

    with tab2:
        st.header("Demographics & Social Determinants of Health (SDOH)")
        df_unique = df_filtered.drop_duplicates(subset=['patient_id'])
        col1, col2 = st.columns(2)
        if not df_unique['age'].dropna().empty:
            col1.plotly_chart(px.histogram(df_unique, x='age', nbins=20, title="Age Distribution (Unique Patients)"), use_container_width=True)
        if not df_unique['gender'].dropna().empty:
            counts = df_unique['gender'].value_counts()
            col2.plotly_chart(px.pie(counts, values=counts.values, names=counts.index, title="Gender Distribution (Unique Patients)"), use_container_width=True)

        st.subheader("Zone Attributes")
        display_zones = zone_attr_main[zone_attr_main['zone_id'].isin(df_filtered['zone_id'].unique())] if not zone_attr_main.empty else pd.DataFrame()
        if not display_zones.empty:
            st.dataframe(display_zones[['name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min']].head(20), use_container_width=True)

    with tab3:
        st.header("Clinical Insights")
        col1, col2 = st.columns(2)
        for col, score_type, title in [(col1, 'ai_risk_score', "AI Risk Score"), (col2, 'ai_followup_priority_score', "AI Priority Score")]:
            scores = df_filtered[score_type].dropna()
            if not scores.empty:
                col.plotly_chart(px.histogram(scores, title=f"{title} Distribution"), use_container_width=True)
            else:
                col.plotly_chart(create_empty_figure(f"No {title} data available."), use_container_width=True)

    with tab4:
        st.header("Systems & Equity Insights")
        enc_by_zone = df_filtered['zone_id'].value_counts().nlargest(20)
        if not enc_by_zone.empty and not zone_attr_main.empty:
            zone_map = zone_attr_main.set_index('zone_id')['name'].to_dict()
            enc_by_zone.index = enc_by_zone.index.map(lambda zid: zone_map.get(zid, f"Zone {zid}"))
            st.plotly_chart(px.bar(enc_by_zone, y=enc_by_zone.index, x=enc_by_zone.values, orientation='h', title="Top 20 Zones by Encounter Volume").update_layout(yaxis={'categoryorder':'total ascending'}), use_container_width=True)

        try:
            req_cols = ['zone_id', 'name', 'avg_travel_time_clinic_min', 'population']
            if zone_attr_main is not None and all(c in zone_attr_main.columns for c in req_cols):
                enc_agg = df_filtered.groupby('zone_id').size().reset_index(name='encounters')
                eq_df = pd.merge(zone_attr_main, enc_agg, on='zone_id', how='left').fillna({'encounters': 0})
                eq_df['population'] = convert_to_numeric(eq_df['population'])
                eq_df = eq_df[eq_df['population'] > 0]
                if not eq_df.empty:
                    eq_df['util_per_1000'] = (eq_df['encounters'] / eq_df['population']) * 1000
                    fig = px.scatter(eq_df.dropna(subset=['avg_travel_time_clinic_min', 'util_per_1000']),
                                     x='avg_travel_time_clinic_min', y='util_per_1000', size='population',
                                     hover_name='name', title="Service Utilization vs. Travel Time by Zone",
                                     labels={'avg_travel_time_clinic_min': 'Avg Travel Time (min)', 'util_per_1000': 'Encounters per 1,000 Population'})
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Equity plot generation failed: {e}", exc_info=True)
            st.caption("Could not generate equity plot.")

    st.divider()
    st.caption(_get_setting('APP_FOOTER_TEXT', "Sentinel Health Co-Pilot."))
    logger.info(f"Population Dashboard rendered. Rows in view: {df_filtered.shape[0]}. Filters: {context_str}")

if __name__ == "__main__":
    run_dashboard()
