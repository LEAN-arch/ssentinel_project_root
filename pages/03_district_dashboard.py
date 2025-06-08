# sentinel_project_root/pages/03_district_dashboard.py
"""
District Health Strategic Command Center for Sentinel Health Co-Pilot.

This Streamlit page provides a district-level overview of health metrics,
geospatial data, trends, and tools for intervention planning.
It aggregates data from various zones to give a high-level strategic view.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta, datetime
from typing import Optional, Dict, Any, List, Tuple
import os
from pathlib import Path
import sys

# --- Sentinel System Imports ---
# SME Note: Added a more robust import block. This helps when running the script
# directly for development by adding the project root to the path.
try:
    from config import settings
    from data_processing.loaders import load_health_records, load_iot_clinic_environment_data, load_zone_data
    from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
    from data_processing.aggregation import get_district_summary_kpis, get_trend_data
    from analytics.orchestrator import apply_ai_models
    from visualization.ui_elements import render_kpi_card
    from visualization.plots import plot_annotated_line_chart, plot_choropleth_map
except ImportError:
    # This makes the script runnable for development if the CWD is the project root
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from config import settings
        from data_processing.loaders import load_health_records, load_iot_clinic_environment_data, load_zone_data
        from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
        from data_processing.aggregation import get_district_summary_kpis, get_trend_data
        from analytics.orchestrator import apply_ai_models
        from visualization.ui_elements import render_kpi_card
        from visualization.plots import plot_annotated_line_chart, plot_choropleth_map
    except ImportError as e:
        st.error(
            "Fatal Error: A required module could not be imported.\n"
            f"Details: {e}\n"
            "Please ensure the script is run from the project root or that the `sentinel_project_root` "
            "directory is in your Python path."
        )
        st.stop()


# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Constants ---
# SME Note: Using constants for column names and keys improves maintainability and reduces errors from typos.
COL_GEOMETRY = 'geometry_obj'
COL_ZONE_ID = 'zone_id'
COL_ENCOUNTER_DATE = 'encounter_date'
COL_IOT_TIMESTAMP = 'timestamp'


# --- Page Specific Component Logic ---

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely retrieve a setting, returning a default if not found."""
    return getattr(settings, attr_name, default_value)

def structure_district_summary_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structures district-level KPIs for display using data from a dictionary."""
    if not kpis:
        return []

    return [
        {"title": "Total Population", "value_str": f"{kpis.get('total_population_district', 0):,}", "icon": "👥"},
        {"title": "Zones Monitored", "value_str": str(kpis.get('total_zones_in_df', 0)), "icon": "🗺️"},
        {
            "title": "Avg. AI Risk Score",
            "value_str": f"{kpis.get('population_weighted_avg_ai_risk_score', 0):.1f}",
            "icon": "🔥",
            "help_text": "Population-weighted average AI-predicted health risk score across all zones."
        },
        {
            "title": "High-Risk Zones",
            "value_str": str(kpis.get('zones_meeting_high_risk_criteria_count', 0)),
            "icon": "🚨",
            "help_text": f"Number of zones with an average risk score greater than {_get_setting('DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 7.5)}."
        },
    ]

def render_district_map_visualization(map_df: pd.DataFrame, metric_key: str, display_name: str):
    """Renders the main choropleth map for the district."""
    if COL_GEOMETRY not in map_df.columns:
        st.warning("Map unavailable: Zone geometry data is missing.")
        return
    if metric_key not in map_df.columns:
        st.warning(f"Metric '{display_name}' not available in the data for mapping.")
        return

    # Ensure metric column is numeric and fill any conversion errors with 0
    map_df[metric_key] = pd.to_numeric(map_df[metric_key], errors='coerce').fillna(0)

    # SME Note: OPTIMIZATION. Replaced slow `iterrows()` with a much faster list comprehension
    # using `to_dict('records')`. This is significantly more performant on large DataFrames.
    features = [
        {"type": "Feature", "geometry": row[COL_GEOMETRY], "id": str(row[COL_ZONE_ID]), "properties": {}}
        for row in map_df[[COL_GEOMETRY, COL_ZONE_ID]].to_dict('records')
    ]
    geojson_data = {"type": "FeatureCollection", "features": features}

    fig = plot_choropleth_map(
        map_df,
        geojson=geojson_data,
        locations=COL_ZONE_ID,
        color=metric_key,
        hover_name="name",
        labels={metric_key: display_name},
        title=f"<b>{display_name} by Zone</b>"
    )
    st.plotly_chart(fig, use_container_width=True)

# SME Note: BUG FIX - This function was called but not defined in the original script.
# This implementation infers its purpose: to provide a mapping of user-friendly names
# to DataFrame columns for the map metric selector.
def _get_district_map_metric_options_config(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Generates a configuration for map metric selection based on available data."""
    potential_metrics = {
        "Average Patient Risk Score": "avg_risk_score",
        "Total Population": "population",
        "Active TB Cases": "active_tb_cases",
        "Facility Coverage Score": "facility_coverage_score",
        "AI-Predicted Outbreak Risk": "ai_outbreak_risk_score",
    }
    # Return only metrics that actually exist as columns in the DataFrame
    return {
        display_name: {"col": col_name}
        for display_name, col_name in potential_metrics.items()
        if col_name in df.columns
    }

def calculate_district_wide_trends(health_df: pd.DataFrame, iot_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates various district-wide trends for key health and environmental metrics.
    SME Note: REFACTOR - Removed unused parameters. Simplified logic.
    """
    trends = {}
    if not health_df.empty:
        trends["avg_patient_ai_risk_trend"] = get_trend_data(
            health_df, value_col='ai_risk_score', date_col=COL_ENCOUNTER_DATE, freq='W-MON'
        )
    if not iot_df.empty:
        trends["avg_clinic_co2_trend"] = get_trend_data(
            iot_df, value_col='avg_co2_ppm', date_col=COL_IOT_TIMESTAMP, freq='D' # Daily avg is more stable for IoT
        )
    return trends

def get_district_intervention_criteria_options(df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    """Defines and filters intervention criteria based on available data columns."""
    options = {
        "High Avg. Patient Risk": {"col": "avg_risk_score", "threshold": _get_setting('DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 7.5), "comparison": "ge"},
        "Low Facility Coverage": {"col": "facility_coverage_score", "threshold": _get_setting('DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT', 0.5), "comparison": "le"},
        "High TB Burden": {"col": "active_tb_cases", "threshold": _get_setting('DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS', 10), "comparison": "ge"}
    }
    # Only offer criteria for which the data column exists in the dataframe
    if df is not None:
        return {name: config for name, config in options.items() if config["col"] in df.columns}
    return options

def identify_priority_zones(df: pd.DataFrame, criteria: List[str], options: Dict) -> pd.DataFrame:
    """
    Identifies zones that meet at least one of the selected intervention criteria.
    SME Note: REFACTOR - Simplified to return just the DataFrame. The criteria list is already known.
    Removed unused `context_str` parameter.
    """
    if df.empty or not criteria:
        return pd.DataFrame()

    # Start with a mask of all False
    final_mask = pd.Series(False, index=df.index)

    for criterion in criteria:
        config = options.get(criterion)
        if config and config["col"] in df.columns:
            col, thr, comp = config["col"], config["threshold"], config["comparison"]
            if comp == "ge":
                final_mask |= (df[col] >= thr)
            elif comp == "le":
                final_mask |= (df[col] <= thr)

    return df[final_mask]

# --- Data Caching and Preparation ---
@st.cache_data(ttl=_get_setting('CACHE_TTL_SECONDS_WEB_REPORTS', 3600), show_spinner="Aggregating district-level operational data...")
def get_dho_command_center_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main data loading and processing pipeline for the DHO dashboard.
    This function is cached to prevent reloading data on every interaction.
    """
    log_ctx = "DHODatasetPrep"
    logger.info(f"({log_ctx}) Initializing full data pipeline for DHO view...")
    # Load raw data
    raw_health_df = load_health_records()
    raw_iot_df = load_iot_clinic_environment_data()
    base_zone_df = load_zone_data()

    if base_zone_df.empty:
        logger.warning(f"({log_ctx}) Base zone data is empty. Aborting pipeline.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    # AI Enrichment
    ai_enriched_health_df, _ = apply_ai_models(raw_health_df.copy()) # Use .copy() to avoid cache mutation issues

    # Data Enrichment and Aggregation
    # SME Note: REFACTOR - Removed the invalid 'source_context' argument from the calls below
    # for consistency and to prevent potential errors if the underlying functions don't accept it.
    enriched_zone_df = enrich_zone_geodata_with_health_aggregates(
        zone_df=base_zone_df,
        health_df=ai_enriched_health_df,
        iot_df=raw_iot_df
    )

    if enriched_zone_df.empty:
        logger.warning(f"({log_ctx}) Zone data enrichment resulted in an empty DataFrame.")
        return base_zone_df, ai_enriched_health_df, raw_iot_df, {}

    # Prepare data for KPI calculation (without heavy geometry objects)
    df_for_kpis = enriched_zone_df.drop(columns=[COL_GEOMETRY], errors='ignore')
    district_summary_kpis = get_district_summary_kpis(df_for_kpis)

    logger.info(f"({log_ctx}) DHO data preparation complete.")
    return enriched_zone_df, ai_enriched_health_df, raw_iot_df, district_summary_kpis

# --- Main Page Execution ---
def main():
    """Main function to render the Streamlit page."""
    st.set_page_config(
        page_title=f"District Command - {_get_setting('APP_NAME', 'Sentinel')}",
        page_icon="🗺️",
        layout="wide"
    )
    st.title(f"🗺️ {_get_setting('APP_NAME', 'Sentinel')} - District Health Strategic Command Center")
    st.markdown("Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.")
    st.divider()

    # --- Load Data ---
    enriched_zone_df, health_df, iot_df, district_kpis = get_dho_command_center_processed_datasets()

    if enriched_zone_df.empty:
        st.error("Could not load or process the necessary zone and health data. The dashboard cannot be displayed.")
        st.stop()

    st.caption(f"Data presented as of: {pd.Timestamp('now', tz='UTC').strftime('%d %b %Y, %H:%M %Z')}")

    # --- Sidebar Filters ---
    st.sidebar.header("Analysis Filters")
    logo_path = _get_setting('APP_LOGO_SMALL_PATH', '')
    if logo_path and os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=120)

    # Determine date range from available data
    min_date = (date.today() - timedelta(days=365*2))
    max_date = date.today()
    if not health_df.empty and pd.api.types.is_datetime64_any_dtype(health_df[COL_ENCOUNTER_DATE]):
        min_date = health_df[COL_ENCOUNTER_DATE].min().date()
        max_date = health_df[COL_ENCOUNTER_DATE].max().date()
    
    default_start = max(min_date, max_date - timedelta(days=90))
    
    selected_start, selected_end = st.sidebar.date_input(
        "Select Date Range for Trend Analysis:",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # --- KPI Section ---
    st.header("📊 District Performance Dashboard")
    kpi_list = structure_district_summary_kpis(district_kpis)
    if kpi_list:
        cols = st.columns(len(kpi_list))
        for col, kpi in zip(cols, kpi_list):
            with col:
                render_kpi_card(**kpi)
    else:
        st.warning("District-wide summary KPIs are unavailable.")
    st.divider()

    # --- Tabbed Main Content ---
    tab_map, tab_trends, tab_compare, tab_intervene = st.tabs([
        "🗺️ Geospatial Overview",
        "📈 District Trends",
        "🆚 Zonal Comparison",
        "🎯 Intervention Planning"
    ])

    with tab_map:
        st.subheader("Interactive District Health Map")
        map_metric_options = _get_district_map_metric_options_config(enriched_zone_df)
        if not map_metric_options:
            st.info("No metrics are available for geospatial visualization.")
        else:
            selected_metric_name = st.selectbox(
                "Select Map Metric:",
                options=list(map_metric_options.keys())
            )
            render_district_map_visualization(
                enriched_zone_df,
                map_metric_options[selected_metric_name]['col'],
                selected_metric_name
            )

    with tab_trends:
        st.subheader("District-Wide Health & Environmental Trends")
        # SME Note: OPTIMIZATION. Filter datetime series directly without converting to .dt.date
        # This is significantly more performant.
        start_ts = pd.Timestamp(selected_start)
        end_ts = pd.Timestamp(selected_end) + pd.Timedelta(days=1) # include the whole end day

        health_trends_df = health_df[health_df[COL_ENCOUNTER_DATE].between(start_ts, end_ts)]
        iot_trends_df = iot_df[iot_df[COL_IOT_TIMESTAMP].between(start_ts, end_ts)]

        trends_data = calculate_district_wide_trends(health_trends_df, iot_trends_df)

        risk_trend = trends_data.get("avg_patient_ai_risk_trend")
        co2_trend = trends_data.get("avg_clinic_co2_trend")

        if risk_trend is None or risk_trend.empty:
            st.info("No patient risk trend data available for the selected period.")
        else:
            fig = plot_annotated_line_chart(risk_trend, "Weekly Average Patient Risk Score", "Avg. Risk Score")
            st.plotly_chart(fig, use_container_width=True)
        
        if co2_trend is None or co2_trend.empty:
            st.info("No clinic environmental trend data available for the selected period.")
        else:
            fig = plot_annotated_line_chart(co2_trend, "Daily Average Clinic CO₂ Levels", "Avg. CO₂ (PPM)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_compare:
        st.subheader("Comparative Zonal Analysis")
        # SME Note: REFACTOR - The original function `prepare_district_zonal_comparison_data` was trivial.
        # Direct usage of the dataframe is cleaner. Drop geometry for a cleaner table view.
        display_df = enriched_zone_df.drop(columns=[COL_GEOMETRY], errors='ignore')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab_intervene:
        st.subheader("Targeted Intervention Planning Assistant")
        criteria_options = get_district_intervention_criteria_options(enriched_zone_df)
        selected_criteria = st.multiselect(
            "Select Criteria to Identify Priority Zones:",
            options=list(criteria_options.keys()),
            help="Zones that meet one or more of the selected criteria will be shown."
        )
        if selected_criteria:
            priority_df = identify_priority_zones(enriched_zone_df, selected_criteria, criteria_options)
            st.markdown(f"###### **{len(priority_df)} Zone(s) Flagged for Intervention**")
            if not priority_df.empty:
                st.dataframe(
                    priority_df.drop(columns=[COL_GEOMETRY], errors='ignore'),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ No zones currently meet the selected high-priority criteria.")
        else:
            st.info("Select one or more criteria to identify priority zones.")

if __name__ == "__main__":
    main()
    logger.info("DHO Strategic Command Center page rendered.")
