# sentinel_project_root/pages/district_components/map_display.py
# Renders the interactive district map for the Sentinel DHO web dashboard.

import streamlit as st
import pandas as pd
import logging
import json # For handling GeoJSON features list
import re # For dynamic column name creation
from typing import Dict, Any, Optional, List

from config import settings
from visualization.plots import plot_choropleth_map, create_empty_figure

logger = logging.getLogger(__name__)


def _get_district_map_metric_options_config(
    district_zone_df_sample: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, str]]:
    """
    Defines metrics available for map display, checking against DataFrame sample columns.
    """
    module_log_prefix = "DistrictMapMetricsConfig"
    all_map_metrics: Dict[str, Dict[str, str]] = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "colorscale": "OrRd", "format_str": "{:.1f}"},
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd", "format_str": "{:.1f}"},
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "colorscale": "Greens", "format_str": "{:.0f}%"},
        "Population (Total by Zone)": {"col": "population", "colorscale": "Blues", "format_str": "{:,.0f}"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "colorscale": "Greens", "format_str": "{:.2f}"},
        "Avg. Clinic CO2 (Zone Avg, ppm)": {"col": "zone_avg_co2", "colorscale": "Oranges", "format_str": "{:.0f}"},
        "Population Density (per sqkm)": {"col": "population_density", "colorscale": "Plasma", "format_str": "{:.1f}"},
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "colorscale": "Reds", "format_str": "{:.1f}"},
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "colorscale": "Greens", "format_str": "{:.0f}%"},
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "colorscale": "Purples", "format_str": "{:,.0f}"},
        "Avg. Patient Daily Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "BuGn", "format_str": "{:,.0f}"}
    }
    for cond_key_map in settings.KEY_CONDITIONS_FOR_ACTION:
        # CORRECTED: Removed the .replace('(severe)','') to match the column name generation in enrichment.py
        col_name_map = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_map.lower().strip())}_cases"
        disp_label_map = cond_key_map.replace("(Severe)", "").strip()
        all_map_metrics[f"Active {disp_label_map} Cases (Zone)"] = {
            "col": col_name_map, "colorscale": "Reds", "format_str": "{:.0f}"
        }

    if not isinstance(district_zone_df_sample, pd.DataFrame) or district_zone_df_sample.empty:
        logger.debug(f"({module_log_prefix}) No zone DF sample. Returning all potential map metrics.")
        return all_map_metrics

    available_metrics: Dict[str, Dict[str, str]] = {}
    for metric_name, props in all_map_metrics.items():
        col = props["col"]
        if col in district_zone_df_sample.columns and district_zone_df_sample[col].notna().any():
            available_metrics[metric_name] = props
        else:
            logger.debug(f"({module_log_prefix}) Map metric '{metric_name}' (col '{col}') excluded: missing/all NaN in sample.")
            
    if not available_metrics: logger.warning(f"({module_log_prefix}) No map metrics available after checking DF sample.")
    return available_metrics


def render_district_map_visualization(
    enriched_district_zone_df: Optional[pd.DataFrame],
    default_metric_col_for_map_display: str = 'avg_risk_score',
    reporting_period_context_str: str = "Latest Aggregated Zonal Data"
) -> None:
    """
    Renders an interactive choropleth map for DHO's district-level visualization.
    """
    module_log_prefix = "DistrictMapVisualizer"
    logger.info(f"({module_log_prefix}) Rendering district map for: {reporting_period_context_str}")

    @st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
    def _load_geojson_features(geojson_path: str) -> Optional[List[Dict[str, Any]]]:
        # This function no longer needs os.path check since settings validation handles it.
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f: geo_data = json.load(f)
            return geo_data.get("features") if isinstance(geo_data.get("features"), list) else None
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error loading/parsing GeoJSON from {geojson_path}: {e}", exc_info=True)
            return None

    geojson_features = _load_geojson_features(settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH)
    map_height = settings.WEB_MAP_DEFAULT_HEIGHT

    if not geojson_features:
        st.warning("Map unavailable: Base geographic boundary data (GeoJSON features) could not be loaded.")
        st.plotly_chart(create_empty_figure("District Health Map", map_height, "Geographic boundary data missing."), use_container_width=True)
        return

    if not isinstance(enriched_district_zone_df, pd.DataFrame) or enriched_district_zone_df.empty or \
       'zone_id' not in enriched_district_zone_df.columns:
        st.warning("Map unavailable: Enriched zone data missing, empty, or lacks 'zone_id' for map linkage.")
        st.plotly_chart(create_empty_figure("District Health Map", map_height, "Enriched zone data not loaded or invalid."), use_container_width=True)
        return

    map_metric_options = _get_district_map_metric_options_config(enriched_district_zone_df.head(2))
    if not map_metric_options:
        st.warning("No metrics available for map display based on current zone data.")
        st.plotly_chart(create_empty_figure("District Health Map", map_height, "No metrics available for map display."), use_container_width=True)
        return

    default_selection_name = next((name for name, props in map_metric_options.items() if props["col"] == default_metric_col_for_map_display), list(map_metric_options.keys())[0])
    
    map_metric_session_key = "dho_map_metric_selection"
    if map_metric_session_key not in st.session_state:
        st.session_state[map_metric_session_key] = default_selection_name
    
    # CORRECTED: Use a consistent key for the widget and for accessing session_state.
    # The `key` argument in Streamlit widgets becomes the dictionary key in `st.session_state`.
    selected_metric_name = st.selectbox(
        "Select Metric for Map Visualization:", options=list(map_metric_options.keys()),
        key=map_metric_session_key
    )
    
    selected_metric_cfg = map_metric_options.get(selected_metric_name)

    if selected_metric_cfg:
        metric_col_plot = selected_metric_cfg["col"]
        base_hover_cols = ['name', 'population', 'num_clinics', 'zone_id']
        hover_data_map = [col for col in base_hover_cols if col in enriched_district_zone_df.columns]
        if metric_col_plot not in hover_data_map and metric_col_plot in enriched_district_zone_df.columns:
            hover_data_map.append(metric_col_plot)

        map_fig = plot_choropleth_map(
            map_data_df=enriched_district_zone_df, geojson_features=geojson_features,
            value_col_name=metric_col_plot, map_title=f"District Map: {selected_metric_name}",
            zone_id_geojson_prop='zone_id', zone_id_df_col='zone_id',
            color_scale_name=selected_metric_cfg["colorscale"],
            hover_name_col='name', hover_data_cols=hover_data_map, map_height=map_height
        )
        st.plotly_chart(map_fig, use_container_width=True)
        logger.info(f"({module_log_prefix}) District map rendered for metric: '{selected_metric_name}' (col: '{metric_col_plot}')")
    else:
        st.info("Please select a valid metric to display on the map.")
        logger.warning(f"({module_log_prefix}) No valid metric config for selected name: '{selected_metric_name}'")
