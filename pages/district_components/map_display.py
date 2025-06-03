# sentinel_project_root/pages/district_components/map_display.py
# Renders the interactive district map for the Sentinel DHO web dashboard.
# Renamed from map_display_district_web.py

import streamlit as st
import pandas as pd # For type hinting, actual data is DataFrame
# GeoPandas is removed. Plotly Express will handle GeoJSON directly.
import logging
from typing import Dict, Any, Optional, List # List for hover_data_cols

from config import settings # Use new settings module
from visualization.plots import plot_choropleth_map, create_empty_figure # Use new plot functions

logger = logging.getLogger(__name__)


def _get_district_map_metric_options_config( # Renamed function
    district_zone_df_sample: Optional[pd.DataFrame] = None # Expects DataFrame, not GDF
) -> Dict[str, Dict[str, str]]:
    """
    Defines metrics available for map display, checking against DataFrame sample columns.
    These metrics should be present in the enriched zone DataFrame.

    Args:
        district_zone_df_sample: A small sample (e.g., .head(2)) of the enriched zone DataFrame.
                                 Used to validate column existence and non-null data.
    Returns:
        Dict[str, Dict[str, str]]: Configuration for available map metrics.
            Format: {Display Name: {"col": actual_col_name_in_df, 
                                    "colorscale": "PlotlyScaleName", 
                                    "format_str": "{:.1f}" (example for hover/labels)}}
    """
    module_log_prefix = "DistrictMapMetricsConfig"
    
    # Define all potential metrics that could be mapped from an enriched zone DataFrame.
    # 'col' names MUST align with output of `data_processing.enrichment.enrich_zone_geodata_with_health_aggregates`.
    all_map_metrics_potential_definitions: Dict[str, Dict[str, str]] = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "colorscale": "OrRd", "format_str": "{:.1f}"},
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd", "format_str": "{:.1f}"},
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "colorscale": "Greens", "format_str": "{:.0f}%"},
        "Population (Total by Zone)": {"col": "population", "colorscale": "Blues", "format_str": "{:,.0f}"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "colorscale": "Greens", "format_str": "{:.2f}"}, # If col exists
        "Avg. Clinic CO2 (Zone Avg, ppm)": {"col": "zone_avg_co2", "colorscale": "Oranges", "format_str": "{:.0f}"},
        "Population Density (per sqkm)": {"col": "population_density", "colorscale": "Plasma", "format_str": "{:.1f}"},
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "colorscale": "Reds", "format_str": "{:.1f}"},
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "colorscale": "Greens", "format_str": "{:.0f}%"},
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "colorscale": "Purples", "format_str": "{:,.0f}"},
        "Avg. Patient Daily Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "BuGn", "format_str": "{:,.0f}"}
    }
    
    # Dynamically add metrics for active cases of each key condition
    for condition_key_map_config_val in settings.KEY_CONDITIONS_FOR_ACTION:
        col_name_for_map_display_metric = f"active_{condition_key_map_config_val.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_label_for_map_cond = condition_key_map_config_val.replace("(Severe)", "").strip()
        all_map_metrics_potential_definitions[f"Active {display_label_for_map_cond} Cases (Zone)"] = {
            "col": col_name_for_map_display_metric, 
            "colorscale": "Reds", # Higher is worse
            "format_str": "{:.0f}"
        }

    if not isinstance(district_zone_df_sample, pd.DataFrame) or district_zone_df_sample.empty:
        logger.debug(f"({module_log_prefix}) No zone DataFrame sample provided. Returning all defined potential map metrics without column validation.")
        return all_map_metrics_potential_definitions

    # Filter metrics: only include if the required column exists in the DataFrame sample
    # AND that column has at least one non-null data point.
    available_metrics_for_map_display: Dict[str, Dict[str, str]] = {}
    for metric_display_name_map, metric_details_map_val in all_map_metrics_potential_definitions.items():
        actual_col_name_map_metric = metric_details_map_val["col"]
        if actual_col_name_map_metric in district_zone_df_sample.columns and \
           district_zone_df_sample[actual_col_name_map_metric].notna().any():
            available_metrics_for_map_display[metric_display_name_map] = metric_details_map_val
        else:
            logger.debug(
                f"({module_log_prefix}) Map metric '{metric_display_name_map}' (column '{actual_col_name_map_metric}') "
                f"excluded: column missing from DataFrame sample or contains only NaN values."
            )
    
    if not available_metrics_for_map_display:
         logger.warning(f"({module_log_prefix}) No map metrics found to be available after checking DataFrame sample.")
         
    return available_metrics_for_map_display


def render_district_map_visualization( # Renamed function
    enriched_district_zone_df: Optional[pd.DataFrame], # Enriched DataFrame (not GeoDataFrame)
    # GeoJSON features will be loaded by `plot_choropleth_map` or passed if already loaded
    default_metric_col_for_map_display: str = 'avg_risk_score', # Internal DataFrame column name for default selection
    reporting_period_context_str: str = "Latest Aggregated Zonal Data" # Renamed for clarity
) -> None:
    """
    Renders an interactive choropleth map for DHO's district-level visualization.
    Manages its own metric selection dropdown.
    The input `enriched_district_zone_df` should contain a 'geometry_obj' column with
    parsed GeoJSON geometry dictionaries, or 'geometry' with GeoJSON strings.
    """
    module_log_prefix = "DistrictMapVisualizer" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Rendering district interactive map for period: {reporting_period_context_str}")

    # The `plot_choropleth_map` function expects a list of GeoJSON features.
    # We need to extract these from the 'geometry_obj' or 'geometry' (JSON string) column
    # of the enriched_district_zone_df if they are stored there, or load the raw GeoJSON file.
    
    # For this refactor, we assume `load_zone_data` (in data_processing.loaders) now returns
    # a DataFrame where 'geometry_obj' holds the parsed GeoJSON geometry for each zone,
    # and 'zone_id' is present. `plot_choropleth_map` will need to be adapted or we
    # pass the raw geojson features list.
    # Let's assume `enriched_district_zone_df` has 'zone_id' and the metrics, and we
    # load the base GeoJSON features separately for the map.

    # --- Load raw GeoJSON features for the map base ---
    # This should ideally be cached.
    @st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
    def _load_geojson_features_for_map(geojson_file_path: str) -> Optional[List[Dict[str, Any]]]:
        if not os.path.exists(geojson_file_path):
            logger.error(f"({module_log_prefix}) GeoJSON file for map base not found: {geojson_file_path}")
            return None
        try:
            with open(geojson_file_path, 'r', encoding='utf-8') as f:
                geojson_data_map_base = json.load(f)
            if geojson_data_map_base and isinstance(geojson_data_map_base.get("features"), list):
                return geojson_data_map_base["features"]
            else:
                logger.error(f"({module_log_prefix}) Invalid GeoJSON structure in {geojson_file_path}. 'features' list missing.")
                return None
        except Exception as e_geojson_load:
            logger.error(f"({module_log_prefix}) Error loading or parsing GeoJSON from {geojson_file_path}: {e_geojson_load}", exc_info=True)
            return None

    geojson_features_list_for_map = _load_geojson_features_for_map(settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH)

    if not geojson_features_list_for_map:
        st.warning("Map visualization unavailable: Base geographic boundary data (GeoJSON features) could not be loaded.")
        st.plotly_chart(
            create_empty_figure("District Health Map", settings.WEB_MAP_DEFAULT_HEIGHT, "Geographic boundary data missing."),
            use_container_width=True
        )
        return

    if not isinstance(enriched_district_zone_df, pd.DataFrame) or enriched_district_zone_df.empty or \
       'zone_id' not in enriched_district_zone_df.columns: # zone_id is crucial for joining data to map
        st.warning("Map visualization unavailable: Enriched district zone data is missing, empty, or lacks 'zone_id' for map linkage.")
        st.plotly_chart(
            create_empty_figure("District Health Map", settings.WEB_MAP_DEFAULT_HEIGHT, "Enriched zone data not loaded or invalid."),
            use_container_width=True
        )
        return

    # Get available metrics for map selection based on columns in the enriched zone DataFrame
    map_metric_options_config_available = _get_district_map_metric_options_config(enriched_district_zone_df.head(2))
    
    if not map_metric_options_config_available:
        st.warning("No metrics available for map display based on the current zone data's columns or content.")
        st.plotly_chart(
            create_empty_figure("District Health Map", settings.WEB_MAP_DEFAULT_HEIGHT, "No metrics available to display on map."),
            use_container_width=True
        )
        return

    # Determine default selection for the selectbox (user-friendly display name)
    default_selection_display_name_map = None
    for display_name_map_opt, details_map_opt in map_metric_options_config_available.items():
        if details_map_opt["col"] == default_metric_col_for_map_display:
            default_selection_display_name_map = display_name_map_opt
            break
    if not default_selection_display_name_map: # Fallback if default col_name not found or not available
        default_selection_display_name_map = list(map_metric_options_config_available.keys())[0]

    # Create selectbox for user to choose which metric to visualize on the map
    selected_metric_display_name_user = st.selectbox(
        "Select Metric for Map Visualization:",
        options=list(map_metric_options_config_available.keys()),
        index=list(map_metric_options_config_available.keys()).index(default_selection_display_name_map),
        key="dho_map_metric_selector_main_key" # Unique key for this widget
    )
    
    selected_metric_config_details = map_metric_options_config_available.get(selected_metric_display_name_user)

    if selected_metric_config_details:
        metric_col_name_to_plot = selected_metric_config_details["col"]
        
        # Define a standard set of columns to show on hover, ensuring they exist in the DataFrame
        base_hover_cols_map_display = ['name', 'population', 'num_clinics', 'zone_id']
        hover_data_cols_for_map_display = [
            col_h for col_h in base_hover_cols_map_display if col_h in enriched_district_zone_df.columns
        ]
        # Ensure the plotted metric itself is included in hover data if not already in base_hover_cols
        if metric_col_name_to_plot not in hover_data_cols_for_map_display and \
           metric_col_name_to_plot in enriched_district_zone_df.columns:
            hover_data_cols_for_map_display.append(metric_col_name_to_plot)

        # The `plot_choropleth_map` function now takes the list of GeoJSON features.
        # The `zone_id_geojson_prop` should match the property name in your GeoJSON features that contains the zone ID.
        # The `zone_id_df_col` should be the column in `enriched_district_zone_df` that contains the zone ID.
        map_figure_to_display = plot_choropleth_map(
            map_data_df=enriched_district_zone_df,
            geojson_features=geojson_features_list_for_map, # Pass the loaded features
            value_col_name=metric_col_name_to_plot, 
            map_title=f"District Map: {selected_metric_display_name_user}",
            zone_id_geojson_prop='zone_id', # Assumes 'zone_id' is the property in GeoJSON features
            zone_id_df_col='zone_id',       # Assumes 'zone_id' is the column in the DataFrame
            color_scale_name=selected_metric_config_details["colorscale"],
            hover_name_col='name', # Use 'name' column from DataFrame for hover titles
            hover_data_cols=hover_data_cols_for_map_display,
            map_height=settings.WEB_MAP_DEFAULT_HEIGHT
            # mapbox_style_override can be passed if specific override needed, else uses theme
        )
        st.plotly_chart(map_figure_to_display, use_container_width=True)
        logger.info(
            f"({module_log_prefix}) District map rendered for metric: '{selected_metric_display_name_user}' "
            f"(using DataFrame column: '{metric_col_name_to_plot}')"
        )
    else: # Should not happen if selectbox options are derived correctly
        st.info("Please select a valid metric from the dropdown to display on the map.")
        logger.warning(
            f"({module_log_prefix}) No valid metric configuration found for selected display name: "
            f"'{selected_metric_display_name_user}' after selectbox interaction."
        )
