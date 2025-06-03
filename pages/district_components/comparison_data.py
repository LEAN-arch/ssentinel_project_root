# sentinel_project_root/pages/district_components/comparison_data.py
# Prepares data for DHO zonal comparative analysis for Sentinel Health Co-Pilot.
# Renamed from comparison_data_preparer_district.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
# No direct data loading or processing helpers needed here, expects enriched_zone_df

logger = logging.getLogger(__name__)


def get_district_comparison_metrics_config( # Renamed from get_comparison_criteria_options_district
    district_zone_df_sample: Optional[pd.DataFrame] = None # Expects DataFrame, not GDF
) -> Dict[str, Dict[str, str]]:
    """
    Defines and returns metrics available for DHO zonal comparison tables/charts.
    Checks against a sample of the zone DataFrame to ensure columns exist and have data.

    Args:
        district_zone_df_sample: A small sample (e.g., .head(2)) of the enriched zone DataFrame.
                                 Used to validate column existence and non-null data.

    Returns:
        Dict[str, Dict[str, str]]: Configuration for available metrics.
            Format: {Display Name: {"col": actual_col_name_in_df, 
                                    "format_str": "{:.1f}", 
                                    "colorscale_hint": "PlotlyScaleName_r_or_normal"}}
    """
    module_log_prefix = "DistrictComparisonMetricsConfig"
    
    # Define all potential comparison metrics that could be derived from an enriched zone DataFrame.
    # Column names ('col') MUST match those produced by data_processing.enrichment.enrich_zone_geodata_with_health_aggregates.
    all_potential_metrics_definitions: Dict[str, Dict[str, str]] = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "format_str": "{:.1f}", "colorscale_hint": "OrRd"}, # Higher is worse (OrRd_r often means reversed, so OrRd)
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "format_str": "{:.1f}", "colorscale_hint": "YlOrRd"}, # Higher is worse
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "format_str": "{:.0f}%", "colorscale_hint": "Greens"}, # Higher is better
        "Population (Total by Zone)": {"col": "population", "format_str": "{:,.0f}", "colorscale_hint": "Blues"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "format_str": "{:.2f}", "colorscale_hint": "Greens"}, # Higher is better (if column exists)
        "Avg. Clinic CO2 (Zone Avg, ppm)": {"col": "zone_avg_co2", "format_str": "{:.0f}", "colorscale_hint": "Oranges"}, # Higher is worse
        "Population Density (per sqkm)": {"col": "population_density", "format_str": "{:.1f}", "colorscale_hint": "Plasma"}, # Higher is denser
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "format_str": "{:.1f}", "colorscale_hint": "Reds"}, # Higher is worse
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "format_str": "{:.0f}%", "colorscale_hint": "Greens"}, # Higher is better
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "format_str": "{:,.0f}", "colorscale_hint": "Purples"},
        "Avg. Patient Daily Steps (Zone)": {"col": "avg_daily_steps_zone", "format_str": "{:,.0f}", "colorscale_hint": "BuGn"} # Higher is better
    }
    
    # Dynamically add metrics for active cases of each key condition from settings.KEY_CONDITIONS_FOR_ACTION
    for condition_key_name_cfg in settings.KEY_CONDITIONS_FOR_ACTION:
        # Construct column name exactly as created in data_processing.enrichment module
        col_name_for_cond_metric = f"active_{condition_key_name_cfg.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_label_for_cond = condition_key_name_cfg.replace("(Severe)", "").strip() # Cleaner label for UI
        all_potential_metrics_definitions[f"Active {display_label_for_cond} Cases (Zone)"] = {
            "col": col_name_for_cond_metric, 
            "format_str": "{:.0f}", # Count of cases
            "colorscale_hint": "Reds" # Default for disease burden: higher is worse
        }

    if not isinstance(district_zone_df_sample, pd.DataFrame) or district_zone_df_sample.empty:
        logger.debug(f"({module_log_prefix}) No zone DataFrame sample provided. Returning all defined potential metrics without column validation.")
        return all_potential_metrics_definitions

    # Filter metrics: only include if the required column exists in the DataFrame sample
    # AND that column has at least one non-null data point in the sample.
    available_metrics_for_comparison: Dict[str, Dict[str, str]] = {}
    for metric_display_name, metric_props in all_potential_metrics_definitions.items():
        actual_column_name = metric_props["col"]
        if actual_column_name in district_zone_df_sample.columns and \
           district_zone_df_sample[actual_column_name].notna().any(): # Check if at least one non-NaN value exists
            available_metrics_for_comparison[metric_display_name] = metric_props
        else:
            logger.debug(
                f"({module_log_prefix}) Comparison metric '{metric_display_name}' (column '{actual_column_name}') "
                f"excluded: column missing from DataFrame sample or contains only NaN values."
            )
            
    if not available_metrics_for_comparison:
        logger.warning(f"({module_log_prefix}) No comparison metrics found to be available after checking DataFrame sample columns and data.")
        
    return available_metrics_for_comparison


def prepare_district_zonal_comparison_data( # Renamed function
    enriched_district_zone_df: Optional[pd.DataFrame], # Enriched DataFrame (not GeoDataFrame)
    reporting_period_context_str: str = "Latest Aggregated Data" # Renamed for clarity
) -> Dict[str, Any]:
    """
    Prepares data for the DHO Zonal Comparison tab, including a comparison table
    and the configuration of metrics used.

    Args:
        enriched_district_zone_df: The DataFrame output from `enrich_zone_geodata_with_health_aggregates`.
                                   Should contain 'zone_id', 'name', and various aggregated metrics.
        reporting_period_context_str: String describing the reporting period.

    Returns:
        Dict[str, Any]: Contains:
            "reporting_period": str
            "comparison_metrics_config": Dict (from get_district_comparison_metrics_config)
            "zonal_comparison_table_df": pd.DataFrame (Zone Name as index, metrics as columns)
            "data_availability_notes": List[str]
    """
    module_log_prefix = "DistrictZonalComparisonPrep" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Preparing zonal comparison data for period: {reporting_period_context_str}")
    
    # Initialize output structure with defaults for DataFrames
    output_zonal_comparison_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "comparison_metrics_config": {},
        "zonal_comparison_table_df": pd.DataFrame(), # Default to empty DF
        "data_availability_notes": []
    }
    
    if not isinstance(enriched_district_zone_df, pd.DataFrame) or enriched_district_zone_df.empty:
        note_msg = "Enriched District Zone DataFrame is missing or empty. Cannot prepare zonal comparison data."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        output_zonal_comparison_data["data_availability_notes"].append(note_msg)
        return output_zonal_comparison_data

    # Get available metrics configuration based on the columns present in the provided DataFrame
    available_metrics_for_table_config = get_district_comparison_metrics_config(enriched_district_zone_df.head(2)) # Pass small sample
    
    if not available_metrics_for_table_config:
        note_msg = "No valid metrics found for zonal comparison based on the columns and data in the provided zone DataFrame."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        output_zonal_comparison_data["data_availability_notes"].append(note_msg)
        return output_zonal_comparison_data
        
    output_zonal_comparison_data["comparison_metrics_config"] = available_metrics_for_table_config
    
    # Determine the zone identifier column for table display (prefer 'name', fallback to 'zone_id')
    zone_display_identifier_col = 'name' # User-friendly name for table index/rows
    if 'name' not in enriched_district_zone_df.columns or enriched_district_zone_df['name'].isnull().all():
        if 'zone_id' in enriched_district_zone_df.columns:
            zone_display_identifier_col = 'zone_id' # Fallback if 'name' is unusable
        else: 
            note_msg = "Critical error: Neither 'name' nor 'zone_id' found in enriched_district_zone_df. Cannot create comparison table."
            logger.error(f"({module_log_prefix}) {note_msg}")
            output_zonal_comparison_data["data_availability_notes"].append(note_msg)
            return output_zonal_comparison_data # Cannot proceed
            
    # Select columns for the comparison table: zone identifier + all available metric columns
    # Ensure 'zone_id' is also included if it's different from zone_display_identifier_col for potential internal use,
    # though not primary for display if 'name' is used.
    columns_for_final_comparison_table = [zone_display_identifier_col] + \
                                         [details['col'] for details in available_metrics_for_table_config.values()]
    if 'zone_id' in enriched_district_zone_df.columns and 'zone_id' not in columns_for_final_comparison_table:
        columns_for_final_comparison_table.append('zone_id') # Ensure 'zone_id' is present if not the display col

    # Ensure all selected columns actually exist in the DataFrame to prevent KeyErrors
    final_table_cols_to_select = [
        col_name for col_name in list(set(columns_for_final_comparison_table)) # Unique columns
        if col_name in enriched_district_zone_df.columns
    ]
    
    # If only zone identifier is left (no actual metrics, or identifier itself missing), create empty table
    num_metric_cols_in_final_selection = len([c for c in final_table_cols_to_select if c != zone_display_identifier_col and c != 'zone_id'])
    if num_metric_cols_in_final_selection == 0 or zone_display_identifier_col not in final_table_cols_to_select:
        note_msg = "No metric columns available in DataFrame for comparison table after filtering, or zone identifier missing."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        output_zonal_comparison_data["data_availability_notes"].append(note_msg)
        # Create an empty DF with just the identifier column if it exists, for schema consistency
        output_zonal_comparison_data["zonal_comparison_table_df"] = pd.DataFrame(
            columns=[zone_display_identifier_col] if zone_display_identifier_col in enriched_district_zone_df.columns else []
        )
        return output_zonal_comparison_data

    df_comparison_table_final = enriched_district_zone_df[final_table_cols_to_select].copy()
    
    # Set the zone identifier as index for better table display in Streamlit, but keep the column too
    if zone_display_identifier_col in df_comparison_table_final.columns:
        df_comparison_table_final = df_comparison_table_final.set_index(zone_display_identifier_col, drop=False) 
        df_comparison_table_final.index.name = "Zone / Sector" # More descriptive index name for display
    
    output_zonal_comparison_data["zonal_comparison_table_df"] = df_comparison_table_final
    
    num_metrics_in_final_table = len(available_metrics_for_table_config) # Number of configured metrics actually used
    logger.info(
        f"({module_log_prefix}) Zonal comparison data prepared with {len(df_comparison_table_final)} zones "
        f"and {num_metrics_in_final_table} metrics in the table."
    )
    return output_zonal_comparison_data
