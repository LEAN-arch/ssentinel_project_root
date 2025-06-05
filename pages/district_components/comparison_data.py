# sentinel_project_root/pages/district_components/comparison_data.py
# Prepares data for DHO zonal comparative analysis for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
import re # For dynamic column name creation consistency
from typing import Dict, Any, Optional, List

from config import settings

logger = logging.getLogger(__name__)


def get_district_comparison_metrics_config(
    district_zone_df_sample: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, str]]:
    """
    Defines and returns metrics available for DHO zonal comparison tables/charts.
    Checks against a sample of the zone DataFrame to ensure columns exist and have data.
    """
    module_log_prefix = "DistrictComparisonMetricsConfig"
    
    all_potential_metrics: Dict[str, Dict[str, str]] = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "format_str": "{:.1f}", "colorscale_hint": "OrRd"},
        "Key Disease Prevalence (/1k pop)": {"col": "prevalence_per_1000", "format_str": "{:.1f}", "colorscale_hint": "YlOrRd"},
        "Facility Coverage Score (%)": {"col": "facility_coverage_score", "format_str": "{:.0f}%", "colorscale_hint": "Greens"},
        "Population (Total by Zone)": {"col": "population", "format_str": "{:,.0f}", "colorscale_hint": "Blues"},
        "CHW Density (/10k pop)": {"col": "chw_density_per_10k", "format_str": "{:.2f}", "colorscale_hint": "Greens"},
        "Avg. Clinic CO2 (Zone Avg, ppm)": {"col": "zone_avg_co2", "format_str": "{:.0f}", "colorscale_hint": "Oranges"},
        "Population Density (per sqkm)": {"col": "population_density", "format_str": "{:.1f}", "colorscale_hint": "Plasma"},
        "Avg. Critical Test TAT (days)": {"col": "avg_test_turnaround_critical", "format_str": "{:.1f}", "colorscale_hint": "Reds"},
        "% Critical Tests TAT Met": {"col": "perc_critical_tests_tat_met", "format_str": "{:.0f}%", "colorscale_hint": "Greens"},
        "Total Patient Encounters (Zone)": {"col": "total_patient_encounters", "format_str": "{:,.0f}", "colorscale_hint": "Purples"},
        "Avg. Patient Daily Steps (Zone)": {"col": "avg_daily_steps_zone", "format_str": "{:,.0f}", "colorscale_hint": "BuGn"}
    }
    
    for cond_key_name in settings.KEY_CONDITIONS_FOR_ACTION:
        # Consistent column name generation (lowercase, underscores, no special chars like parentheses)
        col_name_cond = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_name.lower().replace('(severe)','').strip())}_cases"
        display_label_cond = cond_key_name.replace("(Severe)", "").strip()
        all_potential_metrics[f"Active {display_label_cond} Cases (Zone)"] = {
            "col": col_name_cond, "format_str": "{:.0f}", "colorscale_hint": "Reds"
        }

    if not isinstance(district_zone_df_sample, pd.DataFrame) or district_zone_df_sample.empty:
        logger.debug(f"({module_log_prefix}) No zone DF sample. Returning all potential metrics.")
        return all_potential_metrics

    available_metrics: Dict[str, Dict[str, str]] = {}
    for metric_name, metric_props in all_potential_metrics.items():
        col_name = metric_props["col"]
        if col_name in district_zone_df_sample.columns and district_zone_df_sample[col_name].notna().any():
            available_metrics[metric_name] = metric_props
        else:
            logger.debug(f"({module_log_prefix}) Metric '{metric_name}' (col '{col_name}') excluded: missing or all NaN in sample.")
            
    if not available_metrics:
        logger.warning(f"({module_log_prefix}) No comparison metrics available after checking DF sample.")
    return available_metrics


def prepare_district_zonal_comparison_data(
    enriched_district_zone_df: Optional[pd.DataFrame],
    reporting_period_context_str: str = "Latest Aggregated Data"
) -> Dict[str, Any]:
    """
    Prepares data for the DHO Zonal Comparison tab.
    """
    module_log_prefix = "DistrictZonalComparisonPrep"
    logger.info(f"({module_log_prefix}) Preparing zonal comparison data for: {reporting_period_context_str}")
    
    output_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str, "comparison_metrics_config": {},
        "zonal_comparison_table_df": pd.DataFrame(), "data_availability_notes": []
    }
    
    if not isinstance(enriched_district_zone_df, pd.DataFrame) or enriched_district_zone_df.empty:
        note = "Enriched District Zone DF missing/empty. Cannot prepare zonal comparison."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
        return output_data

    metrics_config = get_district_comparison_metrics_config(enriched_district_zone_df.head(2))
    if not metrics_config:
        note = "No valid metrics for zonal comparison based on provided zone DF."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
        return output_data
    output_data["comparison_metrics_config"] = metrics_config
    
    zone_display_col = 'name' # Prefer 'name' for display
    if 'name' not in enriched_district_zone_df.columns or enriched_district_zone_df['name'].isnull().all():
        if 'zone_id' in enriched_district_zone_df.columns: zone_display_col = 'zone_id'
        else: 
            note = "Critical: Neither 'name' nor 'zone_id' in enriched_district_zone_df for comparison table."
            logger.error(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
            return output_data
            
    # Columns for the table: display identifier + all available metric columns
    table_cols = [zone_display_col] + [details['col'] for details in metrics_config.values()]
    # Ensure 'zone_id' is also included if different from display_col and exists (for internal use/linking)
    if 'zone_id' in enriched_district_zone_df.columns and 'zone_id' not in table_cols:
        table_cols.append('zone_id')
    
    final_table_cols = [col for col in list(set(table_cols)) if col in enriched_district_zone_df.columns] # Unique & existing
    
    num_actual_metric_cols = len([c for c in final_table_cols if c != zone_display_col and c != 'zone_id'])
    if num_actual_metric_cols == 0 or zone_display_col not in final_table_cols:
        note = "No metric columns available or zone identifier missing for comparison table."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
        output_data["zonal_comparison_table_df"] = pd.DataFrame(columns=[zone_display_col] if zone_display_col in final_table_cols else [])
        return output_data

    df_comparison_table = enriched_district_zone_df[final_table_cols].copy()
    
    if zone_display_col in df_comparison_table.columns:
        df_comparison_table = df_comparison_table.set_index(zone_display_col, drop=False) 
        df_comparison_table.index.name = "Zone / Sector"
    
    output_data["zonal_comparison_table_df"] = df_comparison_table
    
    logger.info(f"({module_log_prefix}) Zonal comparison data prepared with {len(df_comparison_table)} zones and {len(metrics_config)} metrics.")
    return output_data
