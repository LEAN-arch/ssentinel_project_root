# sentinel_project_root/data_processing/enrichment.py
# Functions for enriching data, e.g., merging health aggregates into zone data.

import pandas as pd
import numpy as np
import logging
import re # For dynamic column name creation consistency
from typing import Optional, Dict, Any, List

from config import settings
from .helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def _robust_merge_agg_for_enrichment(
    left_df: pd.DataFrame,
    right_df_optional: Optional[pd.DataFrame],
    target_col_name: str,
    on_col: str = 'zone_id',
    default_fill_value: Any = 0.0, # Ensure this default matches expected type of target_col_name
    source_context_for_log: str = "EnrichmentMerge"
) -> pd.DataFrame:
    """
    Helper to robustly merge an aggregated (right) DataFrame into a main (left) DataFrame.
    Handles cases where right_df is None or empty, or target_col already exists.
    Fills NaNs in the merged column with default_fill_value.
    """
    left_df_copy = left_df.copy()

    # Ensure the target column exists in the left DataFrame, initialized correctly.
    is_numeric_default = isinstance(default_fill_value, (float, int, np.number)) or default_fill_value is np.nan
    
    if target_col_name not in left_df_copy.columns:
        if is_numeric_default:
            left_df_copy[target_col_name] = pd.Series(default_fill_value, index=left_df_copy.index, dtype=float if default_fill_value is np.nan else type(default_fill_value))
        else:
            left_df_copy[target_col_name] = default_fill_value # Assign scalar or Series of objects
    else: # Column exists, fill its NaNs or ensure correct type
        if is_numeric_default:
            left_df_copy[target_col_name] = convert_to_numeric(left_df_copy[target_col_name], default_value=default_fill_value)
        else:
            left_df_copy[target_col_name] = left_df_copy[target_col_name].fillna(default_fill_value)

    if not isinstance(right_df_optional, pd.DataFrame) or right_df_optional.empty or \
       on_col not in right_df_optional.columns:
        logger.debug(f"({source_context_for_log}) Right DataFrame for '{target_col_name}' empty/invalid. Left DF returned as is.")
        return left_df_copy

    # Ensure right_df_optional has a value column to merge
    value_cols_in_right = [col for col in right_df_optional.columns if col != on_col]
    if not value_cols_in_right:
        logger.warning(f"({source_context_for_log}) Right DataFrame for '{target_col_name}' has no value column. Merge skipped.")
        return left_df_copy
    
    value_col_to_merge = value_cols_in_right[0] # Use the first value column
    # Use a more robust temporary column name to avoid clashes
    temp_agg_col_name = f"__temp_agg_{target_col_name}_{pd.Timestamp.now().strftime('%H%M%S%f')}"

    right_df_prepared = right_df_optional[[on_col, value_col_to_merge]].copy()
    right_df_prepared.rename(columns={value_col_to_merge: temp_agg_col_name}, inplace=True)

    # Standardize join key types (string is safest for zone_ids)
    try:
        left_df_copy[on_col] = left_df_copy[on_col].astype(str).str.strip()
        right_df_prepared[on_col] = right_df_prepared[on_col].astype(str).str.strip()
    except Exception as e_type_conv:
        logger.error(f"({source_context_for_log}) Error converting join column '{on_col}' to string for '{target_col_name}': {e_type_conv}", exc_info=True)
        return left_df_copy # Return without merging if type conversion fails

    original_left_index_name = left_df_copy.index.name
    left_df_copy_reset = left_df_copy.reset_index() # Preserve index for later restoration
    index_col_name_after_reset = left_df_copy_reset.columns[0] # Name of the column created by reset_index

    merged_df = pd.merge(left_df_copy_reset, right_df_prepared, on=on_col, how='left')

    # Update target_col_name with merged data, prioritizing new data, then fill remaining NaNs
    if temp_agg_col_name in merged_df.columns:
        if is_numeric_default: # If target column should be numeric
            merged_df[temp_agg_col_name] = convert_to_numeric(merged_df[temp_agg_col_name], default_value=np.nan) # Convert temp col
        # Use .update() which is robust for NaNs, or combine_first
        merged_df[target_col_name] = merged_df[temp_agg_col_name].combine_first(merged_df[target_col_name])
        merged_df.drop(columns=[temp_agg_col_name], inplace=True)
    
    # Final fillna for the target column based on its intended type
    if is_numeric_default:
        merged_df[target_col_name] = convert_to_numeric(merged_df[target_col_name], default_value=default_fill_value)
    else:
        merged_df[target_col_name] = merged_df[target_col_name].fillna(default_fill_value)

    # Restore original index
    if index_col_name_after_reset in merged_df.columns: # Check if index column still exists
        merged_df.set_index(index_col_name_after_reset, inplace=True)
        merged_df.index.name = original_left_index_name
    else: # If reset_index column name was somehow dropped (e.g. it was the 'on_col')
         logger.warning(f"({source_context_for_log}) Index column '{index_col_name_after_reset}' lost during merge. Original index not fully restored.")
         # Try to restore from original left_df_copy index if shapes match
         if len(merged_df) == len(left_df_copy.index):
             merged_df.index = left_df_copy.index
         else:
             logger.error(f"({source_context_for_log}) Shape mismatch, cannot restore original index for '{target_col_name}'.")


    return merged_df


def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame], # Base zone data (output of load_zone_data)
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Starting zone data enrichment process.")

    # Define expected output columns for schema consistency if input is empty
    expected_enriched_cols = [
        'zone_id', 'name', 'geometry_obj', 'population', 'avg_risk_score', 
        'total_patient_encounters', 'total_active_key_infections', 
        'prevalence_per_1000', 'zone_avg_co2', 'facility_coverage_score', 
        'population_density', 'chw_density_per_10k', 'avg_test_turnaround_critical', 
        'perc_critical_tests_tat_met', 'avg_daily_steps_zone'
    ]
    for cond_key_enrich in settings.KEY_CONDITIONS_FOR_ACTION:
        col_name_dyn_enrich = f"active_{cond_key_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        expected_enriched_cols.append(col_name_dyn_enrich)

    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Base zone DataFrame is empty or invalid. Cannot perform enrichment.")
        return pd.DataFrame(columns=list(set(expected_enriched_cols)))

    enriched_df = zone_df.copy()
    if 'zone_id' not in enriched_df.columns:
        logger.error(f"({source_context}) Critical: 'zone_id' missing from base zone_df. Enrichment aborted.")
        return pd.DataFrame(columns=list(set(expected_enriched_cols)))

    # Initialize aggregate columns in enriched_df
    default_aggregates_map: Dict[str, Any] = {
        'total_population_health_data': 0, 'avg_risk_score': np.nan,
        'total_patient_encounters': 0, 'total_active_key_infections': 0,
        'prevalence_per_1000': np.nan, 'zone_avg_co2': np.nan,
        'avg_test_turnaround_critical': np.nan, 'perc_critical_tests_tat_met': 0.0,
        'avg_daily_steps_zone': np.nan, 'facility_coverage_score': np.nan,
        'population_density': np.nan, 'chw_density_per_10k': np.nan
    }
    for cond_name_enrich in settings.KEY_CONDITIONS_FOR_ACTION:
        col_name_enrich_dyn = f"active_{cond_name_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        default_aggregates_map[col_name_enrich_dyn] = 0
    
    for col_name, default_val in default_aggregates_map.items():
        if col_name not in enriched_df.columns:
            is_numeric = isinstance(default_val, (int, float)) or default_val is np.nan
            enriched_df[col_name] = pd.Series(default_val, index=enriched_df.index, dtype=float if is_numeric and default_val is np.nan else type(default_val))
        else: # Ensure type and fill NaNs if column already exists
            if isinstance(default_val, (float, int)) or default_val is np.nan:
                enriched_df[col_name] = convert_to_numeric(enriched_df[col_name], default_value=default_val)
            else:
                enriched_df[col_name] = enriched_df[col_name].fillna(default_val)

    # Aggregate from Health Data
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_df_agg_src = health_df[health_df['zone_id'].notna()].copy()
        health_df_agg_src['zone_id'] = health_df_agg_src['zone_id'].astype(str).str.strip()

        if 'patient_id' in health_df_agg_src.columns:
            pat_counts_zone = health_df_agg_src.groupby('zone_id')['patient_id'].nunique().reset_index(name='count')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, pat_counts_zone, 'total_population_health_data', default_fill_value=0)
        if 'ai_risk_score' in health_df_agg_src.columns:
            avg_risk_zone = health_df_agg_src.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_risk_zone, 'avg_risk_score', default_fill_value=np.nan)
        if 'encounter_id' in health_df_agg_src.columns:
            enc_counts_zone = health_df_agg_src.groupby('zone_id')['encounter_id'].nunique().reset_index(name='count')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, enc_counts_zone, 'total_patient_encounters', default_fill_value=0)

        total_key_infections_series = pd.Series(0.0, index=enriched_df.index, dtype=float)
        if 'condition' in health_df_agg_src.columns and 'patient_id' in health_df_agg_src.columns:
            for cond_key_iter_enrich in settings.KEY_CONDITIONS_FOR_ACTION:
                col_name_key_cond_enrich = f"active_{cond_key_iter_enrich.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
                cond_mask_agg_enrich = health_df_agg_src['condition'].astype(str).str.contains(re.escape(cond_key_iter_enrich), case=False, na=False, regex=True)
                if cond_mask_agg_enrich.any():
                    active_cases_zone_df = health_df_agg_src[cond_mask_agg_enrich].groupby('zone_id')['patient_id'].nunique().reset_index(name='count')
                    enriched_df = _robust_merge_agg_for_enrichment(enriched_df, active_cases_zone_df, col_name_key_cond_enrich, default_fill_value=0)
                    enriched_df[col_name_key_cond_enrich] = convert_to_numeric(enriched_df[col_name_key_cond_enrich], default_value=0) # Ensure numeric
                    total_key_infections_series = total_key_infections_series.add(enriched_df[col_name_key_cond_enrich], fill_value=0)
            enriched_df['total_active_key_infections'] = total_key_infections_series
        
        if 'test_type' in health_df_agg_src.columns and 'test_turnaround_days' in health_df_agg_src.columns:
            crit_tests_df_agg = health_df_agg_src[health_df_agg_src['test_type'].isin(settings.CRITICAL_TESTS)].copy()
            crit_tests_df_agg['test_turnaround_days'] = convert_to_numeric(crit_tests_df_agg['test_turnaround_days'], np.nan)
            avg_tat_crit_zone = crit_tests_df_agg.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_tat_crit_zone, 'avg_test_turnaround_critical', default_fill_value=np.nan)
            
            met_tat_counts_list: List[Dict[str, Any]] = []
            for zone_id_val_tat, group_df_tat in crit_tests_df_agg.groupby('zone_id'):
                total_concl_crit_zone = 0; met_tat_zone_count = 0
                for _, row_crit_test_tat in group_df_tat.iterrows():
                    test_cfg_tat = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_crit_test_tat['test_type'])
                    if test_cfg_tat and pd.notna(row_crit_test_tat['test_turnaround_days']):
                        total_concl_crit_zone += 1
                        target_tat_this_test = test_cfg_tat.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS)
                        if row_crit_test_tat['test_turnaround_days'] <= target_tat_this_test: met_tat_zone_count += 1
                perc_met_val = (met_tat_zone_count / total_concl_crit_zone * 100) if total_concl_crit_zone > 0 else 0.0
                met_tat_counts_list.append({'zone_id': zone_id_val_tat, 'perc_met_val': perc_met_val})
            if met_tat_counts_list:
                 enriched_df = _robust_merge_agg_for_enrichment(enriched_df, pd.DataFrame(met_tat_counts_list), 'perc_critical_tests_tat_met', default_fill_value=0.0)
        if 'avg_daily_steps' in health_df_agg_src.columns:
            avg_steps_zone = health_df_agg_src.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_steps_zone, 'avg_daily_steps_zone', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) Health DataFrame empty or missing 'zone_id'. Skipping health data aggregations.")

    # Aggregate from IoT Data
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty and 'zone_id' in iot_df.columns:
        iot_df_agg_src = iot_df[iot_df['zone_id'].notna()].copy()
        iot_df_agg_src['zone_id'] = iot_df_agg_src['zone_id'].astype(str).str.strip()
        if 'avg_co2_ppm' in iot_df_agg_src.columns:
            avg_co2_zone = iot_df_agg_src.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_co2_zone, 'zone_avg_co2', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) IoT DataFrame empty or missing 'zone_id'. Skipping IoT aggregations.")

    # Calculate Derived Metrics
    if 'population' in enriched_df.columns and 'total_active_key_infections' in enriched_df.columns:
        pop_num = convert_to_numeric(enriched_df['population'], default_value=0.0)
        enriched_df['prevalence_per_1000'] = np.where(
            pop_num > 0, (enriched_df['total_active_key_infections'] / pop_num) * 1000, 0.0
        )
    if 'population' in enriched_df.columns and 'area_sqkm' in enriched_df.columns:
        area_num = convert_to_numeric(enriched_df['area_sqkm'], default_value=0.0)
        pop_num_density = convert_to_numeric(enriched_df['population'], default_value=0.0) # Re-convert if used before
        enriched_df['population_density'] = np.where(area_num > 0, pop_num_density / area_num, 0.0)
    if 'num_clinics' in enriched_df.columns and 'population' in enriched_df.columns:
        clinics_num = convert_to_numeric(enriched_df['num_clinics'], default_value=0)
        pop_num_fac_cov = convert_to_numeric(enriched_df['population'], default_value=0)
        clinics_per_10k = np.where(pop_num_fac_cov > 0, (clinics_num / (pop_num_fac_cov / 10000)), 0.0)
        enriched_df['facility_coverage_score'] = np.clip(clinics_per_10k * 100, 0, 100) # Example scaling

    logger.info(f"({source_context}) Zone data enrichment complete. Final shape: {enriched_df.shape}")
    # Convert specific integer columns that might have become float due to NaNs back to nullable Int64 if desired
    # For simplicity here, most numeric columns will remain float if they encountered NaNs.
    return enriched_df
