# sentinel_project_root/data_processing/enrichment.py
# Functions for enriching data, e.g., merging health aggregates into zone data.

import pandas as pd
import numpy as np
import logging
import re
import os
from typing import Optional, Dict, Any, List, Union

try:
    from config import settings
    from .helpers import convert_to_numeric 
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in enrichment.py: {e}. Ensure paths are correct.")
    raise

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets a setting attribute or returns a default."""
    return getattr(settings, attr_name, default_value)

def _robust_merge_agg_for_enrichment(
    left_df: pd.DataFrame,
    right_df_optional: Optional[pd.DataFrame],
    target_col_name: str,
    on_col: str = 'zone_id',
    default_fill_value: Any = 0.0, 
    value_col_in_right_df: Optional[str] = None,
    source_context_for_log: str = "EnrichmentMerge"
) -> pd.DataFrame:
    """
    Helper to robustly merge an aggregated (right) DataFrame into a main (left) DataFrame.
    """
    if not isinstance(left_df, pd.DataFrame):
        logger.error(f"({source_context_for_log}) Left input is not a DataFrame. Cannot merge '{target_col_name}'.")
        return left_df

    left_df_enriched = left_df.copy()
    is_target_numeric = isinstance(default_fill_value, (int, float, np.number)) or pd.isna(default_fill_value)

    if target_col_name not in left_df_enriched.columns:
        if is_target_numeric:
            dtype_for_new_col = float if pd.isna(default_fill_value) else type(default_fill_value)
            left_df_enriched[target_col_name] = pd.Series(default_fill_value, index=left_df_enriched.index, dtype=dtype_for_new_col)
        else:
            left_df_enriched[target_col_name] = default_fill_value
    else:
        if is_target_numeric:
            left_df_enriched[target_col_name] = convert_to_numeric(left_df_enriched[target_col_name], default_value=default_fill_value)
        else:
            left_df_enriched[target_col_name] = left_df_enriched[target_col_name].fillna(default_fill_value)

    if not isinstance(right_df_optional, pd.DataFrame) or right_df_optional.empty or on_col not in right_df_optional.columns:
        return left_df_enriched

    actual_value_col_to_merge = value_col_in_right_df
    if not actual_value_col_to_merge:
        value_cols_in_right = [col for col in right_df_optional.columns if col != on_col]
        if not value_cols_in_right:
            return left_df_enriched
        actual_value_col_to_merge = value_cols_in_right[0]
    elif actual_value_col_to_merge not in right_df_optional.columns:
        return left_df_enriched
        
    temp_merged_value_col = f"__temp_merged_{target_col_name}_{os.urandom(4).hex()}"
    right_df_prepared = right_df_optional[[on_col, actual_value_col_to_merge]].copy()
    right_df_prepared.rename(columns={actual_value_col_to_merge: temp_merged_value_col}, inplace=True)

    original_index = left_df_enriched.index
    # Ensure join keys are consistent string types
    left_df_enriched[on_col] = left_df_enriched[on_col].astype(str).str.strip()
    right_df_prepared[on_col] = right_df_prepared[on_col].astype(str).str.strip()
    
    # Using merge with a temporary column name to avoid conflicts
    merged_df = pd.merge(left_df_enriched, right_df_prepared, on=on_col, how='left')
    
    # Update the target column with new values, then drop the temporary column
    if temp_merged_value_col in merged_df.columns:
        # Update only where the new value is not NaN
        update_mask = merged_df[temp_merged_value_col].notna()
        merged_df.loc[update_mask, target_col_name] = merged_df.loc[update_mask, temp_merged_value_col]
        merged_df.drop(columns=[temp_merged_value_col], inplace=True, errors='ignore')
    
    # Final cleanup of the target column
    if is_target_numeric:
        merged_df[target_col_name] = convert_to_numeric(merged_df[target_col_name], default_value=default_fill_value)
    else:
        merged_df[target_col_name] = merged_df[target_col_name].fillna(default_fill_value)
    
    # Restore original index if dimensions match
    if len(merged_df) == len(original_index):
        merged_df.index = original_index

    return merged_df

def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame], 
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Starting zone data enrichment process.")

    expected_base_cols = ['zone_id', 'name', 'geometry_obj', 'population', 'area_sqkm']
    expected_agg_cols = [
        'avg_risk_score', 'total_patient_encounters', 'total_active_key_infections', 
        'prevalence_per_1000', 'zone_avg_co2', 'facility_coverage_score', 
        'population_density', 'chw_density_per_10k', 'avg_test_turnaround_critical', 
        'perc_critical_tests_tat_met', 'avg_daily_steps_zone'
    ]
    key_conditions_list_setting = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    for cond_key_enrich in key_conditions_list_setting:
        safe_cond_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_enrich.lower().strip())}_cases"
        expected_agg_cols.append(safe_cond_col_name)
    all_expected_cols = list(set(expected_base_cols + expected_agg_cols))

    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Base zone DataFrame is empty or invalid. Cannot perform enrichment.")
        return pd.DataFrame(columns=all_expected_cols)

    enriched_df = zone_df.copy()

    if 'zone_id' not in enriched_df.columns:
        logger.error(f"({source_context}) Critical: 'zone_id' missing from base zone_df. Enrichment aborted.")
        return enriched_df

    default_aggregates_map: Dict[str, Any] = {
        'avg_risk_score': np.nan, 'total_patient_encounters': 0, 'total_active_key_infections': 0, 
        'prevalence_per_1000': np.nan, 'zone_avg_co2': np.nan, 'avg_test_turnaround_critical': np.nan, 
        'perc_critical_tests_tat_met': 0.0, 'avg_daily_steps_zone': np.nan,
        'facility_coverage_score': np.nan, 'population_density': np.nan, 'chw_density_per_10k': np.nan 
    }
    for cond_name_enrich in key_conditions_list_setting:
        col_name_enrich_dyn = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_name_enrich.lower().strip())}_cases"
        default_aggregates_map[col_name_enrich_dyn] = 0
    
    for col_name, default_val in default_aggregates_map.items():
        is_numeric = isinstance(default_val, (int, float, np.number)) or pd.isna(default_val)
        if col_name not in enriched_df.columns:
            dtype_new = float if is_numeric and pd.isna(default_val) else type(default_val)
            enriched_df[col_name] = pd.Series(default_val, index=enriched_df.index, dtype=dtype_new)
        else:
            if is_numeric:
                enriched_df[col_name] = convert_to_numeric(enriched_df[col_name], default_value=default_val)
            else:
                enriched_df[col_name] = enriched_df[col_name].fillna(default_val)

    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_df_agg_src = health_df[health_df['zone_id'].notna()].copy()
        health_df_agg_src['zone_id'] = health_df_agg_src['zone_id'].astype(str).str.strip()
        
        for num_col in ['ai_risk_score', 'test_turnaround_days', 'avg_daily_steps']:
            if num_col in health_df_agg_src.columns:
                health_df_agg_src[num_col] = convert_to_numeric(health_df_agg_src[num_col], default_value=np.nan)

        if 'patient_id' in health_df_agg_src.columns:
            pat_counts_zone = health_df_agg_src.groupby('zone_id')['patient_id'].nunique().reset_index(name='unique_patients')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, pat_counts_zone, 'total_population_health_data', value_col_in_right_df='unique_patients', default_fill_value=0)
        
        if 'ai_risk_score' in health_df_agg_src.columns:
            avg_risk_zone = health_df_agg_src.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_risk')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_risk_zone, 'avg_risk_score', value_col_in_right_df='mean_risk', default_fill_value=np.nan)
        
        enc_id_col = 'encounter_id' if 'encounter_id' in health_df_agg_src.columns else 'patient_id'
        if enc_id_col in health_df_agg_src.columns :
            enc_counts_zone = health_df_agg_src.groupby('zone_id')[enc_id_col].nunique().reset_index(name='enc_count')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, enc_counts_zone, 'total_patient_encounters', value_col_in_right_df='enc_count', default_fill_value=0)

        # CORRECTED: The logic for aggregating total key infections is now robust and correct.
        if 'condition' in health_df_agg_src.columns and 'patient_id' in health_df_agg_src.columns and key_conditions_list_setting:
            all_cond_dfs = []
            for cond_key_iter in key_conditions_list_setting:
                safe_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_iter.lower().strip())}_cases"
                try:
                    cond_mask_agg = health_df_agg_src['condition'].astype(str).str.contains(re.escape(cond_key_iter), case=False, na=False)
                    if cond_mask_agg.any():
                        active_cases_df = health_df_agg_src[cond_mask_agg].groupby('zone_id')['patient_id'].nunique().reset_index(name='case_count')
                        enriched_df = _robust_merge_agg_for_enrichment(enriched_df, active_cases_df, safe_col_name, value_col_in_right_df='case_count', default_fill_value=0)
                        
                        # Collect the counts for total aggregation later
                        if safe_col_name in enriched_df.columns:
                            all_cond_dfs.append(enriched_df[['zone_id', safe_col_name]].set_index('zone_id'))

                except Exception as e_cond_agg:
                     logger.warning(f"({source_context}) Error aggregating condition '{cond_key_iter}': {e_cond_agg}")
            
            # Sum up all individual active case columns to get the total
            if all_cond_dfs:
                total_infections_s = pd.concat(all_cond_dfs, axis=1).sum(axis=1)
                total_infections_df = total_infections_s.reset_index(name='total_infections')
                total_infections_df.rename(columns={'index': 'zone_id'}, inplace=True)
                
                # Use the robust merge helper to add this total to the main DataFrame
                enriched_df = _robust_merge_agg_for_enrichment(
                    enriched_df, total_infections_df, 'total_active_key_infections',
                    value_col_in_right_df='total_infections', default_fill_value=0
                )

    # --- Derived Metric Calculations ---
    if 'population' in enriched_df.columns and 'total_active_key_infections' in enriched_df.columns:
        pop_numeric = convert_to_numeric(enriched_df['population'], default_value=0.0)
        total_infections_numeric = convert_to_numeric(enriched_df['total_active_key_infections'], default_value=0.0)
        
        # Calculate prevalence, avoiding division by zero
        enriched_df['prevalence_per_1000'] = np.where(
            pop_numeric > 0, (total_infections_numeric / pop_numeric) * 1000, 0.0
        )
        enriched_df['prevalence_per_1000'] = enriched_df['prevalence_per_1000'].fillna(0.0)

    # --- Final Column Cleanup ---
    for col in all_expected_cols:
        if col not in enriched_df.columns:
            if "count" in col or "total" in col or "active" in col or "perc" in col:
                enriched_df[col] = 0
            else:
                enriched_df[col] = np.nan
    
    final_cols_present = [col for col in all_expected_cols if col in enriched_df.columns]
    enriched_df = enriched_df[final_cols_present]

    logger.info(f"({source_context}) Zone data enrichment complete. Final shape: {enriched_df.shape}")
    return enriched_df
