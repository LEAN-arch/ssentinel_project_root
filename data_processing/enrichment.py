# sentinel_project_root/data_processing/enrichment.py
# Functions for enriching data, e.g., merging health aggregates into zone data.

import pandas as pd
import numpy as np
import logging
import re
import os  # <--- ADDED THIS IMPORT
from typing import Optional, Dict, Any, List, Union

try:
    from config import settings
    from .helpers import convert_to_numeric # Ensure this helper is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in enrichment.py: {e}. Ensure paths are correct.")
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
    value_col_in_right_df: Optional[str] = None, # Explicitly specify value col if right_df has multiple
    source_context_for_log: str = "EnrichmentMerge"
) -> pd.DataFrame:
    """
    Helper to robustly merge an aggregated (right) DataFrame into a main (left) DataFrame.
    Handles cases where right_df is None or empty, or target_col already exists.
    Fills NaNs in the merged column with default_fill_value.
    Assumes right_df_optional, if provided, has 'on_col' and one primary value column to merge.
    """
    if not isinstance(left_df, pd.DataFrame): # Should not happen if called internally
        logger.error(f"({source_context_for_log}) Left input is not a DataFrame. Cannot merge '{target_col_name}'.")
        return left_df # Or raise error

    left_df_enriched = left_df.copy() # Work on a copy

    # Determine if the target column should be numeric based on default_fill_value
    is_target_numeric = isinstance(default_fill_value, (int, float, np.number)) or pd.isna(default_fill_value)

    # Ensure the target column exists in the left DataFrame, initialized correctly.
    if target_col_name not in left_df_enriched.columns:
        logger.debug(f"({source_context_for_log}) Initializing new column '{target_col_name}' in left_df with default: {default_fill_value}")
        if is_target_numeric:
            # For numeric, ensure float dtype if default is NaN to hold NaNs, else type of default
            dtype_for_new_col = float if pd.isna(default_fill_value) else type(default_fill_value)
            left_df_enriched[target_col_name] = pd.Series(default_fill_value, index=left_df_enriched.index, dtype=dtype_for_new_col)
        else: # String or other object type
            left_df_enriched[target_col_name] = default_fill_value
    else: # Column exists, ensure its NaNs are filled with the default and type is appropriate
        logger.debug(f"({source_context_for_log}) Column '{target_col_name}' exists. Ensuring type and filling NaNs.")
        if is_target_numeric:
            left_df_enriched[target_col_name] = convert_to_numeric(left_df_enriched[target_col_name], default_value=default_fill_value)
        else: # For string or other object types
            left_df_enriched[target_col_name] = left_df_enriched[target_col_name].fillna(default_fill_value).astype(type(default_fill_value) if not pd.isna(default_fill_value) else str)


    if not isinstance(right_df_optional, pd.DataFrame) or right_df_optional.empty or \
       on_col not in right_df_optional.columns:
        logger.debug(f"({source_context_for_log}) Right DataFrame for '{target_col_name}' is empty, invalid, or missing '{on_col}'. No merge performed.")
        # Target column is already initialized with defaults, so just return
        return left_df_enriched

    # Determine the value column from right_df_optional to merge
    actual_value_col_to_merge = value_col_in_right_df
    if not actual_value_col_to_merge: # If not specified, try to infer
        value_cols_in_right = [col for col in right_df_optional.columns if col != on_col]
        if not value_cols_in_right:
            logger.warning(f"({source_context_for_log}) Right DataFrame for '{target_col_name}' has no value column (besides '{on_col}'). Merge skipped.")
            return left_df_enriched
        actual_value_col_to_merge = value_cols_in_right[0]
        if len(value_cols_in_right) > 1:
            logger.debug(f"({source_context_for_log}) Multiple potential value columns in right_df for '{target_col_name}'. Using first: '{actual_value_col_to_merge}'.")
    elif actual_value_col_to_merge not in right_df_optional.columns:
        logger.warning(f"({source_context_for_log}) Specified value_col_in_right_df '{actual_value_col_to_merge}' not found in right_df for '{target_col_name}'. Merge skipped.")
        return left_df_enriched
        
    # Prepare right DataFrame for merge (only on_col and the value column)
    # Use a unique temporary name for the value column from right_df to avoid clashes during merge
    temp_merged_value_col = f"__temp_merged_{target_col_name}_{os.urandom(4).hex()}"
    right_df_prepared = right_df_optional[[on_col, actual_value_col_to_merge]].copy()
    right_df_prepared.rename(columns={actual_value_col_to_merge: temp_merged_value_col}, inplace=True)

    # Standardize join key types (string is often safest for IDs like zone_id)
    original_left_on_col_dtype = left_df_enriched[on_col].dtype # Store original dtype if needed
    try:
        left_df_enriched[on_col] = left_df_enriched[on_col].astype(str).str.strip()
        right_df_prepared[on_col] = right_df_prepared[on_col].astype(str).str.strip()
    except Exception as e_type_conv_on_col:
        logger.error(f"({source_context_for_log}) Error converting join column '{on_col}' to string for '{target_col_name}': {e_type_conv_on_col}", exc_info=True)
        return left_df_enriched # Return without merging if type conversion of join key fails

    # Perform the merge
    # Preserve original index of left_df_enriched
    original_index = left_df_enriched.index
    left_df_enriched.reset_index(drop=True, inplace=True) # Drop original index for clean merge if it's unnamed or conflicting
    
    merged_df = pd.merge(left_df_enriched, right_df_prepared, on=on_col, how='left')

    # Update target_col_name with merged data.
    # Use .loc to ensure alignment and avoid potential SettingWithCopyWarning on slices.
    # Prioritize new data from temp_merged_value_col. If it's NaN, keep existing target_col_name value.
    if temp_merged_value_col in merged_df.columns:
        update_mask = merged_df[temp_merged_value_col].notna()
        merged_df.loc[update_mask, target_col_name] = merged_df.loc[update_mask, temp_merged_value_col]
        merged_df.drop(columns=[temp_merged_value_col], inplace=True)
    
    # Final fillna for the target column based on its intended type and default_fill_value
    if is_target_numeric:
        merged_df[target_col_name] = convert_to_numeric(merged_df[target_col_name], default_value=default_fill_value)
    else:
        merged_df[target_col_name] = merged_df[target_col_name].fillna(default_fill_value)

    # Restore original index if it was simple RangeIndex or compatible
    if len(merged_df) == len(original_index) and not isinstance(original_index, pd.MultiIndex): # Basic check
        merged_df.index = original_index
    else: # If index was complex or shapes changed unexpectedly, log and proceed with new index
        logger.warning(f"({source_context_for_log}) Could not perfectly restore original index for '{target_col_name}'. Merge might have changed row order or count.")

    return merged_df


def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame], 
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Starting zone data enrichment process.")

    # Define expected output columns for schema consistency
    # This helps ensure the final DataFrame has a predictable structure.
    expected_base_cols = ['zone_id', 'name', 'geometry_obj', 'population', 'area_sqkm'] # Assuming these are in zone_df
    expected_agg_cols = [
        'avg_risk_score', 'total_patient_encounters', 'total_active_key_infections', 
        'prevalence_per_1000', 'zone_avg_co2', 'facility_coverage_score', 
        'population_density', 'chw_density_per_10k', 'avg_test_turnaround_critical', 
        'perc_critical_tests_tat_met', 'avg_daily_steps_zone'
    ]
    key_conditions_list_setting = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    for cond_key_enrich in key_conditions_list_setting:
        # Sanitize condition name for use as a column name
        safe_cond_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_enrich.lower().strip())}_cases"
        expected_agg_cols.append(safe_cond_col_name)
    
    # All expected columns (unique)
    all_expected_cols = list(set(expected_base_cols + expected_agg_cols))


    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Base zone DataFrame is empty or invalid. Cannot perform enrichment. Returning empty DataFrame with expected columns.")
        return pd.DataFrame(columns=all_expected_cols)

    enriched_df = zone_df.copy() # Start with a copy of the base zone data

    if 'zone_id' not in enriched_df.columns:
        logger.error(f"({source_context}) Critical: 'zone_id' missing from base zone_df. Enrichment aborted. Returning original zone_df.")
        return enriched_df # Or return empty with expected cols: pd.DataFrame(columns=all_expected_cols)

    # Initialize aggregate columns in enriched_df with appropriate defaults and dtypes
    # Default values map for known aggregate columns
    default_aggregates_map: Dict[str, Any] = {
        'avg_risk_score': np.nan, 'total_patient_encounters': 0, 
        'total_active_key_infections': 0, 'prevalence_per_1000': np.nan, 
        'zone_avg_co2': np.nan, 'avg_test_turnaround_critical': np.nan, 
        'perc_critical_tests_tat_met': 0.0, 'avg_daily_steps_zone': np.nan,
        # Derived metrics will be calculated later, but ensure columns exist if needed by other parts
        'facility_coverage_score': np.nan, 'population_density': np.nan, 
        'chw_density_per_10k': np.nan 
    }
    for cond_name_enrich in key_conditions_list_setting:
        col_name_enrich_dyn = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_name_enrich.lower().strip())}_cases"
        default_aggregates_map[col_name_enrich_dyn] = 0
    
    for col_name, default_val in default_aggregates_map.items():
        is_numeric = isinstance(default_val, (int, float, np.number)) or pd.isna(default_val)
        if col_name not in enriched_df.columns:
            dtype_new = float if is_numeric and pd.isna(default_val) else type(default_val)
            enriched_df[col_name] = pd.Series(default_val, index=enriched_df.index, dtype=dtype_new)
        else: # Ensure correct type and fill existing NaNs
            if is_numeric:
                enriched_df[col_name] = convert_to_numeric(enriched_df[col_name], default_value=default_val)
            else: # String default
                enriched_df[col_name] = enriched_df[col_name].fillna(default_val).astype(type(default_val))


    # --- Aggregate from Health Data ---
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        # Prepare health_df for aggregation: ensure zone_id is string and clean, handle NaNs
        health_df_agg_src = health_df[health_df['zone_id'].notna()].copy() # Use copy for modifications
        health_df_agg_src['zone_id'] = health_df_agg_src['zone_id'].astype(str).str.strip()
        
        # Ensure key numeric columns are numeric for aggregation
        for num_col in ['ai_risk_score', 'test_turnaround_days', 'avg_daily_steps']:
            if num_col in health_df_agg_src.columns:
                health_df_agg_src[num_col] = convert_to_numeric(health_df_agg_src[num_col], default_value=np.nan)

        if 'patient_id' in health_df_agg_src.columns:
            # total_population_health_data implies unique patients from health data, not census population
            pat_counts_zone = health_df_agg_src.groupby('zone_id')['patient_id'].nunique().reset_index(name='unique_patients')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, pat_counts_zone, 'total_population_health_data', value_col_in_right_df='unique_patients', default_fill_value=0)
        
        if 'ai_risk_score' in health_df_agg_src.columns:
            avg_risk_zone = health_df_agg_src.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_risk')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_risk_zone, 'avg_risk_score', value_col_in_right_df='mean_risk', default_fill_value=np.nan)
        
        # Total Encounters (assuming encounter_id exists, else patient_id could be a proxy if each row is an encounter)
        enc_id_col = 'encounter_id' if 'encounter_id' in health_df_agg_src.columns else 'patient_id' # Fallback
        if enc_id_col in health_df_agg_src.columns :
            enc_counts_zone = health_df_agg_src.groupby('zone_id')[enc_id_col].nunique().reset_index(name='enc_count')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, enc_counts_zone, 'total_patient_encounters', value_col_in_right_df='enc_count', default_fill_value=0)

        # Active cases for key conditions
        total_key_infections_by_zone = pd.Series(0, index=enriched_df['zone_id'].unique(), dtype=float) # Temp series for sum
        if 'condition' in health_df_agg_src.columns and 'patient_id' in health_df_agg_src.columns and key_conditions_list_setting:
            for cond_key_iter in key_conditions_list_setting:
                safe_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_iter.lower().strip())}_cases"
                try:
                    cond_mask_agg = health_df_agg_src['condition'].astype(str).str.contains(re.escape(cond_key_iter), case=False, na=False, regex=True)
                    if cond_mask_agg.any():
                        active_cases_df = health_df_agg_src[cond_mask_agg].groupby('zone_id')['patient_id'].nunique().reset_index(name='case_count')
                        enriched_df = _robust_merge_agg_for_enrichment(enriched_df, active_cases_df, safe_col_name, value_col_in_right_df='case_count', default_fill_value=0)
                        
                        # Sum up for total_active_key_infections (aligns with enriched_df's zone_ids)
                        # This requires a bit more care to sum across the correct zones
                        current_cond_sum_per_zone = enriched_df.set_index('zone_id')[safe_col_name]
                        total_key_infections_by_zone = total_key_infections_by_zone.add(current_cond_sum_per_zone, fill_value=0)
                except Exception as e_cond_agg:
                     logger.warning(f"({source_context}) Error aggregating condition '{cond_key_iter}': {e_cond_agg}")
            enriched_df = pd.merge(enriched_df, total_key_infections_by_zone.rename('total_active_key_infections_temp').reset_index(), on='zone_id', how='left')
            enriched_df['total_active_key_infections'] = enriched_df['total_active_key_infections_temp'].fillna(0).combine_first(enriched_df['total_active_key_infections'])
            enriched_df.drop(columns=['total_active_key_infections_temp'], inplace=True, errors='ignore')


        # Test Turnaround Times (TAT)
        if 'test_type' in health_df_agg_src.columns and 'test_turnaround_days' in health_df_agg_src.columns:
            critical_test_names = _get_setting('CRITICAL_TESTS', [])
            if critical_test_names:
                crit_tests_df = health_df_agg_src[health_df_agg_src['test_type'].isin(critical_test_names)].copy() # Use copy for modifications
                crit_tests_df['test_turnaround_days'] = convert_to_numeric(crit_tests_df['test_turnaround_days'], np.nan) # Already done but good practice
                
                if not crit_tests_df.empty and crit_tests_df['test_turnaround_days'].notna().any():
                    avg_tat_crit_zone = crit_tests_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name='mean_tat')
                    enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_tat_crit_zone, 'avg_test_turnaround_critical', value_col_in_right_df='mean_tat', default_fill_value=np.nan)
                
                # Percentage TAT Met for critical tests
                key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
                default_target_tat = _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)
                met_tat_list_per_zone: List[Dict[str, Any]] = []

                for zone_id_val, group_by_zone_df in crit_tests_df.groupby('zone_id'):
                    total_conclusive_critical_in_zone = 0
                    met_tat_in_zone_count = 0
                    for _, test_row in group_by_zone_df.iterrows():
                        test_type_val = test_row.get('test_type')
                        tat_val = test_row.get('test_turnaround_days')
                        if pd.notna(tat_val) and test_type_val: # TAT is valid and test_type exists
                            total_conclusive_critical_in_zone += 1
                            test_specific_config = key_test_configs.get(test_type_val, {})
                            target_tat_for_this_test = test_specific_config.get('target_tat_days', default_target_tat)
                            if tat_val <= target_tat_for_this_test:
                                met_tat_in_zone_count += 1
                    
                    perc_met_for_zone = (met_tat_in_zone_count / total_conclusive_critical_in_zone * 100) if total_conclusive_critical_in_zone > 0 else 0.0
                    met_tat_list_per_zone.append({'zone_id': str(zone_id_val), 'perc_met_tat': perc_met_for_zone})
                
                if met_tat_list_per_zone:
                     met_tat_df_agg = pd.DataFrame(met_tat_list_per_zone)
                     enriched_df = _robust_merge_agg_for_enrichment(enriched_df, met_tat_df_agg, 'perc_critical_tests_tat_met', value_col_in_right_df='perc_met_tat', default_fill_value=0.0)

        if 'avg_daily_steps' in health_df_agg_src.columns: # Assuming this column might exist per patient in health_df
            avg_steps_zone = health_df_agg_src.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name='mean_steps')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_steps_zone, 'avg_daily_steps_zone', value_col_in_right_df='mean_steps', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) Health DataFrame empty or missing 'zone_id'. Skipping health data based aggregations.")

    # --- Aggregate from IoT Data ---
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty and 'zone_id' in iot_df.columns:
        iot_df_agg_src = iot_df[iot_df['zone_id'].notna()].copy()
        iot_df_agg_src['zone_id'] = iot_df_agg_src['zone_id'].astype(str).str.strip()
        if 'avg_co2_ppm' in iot_df_agg_src.columns: # Assuming IoT data has 'avg_co2_ppm' per reading/period
            iot_df_agg_src['avg_co2_ppm'] = convert_to_numeric(iot_df_agg_src['avg_co2_ppm'], default_value=np.nan)
            avg_co2_zone = iot_df_agg_src.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='mean_co2')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_co2_zone, 'zone_avg_co2', value_col_in_right_df='mean_co2', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) IoT DataFrame empty or missing 'zone_id'. Skipping IoT aggregations.")

    # --- Calculate Derived Metrics (using columns now present in enriched_df) ---
    if 'population' in enriched_df.columns and 'total_active_key_infections' in enriched_df.columns:
        pop_numeric = convert_to_numeric(enriched_df['population'], default_value=0.0)
        total_infections_numeric = convert_to_numeric(enriched_df['total_active_key_infections'], default_value=0.0)
        enriched_df['prevalence_per_1000'] = np.where(
            pop_numeric > 0, (total_infections_numeric / pop_numeric) * 1000, 0.0 # Use 0.0 if pop is 0
        )
        enriched_df['prevalence_per_1000'] = enriched_df['prevalence_per_1000'].fillna(0.0)


    if 'population' in enriched_df.columns and 'area_sqkm' in enriched_df.columns:
        area_numeric = convert_to_numeric(enriched_df['area_sqkm'], default_value=0.0)
        pop_density_numeric = convert_to_numeric(enriched_df['population'], default_value=0.0) 
        enriched_df['population_density'] = np.where(area_numeric > 0, pop_density_numeric / area_numeric, 0.0)
        enriched_df['population_density'] = enriched_df['population_density'].fillna(0.0)

    # Facility Coverage Score & CHW Density (example logic - needs 'num_clinics' and 'num_chws' in zone_df)
    if 'num_clinics' in enriched_df.columns and 'population' in enriched_df.columns:
        num_clinics_numeric = convert_to_numeric(enriched_df['num_clinics'], default_value=0)
        pop_facility_numeric = convert_to_numeric(enriched_df['population'], default_value=0.0)
        # Example: clinics per 10k population, scaled to a 0-100 score
        # This is a simplified placeholder; actual calculation would be more nuanced.
        clinics_per_10k_pop = np.where(pop_facility_numeric > 0, (num_clinics_numeric / (pop_facility_numeric / 10000)), 0.0)
        enriched_df['facility_coverage_score'] = np.clip(clinics_per_10k_pop * 20, 0, 100) # Arbitrary scaling to 0-100
        enriched_df['facility_coverage_score'] = enriched_df['facility_coverage_score'].fillna(0.0)


    if 'num_chws' in enriched_df.columns and 'population' in enriched_df.columns: # Assuming 'num_chws' column
        num_chws_numeric = convert_to_numeric(enriched_df['num_chws'], default_value=0)
        pop_chw_numeric = convert_to_numeric(enriched_df['population'], default_value=0.0)
        enriched_df['chw_density_per_10k'] = np.where(pop_chw_numeric > 0, (num_chws_numeric / (pop_chw_numeric / 10000)), 0.0)
        enriched_df['chw_density_per_10k'] = enriched_df['chw_density_per_10k'].fillna(0.0)


    # Ensure all expected columns are present in the final DataFrame, fill with appropriate NaN or 0 if not
    for col in all_expected_cols:
        if col not in enriched_df.columns:
            # Determine default based on if it's typically a count (0) or an average (NaN)
            if "count" in col or "total" in col or "active" in col or "perc" in col: # Heuristic
                enriched_df[col] = 0
            else:
                enriched_df[col] = np.nan
            logger.debug(f"({source_context}) Ensuring expected column '{col}' exists, added with default.")
    
    # Select and reorder to final expected schema
    final_cols_present = [col for col in all_expected_cols if col in enriched_df.columns]
    enriched_df = enriched_df[final_cols_present]


    logger.info(f"({source_context}) Zone data enrichment complete. Final shape: {enriched_df.shape}, Columns: {enriched_df.columns.tolist()}")
    return enriched_df
