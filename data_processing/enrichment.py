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
    value_col_in_right_df: Optional[str] = None,
    source_context_for_log: str = "EnrichmentMerge"
) -> pd.DataFrame:
    # This helper function is complex but assumed correct for this fix.
    # ... (its implementation remains unchanged) ...
    if not isinstance(left_df, pd.DataFrame):
        logger.error(f"({source_context_for_log}) Left input is not a DataFrame. Cannot merge '{target_col_name}'.")
        return left_df

    left_df_enriched = left_df.copy()
    is_target_numeric = isinstance(default_fill_value, (int, float, np.number)) or pd.isna(default_fill_value)

    if target_col_name not in left_df_enriched.columns:
        dtype_for_new_col = float if is_target_numeric and pd.isna(default_fill_value) else type(default_fill_value)
        left_df_enriched[target_col_name] = pd.Series(default_fill_value, index=left_df_enriched.index, dtype=dtype_for_new_col if is_target_numeric else object)
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
        
    temp_merged_value_col = f"__temp_merged_{os.urandom(4).hex()}"
    right_df_prepared = right_df_optional[[on_col, actual_value_col_to_merge]].rename(columns={actual_value_col_to_merge: temp_merged_value_col})
    
    original_index = left_df_enriched.index
    left_df_enriched[on_col] = left_df_enriched[on_col].astype(str)
    right_df_prepared[on_col] = right_df_prepared[on_col].astype(str)

    merged_df = pd.merge(left_df_enriched.reset_index(drop=True), right_df_prepared, on=on_col, how='left')
    
    update_mask = merged_df[temp_merged_value_col].notna()
    merged_df.loc[update_mask, target_col_name] = merged_df.loc[update_mask, temp_merged_value_col]
    merged_df.drop(columns=[temp_merged_value_col], inplace=True)
    
    if is_target_numeric:
        merged_df[target_col_name] = convert_to_numeric(merged_df[target_col_name], default_value=default_fill_value)
    else:
        merged_df[target_col_name] = merged_df[target_col_name].fillna(default_fill_value)
    
    merged_df.index = original_index
    return merged_df

def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame], 
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Starting zone data enrichment process.")
    
    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        return pd.DataFrame()
        
    enriched_df = zone_df.copy()
    if 'zone_id' not in enriched_df.columns:
        logger.error(f"({source_context}) Critical: 'zone_id' missing from base zone_df.")
        return enriched_df

    # Initializing columns logic remains the same...
    
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_df_agg_src = health_df[health_df['zone_id'].notna()].copy()
        health_df_agg_src['zone_id'] = health_df_agg_src['zone_id'].astype(str).str.strip()
        
        # ... Other enrichment steps remain the same ...

        # Active cases for key conditions
        total_key_infections_by_zone = pd.Series(0, index=enriched_df['zone_id'].unique(), dtype=float)
        # CORRECTED: Name the index of the series before resetting it.
        total_key_infections_by_zone.index.name = 'zone_id'
        
        key_conditions_list_setting = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
        if 'condition' in health_df_agg_src.columns and 'patient_id' in health_df_agg_src.columns and key_conditions_list_setting:
            for cond_key_iter in key_conditions_list_setting:
                safe_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key_iter.lower().strip())}_cases"
                try:
                    cond_mask_agg = health_df_agg_src['condition'].astype(str).str.contains(re.escape(cond_key_iter), case=False, na=False, regex=True)
                    if cond_mask_agg.any():
                        active_cases_df = health_df_agg_src[cond_mask_agg].groupby('zone_id')['patient_id'].nunique().reset_index(name='case_count')
                        enriched_df = _robust_merge_agg_for_enrichment(enriched_df, active_cases_df, safe_col_name, value_col_in_right_df='case_count', default_fill_value=0)
                        
                        current_cond_sum_per_zone = enriched_df.set_index('zone_id')[safe_col_name]
                        total_key_infections_by_zone = total_key_infections_by_zone.add(current_cond_sum_per_zone, fill_value=0)
                except Exception as e_cond_agg:
                     logger.warning(f"({source_context}) Error aggregating condition '{cond_key_iter}': {e_cond_agg}")
            
            # CORRECTED: The Series now has a named index, so reset_index() will create a 'zone_id' column.
            enriched_df = pd.merge(enriched_df, total_key_infections_by_zone.rename('total_active_key_infections_temp').reset_index(), on='zone_id', how='left')
            enriched_df['total_active_key_infections'] = enriched_df['total_active_key_infections_temp'].fillna(0).combine_first(enriched_df.get('total_active_key_infections', 0))
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
