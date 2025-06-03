# sentinel_project_root/data_processing/enrichment.py
# Functions for enriching data, e.g., merging health aggregates into zone data.

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

from config import settings # Use the new settings module
from .helpers import convert_to_numeric # Local import

logger = logging.getLogger(__name__)


def _robust_merge_agg_for_enrichment(
    left_df: pd.DataFrame,
    right_df_optional: Optional[pd.DataFrame],
    target_col_name: str,
    on_col: str = 'zone_id', # Common join key
    default_fill_value: Any = 0.0,
    source_context_for_log: str = "EnrichmentMerge"
) -> pd.DataFrame:
    """
    Helper to robustly merge an aggregated (right) DataFrame into a main (left) DataFrame.
    Handles cases where right_df is None or empty, or target_col already exists.
    Fills NaNs in the merged column with default_fill_value.
    """
    left_df_copy = left_df.copy() # Work on a copy to avoid modifying original outside function

    # Ensure the target column exists in the left DataFrame, initialized with default if not.
    if target_col_name not in left_df_copy.columns:
        left_df_copy[target_col_name] = default_fill_value
    else:
        # If it exists, ensure its NaNs are filled with the default (or convert type if needed)
        try:
            if pd.api.types.is_numeric_dtype(type(default_fill_value)) or default_fill_value is np.nan:
                 left_df_copy[target_col_name] = convert_to_numeric(left_df_copy[target_col_name], default_value=default_fill_value)
            else: # String or other type
                 left_df_copy[target_col_name] = left_df_copy[target_col_name].fillna(default_fill_value)
        except Exception as e_fill:
            logger.warning(f"({source_context_for_log}) Could not pre-fill NaNs for existing target_col '{target_col_name}': {e_fill}")
            left_df_copy[target_col_name] = default_fill_value # Fallback to just assigning default

    # If no right_df or it's unsuitable for merge, return the prepared left_df_copy
    if not isinstance(right_df_optional, pd.DataFrame) or right_df_optional.empty or \
       on_col not in right_df_optional.columns:
        logger.debug(f"({source_context_for_log}) Right DataFrame for '{target_col_name}' is empty or missing '{on_col}'. Left DF returned as is (or with new default col).")
        return left_df_copy

    # Prepare right_df: select join key and value column, rename value column to avoid clashes
    # Assume right_df has two columns: the 'on_col' and one value column.
    value_cols_in_right = [col for col in right_df_optional.columns if col != on_col]
    if not value_cols_in_right:
        logger.warning(f"({source_context_for_log}) Right DataFrame for '{target_col_name}' has no value column (only '{on_col}'). Merge skipped.")
        return left_df_copy
    
    value_col_to_merge = value_cols_in_right[0] # Take the first value column
    temp_agg_col_name = f"__temp_agg_{target_col_name}_{pd.Timestamp.now().strftime('%H%M%S%f')}" # Unique temp name

    right_df_prepared = right_df_optional[[on_col, value_col_to_merge]].copy()
    right_df_prepared.rename(columns={value_col_to_merge: temp_agg_col_name}, inplace=True)

    # Standardize join key types (string is safest for zone_ids that might be numeric-like)
    try:
        left_df_copy[on_col] = left_df_copy[on_col].astype(str).str.strip()
        right_df_prepared[on_col] = right_df_prepared[on_col].astype(str).str.strip()
    except Exception as e_type_conv:
        logger.error(f"({source_context_for_log}) Error converting join column '{on_col}' to string for '{target_col_name}': {e_type_conv}")
        return left_df_copy # Return without merging if type conversion fails

    # Perform the merge
    # Keep original index of left_df_copy
    original_left_index_name = left_df_copy.index.name
    left_df_copy_reset = left_df_copy.reset_index() # Store index to reset it later

    merged_df = pd.merge(left_df_copy_reset, right_df_prepared, on=on_col, how='left')

    # Combine the newly merged data with existing target_col_name data if any, then fill NaNs
    if temp_agg_col_name in merged_df.columns:
        # If target_col_name was already numeric, ensure temp_agg_col_name is also numeric before combine_first
        if pd.api.types.is_numeric_dtype(merged_df[target_col_name].dtype):
            merged_df[temp_agg_col_name] = convert_to_numeric(merged_df[temp_agg_col_name], default_value=np.nan) # Use NaN for combine_first behavior
        
        merged_df[target_col_name] = merged_df[temp_agg_col_name].combine_first(merged_df[target_col_name])
        merged_df.drop(columns=[temp_agg_col_name], inplace=True)
    
    # Final fillna for the target column after merging and combining
    if pd.api.types.is_numeric_dtype(type(default_fill_value)) or default_fill_value is np.nan:
        merged_df[target_col_name] = convert_to_numeric(merged_df[target_col_name], default_value=default_fill_value)
    else:
        merged_df[target_col_name] = merged_df[target_col_name].fillna(default_fill_value)

    # Restore original index
    index_col_name_after_reset = left_df_copy_reset.columns[0] # Name of the column created by reset_index
    merged_df.set_index(index_col_name_after_reset, inplace=True)
    merged_df.index.name = original_left_index_name # Restore original index name

    return merged_df


def enrich_zone_geodata_with_health_aggregates( # Renamed for clarity
    zone_df: Optional[pd.DataFrame], # Base zone data (output of load_zone_data)
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame: # Returns a simple DataFrame, not GeoDataFrame
    """
    Enriches zone DataFrame with aggregated health and IoT metrics.
    Assumes zone_df contains 'zone_id' and 'population'.
    Health_df and iot_df should also contain 'zone_id'.
    """
    logger.info(f"({source_context}) Starting zone data enrichment process.")

    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Base zone DataFrame is empty or invalid. Cannot perform enrichment.")
        # Return a DataFrame with expected columns if possible, or just an empty one
        # This schema should align with what downstream DHO components expect.
        expected_cols = ['zone_id', 'name', 'geometry_obj', 'population', 'avg_risk_score', 'total_patient_encounters',
                         'total_active_key_infections', 'prevalence_per_1000', 'zone_avg_co2',
                         'facility_coverage_score', 'population_density', 'chw_density_per_10k',
                         'avg_test_turnaround_critical', 'perc_critical_tests_tat_met', 'avg_daily_steps_zone']
        for cond_key in settings.KEY_CONDITIONS_FOR_ACTION:
            expected_cols.append(f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases")
        return pd.DataFrame(columns=list(set(expected_cols)))


    enriched_df = zone_df.copy()
    if 'zone_id' not in enriched_df.columns:
        logger.error(f"({source_context}) Critical: 'zone_id' missing from base zone_df. Enrichment aborted.")
        return enriched_df # Or an empty DF with schema

    # --- Initialize aggregate columns in enriched_df ---
    # This ensures columns exist even if health/IoT data is missing for some aggregations.
    default_aggregates: Dict[str, Any] = {
        'total_population_health_data': 0, # Count of unique patients in health_df for this zone
        'avg_risk_score': np.nan,
        'total_patient_encounters': 0,
        'total_active_key_infections': 0,
        'prevalence_per_1000': np.nan,
        'zone_avg_co2': np.nan,
        'avg_test_turnaround_critical': np.nan,
        'perc_critical_tests_tat_met': 0.0,
        'avg_daily_steps_zone': np.nan,
        # Placeholder for derived metrics not directly from health_df/iot_df aggregations here:
        'facility_coverage_score': np.nan, # Requires clinic count, population, possibly travel times
        'population_density': np.nan, # Requires area_sqkm and population
        'chw_density_per_10k': np.nan # Requires CHW count per zone and population
    }
    for cond_name in settings.KEY_CONDITIONS_FOR_ACTION: # Specific active case counts
        default_aggregates[f"active_{cond_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"] = 0
    
    for col_name, default_val in default_aggregates.items():
        if col_name not in enriched_df.columns:
            enriched_df[col_name] = default_val
        else: # If column exists from load_zone_data (e.g. population), ensure it's suitable type or fill
            if isinstance(default_val, (float, int)) or default_val is np.nan:
                enriched_df[col_name] = convert_to_numeric(enriched_df[col_name], default_value=default_val)
            else:
                enriched_df[col_name] = enriched_df[col_name].fillna(default_val)


    # --- Aggregate from Health Data ---
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_df_for_agg = health_df[health_df['zone_id'].notna()].copy()
        health_df_for_agg['zone_id'] = health_df_for_agg['zone_id'].astype(str).str.strip()

        # Total unique patients from health data per zone
        if 'patient_id' in health_df_for_agg.columns:
            patient_counts_zone = health_df_for_agg.groupby('zone_id')['patient_id'].nunique().reset_index(name='count')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, patient_counts_zone, 'total_population_health_data', default_fill_value=0)

        # Average AI Risk Score per zone
        if 'ai_risk_score' in health_df_for_agg.columns:
            avg_risk_zone = health_df_for_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_risk_zone, 'avg_risk_score', default_fill_value=np.nan)

        # Total patient encounters per zone
        if 'encounter_id' in health_df_for_agg.columns:
            encounter_counts_zone = health_df_for_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(name='count')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, encounter_counts_zone, 'total_patient_encounters', default_fill_value=0)

        # Active cases for each key condition & total key infections
        total_key_infections_per_zone_accumulator = pd.Series(0, index=enriched_df.index)
        if 'condition' in health_df_for_agg.columns and 'patient_id' in health_df_for_agg.columns:
            for condition_name_key in settings.KEY_CONDITIONS_FOR_ACTION:
                col_name_dynamic = f"active_{condition_name_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
                condition_mask_agg = health_df_for_agg['condition'].astype(str).str.contains(condition_name_key, case=False, na=False)
                if condition_mask_agg.any():
                    active_cases_zone = health_df_for_agg[condition_mask_agg].groupby('zone_id')['patient_id'].nunique().reset_index(name='count')
                    enriched_df = _robust_merge_agg_for_enrichment(enriched_df, active_cases_zone, col_name_dynamic, default_fill_value=0)
                    # Ensure the column is numeric after merge for summation
                    enriched_df[col_name_dynamic] = convert_to_numeric(enriched_df[col_name_dynamic], default_value=0)
                    total_key_infections_per_zone_accumulator = total_key_infections_per_zone_accumulator.add(enriched_df[col_name_dynamic], fill_value=0)
                # If no cases for this condition, the col_name_dynamic in enriched_df remains its default (0)
            enriched_df['total_active_key_infections'] = total_key_infections_per_zone_accumulator
        
        # Average Test Turnaround Time for Critical Tests
        if 'test_type' in health_df_for_agg.columns and 'test_turnaround_days' in health_df_for_agg.columns:
            df_critical_tests_agg = health_df_for_agg[health_df_for_agg['test_type'].isin(settings.CRITICAL_TESTS)].copy()
            df_critical_tests_agg['test_turnaround_days'] = convert_to_numeric(df_critical_tests_agg['test_turnaround_days'])
            avg_tat_critical_zone = df_critical_tests_agg.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_tat_critical_zone, 'avg_test_turnaround_critical', default_fill_value=np.nan)
            
            # Percentage of Critical Tests meeting TAT
            # This requires knowing the target TAT for each critical test, which is in settings.KEY_TEST_TYPES_FOR_ANALYSIS
            met_tat_counts = []
            for zone_id_val, group_df in df_critical_tests_agg.groupby('zone_id'):
                total_conclusive_critical_in_zone = 0
                met_tat_in_zone_count = 0
                for _, row_crit_test in group_df.iterrows():
                    test_config = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_crit_test['test_type'])
                    if test_config and pd.notna(row_crit_test['test_turnaround_days']): # Only consider if TAT is known
                        total_conclusive_critical_in_zone += 1
                        target_tat_for_this_test = test_config.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS)
                        if row_crit_test['test_turnaround_days'] <= target_tat_for_this_test:
                            met_tat_in_zone_count += 1
                perc_met = (met_tat_in_zone_count / total_conclusive_critical_in_zone * 100) if total_conclusive_critical_in_zone > 0 else 0.0
                met_tat_counts.append({'zone_id': zone_id_val, 'perc_met_val': perc_met})
            
            if met_tat_counts:
                 enriched_df = _robust_merge_agg_for_enrichment(enriched_df, pd.DataFrame(met_tat_counts), 'perc_critical_tests_tat_met', default_fill_value=0.0)


        # Average Daily Steps per zone
        if 'avg_daily_steps' in health_df_for_agg.columns:
            avg_steps_zone = health_df_for_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_steps_zone, 'avg_daily_steps_zone', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) Health DataFrame is empty or missing 'zone_id'. Skipping health data aggregations for enrichment.")

    # --- Aggregate from IoT Data ---
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty and 'zone_id' in iot_df.columns:
        iot_df_for_agg = iot_df[iot_df['zone_id'].notna()].copy()
        iot_df_for_agg['zone_id'] = iot_df_for_agg['zone_id'].astype(str).str.strip()

        # Average Clinic CO2 per zone
        if 'avg_co2_ppm' in iot_df_for_agg.columns:
            avg_co2_zone = iot_df_for_agg.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='mean_val')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_co2_zone, 'zone_avg_co2', default_fill_value=np.nan)
    else:
        logger.info(f"({source_context}) IoT DataFrame is empty or missing 'zone_id'. Skipping IoT data aggregations for enrichment.")

    # --- Calculate Derived Metrics (Post-Aggregation) ---
    # Prevalence per 1000
    if 'population' in enriched_df.columns and enriched_df['population'].sum() > 0 : # Ensure population column exists
        enriched_df['population_numeric'] = convert_to_numeric(enriched_df['population'], default_value=0.0)
        # Only calculate if population > 0 to avoid division by zero
        enriched_df['prevalence_per_1000'] = np.where(
            enriched_df['population_numeric'] > 0,
            (enriched_df['total_active_key_infections'] / enriched_df['population_numeric']) * 1000,
            0.0 # Or np.nan if preferred for zones with 0 population
        )
        enriched_df.drop(columns=['population_numeric'], inplace=True, errors='ignore')
    else:
        enriched_df['prevalence_per_1000'] = np.nan


    # Population Density
    if 'population' in enriched_df.columns and 'area_sqkm' in enriched_df.columns:
        enriched_df['population_numeric'] = convert_to_numeric(enriched_df['population'], default_value=0.0)
        enriched_df['area_sqkm_numeric'] = convert_to_numeric(enriched_df['area_sqkm'], default_value=0.0)
        enriched_df['population_density'] = np.where(
            enriched_df['area_sqkm_numeric'] > 0,
            enriched_df['population_numeric'] / enriched_df['area_sqkm_numeric'],
            0.0 # Or np.nan
        )
        enriched_df.drop(columns=['population_numeric', 'area_sqkm_numeric'], inplace=True, errors='ignore')
    else:
        enriched_df['population_density'] = np.nan


    # Facility Coverage Score (Example: simple clinics per 10k population)
    # This is a placeholder. Real calculation might be more complex (travel times, service types).
    if 'num_clinics' in enriched_df.columns and 'population' in enriched_df.columns:
        enriched_df['num_clinics_numeric'] = convert_to_numeric(enriched_df['num_clinics'], default_value=0)
        enriched_df['population_numeric'] = convert_to_numeric(enriched_df['population'], default_value=0)
        
        # Clinics per 10,000 population as a score (0-100, capped)
        # Assuming 1 clinic per 10k is "good" (100%), 0.5 per 10k is 50%, etc. Max score is 100.
        clinics_per_10k_pop = np.where(
            enriched_df['population_numeric'] > 0,
            (enriched_df['num_clinics_numeric'] / (enriched_df['population_numeric'] / 10000)),
            0.0
        )
        # Scale this to a 0-100 score. If 1 clinic per 10k is target (100), then score = clinics_per_10k * 100. Capped at 100.
        # This is an example scaling. The actual definition of "facility_coverage_score" might differ.
        enriched_df['facility_coverage_score'] = np.clip(clinics_per_10k_pop * 100, 0, 100) # Example: 1 clinic/10k = score 100
        
        enriched_df.drop(columns=['num_clinics_numeric', 'population_numeric'], inplace=True, errors='ignore')
    else:
        enriched_df['facility_coverage_score'] = np.nan

    # CHW Density per 10k population (Placeholder: Requires CHW count per zone data)
    # If CHW counts per zone were available, e.g., in `zone_df` or another source:
    # if 'chw_count_zone' in enriched_df.columns and 'population' in enriched_df.columns:
    #     enriched_df['population_numeric'] = convert_to_numeric(enriched_df['population'], default_value=0.0)
    #     enriched_df['chw_count_numeric'] = convert_to_numeric(enriched_df['chw_count_zone'], default_value=0)
    #     enriched_df['chw_density_per_10k'] = np.where(
    #         enriched_df['population_numeric'] > 0,
    #         (enriched_df['chw_count_numeric'] / (enriched_df['population_numeric'] / 10000)),
    #         0.0
    #     )
    # else:
    #     enriched_df['chw_density_per_10k'] = np.nan # Set to NaN if data is missing
    # For now, ensure the column exists as per expected schema.
    if 'chw_density_per_10k' not in enriched_df.columns:
        enriched_df['chw_density_per_10k'] = np.nan


    logger.info(f"({source_context}) Zone data enrichment complete. Final shape: {enriched_df.shape}")
    # Ensure no Pandas dtypes like Int64 that might cause issues with some downstream libraries if not handled
    for col in enriched_df.select_dtypes(include=['Int64','Int32','Int16','Int8']).columns:
        enriched_df[col] = enriched_df[col].astype(float) # Convert to float to handle potential NaNs smoothly in some tools

    return enriched_df
