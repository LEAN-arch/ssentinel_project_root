import pandas as pd
import numpy as np
import logging
import re
import os
from typing import Optional, Dict, Any, List, Union

# --- Module Imports & Setup ---
try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    # FIXED: Use the correct `__name__` magic variable.
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in enrichment.py: {e}. Ensure paths are correct.")
    raise

# FIXED: Use the correct `__name__` magic variable.
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
    value_col_in_right_df: Optional[str] = None
) -> pd.DataFrame:
    """
    Helper to robustly merge an aggregated (right) DataFrame into a main (left) DataFrame.
    """
    if not isinstance(left_df, pd.DataFrame):
        return left_df

    enriched_df = left_df.copy()
    is_numeric = isinstance(default_fill_value, (int, float, np.number)) or pd.isna(default_fill_value)

    if target_col_name not in enriched_df.columns:
        enriched_df[target_col_name] = default_fill_value
    
    if is_numeric:
        enriched_df[target_col_name] = convert_to_numeric(enriched_df[target_col_name], default_value=default_fill_value)
    else:
        enriched_df[target_col_name] = enriched_df[target_col_name].fillna(default_fill_value)

    if not isinstance(right_df_optional, pd.DataFrame) or right_df_optional.empty or on_col not in right_df_optional.columns:
        return enriched_df

    value_col = value_col_in_right_df or next((c for c in right_df_optional.columns if c != on_col), None)
    if not value_col or value_col not in right_df_optional.columns:
        return enriched_df
    
    right_df_prepared = right_df_optional[[on_col, value_col]].copy()
    
    # Ensure join keys are consistent string types
    enriched_df[on_col] = enriched_df[on_col].astype(str).str.strip()
    right_df_prepared[on_col] = right_df_prepared[on_col].astype(str).str.strip()
    
    # Set index for efficient update
    original_index = enriched_df.index
    enriched_df.set_index(on_col, inplace=True)
    right_df_prepared.set_index(on_col, inplace=True)
    
    # Update the target column with new values from the right DataFrame
    enriched_df[target_col_name].update(right_df_prepared[value_col])
    
    enriched_df.reset_index(inplace=True)
    if len(enriched_df) == len(original_index): # Restore original index if dimensions match
        enriched_df.index = original_index
        
    return enriched_df


def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame],
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame:
    """Enriches zone geographical data with aggregated health and environmental metrics."""
    logger.info(f"({source_context}) Starting zone data enrichment process.")
    
    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Base zone DataFrame is empty. Cannot perform enrichment.")
        return pd.DataFrame()

    enriched_df = zone_df.copy()
    if 'zone_id' not in enriched_df.columns:
        logger.error(f"({source_context}) 'zone_id' missing from base zone_df. Enrichment aborted.")
        return enriched_df

    key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    
    # --- Initialize Aggregate Columns ---
    default_aggregates = {
        'avg_risk_score': np.nan, 'total_patient_encounters': 0, 'total_active_key_infections': 0,
        'prevalence_per_1000': 0.0, 'zone_avg_co2': np.nan, 'avg_test_turnaround_critical': np.nan,
        'perc_critical_tests_tat_met': 0.0, 'avg_daily_steps_zone': np.nan,
        'facility_coverage_score': np.nan, 'population_density': np.nan, 'chw_density_per_10k': np.nan
    }
    
    # Add dynamic columns for each key condition
    for cond in key_conditions:
        col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond.lower().strip())}_cases"
        default_aggregates[col_name] = 0

    for col, default in default_aggregates.items():
        if col not in enriched_df.columns:
            enriched_df[col] = default
    
    # --- Health Data Aggregation ---
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_agg = health_df[health_df['zone_id'].notna()].copy()
        health_agg['zone_id'] = health_agg['zone_id'].astype(str).str.strip()
        
        # Aggregate and merge key metrics
        if 'ai_risk_score' in health_agg.columns:
            avg_risk = health_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='mean_risk')
            enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_risk, 'avg_risk_score', value_col_in_right_df='mean_risk', default_fill_value=np.nan)
        
        enc_id_col = 'encounter_id' if 'encounter_id' in health_agg.columns else 'patient_id'
        enc_counts = health_agg.groupby('zone_id')[enc_id_col].nunique().reset_index(name='enc_count')
        enriched_df = _robust_merge_agg_for_enrichment(enriched_df, enc_counts, 'total_patient_encounters', value_col_in_right_df='enc_count', default_fill_value=0)

        # FIXED: Correct and robust aggregation for key infections.
        if 'condition' in health_agg.columns and 'patient_id' in health_agg.columns and key_conditions:
            active_case_cols = []
            for cond in key_conditions:
                safe_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond.lower().strip())}_cases"
                active_case_cols.append(safe_col_name)
                cond_mask = health_agg['condition'].astype(str).str.contains(re.escape(cond), case=False, na=False)
                if cond_mask.any():
                    active_cases_df = health_agg[cond_mask].groupby('zone_id')['patient_id'].nunique().reset_index(name='case_count')
                    enriched_df = _robust_merge_agg_for_enrichment(enriched_df, active_cases_df, safe_col_name, value_col_in_right_df='case_count', default_fill_value=0)
            
            # After all individual merges, sum them up to get the total
            enriched_df['total_active_key_infections'] = enriched_df[active_case_cols].sum(axis=1)

    # --- IoT/Environmental Data Aggregation (Example) ---
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty and 'zone_id' in iot_df.columns:
        iot_agg = iot_df[iot_df['zone_id'].notna()].copy()
        iot_agg['zone_id'] = iot_agg['zone_id'].astype(str).str.strip()
        if 'co2_ppm' in iot_agg.columns:
             avg_co2 = iot_agg.groupby('zone_id')['co2_ppm'].mean().reset_index(name='mean_co2')
             enriched_df = _robust_merge_agg_for_enrichment(enriched_df, avg_co2, 'zone_avg_co2', value_col_in_right_df='mean_co2', default_fill_value=np.nan)

    # --- Derived Metric Calculations ---
    # FIXED: This logic was incorrectly outside the function. Moved it inside.
    if 'population' in enriched_df.columns and 'total_active_key_infections' in enriched_df.columns:
        pop = convert_to_numeric(enriched_df['population'], default_value=0.0)
        infections = convert_to_numeric(enriched_df['total_active_key_infections'], default_value=0.0)
        enriched_df['prevalence_per_1000'] = np.where(pop > 0, (infections / pop) * 1000, 0.0).fillna(0.0)

    logger.info(f"({source_context}) Zone data enrichment complete. Final shape: {enriched_df.shape}")
    return enriched_df
