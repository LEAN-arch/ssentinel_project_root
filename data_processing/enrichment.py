# sentinel_project_root/data_processing/enrichment.py
# Functions for enriching data, e.g., merging health aggregates into zone data.

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Dict, Any

# --- Core Imports ---
try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in enrichment.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)


def _robust_merge(
    left_df: pd.DataFrame,
    right_df: Optional[pd.DataFrame],
    target_col: str,
    on_col: str = 'zone_id',
    value_col: Optional[str] = None,
    default_fill: Any = 0
) -> pd.DataFrame:
    """Helper to robustly merge an aggregated DataFrame into a main one."""
    df = left_df.copy()
    if not isinstance(right_df, pd.DataFrame) or right_df.empty or on_col not in right_df.columns:
        if target_col not in df.columns:
            df[target_col] = default_fill
        return df

    val_col = value_col if value_col and value_col in right_df.columns else right_df.columns[1]
    
    # Ensure join keys are consistent string types
    df[on_col] = df[on_col].astype(str)
    right_df[on_col] = right_df[on_col].astype(str)
    
    # Merge using a temporary suffix to avoid column name collisions
    df = pd.merge(df, right_df[[on_col, val_col]], on=on_col, how='left', suffixes=('', '_new'))
    
    # Coalesce new values into the target column
    new_col_name = f"{val_col}_new"
    if new_col_name in df.columns:
        # If target column doesn't exist, create it from the new one
        if target_col not in df.columns:
            df[target_col] = df[new_col_name]
        else:
            # Otherwise, update existing NaNs with new values
            df[target_col] = df[target_col].fillna(df[new_col_name])
        df.drop(columns=[new_col_name], inplace=True)
    
    # Ensure target column exists and is filled with the default
    if target_col not in df.columns:
        df[target_col] = default_fill
    else:
        df[target_col] = df[target_col].fillna(default_fill)
        
    return df


def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame],
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "ZoneDataEnricher"
) -> pd.DataFrame:
    """
    Enriches zone-level geospatial data with aggregated health, risk, and
    environmental metrics for comprehensive situational awareness.
    """
    logger.info(f"({source_context}) Starting zone data enrichment process.")

    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning(f"({source_context}) Base zone DataFrame is empty or invalid. Cannot perform enrichment.")
        return pd.DataFrame()
    if 'zone_id' not in zone_df.columns:
        logger.error(f"({source_context}) Critical: 'zone_id' missing from base zone_df. Enrichment aborted.")
        return zone_df

    enriched_df = zone_df.copy()

    # --- Calculations from Health Data ---
    if isinstance(health_df, pd.DataFrame) and not health_df.empty and 'zone_id' in health_df.columns:
        health_df['zone_id'] = health_df['zone_id'].astype(str)
        
        # Aggregate AI Risk Score
        if 'ai_risk_score' in health_df.columns:
            avg_risk = health_df.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            enriched_df = _robust_merge(enriched_df, avg_risk, 'avg_risk_score', value_col='ai_risk_score', default_fill=np.nan)

        # Aggregate Patient Encounters
        if 'patient_id' in health_df.columns:
            encounters = health_df.groupby('zone_id')['patient_id'].nunique().reset_index()
            enriched_df = _robust_merge(enriched_df, encounters, 'total_patient_encounters', value_col='patient_id', default_fill=0)
        
        # Aggregate Active Cases for Key Conditions
        if 'condition' in health_df.columns and hasattr(settings, 'KEY_CONDITIONS_FOR_ACTION'):
            total_active_cases = pd.Series(0, index=enriched_df['zone_id'].unique(), dtype=int)
            for condition in settings.KEY_CONDITIONS_FOR_ACTION:
                col_name = f"active_{re.sub(r'[^a-z0-9]+', '_', condition.lower())}_cases"
                condition_mask = health_df['condition'].str.contains(condition, case=False, na=False)
                active_cases = health_df[condition_mask].groupby('zone_id')['patient_id'].nunique()
                enriched_df[col_name] = enriched_df['zone_id'].map(active_cases).fillna(0).astype(int)
                total_active_cases = total_active_cases.add(enriched_df.set_index('zone_id')[col_name], fill_value=0)
            
            enriched_df['total_active_key_infections'] = enriched_df['zone_id'].map(total_active_cases).fillna(0).astype(int)

    # --- Calculations from IoT Data ---
    if isinstance(iot_df, pd.DataFrame) and not iot_df.empty and 'zone_id' in iot_df.columns:
        iot_df['zone_id'] = iot_df['zone_id'].astype(str)
        
        # Aggregate CO2 levels
        if 'avg_co2_ppm' in iot_df.columns:
            avg_co2 = iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index()
            enriched_df = _robust_merge(enriched_df, avg_co2, 'zone_avg_co2', value_col='avg_co2_ppm', default_fill=np.nan)
    
    # --- Derived Metric Calculations (Post-Merge) ---
    if 'population' in enriched_df.columns and 'total_active_key_infections' in enriched_df.columns:
        pop = convert_to_numeric(enriched_df['population'], default_value=0)
        cases = convert_to_numeric(enriched_df['total_active_key_infections'], default_value=0)
        enriched_df['prevalence_per_1000'] = np.where(pop > 0, (cases / pop) * 1000, 0).round(2)
        enriched_df['prevalence_per_1000'] = enriched_df['prevalence_per_1000'].fillna(0)

    if 'population' in enriched_df.columns and 'area_sqkm' in enriched_df.columns:
        pop = convert_to_numeric(enriched_df['population'], default_value=0)
        area = convert_to_numeric(enriched_df['area_sqkm'], default_value=0)
        enriched_df['population_density'] = np.where(area > 0, pop / area, 0).round(1)
        enriched_df['population_density'] = enriched_df['population_density'].fillna(0)

    if 'num_chws' in enriched_df.columns and 'population' in enriched_df.columns:
        chws = convert_to_numeric(enriched_df['num_chws'], default_value=0)
        pop = convert_to_numeric(enriched_df['population'], default_value=0)
        enriched_df['chw_density_per_10k'] = np.where(pop > 0, (chws / pop) * 10000, 0).round(1)
        enriched_df['chw_density_per_10k'] = enriched_df['chw_density_per_10k'].fillna(0)
    
    logger.info(f"({source_context}) Zone data enrichment complete. Final shape: {enriched_df.shape}")
    return enriched_df
