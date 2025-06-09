# sentinel_project_root/data_processing/enrichment.py
# SME PLATINUM STANDARD (V2 - SCHEMA FIX & ARCHITECTURAL REFACTOR)
# This definitive version corrects the critical 'condition' vs 'diagnosis' schema
# mismatch and refactors the orchestration logic into a fluent, readable pipeline.

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Dict, Any, List

# --- Module Imports & Setup ---
try:
    # <<< SME REVISION V2 >>> Use the Pydantic settings object.
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in enrichment.py: {e}.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# <<< SME INTEGRATION >>> This enrichment function was added for KPI calculations.
def enrich_health_records_for_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches health records with boolean flags needed for KPI calculations."""
    if df.empty: return df
    df_enriched = df.copy()
    if 'sample_status' in df_enriched.columns:
        df_enriched['is_rejected'] = (df_enriched['sample_status'].astype(str).str.lower() == 'rejected by lab').astype(int)
    if 'test_type' in df_enriched.columns and 'test_result' in df_enriched.columns and settings.KEY_TEST_TYPES_FOR_ANALYSIS:
        critical_tests = [k for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() if isinstance(v, dict) and v.get("critical")]
        df_enriched['is_critical_and_pending'] = (df_enriched['test_type'].isin(critical_tests) & (df_enriched['test_result'].astype(str).str.lower() == 'pending')).astype(int)
    if 'item_stock_agg_zone' in df_enriched.columns and 'consumption_rate_per_day' in df_enriched.columns:
        consumption_rate = df_enriched['consumption_rate_per_day'].clip(lower=0.001)
        days_of_supply = df_enriched['item_stock_agg_zone'] / consumption_rate
        df_enriched['is_stockout'] = (days_of_supply < settings.CRITICAL_SUPPLY_DAYS_REMAINING).astype(int)
    return df_enriched


class ZoneDataEnricher:
    """Encapsulates logic for enriching zone data using a fluent, vectorized pipeline."""

    def __init__(self, zone_df: pd.DataFrame, health_df: Optional[pd.DataFrame], iot_df: Optional[pd.DataFrame]):
        if not isinstance(zone_df, pd.DataFrame) or zone_df.empty or 'zone_id' not in zone_df.columns:
            raise ValueError("Base zone_df must be a non-empty DataFrame with a 'zone_id' column.")
        
        self.zone_df = zone_df.copy()
        self.health_df = health_df.copy() if isinstance(health_df, pd.DataFrame) else pd.DataFrame()
        self.iot_df = iot_df.copy() if isinstance(iot_df, pd.DataFrame) else pd.DataFrame()
        
        # <<< SME REVISION V2 >>> Corrected variable name to align with settings schema.
        self.key_diagnoses = getattr(settings, 'KEY_DIAGNOSES_FOR_ACTION', [])

    def _aggregate_health_data(self) -> pd.DataFrame:
        if self.health_df.empty or 'zone_id' not in self.health_df.columns: return pd.DataFrame()
        health_agg = self.health_df.dropna(subset=['zone_id']).copy()
        
        aggregations = {
            'avg_risk_score': pd.NamedAgg('ai_risk_score', 'mean'),
            'total_patient_encounters': pd.NamedAgg('encounter_id', 'nunique'),
        }
        health_summary = health_agg.groupby('zone_id').agg(**aggregations)

        # <<< SME REVISION V2 >>> Corrected column name to 'diagnosis'.
        if 'diagnosis' in health_agg.columns and self.key_diagnoses:
            counts = health_agg[health_agg['diagnosis'].isin(self.key_diagnoses)].groupby(['zone_id', 'diagnosis'])['patient_id'].nunique().unstack(fill_value=0)
            counts.columns = [f"active_{re.sub(r'[^a-z0-9_]+', '', c.lower())}_cases" for c in counts.columns]
            health_summary = health_summary.join(counts, how='left')

        active_case_cols = [c for c in health_summary.columns if c.startswith('active_')]
        if active_case_cols:
            health_summary['total_active_key_infections'] = health_summary[active_case_cols].sum(axis=1)

        return health_summary.reset_index()

    def _aggregate_iot_data(self) -> pd.DataFrame:
        if self.iot_df.empty or 'zone_id' not in self.iot_df.columns: return pd.DataFrame()
        return self.iot_df.groupby('zone_id').agg(zone_avg_co2=('avg_co2_ppm', 'mean')).reset_index()

    # --- Pipeline Stages for `enrich` method ---
    # <<< SME REVISION V2 >>> Broke down logic into composable pipeline stages.
    def _merge_aggregates(self, df: pd.DataFrame, health_summary: pd.DataFrame, iot_summary: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 1: Merge all aggregated summaries into the base zone DataFrame."""
        df['zone_id'] = df['zone_id'].astype(str).str.strip()
        if not health_summary.empty:
            df = pd.merge(df, health_summary, on='zone_id', how='left')
        if not iot_summary.empty:
            df = pd.merge(df, iot_summary, on='zone_id', how='left')
        return df

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 2: Calculate metrics that depend on already aggregated or base columns."""
        if 'population' in df.columns:
            df['population'] = convert_to_numeric(df['population'], default_value=0.0)
            if 'total_active_key_infections' in df.columns:
                infections = convert_to_numeric(df['total_active_key_infections'], default_value=0.0)
                df['prevalence_per_1000'] = np.where(df['population'] > 0, (infections / df['population']) * 1000, 0.0)
        return df
    
    def _fill_na_and_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 3: Fill NA values with sensible defaults for a clean final output."""
        default_fills = {'avg_risk_score': np.nan, 'total_patient_encounters': 0, 'total_active_key_infections': 0, 'prevalence_per_1000': 0.0, 'zone_avg_co2': np.nan}
        for diag in self.key_diagnoses:
            default_fills[f"active_{re.sub(r'[^a-z0-9_]+', '', diag.lower())}_cases"] = 0
        # <<< SME REVISION V2 >>> Replaced inplace=True with assignment.
        return df.fillna(value=default_fills)

    def enrich(self) -> pd.DataFrame:
        """Orchestrates the entire enrichment process using a fluent pipeline."""
        logger.info("Starting zone data enrichment process.")
        health_summary = self._aggregate_health_data()
        iot_summary = self._aggregate_iot_data()
        
        # <<< SME REVISION V2 >>> Use a fluent .pipe() based pipeline.
        final_df = (self.zone_df.copy()
                    .pipe(self._merge_aggregates, health_summary=health_summary, iot_summary=iot_summary)
                    .pipe(self._calculate_derived_metrics)
                    .pipe(self._fill_na_and_finalize)
        )
        
        logger.info(f"Zone data enrichment complete. Final shape: {final_df.shape}")
        return final_df

def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame], health_df: Optional[pd.DataFrame], iot_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Public factory function to enrich zone geographical data."""
    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        return pd.DataFrame()
    try:
        return ZoneDataEnricher(zone_df, health_df, iot_df).enrich()
    except ValueError as ve:
        logger.error(f"Failed to initialize ZoneDataEnricher: {ve}")
        return zone_df 
    except Exception as e:
        logger.error(f"An unexpected error occurred during zone data enrichment: {e}", exc_info=True)
        return zone_df
