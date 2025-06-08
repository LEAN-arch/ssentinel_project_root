# sentinel_project_root/data_processing/enrichment.py
# Functions for enriching geographical data with aggregated health and environmental metrics.

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Dict, Any

# --- Module Imports & Setup ---
try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in enrichment.py: {e}. Check paths/dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ZoneDataEnricher:
    """
    Encapsulates the logic for enriching zone geographical data with aggregated health
    and environmental metrics using efficient, vectorized operations.
    """
    def __init__(self, zone_df: pd.DataFrame, health_df: Optional[pd.DataFrame], iot_df: Optional[pd.DataFrame]):
        if not isinstance(zone_df, pd.DataFrame) or zone_df.empty or 'zone_id' not in zone_df.columns:
            raise ValueError("Base zone_df must be a non-empty DataFrame with a 'zone_id' column.")
        
        self.zone_df = zone_df.copy()
        self.health_df = health_df.copy() if isinstance(health_df, pd.DataFrame) and not health_df.empty else pd.DataFrame()
        self.iot_df = iot_df.copy() if isinstance(iot_df, pd.DataFrame) and not iot_df.empty else pd.DataFrame()
        
        self.key_conditions = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])

    def _aggregate_health_data(self) -> pd.DataFrame:
        """Aggregates all necessary metrics from the health data in a single pass."""
        if self.health_df.empty or 'zone_id' not in self.health_df.columns:
            return pd.DataFrame(columns=['zone_id'])

        health_agg = self.health_df[self.health_df['zone_id'].notna()].copy()
        health_agg['zone_id'] = health_agg['zone_id'].astype(str).str.strip()

        aggregations: Dict[str, Any] = {
            'avg_risk_score': pd.NamedAgg(column='ai_risk_score', aggfunc='mean'),
            'total_patient_encounters': pd.NamedAgg(column='encounter_id', aggfunc='nunique'),
            'avg_daily_steps_zone': pd.NamedAgg(column='avg_daily_steps', aggfunc='mean')
        }
        
        health_summary = health_agg.groupby('zone_id').agg(**aggregations)

        if 'condition' in health_agg.columns and self.key_conditions:
            condition_counts = health_agg[health_agg['condition'].isin(self.key_conditions)].groupby(['zone_id', 'condition'])['patient_id'].nunique().unstack(fill_value=0)
            condition_counts.columns = [f"active_{re.sub(r'[^a-z0-9_]+', '_', c.lower().strip())}_cases" for c in condition_counts.columns]
            health_summary = health_summary.join(condition_counts, how='left')

        active_case_cols = [col for col in health_summary.columns if col.startswith('active_')]
        if active_case_cols:
            health_summary['total_active_key_infections'] = health_summary[active_case_cols].sum(axis=1)

        return health_summary.reset_index()

    def _aggregate_iot_data(self) -> pd.DataFrame:
        """Aggregates all necessary metrics from the IoT data."""
        if self.iot_df.empty or 'zone_id' not in self.iot_df.columns or 'avg_co2_ppm' not in self.iot_df.columns:
            return pd.DataFrame(columns=['zone_id'])
        iot_agg = self.iot_df[['zone_id', 'avg_co2_ppm']].dropna().copy()
        iot_agg['zone_id'] = iot_agg['zone_id'].astype(str).strip()
        return iot_agg.groupby('zone_id').agg(zone_avg_co2=('avg_co2_ppm', 'mean')).reset_index()

    def _merge_aggregates(self, health_summary: pd.DataFrame, iot_summary: pd.DataFrame) -> pd.DataFrame:
        """Merges all aggregated summaries into the base zone DataFrame."""
        enriched_df = self.zone_df.copy()
        enriched_df['zone_id'] = enriched_df['zone_id'].astype(str).strip()
        if not health_summary.empty:
            enriched_df = pd.merge(enriched_df, health_summary, on='zone_id', how='left')
        if not iot_summary.empty:
            enriched_df = pd.merge(enriched_df, iot_summary, on='zone_id', how='left')
        return enriched_df

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates metrics that depend on already aggregated or base columns."""
        if 'population' in df.columns:
            df['population'] = convert_to_numeric(df['population'], default_value=0.0)
            if 'total_active_key_infections' in df.columns:
                infections = convert_to_numeric(df['total_active_key_infections'], default_value=0.0)
                df['prevalence_per_1000'] = np.where(df['population'] > 0, (infections / df['population']) * 1000, 0.0)
        return df
        
    def enrich(self) -> pd.DataFrame:
        """Orchestrates the entire enrichment process: aggregate, merge, and derive."""
        logger.info("Starting zone data enrichment process.")
        health_summary = self._aggregate_health_data()
        iot_summary = self._aggregate_iot_data()
        
        enriched_df = self._merge_aggregates(health_summary, iot_summary)
        final_df = self._calculate_derived_metrics(enriched_df)
        
        default_fills = {'avg_risk_score': np.nan, 'total_patient_encounters': 0, 'total_active_key_infections': 0, 'prevalence_per_1000': 0.0, 'zone_avg_co2': np.nan, 'avg_daily_steps_zone': np.nan}
        for cond in self.key_conditions:
            default_fills[f"active_{re.sub(r'[^a-z0-9_]+', '_', cond.lower().strip())}_cases"] = 0
        
        final_df.fillna(value=default_fills, inplace=True)
        logger.info(f"Zone data enrichment complete. Final shape: {final_df.shape}")
        return final_df

def enrich_zone_geodata_with_health_aggregates(
    zone_df: Optional[pd.DataFrame],
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Public factory function to enrich zone geographical data.
    """
    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        logger.warning("Base zone DataFrame is empty. Cannot perform enrichment.")
        return pd.DataFrame()
    try:
        enricher = ZoneDataEnricher(zone_df, health_df, iot_df)
        return enricher.enrich()
    except ValueError as ve:
        logger.error(f"Failed to initialize ZoneDataEnricher: {ve}")
        return zone_df 
    except Exception as e:
        logger.error(f"An unexpected error occurred during zone data enrichment: {e}", exc_info=True)
        return zone_df
