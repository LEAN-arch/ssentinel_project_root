# sentinel_project_root/data_processing/enrichment.py
# SME PLATINUM STANDARD - VECTORIZED DATA ENRICHMENT

import logging
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import settings
from .helpers import convert_to_numeric

logger = logging.getLogger(__name__)

# --- Primary Enrichment Functions ---

def enrich_health_records_with_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches health records with pre-calculated boolean and value columns
    needed for efficient KPI calculations and analytics. This is a core
    step in creating an 'analytics-ready' DataFrame.
    """
    if df.empty:
        return df
    
    enriched_df = df.copy()
    
    # 1. Sample Rejection Flag
    if 'sample_status' in enriched_df.columns:
        enriched_df['is_rejected'] = (enriched_df['sample_status'].astype(str).str.lower() == 'rejected by lab').astype(int)
    
    # 2. Critical & Pending Test Flag
    if all(c in enriched_df.columns for c in ['test_type', 'test_result']):
        enriched_df['is_critical_and_pending'] = (
            enriched_df['test_type'].isin(settings.CRITICAL_TESTS) &
            (enriched_df['test_result'].astype(str).str.lower() == 'pending')
        ).astype(int)

    # 3. Supply Chain Risk Flags
    if all(c in enriched_df.columns for c in ['item_stock_agg_zone', 'consumption_rate_per_day', 'item']):
        key_items_pattern = '|'.join(map(re.escape, settings.KEY_SUPPLY_ITEMS))
        is_key_item = enriched_df['item'].str.contains(key_items_pattern, case=False, na=False)
        
        # Avoid division by zero; treat zero consumption as infinite supply
        rate = enriched_df['consumption_rate_per_day'].clip(lower=0.001)
        days_of_supply = enriched_df['item_stock_agg_zone'] / rate
        
        is_at_risk = days_of_supply < settings.ANALYTICS.supply_low_threshold_days
        enriched_df['is_supply_at_risk'] = (is_key_item & is_at_risk).astype(int)
        
        is_critical = days_of_supply < settings.ANALYTICS.supply_critical_threshold_days
        enriched_df['is_supply_critical'] = (is_key_item & is_critical).astype(int)
        
    # 4. Test Positivity Flag
    if 'test_result' in enriched_df.columns:
        enriched_df['is_positive'] = (enriched_df['test_result'].astype(str).str.lower() == 'positive').astype(int)
        
    # 5. Symptom Cluster Flags
    for cluster_name, symptoms in settings.SYMPTOM_CLUSTERS.items():
        pattern = '|'.join(map(re.escape, symptoms))
        if 'patient_reported_symptoms' in enriched_df.columns:
            enriched_df[f'has_symptom_cluster_{cluster_name}'] = enriched_df['patient_reported_symptoms'].str.contains(
                pattern, case=False, na=False
            ).astype(int)

    logger.debug(f"Enriched {len(df)} health records with KPI flags.")
    return enriched_df


def enrich_zone_data_with_aggregates(
    zone_df: Optional[pd.DataFrame],
    health_df: Optional[pd.DataFrame],
    iot_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Public factory function to enrich zone geographical data with aggregated
    health and environmental metrics for choropleth maps and district-level views.
    """
    if not isinstance(zone_df, pd.DataFrame) or zone_df.empty:
        return pd.DataFrame(columns=['zone_id', 'geometry']) # Return a valid empty frame
    try:
        return ZoneDataEnricher(zone_df, health_df, iot_df).enrich()
    except ValueError as ve:
        logger.error(f"Failed to initialize ZoneDataEnricher: {ve}")
        return zone_df
    except Exception as e:
        logger.error(f"Unexpected error during zone data enrichment: {e}", exc_info=True)
        return zone_df


class ZoneDataEnricher:
    """
    Encapsulates logic for enriching zone data using a fluent, vectorized pipeline.
    """
    def __init__(self, zone_df: pd.DataFrame, health_df: Optional[pd.DataFrame], iot_df: Optional[pd.DataFrame]):
        if 'zone_id' not in zone_df.columns:
            raise ValueError("Base zone_df must have a 'zone_id' column.")
        
        self.zone_df = zone_df.copy()
        self.health_df = health_df.copy() if isinstance(health_df, pd.DataFrame) else pd.DataFrame()
        self.iot_df = iot_df.copy() if isinstance(iot_df, pd.DataFrame) else pd.DataFrame()

    def _aggregate_health_data(self) -> pd.DataFrame:
        if self.health_df.empty or 'zone_id' not in self.health_df.columns:
            return pd.DataFrame()
            
        health_agg = self.health_df.dropna(subset=['zone_id']).copy()
        
        aggregations = {
            'avg_risk_score': pd.NamedAgg('ai_risk_score', 'mean'),
            'total_encounters': pd.NamedAgg('encounter_id', 'nunique'),
        }
        summary = health_agg.groupby('zone_id').agg(**aggregations)

        if 'diagnosis' in health_agg.columns:
            # Vectorized case counting for key diagnoses
            counts = health_agg[health_agg['diagnosis'].isin(settings.KEY_DIAGNOSES)]\
                .groupby(['zone_id', 'diagnosis'])['patient_id']\
                .nunique().unstack(fill_value=0)
            
            # Sanitize column names for easy access
            counts.columns = [f"active_cases_{re.sub(r'[^a-z0-9_]+', '', c.lower())}" for c in counts.columns]
            summary = summary.join(counts, how='left')

        case_cols = [c for c in summary.columns if c.startswith('active_cases_')]
        if case_cols:
            summary['total_active_key_cases'] = summary[case_cols].sum(axis=1)
        
        return summary.reset_index()

    def _aggregate_iot_data(self) -> pd.DataFrame:
        if self.iot_df.empty or 'zone_id' not in self.iot_df.columns:
            return pd.DataFrame()
        return self.iot_df.groupby('zone_id').agg(
            zone_avg_co2_ppm=('avg_co2_ppm', 'mean'),
            zone_avg_pm25=('avg_pm25', 'mean')
        ).reset_index()

    def enrich(self) -> pd.DataFrame:
        """Orchestrates the enrichment process via a data pipeline."""
        logger.info("Starting zone data enrichment.")
        health_summary = self._aggregate_health_data()
        iot_summary = self._aggregate_iot_data()
        
        final_df = (self.zone_df
                    .pipe(self._merge_aggregates, health_summary, iot_summary)
                    .pipe(self._calculate_derived_metrics)
                    .pipe(self._finalize_schema))
        
        logger.info(f"Zone data enrichment complete. Final shape: {final_df.shape}")
        return final_df

    # --- Pipeline Stages ---
    def _merge_aggregates(self, df: pd.DataFrame, health_summary: pd.DataFrame, iot_summary: pd.DataFrame) -> pd.DataFrame:
        df['zone_id'] = df['zone_id'].astype(str)
        if not health_summary.empty:
            df = pd.merge(df, health_summary, on='zone_id', how='left')
        if not iot_summary.empty:
            df = pd.merge(df, iot_summary, on='zone_id', how='left')
        return df

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'population' in df.columns:
            df['population'] = convert_to_numeric(df['population'], default_value=0.0)
            if 'total_active_key_cases' in df.columns:
                cases = convert_to_numeric(df['total_active_key_cases'], default_value=0.0)
                df['prevalence_per_1000_pop'] = np.where(df['population'] > 0, (cases / df['population']) * 1000, 0.0)
        return df

    def _finalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        fill_values = {
            'avg_risk_score': np.nan,
            'total_encounters': 0,
            'total_active_key_cases': 0,
            'prevalence_per_1000_pop': 0.0,
            'zone_avg_co2_ppm': np.nan,
            'zone_avg_pm25': np.nan
        }
        # Dynamically create fill values for all possible active case columns
        for diag in settings.KEY_DIAGNOSES:
            col_name = f"active_cases_{re.sub(r'[^a-z0-9_]+', '', diag.lower())}"
            fill_values[col_name] = 0
            
        # Fill NaNs and ensure integer columns are integers
        df.fillna(value=fill_values, inplace=True)
        int_cols = [c for c in df.columns if 'cases' in c or 'encounters' in c or 'count' in c]
        for col in int_cols:
            df[col] = df[col].astype(int)
            
        return df
