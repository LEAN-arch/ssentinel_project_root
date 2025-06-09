# sentinel_project_root/data_processing/aggregation.py
# SME PLATINUM STANDARD - DECISION-GRADE AGGREGATIONS

import logging
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from .helpers import convert_to_numeric, hash_dataframe

logger = logging.getLogger(__name__)

# --- Core KPI Calculation Functions ---

def calculate_clinic_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates a comprehensive set of decision-grade KPIs for a clinic context.

    This function expects a pre-enriched DataFrame from `enrich_health_records_with_kpis`.

    Returns:
        A dictionary of calculated KPIs.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    kpis: Dict[str, Any] = {}

    # 1. Testing Efficiency KPIs
    conclusive_tests = df[df['sample_status'].str.lower().isin(['completed', 'rejected by lab'])]
    if not conclusive_tests.empty:
        kpis['avg_test_tat_days'] = conclusive_tests['test_turnaround_days'].mean()
        
        # Calculate % of tests meeting target TAT
        # This requires a join-like operation if targets vary by test_type
        test_type_configs = settings.KEY_TEST_TYPES
        target_map = {name: config.target_tat_days for name, config in test_type_configs.items()}
        
        # Create a 'target_tat' column for vectorized comparison
        conclusive_tests['target_tat'] = conclusive_tests['test_type'].map(target_map)
        met_tat = (conclusive_tests['test_turnaround_days'] <= conclusive_tests['target_tat']).sum()
        kpis['perc_tests_within_tat'] = (met_tat / len(conclusive_tests.dropna(subset=['target_tat']))) * 100 if len(conclusive_tests.dropna(subset=['target_tat'])) > 0 else 0
    else:
        kpis['avg_test_tat_days'] = np.nan
        kpis['perc_tests_within_tat'] = 0.0

    # 2. Test Quality & Demand KPIs
    kpis['sample_rejection_rate_perc'] = df['is_rejected'].mean() * 100 if 'is_rejected' in df else 0.0
    kpis['pending_critical_tests_count'] = df['is_critical_and_pending'].sum() if 'is_critical_and_pending' in df else 0

    # 3. Supply Chain KPIs
    kpis['key_items_at_risk_count'] = df['is_supply_at_risk'].sum() if 'is_supply_at_risk' in df else 0
    
    # 4. Test Positivity Rates (as a dictionary for detailed breakdown)
    positivity_breakdown = {}
    test_df = df.dropna(subset=['test_type', 'is_positive'])
    if not test_df.empty:
        # Calculate positivity rate per test type
        positivity_groups = test_df.groupby('test_type')['is_positive'].agg(['mean', 'count'])
        for test_type, data in positivity_groups.iterrows():
            positivity_breakdown[test_type] = {
                'positivity_rate_perc': data['mean'] * 100,
                'total_conclusive_tests': data['count']
            }
    kpis['positivity_rates'] = positivity_breakdown

    return kpis


def calculate_environmental_kpis(iot_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates summary KPIs from IoT environmental data."""
    if not isinstance(iot_df, pd.DataFrame) or iot_df.empty:
        return {}

    kpis: Dict[str, Any] = {
        'avg_co2_ppm': iot_df['avg_co2_ppm'].mean(),
        'avg_pm25_ugm3': iot_df['avg_pm25'].mean(),
        'avg_waiting_room_occupancy': iot_df['waiting_room_occupancy'].mean(),
    }
    
    # Get latest noise reading per room and count high-noise alerts
    if 'timestamp' in iot_df.columns and 'room_name' in iot_df.columns and 'avg_noise_db' in iot_df.columns:
        latest_readings = iot_df.sort_values('timestamp').drop_duplicates('room_name', keep='last')
        high_noise_threshold = getattr(settings.ANALYTICS, 'noise_high_threshold_db', 80)
        kpis['rooms_with_high_noise_count'] = (latest_readings['avg_noise_db'] > high_noise_threshold).sum()
    else:
        kpis['rooms_with_high_noise_count'] = 0

    return kpis

def calculate_district_kpis(enriched_zone_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates high-level summary KPIs for an entire district from enriched zone data."""
    if not isinstance(enriched_zone_df, pd.DataFrame) or enriched_zone_df.empty:
        return {}

    kpis: Dict[str, Any] = {}
    population = enriched_zone_df['population'].sum()
    kpis['total_population'] = int(population)
    kpis['total_zones'] = len(enriched_zone_df)
    
    # Population-weighted average risk score
    if population > 0 and 'avg_risk_score' in enriched_zone_df.columns:
        weighted_avg = np.average(
            enriched_zone_df['avg_risk_score'].fillna(0),
            weights=enriched_zone_df['population']
        )
        kpis['population_weighted_avg_risk_score'] = weighted_avg
    else:
        kpis['population_weighted_avg_risk_score'] = enriched_zone_df['avg_risk_score'].mean()

    # Count zones exceeding risk threshold
    high_risk_threshold = settings.ANALYTICS.risk_score_moderate_threshold
    kpis['zones_in_high_risk_count'] = (enriched_zone_df['avg_risk_score'] > high_risk_threshold).sum()
    
    # Aggregate total key disease cases across the district
    active_case_cols = [col for col in enriched_zone_df.columns if col.startswith('active_cases_')]
    for col in active_case_cols:
        disease_name = col.replace('active_cases_', '').replace('_', ' ').title()
        kpis[f'total_{disease_name.lower().replace(" ", "_")}_cases'] = int(enriched_zone_df[col].sum())
        
    return kpis

# --- Generic Trend Calculation Utility ---

def calculate_trend(
    df: Optional[pd.DataFrame],
    value_col: str,
    date_col: str,
    freq: str = 'D',
    agg_func: Union[str, Callable] = 'mean'
) -> pd.Series:
    """
    Calculates a time-series trend for a given column, aggregated by a specified period.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=np.float64)

    df_trend = df[[date_col, value_col]].copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend[value_col] = convert_to_numeric(df_trend[value_col])
    df_trend.dropna(subset=[date_col, value_col], inplace=True)

    if df_trend.empty:
        return pd.Series(dtype=np.float64)

    try:
        trend_series = df_trend.set_index(date_col)[value_col].resample(freq).agg(agg_func)
        # For counts, fill missing periods with 0 for a continuous trend line
        if isinstance(agg_func, str) and agg_func in ['count', 'size', 'nunique', 'sum']:
            trend_series = trend_series.fillna(0)
            if not pd.api.types.is_float_dtype(trend_series.dtype):
                 trend_series = trend_series.astype(int)
        return trend_series
    except Exception as e:
        logger.error(f"Error generating trend for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype=np.float64)

# --- Streamlit-Cached Wrapper Functions for UI ---

@st.cache_data(ttl=settings.WEB_CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_clinic_kpis(df: Optional[pd.DataFrame], _source_context: str = "") -> Dict[str, Any]:
    """Cached wrapper for calculate_clinic_kpis for Streamlit performance."""
    return calculate_clinic_kpis(df)

@st.cache_data(ttl=settings.WEB_CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_environmental_kpis(iot_df: Optional[pd.DataFrame], _source_context: str = "") -> Dict[str, Any]:
    """Cached wrapper for calculate_environmental_kpis for Streamlit performance."""
    return calculate_environmental_kpis(iot_df)

@st.cache_data(ttl=settings.WEB_CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_district_kpis(enriched_zone_df: Optional[pd.DataFrame], _source_context: str = "") -> Dict[str, Any]:
    """Cached wrapper for calculate_district_kpis for Streamlit performance."""
    return calculate_district_kpis(enriched_zone_df)

@st.cache_data(ttl=settings.WEB_CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_trend(df: Optional[pd.DataFrame], **kwargs) -> pd.Series:
    """Cached wrapper for calculate_trend for Streamlit performance."""
    return calculate_trend(df, **kwargs)
