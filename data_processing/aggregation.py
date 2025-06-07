# ssentinel_project_root/data_processing/aggregation.py
"""
Contains functions for aggregating and summarizing processed data into meaningful KPIs.
This module performs the core business logic calculations for dashboards.
"""
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, Union, Callable

# --- Module Imports & Setup ---
try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in aggregation.py: {e}. Ensure config and helpers are accessible.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely get a setting attribute, falling back to a default value."""
    return getattr(settings, attr_name, default_value)


def get_trend_data(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = 'encounter_date',
    period: str = 'D',
    agg_func: Union[str, Callable[[Any], Any]] = 'mean'
) -> pd.Series:
    """
    Calculates a trend Series by resampling and aggregating time-series data.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)

    required_cols = [date_col] if agg_func == 'size' else [date_col, value_col]
    if any(col not in df.columns for col in required_cols):
        logger.warning(f"TrendCalculator: Missing required columns. Required: {required_cols}")
        return pd.Series(dtype=float)

    df_trend = df.copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend.dropna(subset=[date_col], inplace=True)
    if df_trend.empty:
        return pd.Series(dtype=float)

    df_trend.set_index(date_col, inplace=True)
    
    try:
        if isinstance(agg_func, str) and agg_func not in ['size', 'count', 'nunique']:
             df_trend[value_col] = convert_to_numeric(df_trend[value_col])

        trend_series = df_trend[value_col].resample(period).agg(agg_func)
        
        if agg_func in ['size', 'count', 'nunique']:
            trend_series = trend_series.fillna(0).astype(int)
            
    except Exception as e:
        logger.error(f"TrendCalculator: Error during resampling/aggregation: {e}", exc_info=True)
        return pd.Series(dtype=float)
    
    return trend_series


def get_clinic_summary_kpis(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    """Calculates a dictionary of summary KPIs for a clinical setting using vectorized operations."""
    kpis: Dict[str, Any] = {
        "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests": 0,
        "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0,
    }
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return kpis

    df = health_df_period.copy()
    
    # --- Testing KPIs ---
    if 'test_result' in df.columns and 'test_turnaround_days' in df.columns:
        non_conclusive_results = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', ['pending', 'rejected'])
        df['test_turnaround_days'] = convert_to_numeric(df['test_turnaround_days'], np.nan)
        df_conclusive = df[~df['test_result'].astype(str).str.lower().isin(non_conclusive_results)]
        
        if not df_conclusive.empty:
            kpis["overall_avg_test_turnaround_conclusive_days"] = df_conclusive['test_turnaround_days'].mean()

            critical_tests = _get_setting('CRITICAL_TESTS', [])
            df_critical_conclusive = df_conclusive[df_conclusive['test_type'].isin(critical_tests)]
            
            if not df_critical_conclusive.empty:
                key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
                default_target_tat = _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)
                
                target_tats = df_critical_conclusive['test_type'].map(lambda t: key_test_configs.get(t, {}).get('target_tat_days', default_target_tat))
                tat_met_count = (df_critical_conclusive['test_turnaround_days'] <= target_tats).sum()
                kpis["perc_critical_tests_tat_met"] = (tat_met_count / len(df_critical_conclusive)) * 100

        kpis["total_pending_critical_tests"] = df[(df['test_result'].astype(str).str.lower() == 'pending') & (df['test_type'].isin(critical_tests))].shape[0]

    if 'sample_status' in df.columns:
        total_samples = df[df['sample_status'].notna() & (df['sample_status'] != 'Unknown')].shape[0]
        if total_samples > 0:
            rejected_samples = df[df['sample_status'].astype(str).str.lower().str.contains('rejected', na=False)].shape[0]
            kpis["sample_rejection_rate_perc"] = (rejected_samples / total_samples) * 100

    # --- Drug Stock KPIs ---
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns:
        key_drugs = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
        if key_drugs:
            drug_pattern = '|'.join([re.escape(drug) for drug in key_drugs])
            df_drugs = df[df['item'].str.contains(drug_pattern, case=False, na=False)].copy()
            
            if not df_drugs.empty and all(c in df_drugs for c in ['consumption_rate_per_day', 'encounter_date']):
                df_drugs['consumption_rate_per_day'] = convert_to_numeric(df_drugs['consumption_rate_per_day'], 1e-6)
                df_drugs['days_of_supply'] = convert_to_numeric(df_drugs['item_stock_agg_zone']) / df_drugs['consumption_rate_per_day']
                
                latest_stock = df_drugs.sort_values('encounter_date').drop_duplicates(subset='item', keep='last')
                stockout_count = (latest_stock['days_of_supply'] < _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)).sum()
                kpis['key_drug_stockouts_count'] = int(stockout_count)

    return kpis


def get_clinic_environmental_summary_kpis(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    """Calculates a dictionary of summary KPIs for clinic environmental data."""
    kpis = {
        'avg_co2_overall_ppm': np.nan, 'avg_pm25_overall_ugm3': np.nan,
        'avg_waiting_room_occupancy_overall_persons': np.nan,
        'rooms_noise_high_alert_latest_count': 0
    }
    if not isinstance(iot_df_period, pd.DataFrame) or iot_df_period.empty:
        return kpis
        
    df = iot_df_period.copy()

    kpis['avg_co2_overall_ppm'] = convert_to_numeric(df.get('co2_ppm'), np.nan).mean()
    kpis['avg_pm25_overall_ugm3'] = convert_to_numeric(df.get('pm2_5_ug_m3'), np.nan).mean()
    kpis['avg_waiting_room_occupancy_overall_persons'] = convert_to_numeric(df.get('occupancy_count'), np.nan).mean()

    if 'room_id' in df.columns and 'noise_dba' in df.columns:
        latest_readings = df.sort_values('timestamp').drop_duplicates(subset='room_id', keep='last')
        if not latest_readings.empty:
            high_noise_threshold = _get_setting('ALERT_AMBIENT_NOISE_HIGH_DBA', 85)
            high_noise_count = (convert_to_numeric(latest_readings['noise_dba'], 0) >= high_noise_threshold).sum()
            kpis['rooms_noise_high_alert_latest_count'] = int(high_noise_count)
        
    return kpis

# --- Preserving Other Function Signatures for Backward Compatibility ---
# These functions were placeholders in the original file. Providing a minimal but functional
# implementation ensures that any un-seen code calling them will not break.

def get_overall_kpis(health_df_period: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_overall_kpis called (compatibility stub).")
    return {"record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}

def get_chw_summary_kpis(health_df_period: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_chw_summary_kpis called (compatibility stub).")
    return {"record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}

def get_district_summary_kpis(health_df_period: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_district_summary_kpis called (compatibility stub).")
    return {"record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}
