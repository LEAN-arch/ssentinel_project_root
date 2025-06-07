# sentinel_project_root/data_processing/aggregation.py
# Functions for aggregating data to compute KPIs and summaries for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, Union, Callable, List

# --- Core Imports ---
try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in aggregation.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)


def get_trend_data(
    df: Optional[pd.DataFrame],
    value_col: str,
    date_col: str = 'encounter_date',
    period: str = 'D',
    agg_func: Union[str, Callable[[Any], Any]] = 'mean',
    filter_col: Optional[str] = None,
    filter_val: Optional[Any] = None,
    source_context: str = "TrendCalculator"
) -> pd.Series:
    """
    Calculates a trend Series from a DataFrame by resampling and aggregating time-series data.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype='float64')

    required_cols = [date_col]
    if agg_func != 'size': required_cols.append(value_col)
    if filter_col: required_cols.append(filter_col)
    
    if any(col not in df.columns for col in required_cols):
        logger.warning(f"({source_context}) Missing required columns. Returning empty Series.")
        return pd.Series(dtype='float64')
    
    df_trend = df.copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend.dropna(subset=[date_col], inplace=True)
    if df_trend.empty: return pd.Series(dtype='float64')

    if isinstance(agg_func, str) and agg_func in ['mean', 'sum', 'median', 'std', 'var', 'min', 'max']:
        df_trend[value_col] = convert_to_numeric(df_trend[value_col])
        df_trend.dropna(subset=[value_col], inplace=True)
    
    df_to_agg = df_trend
    if filter_col and filter_val is not None:
        try:
            df_to_agg = df_trend[df_trend[filter_col] == filter_val]
        except Exception: # Broad exception for varied comparison errors
             df_to_agg = df_trend[df_trend[filter_col].astype(str) == str(filter_val)]

    if df_to_agg.empty: return pd.Series(dtype='float64')

    try:
        trend_series = df_to_agg.set_index(date_col)[value_col].resample(period).agg(agg_func)
        if agg_func in ['count', 'nunique', 'size']:
            trend_series = trend_series.fillna(0).astype(int)
        return trend_series
    except Exception as e:
        logger.error(f"({source_context}) Error during resampling/aggregation: {e}", exc_info=True)
        return pd.Series(dtype='float64')


def get_clinic_summary_kpis(
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicSummaryKPIs"
) -> Dict[str, Any]:
    """
    Calculates a dictionary of summary KPIs for a clinic over a given period.
    """
    logger.info(f"({source_context}) Calculating clinic summary KPIs.")
    
    kpis: Dict[str, Any] = {
        "test_summary_details": {}, "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": np.nan, "total_pending_critical_tests_patients": 0,
        "sample_rejection_rate_perc": np.nan, "key_drug_stockouts_count": 0
    }

    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) Health DataFrame is empty. Cannot calculate clinic KPIs.")
        return kpis

    df = health_df_period.copy()
    
    # --- Test-related KPIs ---
    if 'test_type' in df.columns and 'test_result' in df.columns:
        df['test_turnaround_days'] = convert_to_numeric(df.get('test_turnaround_days'), np.nan)
        non_conclusive = ['pending', 'rejected', 'unknown', 'n/a', 'indeterminate', 'invalid']
        
        df_conclusive = df[~df['test_result'].astype(str).str.lower().isin(non_conclusive)]
        if df_conclusive['test_turnaround_days'].notna().any():
            kpis["overall_avg_test_turnaround_conclusive_days"] = df_conclusive['test_turnaround_days'].mean()

        df_critical_conclusive = df_conclusive[df_conclusive['test_type'].isin(settings.CRITICAL_TESTS)]
        if not df_critical_conclusive.empty:
            met_tat_count = 0
            for _, row in df_critical_conclusive.iterrows():
                target_tat = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(row['test_type'], {}).get('target_tat_days', 2)
                if pd.notna(row['test_turnaround_days']) and row['test_turnaround_days'] <= target_tat:
                    met_tat_count += 1
            kpis["perc_critical_tests_tat_met"] = (met_tat_count / len(df_critical_conclusive)) * 100

        df_pending_critical = df[(df['test_result'].astype(str).str.lower() == 'pending') & (df['test_type'].isin(settings.CRITICAL_TESTS))]
        if 'patient_id' in df_pending_critical.columns:
            kpis["total_pending_critical_tests_patients"] = df_pending_critical['patient_id'].nunique()

    # --- Sample Rejection Rate ---
    if 'sample_status' in df.columns:
        total_samples = df[df['sample_status'] != 'Unknown']['sample_status'].count()
        rejected_samples = df[df['sample_status'].str.lower() == 'rejected by lab'].shape[0]
        if total_samples > 0:
            kpis["sample_rejection_rate_perc"] = (rejected_samples / total_samples) * 100
        
    # --- Supply Stockout Count ---
    if 'item' in df.columns and hasattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY'):
        drug_pattern = '|'.join([re.escape(drug) for drug in settings.KEY_DRUG_SUBSTRINGS_SUPPLY])
        df_drugs = df[df['item'].str.contains(drug_pattern, case=False, na=False)].copy()
        if not df_drugs.empty and all(c in df_drugs for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']):
            df_drugs['days_of_supply'] = convert_to_numeric(df_drugs['item_stock_agg_zone']) / convert_to_numeric(df_drugs['consumption_rate_per_day']).replace(0, np.nan)
            latest_stock = df_drugs.sort_values('encounter_date').drop_duplicates(subset='item', keep='last')
            kpis['key_drug_stockouts_count'] = latest_stock[latest_stock['days_of_supply'] < 7].shape[0]

    logger.info(f"({source_context}) Clinic KPIs calculated successfully.")
    return kpis

# --- Placeholder/Simplified Functions for other dashboards ---
def get_overall_kpis(health_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    return {"status": "placeholder", "record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}

def get_chw_summary_kpis(health_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    return {"status": "placeholder", "record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}

def get_clinic_environmental_summary_kpis(iot_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    return {"status": "placeholder", "record_count": len(iot_df_period) if isinstance(iot_df_period, pd.DataFrame) else 0}

def get_district_summary_kpis(health_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    return {"status": "placeholder", "record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}
