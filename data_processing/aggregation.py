import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, Union, Callable, List
from datetime import date as date_type, datetime

# --- Module Imports & Setup ---
try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    # FIXED: Use the correct `__name__` magic variable.
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in aggregation.py: {e}. Ensure paths/project are correct.")

    class FallbackSettings:
        KEY_CONDITIONS_FOR_ACTION = ["Malaria", "TB", "Pneumonia"]
        KEY_TEST_TYPES_FOR_ANALYSIS = {"RDT-Malaria": {"critical": True, "target_tat_days": 1}}
        CRITICAL_TESTS = ["RDT-Malaria"]
        NON_CONCLUSIVE_TEST_RESULTS = ['pending', 'rejected', 'unknown', 'n/a', 'indeterminate', 'invalid']
        TARGET_TEST_TURNAROUND_DAYS = 2
        TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
        TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0
        KEY_DRUG_SUBSTRINGS_SUPPLY = ["ACT", "Amox", "ORS"]
        CRITICAL_SUPPLY_DAYS_REMAINING = 7
        DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 60.0
    
    settings = FallbackSettings()
    logger_init.warning("Using fallback settings due to import error with 'config.settings'.")

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely get a setting attribute, falling back to a default value."""
    return getattr(settings, attr_name, default_value)


def get_trend_data(
    df: Optional[pd.DataFrame], value_col: str, date_col: str = 'encounter_date',
    period: str = 'D', agg_func: Union[str, Callable[[Any], Any]] = 'mean',
    filter_col: Optional[str] = None, filter_val: Optional[Any] = None,
    source_context: str = "TrendCalculator"
) -> pd.Series:
    """
    Calculates a trend Series by resampling and aggregating time-series data.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.debug(f"({source_context}) Input DataFrame is empty. Returning empty Series.")
        return pd.Series(dtype='float64')

    df_trend = df.copy()
    required_cols = [date_col]
    if agg_func != 'size': required_cols.append(value_col)
    if filter_col: required_cols.append(filter_col)

    if any(col not in df_trend.columns for col in required_cols):
        logger.warning(f"({source_context}) Missing required columns. Returning empty Series.")
        return pd.Series(dtype='float64')

    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend.dropna(subset=[date_col], inplace=True)
    if df_trend.empty: return pd.Series(dtype='float64')

    numeric_agg_funcs = ['mean', 'sum', 'median', 'std', 'var', 'min', 'max']
    if isinstance(agg_func, str) and agg_func in numeric_agg_funcs:
        df_trend[value_col] = convert_to_numeric(df_trend[value_col], default_value=np.nan)
        df_trend.dropna(subset=[value_col], inplace=True)

    if df_trend.empty and agg_func not in ['size', 'count']:
        return pd.Series(dtype='float64')

    df_to_aggregate = df_trend
    if filter_col and filter_val is not None:
        try:
            if df_trend[filter_col].dtype == 'object' and isinstance(filter_val, str):
                df_to_aggregate = df_trend[df_trend[filter_col].str.lower() == filter_val.lower()]
            else:
                df_to_aggregate = df_trend[df_trend[filter_col] == filter_val]
        except Exception: # Catch errors from incompatible type comparisons
            df_to_aggregate = pd.DataFrame(columns=df_trend.columns)

    if df_to_aggregate.empty:
        return pd.Series(dtype='float64')

    try:
        trend_series = df_to_aggregate.set_index(date_col).resample(period)[value_col].agg(agg_func)
        if agg_func in ['count', 'nunique', 'size']:
            trend_series = trend_series.fillna(0).astype(int)
    except Exception as e:
        logger.error(f"({source_context}) Error during resampling/aggregation: {e}", exc_info=True)
        return pd.Series(dtype='float64')
    
    return trend_series


def get_clinic_summary_kpis(
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicSummaryKPIs"
) -> Dict[str, Any]:
    """Calculates a dictionary of summary KPIs for a clinical setting."""
    logger.info(f"({source_context}) Calculating clinic summary KPIs.")
    kpis: Dict[str, Any] = {
        "test_summary_details": {}, "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": np.nan, "total_pending_critical_tests_patients": 0,
        "sample_rejection_rate_perc": np.nan, "key_drug_stockouts_count": 0
    }
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return kpis

    df = health_df_period.copy()
    test_cols = ['test_type', 'test_result', 'test_turnaround_days', 'patient_id', 'sample_status']
    for col in test_cols:
        if col not in df.columns:
            df[col] = "Unknown" if col in ['test_type', 'test_result', 'sample_status'] else np.nan
    df['test_turnaround_days'] = convert_to_numeric(df['test_turnaround_days'], np.nan)

    non_conclusive_results = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', [])
    df_conclusive = df[~df['test_result'].astype(str).str.lower().isin(non_conclusive_results)]
    
    if not df_conclusive.empty and 'test_turnaround_days' in df_conclusive.columns:
        kpis["overall_avg_test_turnaround_conclusive_days"] = df_conclusive['test_turnaround_days'].mean()

    critical_tests = _get_setting('CRITICAL_TESTS', [])
    df_critical_conclusive = df_conclusive[df_conclusive['test_type'].isin(critical_tests)]
    if not df_critical_conclusive.empty:
        key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
        target_tat = _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)
        tat_met_mask = df_critical_conclusive.apply(
            lambda row: row['test_turnaround_days'] <= key_test_configs.get(row['test_type'], {}).get('target_tat_days', target_tat),
            axis=1
        )
        kpis["perc_critical_tests_tat_met"] = (tat_met_mask.sum() / len(df_critical_conclusive)) * 100

    df_pending_critical = df[(df['test_result'].astype(str).str.lower() == 'pending') & (df['test_type'].isin(critical_tests))]
    if 'patient_id' in df_pending_critical.columns:
        kpis["total_pending_critical_tests_patients"] = df_pending_critical['patient_id'].nunique()

    total_samples = df[df['sample_status'] != 'Unknown'].shape[0]
    if total_samples > 0:
        rejected_samples = df[df['sample_status'].astype(str).str.lower() == 'rejected by lab'].shape[0]
        kpis["sample_rejection_rate_perc"] = (rejected_samples / total_samples) * 100
        
    test_details = {}
    if 'test_type' in df.columns:
        for test_name, group_df in df.groupby('test_type'):
            if test_name == 'Unknown': continue
            group_conclusive = group_df[~group_df['test_result'].astype(str).str.lower().isin(non_conclusive_results)]
            positives = group_conclusive[group_conclusive['test_result'].astype(str).str.lower() == 'positive'].shape[0]
            test_details[test_name] = {
                "positive_rate_perc": (positives / len(group_conclusive)) * 100 if len(group_conclusive) > 0 else 0.0,
                "avg_tat_days": group_conclusive['test_turnaround_days'].mean(),
                "pending_count_patients": group_df[group_df['test_result'].astype(str).str.lower() == 'pending']['patient_id'].nunique()
            }
    kpis["test_summary_details"] = test_details

    key_drugs = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if key_drugs and 'item' in df.columns:
        drug_pattern = '|'.join([re.escape(drug) for drug in key_drugs])
        df_drugs = df[df['item'].str.contains(drug_pattern, case=False, na=False)].copy()
        if not df_drugs.empty and all(c in df_drugs for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']):
            df_drugs['days_of_supply'] = df_drugs['item_stock_agg_zone'] / df_drugs['consumption_rate_per_day'].replace(0, np.nan)
            latest_stock = df_drugs.sort_values('encounter_date').drop_duplicates(subset='item', keep='last')
            stockout_count = latest_stock[latest_stock['days_of_supply'] < _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)].shape[0]
            kpis['key_drug_stockouts_count'] = stockout_count

    logger.info(f"({source_context}) Clinic KPIs calculated successfully.")
    return kpis


# Placeholder Functions
def get_overall_kpis(health_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_overall_kpis called (placeholder).")
    return {"status": "placeholder", "record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}

def get_chw_summary_kpis(health_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_chw_summary_kpis called (placeholder).")
    return {"status": "placeholder", "record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}

def get_clinic_environmental_summary_kpis(iot_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_clinic_environmental_summary_kpis called (placeholder).")
    return {"status": "placeholder", "record_count": len(iot_df_period) if isinstance(iot_df_period, pd.DataFrame) else 0}

def get_district_summary_kpis(health_df_period: Optional[pd.DataFrame], *args, **kwargs) -> Dict[str, Any]:
    logger.debug("get_district_summary_kpis called (placeholder).")
    return {"status": "placeholder", "record_count": len(health_df_period) if isinstance(health_df_period, pd.DataFrame) else 0}
