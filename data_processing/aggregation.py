# sentinel_project_root/data_processing/aggregation.py
"""
A collection of robust, high-performance, and encapsulated components for
aggregating data to compute KPIs and summaries for Sentinel dashboards.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Callable
import streamlit as st
import re

try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.critical(f"Critical import error in aggregation.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

def hash_dataframe_safe(df: pd.DataFrame) -> int:
    return pd.util.hash_pandas_object(df, index=True).sum()

def get_trend_data(df: Optional[pd.DataFrame], value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: Union[str, Callable] = 'mean') -> pd.Series:
    """Calculates a time-series trend for a given column, aggregated by a specified period."""
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=np.float64)
    df_trend = df[[date_col, value_col]].copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend[value_col] = convert_to_numeric(df_trend[value_col])
    df_trend.dropna(subset=[date_col, value_col], inplace=True)
    if df_trend.empty: return pd.Series(dtype=np.float64)
    try:
        if period == 'H': period = 'h' # Use modern, non-deprecated frequency string
        trend_series = df_trend.set_index(date_col)[value_col].resample(period).agg(agg_func)
        if isinstance(agg_func, str) and agg_func in ['count', 'size', 'nunique']:
            trend_series = trend_series.fillna(0).astype(int)
        return trend_series
    except Exception as e:
        logger.error(f"Error generating trend for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype=np.float64)

class ClinicKPIPreparer:
    """Encapsulates all logic for calculating clinic-level summary KPIs."""
    def __init__(self, health_df: pd.DataFrame):
        self.df = health_df.copy() if isinstance(health_df, pd.DataFrame) else pd.DataFrame()
        self.summary: Dict[str, Any] = {"overall_avg_test_turnaround_conclusive_days": np.nan, "perc_critical_tests_tat_met": 0.0, "total_pending_critical_tests_patients": 0, "sample_rejection_rate_perc": 0.0, "key_drug_stockouts_count": 0, "test_summary_details": {}}

    def _calculate_testing_kpis(self):
        if not all(c in self.df.columns for c in ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'patient_id']): return
        conclusive = self.df[~self.df['test_result'].str.lower().isin(['pending', 'rejected', 'unknown'])]
        if not conclusive.empty: self.summary["overall_avg_test_turnaround_conclusive_days"] = conclusive['test_turnaround_days'].mean()
        
        critical_keys = [k for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() if isinstance(v, dict) and v.get("critical")]
        critical_df = self.df[self.df['test_type'].isin(critical_keys)]
        if not critical_df.empty:
            conclusive_crit = critical_df[~critical_df['test_result'].str.lower().isin(['pending', 'rejected'])]
            if not conclusive_crit.empty:
                target_map = {k: v.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS) for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items()}
                conclusive_crit['target_tat'] = conclusive_crit['test_type'].map(target_map)
                self.summary["perc_critical_tests_tat_met"] = (conclusive_crit['test_turnaround_days'] <= conclusive_crit['target_tat']).mean() * 100
            self.summary["total_pending_critical_tests_patients"] = critical_df[critical_df['test_result'].str.lower() == 'pending']['patient_id'].nunique()
        
        valid_status = self.df[self.df['sample_status'].notna() & ~self.df['sample_status'].str.lower().isin(['unknown', ''])]
        if not valid_status.empty and valid_status['patient_id'].nunique() > 0:
            rejected = valid_status[valid_status['sample_status'].str.lower() == 'rejected by lab']['patient_id'].nunique()
            self.summary["sample_rejection_rate_perc"] = (rejected / valid_status['patient_id'].nunique()) * 100

    def _calculate_supply_kpis(self):
        if not all(c in self.df.columns for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date', 'zone_id']): return
        latest = self.df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
        latest['consumption_rate_per_day'] = latest['consumption_rate_per_day'].replace(0, 0.001)
        latest['days_of_supply'] = latest['item_stock_agg_zone'] / latest['consumption_rate_per_day']
        key_drugs_pattern = '|'.join(settings.KEY_DRUG_SUBSTRINGS_SUPPLY)
        key_drugs_stock_df = latest[latest['item'].str.contains(key_drugs_pattern, case=False, na=False)]
        if not key_drugs_stock_df.empty:
            self.summary["key_drug_stockouts_count"] = key_drugs_stock_df[key_drugs_stock_df['days_of_supply'] < settings.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()

    def _calculate_test_breakdown(self):
        if not all(c in self.df.columns for c in ['test_type', 'test_result', 'patient_id', 'sample_status', 'test_turnaround_days']): return
        
        tests_df = self.df[self.df['test_type'].isin(settings.KEY_TEST_TYPES_FOR_ANALYSIS.keys())].copy()
        if tests_df.empty: return
        
        tests_df['is_positive'] = (tests_df['test_result'].str.lower() == 'positive')
        
        conclusive_df = tests_df[~tests_df['test_result'].str.lower().isin(['pending', 'rejected', 'unknown'])]
        if not conclusive_df.empty:
            agg_spec = {'positive_rate_perc': pd.NamedAgg(column='is_positive', aggfunc=lambda x: x.mean() * 100 if len(x) > 0 else 0.0)}
            breakdown = conclusive_df.groupby('test_type').agg(**agg_spec)
            self.summary['test_summary_details'] = breakdown.to_dict('index')
        else:
            self.summary['test_summary_details'] = {}

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return self.summary
        self._calculate_testing_kpis()
        self._calculate_supply_kpis()
        self._calculate_test_breakdown()
        return self.summary

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_summary_kpis(health_df_period: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    return ClinicKPIPreparer(health_df_period).prepare()
