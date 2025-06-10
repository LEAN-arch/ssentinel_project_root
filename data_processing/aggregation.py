# sentinel_project_root/data_processing/aggregation.py
# SME PLATINUM STANDARD - DECISION-GRADE AGGREGATIONS (V7 - DEFINITIVE CACHE FIX)

import logging
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from .helpers import convert_to_numeric, hash_dataframe

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3600

# --- Core, Non-Cached Calculation Functions ---
def _calculate_clinic_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    # ... [This function is correct and remains unchanged] ...
    if not isinstance(df, pd.DataFrame) or df.empty: return {}
    kpis: Dict[str, Any] = {}
    conclusive_tests = df[df['sample_status'].str.lower().isin(['completed', 'rejected by lab'])]
    if not conclusive_tests.empty:
        kpis['avg_test_tat_days'] = conclusive_tests['test_turnaround_days'].mean()
        target_map = {name: config.target_tat_days for name, config in settings.KEY_TEST_TYPES.items()}
        conclusive_tests['target_tat'] = conclusive_tests['test_type'].map(target_map)
        met_tat = (conclusive_tests['test_turnaround_days'] <= conclusive_tests['target_tat']).sum()
        non_na_targets = conclusive_tests.dropna(subset=['target_tat'])
        kpis['perc_tests_within_tat'] = (met_tat / len(non_na_targets)) * 100 if len(non_na_targets) > 0 else 0
    else: kpis['avg_test_tat_days'] = np.nan; kpis['perc_tests_within_tat'] = 0.0
    kpis['sample_rejection_rate_perc'] = df.get('is_rejected', pd.Series(dtype=float)).mean() * 100
    kpis['pending_critical_tests_count'] = df.get('is_critical_and_pending', pd.Series(dtype=int)).sum()
    kpis['key_items_at_risk_count'] = df.get('is_supply_at_risk', pd.Series(dtype=int)).sum()
    positivity_breakdown = {}
    test_df = df.dropna(subset=['test_type', 'is_positive'])
    if not test_df.empty:
        positivity_groups = test_df.groupby('test_type')['is_positive'].agg(['mean', 'count'])
        for test_type, data in positivity_groups.iterrows():
            positivity_breakdown[test_type] = {'positivity_rate_perc': data['mean'] * 100, 'total_conclusive_tests': data['count']}
    kpis['positivity_rates'] = positivity_breakdown
    return kpis

def _calculate_trend(df: Optional[pd.DataFrame], value_col: str, date_col: str, freq: str = 'D', agg_func: Union[str, Callable] = 'mean') -> pd.Series:
    # ... [This function is correct and remains unchanged] ...
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns: return pd.Series(dtype=np.float64)
    df_trend = df[[date_col, value_col]].copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend[value_col] = convert_to_numeric(df_trend[value_col])
    df_trend.dropna(subset=[date_col, value_col], inplace=True)
    if df_trend.empty: return pd.Series(dtype=np.float64)
    try:
        trend_series = df_trend.set_index(date_col)[value_col].resample(freq).agg(agg_func)
        if isinstance(agg_func, str) and agg_func in ['count', 'size', 'nunique', 'sum']:
            trend_series = trend_series.fillna(0).astype(int)
        return trend_series
    except Exception as e:
        logger.error(f"Error generating trend for '{value_col}': {e}", exc_info=True); return pd.Series(dtype=np.float64)

# --- Streamlit-Cached Wrapper Functions for UI ---
@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_clinic_kpis(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    return _calculate_clinic_kpis(df)

# SME FIX: The cached trend function now only accepts hashable string arguments for `agg_func`.
# The non-cached `_calculate_trend` remains available for custom, unhashable aggregations.
@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: hash_dataframe})
def get_cached_trend(
    df: Optional[pd.DataFrame],
    value_col: str,
    date_col: str,
    freq: str = 'D',
    agg_func: str = 'mean'
) -> pd.Series:
    """
    Cached wrapper for calculating trends. Only accepts hashable (string)
    aggregation function names.
    """
    if not isinstance(agg_func, str):
        raise ValueError(
            "get_cached_trend only accepts string values for agg_func "
            "(e.g., 'mean', 'sum', 'nunique'). For custom callables, "
            "use the non-cached '_calculate_trend' function."
        )
    return _calculate_trend(df, value_col, date_col, freq, agg_func)
