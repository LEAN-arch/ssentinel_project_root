# sentinel_project_root/data_processing/aggregation.py
# Functions for aggregating data to compute KPIs and summaries for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, Union, Callable, List
from datetime import date as date_type, datetime 

try:
    from config import settings
    from .helpers import convert_to_numeric # Ensure this helper is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__) 
    logger_init.error(f"Critical import error in aggregation.py: {e}. Ensure paths are correct.")
    class FallbackSettings: # Minimal fallback
        KEY_CONDITIONS_FOR_ACTION = ["Malaria", "TB", "Pneumonia"] # Example
        MALARIA_RDT_TEST_NAME_IDENTIFIER = "RDT-Malaria"
        NON_CONCLUSIVE_TEST_RESULTS = ['pending', 'rejected', 'unknown', 'n/a', 'indeterminate', 'invalid']
        KEY_DRUG_SUBSTRINGS_SUPPLY = ["ACT", "Amox", "ORS"]
        CRITICAL_SUPPLY_DAYS_REMAINING = 7
        DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 60.0
    settings = FallbackSettings()
    logger_init.warning("aggregation.py: Using fallback settings due to import error with 'config.settings'.")

logger = logging.getLogger(__name__)

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


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
    Calculates a trend (time series) for a given value column, aggregated by period.
    Handles missing data and ensures date column is properly formatted.
    Returns an empty Series with dtype 'float64' on failure or if no data.
    """
    logger.debug(f"({source_context}) Trend for '{value_col}' by '{period}'. Agg: {agg_func}. Input DF shape: {df.shape if isinstance(df, pd.DataFrame) else 'N/A'}")
    empty_series = pd.Series(dtype='float64', name=value_col if isinstance(value_col, str) else "trend_value") # Name the empty series

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"({source_context}) Input DataFrame empty/invalid for '{value_col}'.")
        return empty_series

    df_trend = df.copy() # Work on a copy

    if date_col not in df_trend.columns:
        logger.error(f"({source_context}) Date column '{date_col}' not found for trend of '{value_col}'.")
        return empty_series
    
    is_row_count_agg = isinstance(agg_func, str) and agg_func in ['size', 'count']
    
    if not is_row_count_agg and value_col not in df_trend.columns:
        logger.error(f"({source_context}) Value column '{value_col}' not found for trend aggregation '{agg_func}'.")
        return empty_series

    try:
        if not pd.api.types.is_datetime64_any_dtype(df_trend[date_col]):
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
        if df_trend[date_col].dt.tz is not None:
            df_trend[date_col] = df_trend[date_col].dt.tz_localize(None)
        df_trend.dropna(subset=[date_col], inplace=True)
    except Exception as e_date_conv_trend:
        logger.error(f"({source_context}) Error converting/processing date_col '{date_col}': {e_date_conv_trend}", exc_info=True)
        return empty_series

    if filter_col and filter_val is not None:
        if filter_col in df_trend.columns:
            try:
                col_dtype = df_trend[filter_col].dtype
                if pd.api.types.is_numeric_dtype(col_dtype) and isinstance(filter_val, (int, float)):
                    df_trend = df_trend[df_trend[filter_col] == filter_val]
                elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                     filter_val_dt = pd.to_datetime(filter_val, errors='coerce')
                     if pd.notna(filter_val_dt):
                         df_trend = df_trend[df_trend[filter_col] == filter_val_dt]
                     else: logger.warning(f"({source_context}) Invalid filter_val '{filter_val}' for datetime column '{filter_col}'.")
                else: 
                    df_trend = df_trend[df_trend[filter_col].astype(str) == str(filter_val)]
            except Exception as e_filter_trend:
                logger.warning(f"({source_context}) Error applying filter '{filter_col}'=='{filter_val}': {e_filter_trend}. Proceeding without this filter part.", exc_info=True)
        else:
            logger.warning(f"({source_context}) Filter column '{filter_col}' not found. Trend calculated without this filter.")

    if df_trend.empty:
        logger.info(f"({source_context}) DataFrame empty after date cleaning/filtering for '{value_col}'.")
        return empty_series

    try:
        numeric_agg_functions = ['mean', 'sum', 'median', 'std', 'var', 'min', 'max']
        # CORRECTED: The `convert_to_numeric` call is now ONLY for actual numeric aggregations.
        # It no longer incorrectly processes the `value_col` for 'nunique', 'count', or 'size'.
        if isinstance(agg_func, str) and agg_func in numeric_agg_functions:
            if value_col in df_trend.columns:
                df_trend[value_col] = convert_to_numeric(df_trend[value_col], default_value=np.nan)
                df_trend.dropna(subset=[value_col], inplace=True)
            elif not is_row_count_agg: 
                 logger.error(f"({source_context}) Value column '{value_col}' missing, cannot perform '{agg_func}'.")
                 return empty_series

        if df_trend.empty and not is_row_count_agg :
            logger.info(f"({source_context}) DataFrame empty after numeric conversion/dropna of '{value_col}' for trend.")
            return empty_series
        
        resampler = df_trend.set_index(date_col).resample(period)
        
        if is_row_count_agg and agg_func == 'size': 
            trend_series = resampler.size()
        elif value_col in df_trend.columns:
            trend_series = resampler[value_col].agg(agg_func)
        else:
             logger.error(f"({source_context}) Fallback: Value column '{value_col}' not found for aggregation '{agg_func}'.")
             return empty_series

        count_like_aggs = ['count', 'nunique', 'size']
        if isinstance(agg_func, str) and agg_func in count_like_aggs:
            trend_series = trend_series.fillna(0)
            try:
                if not trend_series.isnull().any():
                    trend_series = trend_series.astype(int)
                else:
                    trend_series = trend_series.astype(pd.Int64Dtype()) 
            except Exception:
                logger.debug(f"({source_context}) Could not convert count-like trend for '{value_col}' to integer type. Kept as {trend_series.dtype}.")
        
        if isinstance(trend_series, pd.Series) and not trend_series.name:
            trend_series.name = value_col if isinstance(value_col, str) else "trend_value"

        logger.debug(f"({source_context}) Trend for '{value_col}' generated, {len(trend_series)} points. Sample:\n{trend_series.head().to_string() if not trend_series.empty else 'Empty Series'}")
        return trend_series
    except Exception as e_agg_trend:
        logger.error(f"({source_context}) Error during resampling/aggregation for '{value_col}': {e_agg_trend}", exc_info=True)
        return empty_series


# --- Placeholder definitions for other KPI functions ---
def get_overall_kpis(health_df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    # This remains a placeholder as its implementation is not the source of the error.
    return {}

def get_chw_summary_kpis(chw_daily_encounter_df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    return {}

def get_clinic_summary_kpis(health_df_period: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    return {}

def get_clinic_environmental_summary_kpis(iot_df_period: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    return {}

def get_district_summary_kpis(enriched_zone_df: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
    return {}
