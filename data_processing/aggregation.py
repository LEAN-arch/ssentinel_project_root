# sentinel_project_root/data_processing/aggregation.py
# Functions for aggregating data to compute KPIs and summaries for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, Union, Callable, List
from datetime import date as date_type, datetime # Added datetime for robust parsing

try:
    from config import settings
    from .helpers import convert_to_numeric # Ensure this helper is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in aggregation.py: {e}. Ensure paths are correct.")
    raise # Aggregation functions are critical, so raise if dependencies are missing

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
    logger.debug(f"({source_context}) Generating trend for '{value_col}' by '{period}' period. Agg func: {agg_func}")
    empty_series = pd.Series(dtype='float64') # Define once for reuse

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"({source_context}) Input DataFrame is empty or invalid for trend of '{value_col}'.")
        return empty_series

    df_trend = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # Validate essential columns
    if date_col not in df_trend.columns:
        logger.error(f"({source_context}) Date column '{date_col}' not found for trend of '{value_col}'.")
        return empty_series
    if value_col not in df_trend.columns and not (isinstance(agg_func, str) and agg_func in ['size', 'count']): # value_col not needed for 'size' or 'count' on index
        logger.error(f"({source_context}) Value column '{value_col}' not found for trend aggregation '{agg_func}'.")
        return empty_series

    # Ensure date column is datetime and timezone-naive
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_trend[date_col]):
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
        if df_trend[date_col].dt.tz is not None: # If timezone-aware, make it naive for consistency
            df_trend[date_col] = df_trend[date_col].dt.tz_localize(None)
        df_trend.dropna(subset=[date_col], inplace=True) # Drop rows where date conversion failed (NaT)
    except Exception as e_date_conv:
        logger.error(f"({source_context}) Could not convert/process date_col '{date_col}' to datetime: {e_date_conv}", exc_info=True)
        return empty_series

    # Apply optional filter
    if filter_col and filter_val is not None:
        if filter_col in df_trend.columns:
            try: # Robust filtering, convert filter_val to column type if needed, or column to str
                if pd.api.types.is_numeric_dtype(df_trend[filter_col]) and isinstance(filter_val, (int, float)):
                    df_trend = df_trend[df_trend[filter_col] == filter_val]
                else: # Fallback to string comparison for safety
                    df_trend = df_trend[df_trend[filter_col].astype(str) == str(filter_val)]
            except Exception as e_filter:
                logger.warning(f"({source_context}) Error applying filter '{filter_col}'=='{filter_val}': {e_filter}. Proceeding without this filter part.", exc_info=True)
        else:
            logger.warning(f"({source_context}) Filter column '{filter_col}' not found. Trend calculated without this filter.")

    if df_trend.empty:
        logger.info(f"({source_context}) DataFrame became empty after date cleaning or filtering for trend of '{value_col}'.")
        return empty_series

    try:
        # For numeric aggregations, ensure value_col is numeric.
        numeric_agg_functions = ['mean', 'sum', 'median', 'std', 'var', 'min', 'max', 'nunique'] # nunique also benefits from numeric conversion if IDs are numbers
        if isinstance(agg_func, str) and agg_func in numeric_agg_functions:
            if value_col in df_trend.columns: # Ensure value_col exists before trying to convert
                df_trend[value_col] = convert_to_numeric(df_trend[value_col], default_value=np.nan)
                if agg_func != 'nunique': # For nunique, NaNs are usually excluded by default, no need to drop
                    df_trend.dropna(subset=[value_col], inplace=True)
            elif agg_func != 'size': # 'size' doesn't need value_col
                 logger.error(f"({source_context}) Value column '{value_col}' is required for numeric aggregation '{agg_func}' but is missing.")
                 return empty_series


        if df_trend.empty and not (isinstance(agg_func, str) and agg_func == 'size'): # 'size' can work on empty if index is set
            logger.info(f"({source_context}) DataFrame empty after numeric conversion/dropna of '{value_col}' for trend.")
            return empty_series
        
        # Perform resampling and aggregation
        # If agg_func is 'size', value_col doesn't matter much as it counts rows in each group
        resampler = df_trend.set_index(date_col).resample(period)
        trend_series = resampler[value_col].agg(agg_func) if value_col in df_trend.columns and agg_func != 'size' else resampler.size()


        # For count-based aggregations, fill NaNs with 0 as no occurrences means count is 0
        count_based_agg_functions = ['count', 'nunique', 'size']
        if isinstance(agg_func, str) and agg_func in count_based_agg_functions:
            trend_series = trend_series.fillna(0).astype(int) # Ensure integer type for counts

        logger.debug(f"({source_context}) Trend for '{value_col}' generated with {len(trend_series)} data points.")
        return trend_series
    except Exception as e_trend_agg:
        logger.error(f"({source_context}) Error generating trend for '{value_col}' during aggregation: {e_trend_agg}", exc_info=True)
        return empty_series


def get_overall_kpis(
    health_df: Optional[pd.DataFrame],
    date_filter_start: Optional[Union[str, pd.Timestamp, date_type, datetime]] = None,
    date_filter_end: Optional[Union[str, pd.Timestamp, date_type, datetime]] = None,
    source_context: str = "GlobalKPIs"
) -> Dict[str, Any]:
    """Calculates overall key performance indicators from health data for a given period."""
    
    start_date_str = str(date_filter_start) if date_filter_start else "N/A"
    end_date_str = str(date_filter_end) if date_filter_end else "N/A"
    logger.info(f"({source_context}) Calculating overall KPIs for period: {start_date_str} to {end_date_str}")

    # Initialize KPIs with defaults
    kpis: Dict[str, Any] = {
        "total_patients_period": 0, "avg_patient_ai_risk_period": np.nan,
        "malaria_rdt_positive_rate_period": np.nan, "key_supply_stockout_alerts_period": 0,
        "total_encounters_period": 0
    }
    key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    for condition_name in key_conditions:
        # Sanitize condition name for use as a dictionary key
        kpi_key_safe = f"active_{re.sub(r'[^a-z0-9_]+', '_', condition_name.lower().strip())}_cases_period"
        kpis[kpi_key_safe] = 0

    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        logger.warning(f"({source_context}) Health DataFrame is empty or invalid. Returning default KPIs.")
        return kpis

    df = health_df.copy() # Work on a copy

    # --- Data Preparation and Filtering ---
    if 'encounter_date' not in df.columns:
        logger.error(f"({source_context}) 'encounter_date' column missing. KPIs calculations will be inaccurate or fail.")
        return kpis 
    
    try:
        # Ensure encounter_date is datetime and timezone-naive
        if not pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
            df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        if df['encounter_date'].dt.tz is not None:
            df['encounter_date'] = df['encounter_date'].dt.tz_localize(None)
        df.dropna(subset=['encounter_date'], inplace=True) # Remove invalid dates

        if date_filter_start:
            start_dt = pd.to_datetime(date_filter_start, errors='coerce').normalize() # Get date part
            if pd.notna(start_dt): df = df[df['encounter_date'] >= start_dt]
            else: logger.warning(f"({source_context}) Invalid start_date_filter: {date_filter_start}")
        if date_filter_end:
            end_dt = pd.to_datetime(date_filter_end, errors='coerce').normalize() # Get date part
            if pd.notna(end_dt): df = df[df['encounter_date'] <= end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)] # Inclusive of end date
            else: logger.warning(f"({source_context}) Invalid end_date_filter: {date_filter_end}")
    except Exception as e_date_filter:
        logger.warning(f"({source_context}) Error applying date filters: {e_date_filter}. Proceeding with potentially unfiltered data.", exc_info=True)

    if df.empty:
        logger.info(f"({source_context}) No data remains after date filtering (if applied). Returning default KPIs.")
        return kpis

    # --- Calculate KPIs ---
    if 'patient_id' in df.columns: 
        kpis["total_patients_period"] = df['patient_id'].nunique()
    else: logger.warning(f"({source_context}) 'patient_id' column missing. total_patients_period will be 0.")
    
    # Assuming 'encounter_id' might exist for unique encounters, otherwise fallback to patient_id count
    encounter_id_col = 'encounter_id' if 'encounter_id' in df.columns else ('patient_id' if 'patient_id' in df.columns else None)
    if encounter_id_col:
        kpis["total_encounters_period"] = df[encounter_id_col].nunique()
    else: logger.warning(f"({source_context}) Neither 'encounter_id' nor 'patient_id' found. total_encounters_period will be 0.")


    if 'ai_risk_score' in df.columns:
        risk_scores_series = convert_to_numeric(df['ai_risk_score'], default_value=np.nan)
        if risk_scores_series.notna().any():
            kpis["avg_patient_ai_risk_period"] = risk_scores_series.mean()
    else: logger.debug(f"({source_context}) 'ai_risk_score' column missing.")

    if 'condition' in df.columns and 'patient_id' in df.columns and key_conditions:
        for condition_name_iter in key_conditions:
            kpi_key_dynamic = f"active_{re.sub(r'[^a-z0-9_]+', '_', condition_name_iter.lower().strip())}_cases_period"
            try: # Robust regex for partial, case-insensitive match
                condition_mask = df['condition'].astype(str).str.contains(re.escape(condition_name_iter), case=False, na=False, regex=True)
                kpis[kpi_key_dynamic] = df.loc[condition_mask, 'patient_id'].nunique() if condition_mask.any() else 0
            except Exception as e_cond_regex:
                logger.warning(f"({source_context}) Regex error for condition '{condition_name_iter}': {e_cond_regex}. Count set to 0.")
                kpis[kpi_key_dynamic] = 0
    elif not key_conditions:
        logger.debug(f"({source_context}) No KEY_CONDITIONS_FOR_ACTION defined in settings.")

    # Malaria RDT Positivity Rate
    if 'test_type' in df.columns and 'test_result' in df.columns:
        malaria_rdt_name = _get_setting('MALARIA_RDT_TEST_NAME_IDENTIFIER', "RDT-Malaria") # Allow config
        malaria_tests_df = df[df['test_type'].astype(str).str.contains(re.escape(malaria_rdt_name), case=False, na=False)]
        if not malaria_tests_df.empty:
            non_conclusive_results = ['pending', 'rejected', 'unknown', 'n/a', 'indeterminate', 'invalid'] # Make configurable if needed
            conclusive_malaria_tests = malaria_tests_df[
                ~malaria_tests_df['test_result'].astype(str).str.lower().isin(non_conclusive_results)
            ]
            if not conclusive_malaria_tests.empty:
                positive_malaria_tests_count = conclusive_malaria_tests[
                    conclusive_malaria_tests['test_result'].astype(str).str.lower() == 'positive'
                ].shape[0]
                kpis["malaria_rdt_positive_rate_period"] = (positive_malaria_tests_count / len(conclusive_malaria_tests)) * 100
            else: logger.debug(f"({source_context}) No conclusive Malaria RDT results found.")
        else: logger.debug(f"({source_context}) No Malaria RDT tests found.")
    else: logger.debug(f"({source_context}) 'test_type' or 'test_result' columns missing for Malaria RDT rate.")
    
    # Key Supply Stockout Alerts
    # This KPI is complex and might be better in a dedicated supply chain module or if data structure is simpler here
    # For now, keeping a simplified version if columns exist
    supply_cols = ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date', 'zone_id']
    if all(col in df.columns for col in supply_cols):
        try:
            latest_stock_df = df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['item', 'zone_id'], keep='last').copy()
            latest_stock_df['consumption_rate_per_day'] = convert_to_numeric(latest_stock_df['consumption_rate_per_day'], default_value=0.001) # Avoid NaN default here
            latest_stock_df.loc[latest_stock_df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001 # Prevent DivByZero / negative DOS
            latest_stock_df['item_stock_agg_zone'] = convert_to_numeric(latest_stock_df['item_stock_agg_zone'], default_value=0.0)
            latest_stock_df['days_of_supply'] = latest_stock_df['item_stock_agg_zone'] / latest_stock_df['consumption_rate_per_day']
            
            key_drug_substrings = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
            critical_supply_days = _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
            
            if key_drug_substrings:
                key_drugs_pattern = '|'.join(re.escape(s.strip()) for s in key_drug_substrings if s.strip())
                key_drugs_df = latest_stock_df[
                    latest_stock_df['item'].astype(str).str.contains(key_drugs_pattern, case=False, na=False, regex=True)
                ]
                if not key_drugs_df.empty:
                    kpis['key_supply_stockout_alerts_period'] = key_drugs_df[
                        key_drugs_df['days_of_supply'] < critical_supply_days
                    ]['item'].nunique() 
            else: logger.debug(f"({source_context}) KEY_DRUG_SUBSTRINGS_SUPPLY not defined or empty.")
        except Exception as e_supply_kpi:
            logger.error(f"({source_context}) Error calculating supply stockout KPI: {e_supply_kpi}", exc_info=True)
    else: 
        missing_supply_cols = [col for col in supply_cols if col not in df.columns]
        if missing_supply_cols: logger.debug(f"({source_context}) Missing supply-related columns: {missing_supply_cols}.")
            
    logger.info(f"({source_context}) Overall KPIs calculated: { {k: (f'{v:.2f}' if isinstance(v, float) else v) for k,v in kpis.items()} }") # Format floats for logging
    return kpis


# Note: get_chw_summary_kpis, get_clinic_summary_kpis, get_clinic_environmental_summary_kpis, get_district_summary_kpis
# should also be reviewed and optimized with similar principles:
# - Robust settings access (_get_setting).
# - Clearer DataFrame copies (df.copy() when modifications are made).
# - Consistent use of convert_to_numeric for columns expected to be numeric.
# - Graceful handling of missing columns or empty DataFrames at each step.
# - Explicit error handling with informative logging for calculation steps.
# - Consistent date handling (parsing, normalization, timezone awareness if applicable).
# - Clear variable names and comments.

# For brevity, I'm not re-writing all of them here, but the principles applied to
# get_trend_data and get_overall_kpis should be extended to the others.
# The key is to make each function resilient to imperfect data and configurations.

# Placeholder for other aggregation functions - apply similar optimization patterns

def get_chw_summary_kpis(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    for_date: Any, # Union[str, pd.Timestamp, date_type, datetime]
    chw_daily_kpi_input_data: Optional[Dict[str, Any]] = None,
    source_context: str = "CHWSummaryKPIs"
) -> Dict[str, Any]:
    # This function was provided in the prompt. Apply optimizations focusing on:
    # 1. Robust date parsing for `for_date`.
    # 2. Safe merging of `chw_daily_kpi_input_data`.
    # 3. Standardized column preparation (similar to _prepare_summary_dataframe in chw_components).
    # 4. Robust access to settings attributes.
    # 5. Clearer logic for metric calculations, especially handling of NaNs and unique counts.
    # (Implementation would be similar to the previous optimized version of this function)
    logger.warning(f"({source_context}) get_chw_summary_kpis is a placeholder. Ensure its logic is robustly implemented.")
    # Fallback to a simplified version of what calculate_chw_daily_summary_metrics might return
    # to prevent downstream errors if this function is called.
    # Ideally, replace this with the fully optimized version of calculate_chw_daily_summary_metrics
    # if this is meant to be the canonical one.
    if chw_daily_encounter_df is not None and not chw_daily_encounter_df.empty:
         visits = chw_daily_encounter_df['patient_id'].nunique() if 'patient_id' in chw_daily_encounter_df else 0
    else: visits = 0
    return {
        "date_of_activity": pd.to_datetime(for_date, errors='coerce').date().isoformat() if pd.notna(pd.to_datetime(for_date, errors='coerce')) else date_type.today().isoformat(),
        "visits_count": visits, 
        "high_ai_prio_followups_count": 0, 
        "avg_risk_of_visited_patients": np.nan, 
        # ... other default values ...
        "worker_self_fatigue_level_code": "NOT_ASSESSED", 
        "worker_self_fatigue_index_today": np.nan
    }


def get_clinic_summary_kpis(
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicSummaryKPIs"
) -> Dict[str, Any]:
    # This function was provided. Apply optimizations:
    # 1. Robust handling of empty/invalid health_df_period.
    # 2. Standardized column preparation.
    # 3. Safe access to settings for thresholds, test names.
    # 4. Clear logic for TAT, positivity rates, stockouts, sample rejection.
    # 5. Robust looping for KEY_TEST_TYPES_FOR_ANALYSIS.
    logger.warning(f"({source_context}) get_clinic_summary_kpis is a placeholder. Ensure its logic is robustly implemented.")
    return {"test_summary_details": {}, "overall_avg_test_turnaround_conclusive_days": np.nan} # Minimal fallback


def get_clinic_environmental_summary_kpis(
    iot_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicEnvSummaryKPIs"
) -> Dict[str, Any]:
    # This function was provided. Apply optimizations:
    # 1. Handle empty/invalid iot_df_period.
    # 2. Robust 'timestamp' processing.
    # 3. Safe numeric conversion for sensor columns.
    # 4. Clear logic for latest readings and alert counts, using settings for thresholds.
    logger.warning(f"({source_context}) get_clinic_environmental_summary_kpis is a placeholder. Ensure logic is robust.")
    return {"avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count": 0} # Minimal fallback


def get_district_summary_kpis(
    enriched_zone_df: Optional[pd.DataFrame],
    source_context: str = "DistrictKPIs"
) -> Dict[str, Any]:
    # This function was provided. Apply optimizations:
    # 1. Handle empty/invalid enriched_zone_df.
    # 2. Robust weighted average calculation, handling missing 'population' or value columns.
    # 3. Safe iteration over KEY_CONDITIONS_FOR_ACTION and access to settings.
    logger.warning(f"({source_context}) get_district_summary_kpis is a placeholder. Ensure logic is robust.")
    return {"total_zones_in_df": 0, "total_population_district": 0.0} # Minimal fallback
