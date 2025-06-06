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
    # Define FallbackSettings if settings are absolutely critical for this module to even load
    class FallbackSettings: # Minimal fallback
        KEY_CONDITIONS_FOR_ACTION = ["Malaria", "TB", "Pneumonia"] # Example
        MALARIA_RDT_TEST_NAME_IDENTIFIER = "RDT-Malaria"
        NON_CONCLUSIVE_TEST_RESULTS = ['pending', 'rejected', 'unknown', 'n/a', 'indeterminate', 'invalid']
        KEY_DRUG_SUBSTRINGS_SUPPLY = ["ACT", "Amox", "ORS"]
        CRITICAL_SUPPLY_DAYS_REMAINING = 7
        DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 60.0
        # Add other critical settings used directly in this file if any
    settings = FallbackSettings()
    logger_init.warning("aggregation.py: Using fallback settings due to import error with 'config.settings'.")
    # raise # Or re-raise if module cannot function without real settings

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
                # Make comparison type-robust
                col_dtype = df_trend[filter_col].dtype
                if pd.api.types.is_numeric_dtype(col_dtype) and isinstance(filter_val, (int, float)):
                    df_trend = df_trend[df_trend[filter_col] == filter_val]
                elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                     filter_val_dt = pd.to_datetime(filter_val, errors='coerce')
                     if pd.notna(filter_val_dt):
                         df_trend = df_trend[df_trend[filter_col] == filter_val_dt]
                     else: logger.warning(f"({source_context}) Invalid filter_val '{filter_val}' for datetime column '{filter_col}'.")
                else: # Fallback to string comparison
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
        # For 'nunique', conversion can be helpful if IDs are numeric but stored as objects
        if isinstance(agg_func, str) and (agg_func in numeric_agg_functions or agg_func == 'nunique'):
            if value_col in df_trend.columns: # Ensure value_col exists before conversion
                df_trend[value_col] = convert_to_numeric(df_trend[value_col], default_value=np.nan)
                # For most numeric aggs (not nunique/count/size), drop rows where value became NaN
                if agg_func not in ['nunique', 'count', 'size']:
                    df_trend.dropna(subset=[value_col], inplace=True)
            elif not is_row_count_agg: # Value col missing and not a row count agg
                 logger.error(f"({source_context}) Value column '{value_col}' missing, cannot perform '{agg_func}'.")
                 return empty_series

        if df_trend.empty and not is_row_count_agg :
            logger.info(f"({source_context}) DataFrame empty after numeric conversion/dropna of '{value_col}' for trend.")
            return empty_series
        
        resampler = df_trend.set_index(date_col).resample(period)
        
        if is_row_count_agg and agg_func == 'size': 
            trend_series = resampler.size()
        elif value_col in df_trend.columns: # Ensure value_col exists for other aggs
            trend_series = resampler[value_col].agg(agg_func)
        else: # Should be caught earlier, but as a safeguard
             logger.error(f"({source_context}) Fallback: Value column '{value_col}' not found for aggregation '{agg_func}'.")
             return empty_series

        count_like_aggs = ['count', 'nunique', 'size']
        if isinstance(agg_func, str) and agg_func in count_like_aggs:
            trend_series = trend_series.fillna(0)
            try:
                # Attempt to convert to Int64 (nullable int) first, then standard int if no NaNs
                trend_series = trend_series.astype(pd.Int64Dtype()) 
                if not trend_series.isnull().any():
                    trend_series = trend_series.astype(int)
            except Exception: # Fallback if Int64Dtype or int conversion fails
                logger.debug(f"({source_context}) Could not convert count-like trend for '{value_col}' to integer type. Kept as {trend_series.dtype}.")
        
        # Ensure Series has a name
        if isinstance(trend_series, pd.Series) and not trend_series.name:
            trend_series.name = value_col if isinstance(value_col, str) else "trend_value"

        logger.debug(f"({source_context}) Trend for '{value_col}' generated, {len(trend_series)} points. Sample:\n{trend_series.head().to_string() if not trend_series.empty else 'Empty Series'}")
        return trend_series
    except Exception as e_agg_trend:
        logger.error(f"({source_context}) Error during resampling/aggregation for '{value_col}': {e_agg_trend}", exc_info=True)
        return empty_series


def get_overall_kpis(
    health_df: Optional[pd.DataFrame],
    date_filter_start: Optional[Union[str, pd.Timestamp, date_type, datetime]] = None,
    date_filter_end: Optional[Union[str, pd.Timestamp, date_type, datetime]] = None,
    source_context: str = "GlobalKPIs"
) -> Dict[str, Any]:
    """Calculates overall key performance indicators from health data for a given period."""
    
    start_date_str = str(date_filter_start.date() if hasattr(date_filter_start, 'date') else date_filter_start) if date_filter_start else "N/A"
    end_date_str = str(date_filter_end.date() if hasattr(date_filter_end, 'date') else date_filter_end) if date_filter_end else "N/A"
    logger.info(f"({source_context}) Calculating overall KPIs for period: {start_date_str} to {end_date_str}")

    kpis: Dict[str, Any] = {
        "total_patients_period": 0, "avg_patient_ai_risk_period": np.nan,
        "malaria_rdt_positive_rate_period": np.nan, "key_supply_stockout_alerts_period": 0,
        "total_encounters_period": 0
    }
    key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    if not isinstance(key_conditions, list): key_conditions = [] # Ensure it's a list

    for condition_name in key_conditions:
        if isinstance(condition_name, str) and condition_name.strip():
            kpi_key_safe = f"active_{re.sub(r'[^a-z0-9_]+', '_', condition_name.lower().strip())}_cases_period"
            kpis[kpi_key_safe] = 0
        else:
            logger.warning(f"({source_context}) Invalid condition name found in KEY_CONDITIONS_FOR_ACTION: {condition_name}")


    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        logger.warning(f"({source_context}) Health DataFrame is empty or invalid. Returning default KPIs.")
        return kpis

    df = health_df.copy() 
    date_col_kpi = 'encounter_date'
    if date_col_kpi not in df.columns:
        logger.error(f"({source_context}) '{date_col_kpi}' column missing. KPIs calculations will be inaccurate or fail.")
        return kpis 
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col_kpi]):
            df[date_col_kpi] = pd.to_datetime(df[date_col_kpi], errors='coerce')
        if df[date_col_kpi].dt.tz is not None:
            df[date_col_kpi] = df[date_col_kpi].dt.tz_localize(None)
        df.dropna(subset=[date_col_kpi], inplace=True)

        if date_filter_start:
            start_dt_filter = pd.to_datetime(date_filter_start, errors='coerce').normalize() 
            if pd.notna(start_dt_filter): df = df[df[date_col_kpi] >= start_dt_filter]
            else: logger.warning(f"({source_context}) Invalid start_date_filter '{date_filter_start}' for KPIs.")
        if date_filter_end:
            end_dt_filter = pd.to_datetime(date_filter_end, errors='coerce').normalize() 
            if pd.notna(end_dt_filter): df = df[df[date_col_kpi] < (end_dt_filter + pd.Timedelta(days=1))] # Inclusive of end date
            else: logger.warning(f"({source_context}) Invalid end_date_filter '{date_filter_end}' for KPIs.")
    except Exception as e_date_filter_kpi:
        logger.warning(f"({source_context}) Error applying date filters for KPIs: {e_date_filter_kpi}. Proceeding with potentially broader data.", exc_info=True)

    if df.empty:
        logger.info(f"({source_context}) No data remains after date filtering for KPIs. Returning default KPIs.")
        return kpis

    if 'patient_id' in df.columns: 
        kpis["total_patients_period"] = df['patient_id'].nunique()
    else: logger.warning(f"({source_context}) 'patient_id' missing. total_patients_period KPI affected.")
    
    encounter_id_col_kpi = 'encounter_id' if 'encounter_id' in df.columns else ('patient_id' if 'patient_id' in df.columns else None)
    if encounter_id_col_kpi:
        kpis["total_encounters_period"] = df[encounter_id_col_kpi].nunique()
    else: logger.warning(f"({source_context}) 'encounter_id'/'patient_id' missing. total_encounters_period KPI affected.")

    if 'ai_risk_score' in df.columns:
        risk_scores_series_kpi = convert_to_numeric(df['ai_risk_score'], default_value=np.nan)
        if risk_scores_series_kpi.notna().any():
            kpis["avg_patient_ai_risk_period"] = risk_scores_series_kpi.mean()
    else: logger.debug(f"({source_context}) 'ai_risk_score' column missing for KPIs.")

    if 'condition' in df.columns and 'patient_id' in df.columns and key_conditions:
        for condition_name_iter_kpi in key_conditions:
            if isinstance(condition_name_iter_kpi, str) and condition_name_iter_kpi.strip():
                kpi_key_cond_dyn = f"active_{re.sub(r'[^a-z0-9_]+', '_', condition_name_iter_kpi.lower().strip())}_cases_period"
                try: 
                    condition_mask_kpi = df['condition'].astype(str).str.contains(re.escape(condition_name_iter_kpi), case=False, na=False, regex=True)
                    kpis[kpi_key_cond_dyn] = df.loc[condition_mask_kpi, 'patient_id'].nunique() if condition_mask_kpi.any() else 0
                except Exception as e_cond_regex_kpi:
                    logger.warning(f"({source_context}) Regex error for KPI condition '{condition_name_iter_kpi}': {e_cond_regex_kpi}. Count set to 0.")
                    kpis[kpi_key_cond_dyn] = 0
    elif not key_conditions:
        logger.debug(f"({source_context}) No KEY_CONDITIONS_FOR_ACTION defined for active cases KPIs.")

    if 'test_type' in df.columns and 'test_result' in df.columns:
        malaria_rdt_id = _get_setting('MALARIA_RDT_TEST_NAME_IDENTIFIER', "RDT-Malaria")
        malaria_tests_df_kpi = df[df['test_type'].astype(str).str.contains(re.escape(malaria_rdt_id), case=False, na=False)]
        if not malaria_tests_df_kpi.empty:
            non_conclusive_list = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', ['pending', 'rejected', 'unknown', 'n/a', 'indeterminate', 'invalid'])
            conclusive_malaria_tests_kpi = malaria_tests_df_kpi[
                ~malaria_tests_df_kpi['test_result'].astype(str).str.lower().isin(non_conclusive_list)
            ]
            if not conclusive_malaria_tests_kpi.empty:
                positive_malaria_count_kpi = conclusive_malaria_tests_kpi[
                    conclusive_malaria_tests_kpi['test_result'].astype(str).str.lower() == 'positive'
                ].shape[0]
                kpis["malaria_rdt_positive_rate_period"] = (positive_malaria_count_kpi / len(conclusive_malaria_tests_kpi)) * 100
            else: logger.debug(f"({source_context}) No conclusive '{malaria_rdt_id}' results found for positivity rate.")
        else: logger.debug(f"({source_context}) No '{malaria_rdt_id}' tests found for positivity rate.")
    else: logger.debug(f"({source_context}) 'test_type' or 'test_result' columns missing for Malaria RDT KPI.")
    
    supply_cols_kpi = ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', date_col_kpi, 'zone_id']
    if all(col in df.columns for col in supply_cols_kpi):
        try:
            latest_stock_df_kpi = df.sort_values(date_col_kpi, na_position='first').drop_duplicates(subset=['item', 'zone_id'], keep='last').copy()
            latest_stock_df_kpi['consumption_rate_per_day'] = convert_to_numeric(latest_stock_df_kpi['consumption_rate_per_day'], default_value=0.001)
            latest_stock_df_kpi.loc[latest_stock_df_kpi['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001 
            latest_stock_df_kpi['item_stock_agg_zone'] = convert_to_numeric(latest_stock_df_kpi['item_stock_agg_zone'], default_value=0.0)
            latest_stock_df_kpi['days_of_supply'] = latest_stock_df_kpi['item_stock_agg_zone'] / latest_stock_df_kpi['consumption_rate_per_day']
            
            key_drug_subs_list = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
            critical_supply_days_val = _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
            
            if key_drug_subs_list:
                key_drugs_pattern_kpi = '|'.join(re.escape(s.strip()) for s in key_drug_subs_list if s.strip())
                if key_drugs_pattern_kpi: # Ensure pattern is not empty
                    key_drugs_df_kpi = latest_stock_df_kpi[
                        latest_stock_df_kpi['item'].astype(str).str.contains(key_drugs_pattern_kpi, case=False, na=False, regex=True)
                    ]
                    if not key_drugs_df_kpi.empty:
                        kpis['key_supply_stockout_alerts_period'] = key_drugs_df_kpi[
                            key_drugs_df_kpi['days_of_supply'] < critical_supply_days_val
                        ]['item'].nunique() 
            else: logger.debug(f"({source_context}) KEY_DRUG_SUBSTRINGS_SUPPLY not defined or empty for stockout KPI.")
        except Exception as e_supply_kpi_calc:
            logger.error(f"({source_context}) Error calculating supply stockout KPI: {e_supply_kpi_calc}", exc_info=True)
    else: 
        missing_supply_cols_kpi = [col for col in supply_cols_kpi if col not in df.columns]
        if missing_supply_cols_kpi: logger.debug(f"({source_context}) Missing supply-related columns for KPIs: {missing_supply_cols_kpi}.")
            
    logger.info(f"({source_context}) Overall KPIs calculated: { {k: (f'{v:.2f}' if isinstance(v, float) and pd.notna(v) else v) for k,v in kpis.items()} }")
    return kpis


# --- Placeholder definitions for other KPI functions ---
# These should be implemented with similar robustness patterns as get_overall_kpis and get_trend_data

def get_chw_summary_kpis(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    for_date: Any, 
    chw_daily_kpi_input_data: Optional[Dict[str, Any]] = None,
    source_context: str = "CHWSummaryKPIs"
) -> Dict[str, Any]:
    logger.warning(f"({source_context}) get_chw_summary_kpis is a placeholder in aggregation.py. Ensure actual implementation from chw_components is used or this is fully built out.")
    # Return a structure that matches what chw_dashboard expects to avoid TypeErrors
    target_date_iso = pd.to_datetime(for_date, errors='coerce').date().isoformat() if pd.notna(pd.to_datetime(for_date, errors='coerce')) else date_type.today().isoformat()
    return {
        "date_of_activity": target_date_iso, "visits_count": 0, 
        "high_ai_prio_followups_count": 0, "avg_risk_of_visited_patients": np.nan, 
        "fever_cases_identified_count": 0, "high_fever_cases_identified_count": 0, 
        "critical_spo2_cases_identified_count": 0, "avg_steps_of_visited_patients": np.nan, 
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", 
        "worker_self_fatigue_index_today": np.nan
    }

def get_clinic_summary_kpis(
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicSummaryKPIs"
) -> Dict[str, Any]:
    logger.warning(f"({source_context}) get_clinic_summary_kpis is a placeholder in aggregation.py. Ensure actual implementation is robust.")
    return {
        "test_summary_details": {}, # Expected nested dict
        "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests_patients": 0,
        "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0
    }

def get_clinic_environmental_summary_kpis(
    iot_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicEnvSummaryKPIs"
) -> Dict[str, Any]:
    logger.warning(f"({source_context}) get_clinic_environmental_summary_kpis is a placeholder in aggregation.py. Ensure actual implementation is robust.")
    return {
        "avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count": 0,
        "rooms_co2_high_alert_latest_count":0, "avg_pm25_overall_ugm3": np.nan,
        "rooms_pm25_very_high_alert_latest_count": 0, "rooms_pm25_high_alert_latest_count":0,
        "avg_waiting_room_occupancy_overall_persons": np.nan,
        "waiting_room_high_occupancy_alert_latest_flag": False,
        "avg_noise_overall_dba": np.nan, "rooms_noise_high_alert_latest_count": 0,
        "latest_readings_timestamp": None
    }

def get_district_summary_kpis(
    enriched_zone_df: Optional[pd.DataFrame],
    source_context: str = "DistrictKPIs"
) -> Dict[str, Any]:
    logger.warning(f"({source_context}) get_district_summary_kpis is a placeholder in aggregation.py. Ensure actual implementation is robust.")
    return {
        "total_zones_in_df": 0, "total_population_district": 0.0,
        "population_weighted_avg_ai_risk_score": np.nan,
        # Add other keys with default values as expected by the calling page
    }
