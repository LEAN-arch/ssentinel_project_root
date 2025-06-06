# sentinel_project_root/pages/chw_components/activity_trends.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union 
from datetime import date as date_type, datetime 

try:
    from config import settings
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric 
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__) 
    logger_init.error(f"Critical import error in activity_trends.py: {e}. Ensure paths are correct.")
    raise

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

def calculate_chw_activity_trends_data(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Union[str, pd.Timestamp, date_type, datetime],
    trend_end_date_input: Union[str, pd.Timestamp, date_type, datetime],
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D', 
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]:
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
    }
    empty_series = pd.Series(dtype='float64') # For consistent empty returns

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) Input chw_historical_health_df is empty or invalid. Trend calculation skipped.")
        return trends_output
    
    valid_agg_periods = ['D', 'B', 'W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 
                         'SM', 'SMS', 'M', 'MS', 'Q', 'QS', 'A', 'AS', 'Y', 'YS', 'H', 'T', 'MIN', 'S', 'L', 'MS', 'US', 'NS']
    if time_period_aggregation.upper() not in valid_agg_periods:
        logger.error(f"({source_context_log_prefix}) Invalid time_period_aggregation: '{time_period_aggregation}'.")
        return trends_output

    try:
        start_date_dt = pd.to_datetime(trend_start_date_input, errors='coerce')
        end_date_dt = pd.to_datetime(trend_end_date_input, errors='coerce')
        if pd.NaT in [start_date_dt, end_date_dt]: raise ValueError("Unparseable date inputs.")
        start_date = start_date_dt.date() if isinstance(start_date_dt, pd.Timestamp) else start_date_dt
        end_date = end_date_dt.date() if isinstance(end_date_dt, pd.Timestamp) else end_date_dt
        if start_date > end_date: start_date, end_date = end_date, start_date
    except Exception as e_date:
        logger.error(f"({source_context_log_prefix}) Invalid date inputs: {e_date}. Start: '{trend_start_date_input}', End: '{trend_end_date_input}'.", exc_info=True)
        return trends_output
        
    logger.info(f"({source_context_log_prefix}) Calculating trends: {start_date.isoformat()} to {end_date.isoformat()}, Zone: {zone_filter or 'All'}, Agg: {time_period_aggregation}")
    
    df_processed = chw_historical_health_df.copy()
    required_date_col = 'encounter_date'
    if required_date_col not in df_processed.columns:
        logger.error(f"({source_context_log_prefix}) Missing '{required_date_col}'. Cannot calculate trends.")
        return trends_output
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_processed[required_date_col]):
            df_processed[required_date_col] = pd.to_datetime(df_processed[required_date_col], errors='coerce')
        if df_processed[required_date_col].dt.tz is not None:
            df_processed[required_date_col] = df_processed[required_date_col].dt.tz_localize(None)
        df_processed.dropna(subset=[required_date_col], inplace=True)
    except Exception as e_date_col_proc:
        logger.error(f"({source_context_log_prefix}) Error processing '{required_date_col}': {e_date_col_proc}.", exc_info=True)
        return trends_output
    
    if df_processed.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning.")
        return trends_output

    df_period_filtered = df_processed[
        (df_processed[required_date_col].dt.date >= start_date) &
        (df_processed[required_date_col].dt.date <= end_date)
    ]
    logger.debug(f"({source_context_log_prefix}) Shape after date range filter: {df_period_filtered.shape}")

    if zone_filter:
        if 'zone_id' in df_period_filtered.columns:
            df_period_filtered = df_period_filtered[df_period_filtered['zone_id'].astype(str) == str(zone_filter)]
            logger.debug(f"({source_context_log_prefix}) Shape after zone filter '{zone_filter}': {df_period_filtered.shape}")
        else:
            logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter.")

    if df_period_filtered.empty:
        logger.info(f"({source_context_log_prefix}) No data after period/zone filtering. Trend Series will be empty.")
        # trends_output will retain its initial None/empty Series values
        return trends_output

    patient_id_col = 'patient_id'
    if patient_id_col in df_period_filtered.columns:
        try:
            logger.debug(f"({source_context_log_prefix}/PatientVisits) Calculating trend. Input DF shape: {df_period_filtered.shape}")
            visits_trend_series = get_trend_data(
                df=df_period_filtered, value_col=patient_id_col, date_col=required_date_col,
                period=time_period_aggregation, agg_func='nunique',
                source_context=f"{source_context_log_prefix}/PatientVisits"
            )
            logger.debug(f"({source_context_log_prefix}/PatientVisits) Result from get_trend_data (type: {type(visits_trend_series)}, empty: {visits_trend_series.empty if isinstance(visits_trend_series, pd.Series) else 'N/A'}):\n{visits_trend_series}")
            if isinstance(visits_trend_series, pd.Series) and not visits_trend_series.empty:
                trends_output["patient_visits_trend"] = visits_trend_series.rename("unique_patient_visits_count")
        except Exception as e_pv_trend:
            logger.error(f"({source_context_log_prefix}) Error calculating patient visits trend: {e_pv_trend}", exc_info=True)
    else:
        logger.warning(f"({source_context_log_prefix}) '{patient_id_col}' missing. Cannot calculate patient visits trend.")

    prio_score_col = 'ai_followup_priority_score'
    if patient_id_col in df_period_filtered.columns and prio_score_col in df_period_filtered.columns:
        try:
            df_prio_analysis = df_period_filtered[[patient_id_col, required_date_col, prio_score_col]].copy()
            df_prio_analysis[prio_score_col] = convert_to_numeric(df_prio_analysis[prio_score_col], default_value=np.nan)
            df_prio_analysis.dropna(subset=[prio_score_col], inplace=True)
            
            high_prio_threshold = _get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 0.7) * 100 # Assuming 0-100 scale if setting is 0-1
            if _get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 0.7) > 1: # if setting is already 0-100 scale
                 high_prio_threshold = _get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 70)

            df_high_prio_encounters = df_prio_analysis[df_prio_analysis[prio_score_col] >= high_prio_threshold]
            logger.debug(f"({source_context_log_prefix}/HighPrio) Num high priority encounters (score >= {high_prio_threshold}): {len(df_high_prio_encounters)}")
            
            if not df_high_prio_encounters.empty:
                high_prio_trend_series = get_trend_data(
                    df=df_high_prio_encounters, value_col=patient_id_col, date_col=required_date_col,
                    period=time_period_aggregation, agg_func='nunique',
                    source_context=f"{source_context_log_prefix}/HighPrioFollowups"
                )
                logger.debug(f"({source_context_log_prefix}/HighPrio) Result from get_trend_data (type: {type(high_prio_trend_series)}, empty: {high_prio_trend_series.empty if isinstance(high_prio_trend_series, pd.Series) else 'N/A'}):\n{high_prio_trend_series}")
                if isinstance(high_prio_trend_series, pd.Series) and not high_prio_trend_series.empty:
                    trends_output["high_priority_followups_trend"] = high_prio_trend_series.rename("high_priority_followups_count")
            else:
                logger.info(f"({source_context_log_prefix}) No high priority encounters met criteria for trend.")
        except Exception as e_hp_trend:
            logger.error(f"({source_context_log_prefix}) Error calculating high priority followups trend: {e_hp_trend}", exc_info=True)
    else:
        missing_prio_cols = [col for col in [patient_id_col, prio_score_col] if col not in df_period_filtered.columns]
        if missing_prio_cols: logger.warning(f"({source_context_log_prefix}) Missing columns for high prio trend: {missing_prio_cols}.")

    num_gen = sum(1 for ts in trends_output.values() if isinstance(ts, pd.Series) and not ts.empty)
    logger.info(f"({source_context_log_prefix}) Trends calculation complete. {num_gen} trend(s) generated.")
    return trends_output
