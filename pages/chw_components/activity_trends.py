# sentinel_project_root/pages/chw_components/activity_trends.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union # Added Union for date inputs
from datetime import date as date_type, datetime # For type hinting and conversion

# Assuming settings and helpers are in the correct path relative to this file or project root is in sys.path
try:
    from config import settings
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric # Ensure this function is robust
except ImportError as e:
    logger = logging.getLogger(__name__) # Basic logger if import fails early
    logger.error(f"Critical import error in activity_trends.py: {e}. Ensure paths are correct.")
    # Consider re-raising or exiting if these are absolutely critical for the module to function
    raise

logger = logging.getLogger(__name__)


def calculate_chw_activity_trends_data(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Union[str, pd.Timestamp, date_type, datetime],
    trend_end_date_input: Union[str, pd.Timestamp, date_type, datetime],
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D', # Default to Daily ('D')
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]:
    """
    Calculates CHW activity trends over a specified period.

    Args:
        chw_historical_health_df: DataFrame with historical health records.
                                  Expected columns: 'encounter_date', 'patient_id', 'zone_id' (optional),
                                                    'ai_followup_priority_score' (optional).
        trend_start_date_input: Start date for trend analysis (str, Timestamp, date, or datetime).
        trend_end_date_input: End date for trend analysis (str, Timestamp, date, or datetime).
        zone_filter: Optional zone to filter by.
        time_period_aggregation: Aggregation period (e.g., 'D', 'W-MON').
        source_context_log_prefix: Logging prefix.

    Returns:
        A dictionary where keys are trend names (e.g., "patient_visits_trend")
        and values are pandas Series representing the trend, or None if calculation failed or no data.
    """
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
        # Add other trend keys here if more are calculated
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided. Trend calculation skipped.")
        return trends_output
    
    # Validate time_period_aggregation early
    # List of common Pandas frequency aliases. This can be expanded if needed.
    valid_agg_periods = ['D', 'B', 'W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 
                         'SM', 'SMS', 'M', 'MS', 'Q', 'QS', 'A', 'AS', 'Y', 'YS', 'H', 'T', 'MIN', 'S', 'L', 'MS', 'US', 'NS']
    if time_period_aggregation.upper() not in valid_agg_periods: # Using upper() for case-insensitivity of input
        logger.error(f"({source_context_log_prefix}) Invalid time_period_aggregation: '{time_period_aggregation}'. Must be a valid Pandas offset alias.")
        return trends_output

    # Convert and validate date inputs robustly
    try:
        # pd.to_datetime is versatile. errors='coerce' will turn unparseable dates into NaT.
        start_date_dt = pd.to_datetime(trend_start_date_input, errors='coerce')
        end_date_dt = pd.to_datetime(trend_end_date_input, errors='coerce')

        if pd.NaT in [start_date_dt, end_date_dt]: # Check for NaT (Not a Time)
            raise ValueError("One or both date inputs are unparseable.")
        
        # Convert to date objects for comparison if they are datetimes
        start_date = start_date_dt.date() if isinstance(start_date_dt, pd.Timestamp) else start_date_dt
        end_date = end_date_dt.date() if isinstance(end_date_dt, pd.Timestamp) else end_date_dt

        if start_date > end_date:
            logger.warning(f"({source_context_log_prefix}) Start date ({start_date}) is after end date ({end_date}). Swapping dates.")
            start_date, end_date = end_date, start_date # Swap them for user convenience
            
    except Exception as e_date:
        logger.error(f"({source_context_log_prefix}) Invalid date inputs for trend calculation: {e_date}. "
                     f"Start: '{trend_start_date_input}', End: '{trend_end_date_input}'. Processing stopped.", exc_info=True)
        return trends_output
        
    logger.info(
        f"({source_context_log_prefix}) Calculating trends for period: {start_date.isoformat()} to {end_date.isoformat()}, "
        f"Zone: {zone_filter or 'All'}, Aggregation: {time_period_aggregation}"
    )
    
    # Work on a copy only if modifications are certain or extensive.
    # For filtering, direct slicing is often more memory-efficient if original df is not needed later.
    # However, given subsequent operations, a copy might be safer to avoid SettingWithCopyWarning.
    df_processed = chw_historical_health_df.copy()

    # --- Data Preparation and Validation ---
    required_date_col = 'encounter_date'
    if required_date_col not in df_processed.columns:
        logger.error(f"({source_context_log_prefix}) Critical column '{required_date_col}' missing. Cannot calculate trends.")
        return trends_output
    
    try:
        # Ensure encounter_date is datetime and timezone-naive
        if not pd.api.types.is_datetime64_any_dtype(df_processed[required_date_col]):
            df_processed[required_date_col] = pd.to_datetime(df_processed[required_date_col], errors='coerce')
        if df_processed[required_date_col].dt.tz is not None:
            df_processed[required_date_col] = df_processed[required_date_col].dt.tz_localize(None)
        
        df_processed.dropna(subset=[required_date_col], inplace=True) # Remove rows where date conversion failed
    except Exception as e_date_col:
        logger.error(f"({source_context_log_prefix}) Error processing '{required_date_col}': {e_date_col}. Trends aborted.", exc_info=True)
        return trends_output
    
    if df_processed.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning. Trends cannot be calculated.")
        return trends_output

    # --- Filter by Date Range and Zone ---
    # Using .dt.date for comparison ensures we compare dates with dates, not datetimes with dates.
    df_period_filtered = df_processed[
        (df_processed[required_date_col].dt.date >= start_date) &
        (df_processed[required_date_col].dt.date <= end_date)
    ]

    if zone_filter:
        if 'zone_id' in df_period_filtered.columns:
            # Ensure consistent type for comparison if zone_filter might be int/str and column is int/str
            df_period_filtered = df_period_filtered[df_period_filtered['zone_id'].astype(str) == str(zone_filter)]
        else:
            logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter '{zone_filter}'. Processing for all zones.")

    if df_period_filtered.empty:
        logger.info(f"({source_context_log_prefix}) No data remains for the specified trend period/zone after filtering.")
        return trends_output

    # --- Calculate Patient Visits Trend ---
    patient_id_col = 'patient_id'
    if patient_id_col in df_period_filtered.columns:
        try:
            visits_trend = get_trend_data(
                df=df_period_filtered, 
                value_col=patient_id_col, 
                date_col=required_date_col,
                period=time_period_aggregation, 
                agg_func='nunique', # Count unique patients
                source_context=f"{source_context_log_prefix}/PatientVisits"
            )
            if isinstance(visits_trend, pd.Series) and not visits_trend.empty:
                trends_output["patient_visits_trend"] = visits_trend.rename("unique_patient_visits_count")
            elif isinstance(visits_trend, pd.Series): # Empty series
                logger.info(f"({source_context_log_prefix}) Patient visits trend resulted in an empty series.")
            else: # Not a series
                logger.warning(f"({source_context_log_prefix}) get_trend_data for patient visits did not return a Series.")
        except Exception as e_visits_trend:
            logger.error(f"({source_context_log_prefix}) Error calculating patient visits trend: {e_visits_trend}", exc_info=True)
    else:
        logger.warning(f"({source_context_log_prefix}) '{patient_id_col}' column missing. Cannot calculate patient visits trend.")

    # --- Calculate High Priority Follow-ups Trend ---
    prio_score_col = 'ai_followup_priority_score'
    # Check for both patient_id (for nunique) and the priority score column
    if patient_id_col in df_period_filtered.columns and prio_score_col in df_period_filtered.columns:
        try:
            # Create a working copy for priority calculation if modifications are needed
            df_prio_analysis = df_period_filtered[[patient_id_col, required_date_col, prio_score_col]].copy()
            
            # Ensure priority score is numeric, defaulting non-convertible values to 0 or NaN.
            # Using 0 might be okay if scores are non-negative; NaN might be better if 0 is a valid low score.
            df_prio_analysis[prio_score_col] = convert_to_numeric(
                df_prio_analysis[prio_score_col], default_value=np.nan # Use NaN to exclude non-numeric from threshold check
            )
            df_prio_analysis.dropna(subset=[prio_score_col], inplace=True) # Remove rows where conversion failed to NaN

            high_prio_threshold = settings.FATIGUE_INDEX_HIGH_THRESHOLD if hasattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD') else 0.7 # Default example
            
            df_high_prio_encounters = df_prio_analysis[df_prio_analysis[prio_score_col] >= high_prio_threshold]
            
            if not df_high_prio_encounters.empty:
                high_prio_trend = get_trend_data(
                    df=df_high_prio_encounters, 
                    value_col=patient_id_col, 
                    date_col=required_date_col,
                    period=time_period_aggregation, 
                    agg_func='nunique', # Count unique patients needing high-priority follow-up
                    source_context=f"{source_context_log_prefix}/HighPrioFollowups"
                )
                if isinstance(high_prio_trend, pd.Series) and not high_prio_trend.empty:
                    trends_output["high_priority_followups_trend"] = high_prio_trend.rename("high_priority_followups_count")
                elif isinstance(high_prio_trend, pd.Series):
                     logger.info(f"({source_context_log_prefix}) High priority follow-ups trend resulted in an empty series.")
                else:
                    logger.warning(f"({source_context_log_prefix}) get_trend_data for high priority follow-ups did not return a Series.")
            else:
                logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria (≥{high_prio_threshold}) for trend calculation.")
        except Exception as e_prio_trend:
            logger.error(f"({source_context_log_prefix}) Error calculating high priority follow-ups trend: {e_prio_trend}", exc_info=True)
    else:
        missing_cols_prio = [col for col in [patient_id_col, prio_score_col] if col not in df_period_filtered.columns]
        if missing_cols_prio:
            logger.warning(f"({source_context_log_prefix}) Missing columns for high priority follow-ups trend: {missing_cols_prio}.")

    num_generated_trends = sum(1 for trend_series in trends_output.values() if isinstance(trend_series, pd.Series) and not trend_series.empty)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {num_generated_trends} trend(s) generated.")
    return trends_output
