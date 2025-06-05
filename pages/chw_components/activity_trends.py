# sentinel_project_root/pages/chw_components/activity_trends.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import date as date_type # For type hinting clarity

from config import settings
from data_processing.aggregation import get_trend_data
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def calculate_chw_activity_trends_data(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Any, # Can be date, str, pd.Timestamp
    trend_end_date_input: Any,   # Can be date, str, pd.Timestamp
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D', # Default to Daily ('D')
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]:
    """
    Calculates CHW activity trends over a specified period.

    Args:
        chw_historical_health_df: DataFrame with historical health records.
        trend_start_date_input: Start date for trend analysis.
        trend_end_date_input: End date for trend analysis.
        zone_filter: Optional zone to filter by.
        time_period_aggregation: Aggregation period ('D', 'W-Mon', etc.).
        source_context_log_prefix: Logging prefix.

    Returns:
        Dict with trend Series (e.g., "patient_visits_trend") or None if calculation failed.
    """
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided. Trend calculation skipped.")
        return trends_output
    
    valid_agg_periods = ['D', 'B', 'W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'M', 'MS', 'Q', 'QS', 'A', 'AS']
    if time_period_aggregation.upper() not in valid_agg_periods:
        logger.error(f"({source_context_log_prefix}) Invalid time_period_aggregation: '{time_period_aggregation}'.")
        return trends_output

    try:
        start_date = pd.to_datetime(trend_start_date_input, errors='coerce').date()
        end_date = pd.to_datetime(trend_end_date_input, errors='coerce').date()
        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
            raise ValueError("Invalid trend date period after conversion or start > end.")
    except Exception as e_date:
        logger.error(f"({source_context_log_prefix}) Invalid date inputs for trend: {e_date}. Start: '{trend_start_date_input}', End: '{trend_end_date_input}'.")
        return trends_output
        
    logger.info(
        f"({source_context_log_prefix}) Calculating trends for: {start_date.isoformat()} to {end_date.isoformat()}, "
        f"Zone: {zone_filter or 'All'}, Agg: {time_period_aggregation}"
    )
    
    df_trends = chw_historical_health_df.copy()

    # Data Preparation
    if 'encounter_date' not in df_trends.columns:
        logger.error(f"({source_context_log_prefix}) 'encounter_date' column missing. Cannot calculate trends.")
        return trends_output
    try:
        df_trends['encounter_date'] = pd.to_datetime(df_trends['encounter_date'], errors='coerce')
        df_trends.dropna(subset=['encounter_date'], inplace=True)
    except Exception as e_date_col:
        logger.error(f"({source_context_log_prefix}) Error processing 'encounter_date': {e_date_col}")
        return trends_output
    
    if df_trends.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning.")
        return trends_output

    # Filter by date range and zone
    df_period_filtered = df_trends[
        (df_trends['encounter_date'].dt.date >= start_date) &
        (df_trends['encounter_date'].dt.date <= end_date)
    ]
    if zone_filter and 'zone_id' in df_period_filtered.columns:
        df_period_filtered = df_period_filtered[df_period_filtered['zone_id'] == zone_filter]
    elif zone_filter:
        logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter '{zone_filter}'.")

    if df_period_filtered.empty:
        logger.info(f"({source_context_log_prefix}) No data for the specified trend period/zone after filtering.")
        return trends_output

    # Calculate Patient Visits Trend
    if 'patient_id' in df_period_filtered.columns:
        visits_trend = get_trend_data(df=df_period_filtered, value_col='patient_id', date_col='encounter_date',
                                      period=time_period_aggregation, agg_func='nunique',
                                      source_context=f"{source_context_log_prefix}/PatientVisits")
        if isinstance(visits_trend, pd.Series) and not visits_trend.empty:
            trends_output["patient_visits_trend"] = visits_trend.rename("unique_patient_visits_count")
    else:
        logger.warning(f"({source_context_log_prefix}) 'patient_id' column missing. Cannot calculate patient visits trend.")

    # Calculate High Priority Follow-ups Trend
    prio_cols = ['ai_followup_priority_score', 'patient_id']
    if all(col in df_period_filtered.columns for col in prio_cols):
        df_prio_trend = df_period_filtered.copy() # Use copy for modification
        df_prio_trend['ai_followup_priority_score'] = convert_to_numeric(
            df_prio_trend['ai_followup_priority_score'], default_value=0.0
        )
        df_high_prio = df_prio_trend[df_prio_trend['ai_followup_priority_score'] >= settings.FATIGUE_INDEX_HIGH_THRESHOLD]
        
        if not df_high_prio.empty:
            high_prio_trend = get_trend_data(df=df_high_prio, value_col='patient_id', date_col='encounter_date',
                                             period=time_period_aggregation, agg_func='nunique',
                                             source_context=f"{source_context_log_prefix}/HighPrioFollowups")
            if isinstance(high_prio_trend, pd.Series) and not high_prio_trend.empty:
                trends_output["high_priority_followups_trend"] = high_prio_trend.rename("high_priority_followups_count")
        else:
            logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria for trend.")
    else:
        missing = [col for col in prio_cols if col not in df_period_filtered.columns]
        logger.warning(f"({source_context_log_prefix}) Missing columns for high priority follow-ups trend: {missing}.")

    num_generated = sum(1 for t in trends_output.values() if isinstance(t, pd.Series) and not t.empty)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {num_generated} trend(s) generated.")
    return trends_output
