# ssentinel_project_root/pages/chw_components/activity_trends.py
"""
[DEPRECATED] This module is deprecated as of v4.1.0.
Its functionality has been moved to the centralized `get_trend_data` function
in `ssentinel_project_root/data_processing/aggregation.py`.
This wrapper is maintained for backward compatibility only.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union
from datetime import date as date_type, datetime

logger = logging.getLogger(__name__)

try:
    # Attempt to import the new, centralized function
    from data_processing.aggregation import get_trend_data
except ImportError:
    # Fallback if the new structure isn't available, to prevent crashing
    def get_trend_data(*args, **kwargs) -> pd.Series:
        logger.error("Could not import centralized get_trend_data function. Returning empty Series.")
        return pd.Series(dtype=float)

def calculate_chw_activity_trends_data(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Union[str, pd.Timestamp, date_type, datetime],
    trend_end_date_input: Union[str, pd.Timestamp, date_type, datetime],
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D',
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]:
    """
    [DEPRECATED] This function is a wrapper for backward compatibility.
    Please use data_processing.aggregation.get_trend_data directly.
    """
    logger.warning(
        "Call to deprecated function 'calculate_chw_activity_trends_data'. "
        "Please refactor to use 'data_processing.aggregation.get_trend_data'."
    )
    
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
    }
    
    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        return trends_output
        
    df = chw_historical_health_df.copy()

    # Apply zone filter if provided
    if zone_filter and 'zone_id' in df.columns:
        df = df[df['zone_id'].astype(str) == str(zone_filter)]

    # Filter by date range
    try:
        start_date = pd.to_datetime(trend_start_date_input).date()
        end_date = pd.to_datetime(trend_end_date_input).date()
        if 'encounter_date' in df.columns:
            df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
            df = df[
                (df['encounter_date'].dt.date >= start_date) &
                (df['encounter_date'].dt.date <= end_date)
            ]
    except Exception as e:
        logger.error(f"Date filtering failed in deprecated activity_trends wrapper: {e}")
        return trends_output

    if df.empty:
        return trends_output

    # Calculate Patient Visits Trend
    if 'patient_id' in df.columns:
        trends_output["patient_visits_trend"] = get_trend_data(
            df=df,
            value_col='patient_id',
            date_col='encounter_date',
            period=time_period_aggregation,
            agg_func='nunique'
        )

    # Calculate High Priority Follow-ups Trend
    if 'ai_followup_priority_score' in df.columns:
        try:
            from config import settings
            high_prio_threshold = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80)
            df_high_prio = df[df['ai_followup_priority_score'] >= high_prio_threshold]
            
            trends_output["high_priority_followups_trend"] = get_trend_data(
                df=df_high_prio,
                value_col='patient_id',
                date_col='encounter_date',
                period=time_period_aggregation,
                agg_func='nunique'
            )
        except Exception as e:
            logger.error(f"Could not calculate high priority followups trend in deprecated wrapper: {e}")

    return trends_output
