# sentinel_project_root/pages/chw_components/activity_trends.py
# Calculates CHW activity trend data for Sentinel Health Co-Pilot.
# Renamed from activity_trend_calculator.py

import pandas as pd
import numpy as np # For np.nan if used, less direct use here
import logging
from typing import Dict, Any, Optional # Union not used here
from datetime import date as date_type # For type hinting clarity
# datetime class from datetime module is not directly used here, but pd.to_datetime handles various inputs

from config import settings # Use new settings module
from data_processing.aggregation import get_trend_data # Use centralized trend utility
from data_processing.helpers import convert_to_numeric # For ensuring numeric types

logger = logging.getLogger(__name__)


def calculate_chw_activity_trends_data( # Renamed function
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Any, # Can be date, str, pd.Timestamp
    trend_end_date_input: Any,   # Can be date, str, pd.Timestamp
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D', # Default to Daily ('D'), common for CHW trends
    source_context_log_prefix: str = "CHWActivityTrends"
) -> Dict[str, Optional[pd.Series]]:
    """
    Calculates CHW activity trends (e.g., patient visits, high-priority follow-ups)
    over a specified period using the get_trend_data utility.

    Args:
        chw_historical_health_df: DataFrame with historical health records for CHW/team.
                                  Expected columns: 'encounter_date', 'patient_id', 
                                  'ai_followup_priority_score', 'zone_id' (if zone_filter used).
        trend_start_date_input: Start date for the trend analysis.
        trend_end_date_input: End date for the trend analysis.
        zone_filter: Optional. If provided, filters data for a specific zone.
        time_period_aggregation: Aggregation period ('D' for daily, 'W-Mon' for weekly starting Monday, etc.).
        source_context_log_prefix: Prefix for log messages for traceability.

    Returns:
        Dict[str, Optional[pd.Series]]: A dictionary where keys are trend names
                                        (e.g., "patient_visits_trend") and values are
                                        pandas Series (index=date, value=metric count/value),
                                        or None if a trend could not be calculated.
    """
    trends_output_map: Dict[str, Optional[pd.Series]] = { # Renamed for clarity
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
        # Add other trend keys here as needed
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({source_context_log_prefix}) No historical health data provided. Trend calculation skipped.")
        return trends_output_map
    
    # Validate aggregation period against common Plotly/Pandas resampling codes
    valid_aggregation_periods = ['D', 'B', 'W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'M', 'MS', 'Q', 'QS', 'A', 'AS']
    if time_period_aggregation.upper() not in valid_aggregation_periods: # Check uppercase for flexibility
        logger.error(f"({source_context_log_prefix}) Invalid time_period_aggregation: '{time_period_aggregation}'. Must be one of {valid_aggregation_periods}.")
        # Consider raising ValueError or returning empty if critical, for now just logs and continues (might result in Nones)
        return trends_output_map


    # Standardize input dates robustly using pd.to_datetime and then .date()
    try:
        actual_trend_start_date = pd.to_datetime(trend_start_date_input, errors='coerce').date()
        actual_trend_end_date = pd.to_datetime(trend_end_date_input, errors='coerce').date()

        if pd.isna(actual_trend_start_date) or pd.isna(actual_trend_end_date): # Check if NaT after coercion
            raise ValueError("Date conversion resulted in NaT for trend start or end date.")
        if actual_trend_start_date > actual_trend_end_date:
            logger.error(f"({source_context_log_prefix}) Trend period error: Start date ({actual_trend_start_date}) is after end date ({actual_trend_end_date}).")
            return trends_output_map # Return empty if dates invalid
    except Exception as e_date_conv:
        logger.error(f"({source_context_log_prefix}) Invalid date inputs for trend period: {e_date_conv}. Start: '{trend_start_date_input}', End: '{trend_end_date_input}'.")
        return trends_output_map
        
    logger.info(
        f"({source_context_log_prefix}) Calculating CHW activity trends for period: "
        f"{actual_trend_start_date.isoformat()} to {actual_trend_end_date.isoformat()}, Zone: {zone_filter or 'All'}, Agg: {time_period_aggregation}"
    )
    
    # Work on a copy of the DataFrame to avoid modifying the original
    df_for_trends_analysis = chw_historical_health_df.copy()

    # --- Data Preparation ---
    # 1. Ensure 'encounter_date' is present and valid datetime
    if 'encounter_date' not in df_for_trends_analysis.columns:
        logger.error(f"({source_context_log_prefix}) Critical: 'encounter_date' column missing. Cannot calculate trends.")
        return trends_output_map
    try:
        df_for_trends_analysis['encounter_date'] = pd.to_datetime(df_for_trends_analysis['encounter_date'], errors='coerce')
        df_for_trends_analysis.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed
    except Exception as e_date_col:
        logger.error(f"({source_context_log_prefix}) Error processing 'encounter_date' column: {e_date_col}")
        return trends_output_map
    
    if df_for_trends_analysis.empty:
        logger.info(f"({source_context_log_prefix}) No records with valid encounter dates after cleaning for trend calculation.")
        return trends_output_map

    # 2. Filter by the specified trend date range
    df_period_filtered = df_for_trends_analysis[
        (df_for_trends_analysis['encounter_date'].dt.date >= actual_trend_start_date) &
        (df_for_trends_analysis['encounter_date'].dt.date <= actual_trend_end_date)
    ]

    # 3. Apply zone filter if provided
    if zone_filter:
        if 'zone_id' in df_period_filtered.columns:
            df_period_filtered = df_period_filtered[df_period_filtered['zone_id'] == zone_filter]
        else:
            logger.warning(f"({source_context_log_prefix}) 'zone_id' column missing, cannot apply zone filter '{zone_filter}'. Using all data in period.")
            # Potentially st.warning here if in Streamlit context and this is unexpected by user.

    if df_period_filtered.empty:
        logger.info(f"({source_context_log_prefix}) No CHW data found for the specified trend period/zone after filtering.")
        return trends_output_map

    # --- Calculate Specific Trends using get_trend_data utility ---

    # 1. Trend for Patient Visits (unique patients per period)
    if 'patient_id' in df_period_filtered.columns:
        visits_trend_series_result = get_trend_data(
            df=df_period_filtered,
            value_col='patient_id', # We want to count unique patients
            date_col='encounter_date',
            period=time_period_aggregation,
            agg_func='nunique', # Count unique patient IDs
            source_context=f"{source_context_log_prefix}/PatientVisitsTrend"
        )
        if isinstance(visits_trend_series_result, pd.Series) and not visits_trend_series_result.empty:
            trends_output_map["patient_visits_trend"] = visits_trend_series_result.rename("unique_patient_visits_count")
    else:
        logger.warning(f"({source_context_log_prefix}) 'patient_id' column missing. Cannot calculate patient visits trend.")

    # 2. Trend for High Priority Follow-ups (unique patients needing high prio follow-up)
    required_cols_for_prio_trend = ['ai_followup_priority_score', 'patient_id'] # Need patient_id to count unique patients
    if all(col in df_period_filtered.columns for col in required_cols_for_prio_trend):
        # Ensure priority score is numeric
        df_period_filtered['ai_followup_priority_score_numeric'] = convert_to_numeric(
            df_period_filtered['ai_followup_priority_score'], default_value=0.0 # Default non-prio to 0
        )
        
        df_high_prio_records = df_period_filtered[
            df_period_filtered['ai_followup_priority_score_numeric'] >= settings.FATIGUE_INDEX_HIGH_THRESHOLD # Using this as general high prio
        ]
        
        if not df_high_prio_records.empty:
            high_prio_trend_series_result = get_trend_data(
                df=df_high_prio_records,
                value_col='patient_id', # Count unique patients with high priority follow-ups
                date_col='encounter_date',
                period=time_period_aggregation,
                agg_func='nunique',
                source_context=f"{source_context_log_prefix}/HighPrioFollowupsTrend"
            )
            if isinstance(high_prio_trend_series_result, pd.Series) and not high_prio_trend_series_result.empty:
                trends_output_map["high_priority_followups_trend"] = high_prio_trend_series_result.rename("high_priority_followups_count")
        else:
            logger.info(f"({source_context_log_prefix}) No encounters met high AI follow-up priority criteria for trend calculation.")
    else:
        missing_cols_str = ", ".join([col for col in required_cols_for_prio_trend if col not in df_period_filtered.columns])
        logger.warning(f"({source_context_log_prefix}) Missing columns for high priority follow-ups trend: [{missing_cols_str}]. Calculation skipped.")

    # --- Final Logging ---
    num_trends_generated = sum(1 for trend_val in trends_output_map.values() if isinstance(trend_val, pd.Series) and not trend_val.empty)
    logger.info(f"({source_context_log_prefix}) CHW activity trends calculation complete. {num_trends_generated} trend(s) generated.")
    return trends_output_map
