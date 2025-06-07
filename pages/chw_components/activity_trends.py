# sentinel_project_root/pages/chw_components/activity_trends.py
# Calculates extended CHW activity trend data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from datetime import date as date_type, datetime

# --- Core Imports ---
try:
    from config import settings
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    # This fallback is for isolated testing; in production, app entry point handles this.
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in activity_trends.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

# --- Constants for Column Names ---
DATE_COL = 'encounter_date'
PATIENT_ID_COL = 'patient_id'
ZONE_ID_COL = 'zone_id'
PRIORITY_SCORE_COL = 'ai_followup_priority_score'
RISK_SCORE_COL = 'ai_risk_score'
TASK_PRIORITY_COL = 'priority_score'
TASK_ID_COL = 'task_id'


def _calculate_trend(
    df: pd.DataFrame,
    value_col: str,
    agg_func: str,
    period: str,
    trend_name: str,
    log_context: str,
    filter_series: Optional[pd.Series] = None
) -> Optional[pd.Series]:
    """
    Internal helper to calculate a single trend series from the DataFrame.
    Encapsulates filtering, calculation, and logging for robustness.
    """
    df_filtered = df if filter_series is None else df.loc[filter_series]

    if df_filtered.empty:
        logger.info(f"({log_context}) No data available for '{trend_name}' trend after filtering.")
        return None

    if value_col not in df_filtered.columns:
        logger.warning(f"({log_context}) Value column '{value_col}' not found for '{trend_name}' trend.")
        return None

    try:
        trend_series = get_trend_data(
            df=df_filtered,
            value_col=value_col,
            date_col=DATE_COL,
            period=period,
            agg_func=agg_func,
            source_context=f"{log_context}/{trend_name}"
        )
        if isinstance(trend_series, pd.Series) and not trend_series.empty:
            return trend_series.rename(trend_name)
    except Exception as e:
        logger.error(f"({log_context}) Failed to calculate '{trend_name}' trend: {e}", exc_info=True)

    return None


def calculate_chw_activity_trends_data(
    chw_historical_health_df: Optional[pd.DataFrame],
    trend_start_date_input: Union[str, pd.Timestamp, date_type, datetime],
    trend_end_date_input: Union[str, pd.Timestamp, date_type, datetime],
    zone_filter: Optional[str] = None,
    time_period_aggregation: str = 'D'
) -> Dict[str, Optional[pd.Series]]:
    """
    Calculates a comprehensive set of CHW activity and performance trends.

    This function is enhanced to provide not just activity volume but also insights into
    patient acuity and CHW workload over time.

    Returns:
        A dictionary of pandas Series, each representing a specific trend:
        - `patient_visits_trend`: Count of unique patients visited per period.
        - `high_priority_followups_trend`: Count of unique patients with high AI follow-up scores.
        - `patient_acuity_trend`: Average AI risk score of patients visited per period.
        - `new_high_priority_tasks_trend`: Count of new high-priority tasks generated per period.
    """
    log_context = "CHWActivityTrends"
    trends_output: Dict[str, Optional[pd.Series]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None,
        "patient_acuity_trend": None,
        "new_high_priority_tasks_trend": None,
    }

    if not isinstance(chw_historical_health_df, pd.DataFrame) or chw_historical_health_df.empty:
        logger.warning(f"({log_context}) Input DataFrame is empty or invalid. Trend calculation skipped.")
        return trends_output

    try:
        start_date = pd.to_datetime(trend_start_date_input).date()
        end_date = pd.to_datetime(trend_end_date_input).date()
    except (ValueError, TypeError) as e:
        logger.error(f"({log_context}) Invalid date inputs: {e}. Start: '{trend_start_date_input}', End: '{trend_end_date_input}'.")
        return trends_output

    df = chw_historical_health_df.copy()

    # --- Data Preparation ---
    if DATE_COL not in df.columns:
        logger.error(f"({log_context}) Missing required date column '{DATE_COL}'. Cannot calculate trends.")
        return trends_output
    
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df.dropna(subset=[DATE_COL], inplace=True)
    if df.empty:
        logger.warning(f"({log_context}) No records with valid dates after cleaning.")
        return trends_output
    
    # --- Filtering ---
    mask = (df[DATE_COL].dt.date >= start_date) & (df[DATE_COL].dt.date <= end_date)
    if zone_filter and ZONE_ID_COL in df.columns:
        mask &= (df[ZONE_ID_COL].astype(str) == str(zone_filter))
    df_period = df[mask].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_period.empty:
        logger.info(f"({log_context}) No data after period/zone filtering. Trend Series will be empty.")
        return trends_output

    # --- Trend Calculations ---
    
    # 1. Patient Visits Trend
    trends_output["patient_visits_trend"] = _calculate_trend(
        df_period, PATIENT_ID_COL, 'nunique', time_period_aggregation, 'patient_visits_trend', log_context
    )

    # 2. High-Priority Follow-ups Trend
    prio_threshold = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80)
    if PRIORITY_SCORE_COL in df_period.columns:
        df_period[PRIORITY_SCORE_COL] = convert_to_numeric(df_period[PRIORITY_SCORE_COL])
        high_prio_mask = df_period[PRIORITY_SCORE_COL] >= prio_threshold
        trends_output["high_priority_followups_trend"] = _calculate_trend(
            df_period, PATIENT_ID_COL, 'nunique', time_period_aggregation, 'high_priority_followups_trend', log_context, filter_series=high_prio_mask
        )

    # 3. Patient Acuity Trend (NEW & ACTIONABLE)
    if RISK_SCORE_COL in df_period.columns:
        df_period[RISK_SCORE_COL] = convert_to_numeric(df_period[RISK_SCORE_COL])
        trends_output["patient_acuity_trend"] = _calculate_trend(
            df_period, RISK_SCORE_COL, 'mean', time_period_aggregation, 'patient_acuity_trend', log_context
        )

    # 4. New High-Priority Tasks Trend (NEW & ACTIONABLE)
    task_prio_threshold = getattr(settings, 'TASK_PRIORITY_HIGH_THRESHOLD', 70)
    if TASK_PRIORITY_COL in df_period.columns and TASK_ID_COL in df_period.columns:
        df_period[TASK_PRIORITY_COL] = convert_to_numeric(df_period[TASK_PRIORITY_COL])
        high_task_prio_mask = df_period[TASK_PRIORITY_COL] >= task_prio_threshold
        trends_output["new_high_priority_tasks_trend"] = _calculate_trend(
            df_period, TASK_ID_COL, 'nunique', time_period_aggregation, 'new_high_priority_tasks_trend', log_context, filter_series=high_task_prio_mask
        )

    num_gen = sum(1 for ts in trends_output.values() if isinstance(ts, pd.Series))
    logger.info(f"({log_context}) Trends calculation complete. {num_gen} trend(s) generated.")
    return trends_output
