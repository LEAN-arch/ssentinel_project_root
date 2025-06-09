# ssentinel_project_root/pages/chw_components/summary_metrics.py
"""
SME FINAL VERSION: This component calculates daily summary KPIs for the CHW dashboard.
The function signature has been corrected and simplified to resolve the TypeError
by removing the redundant 'for_date' parameter.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

# --- Module Setup ---
logger = logging.getLogger(__name__)

# --- Safe Setting Import ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError:
    logging.warning("summary_metrics.py: Could not import dependencies. Using mock fallbacks.")
    class MockSettings:
        FATIGUE_INDEX_HIGH_THRESHOLD = 80
        ALERT_SPO2_CRITICAL_LOW_PCT = 90
        ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5
    settings = MockSettings()
    def convert_to_numeric(series, **kwargs):
        return pd.to_numeric(series, errors='coerce')


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets a configuration value from the global settings object."""
    return getattr(settings, attr_name, default_value)


def calculate_chw_daily_summary_metrics(daily_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """
    Calculates a dictionary of summary KPIs from a daily CHW activity DataFrame.
    
    SME NOTE: The 'for_date' and other unused arguments have been removed to align
    with its usage in the dashboard (`01_chw_dashboard.py`) and to fix the TypeError.
    This function now only requires the pre-filtered daily DataFrame.
    """
    # If the input DataFrame is not valid or is empty, return a default structure.
    if not isinstance(daily_df, pd.DataFrame) or daily_df.empty:
        return {
            "visits_count": 0,
            "high_ai_prio_followups_count": 0,
            "critical_spo2_cases_identified_count": 0,
            "high_fever_cases_identified_count": 0,
        }

    # Create a safe copy to avoid modifying the original DataFrame
    df_safe = daily_df.copy()

    # Standardize a temperature column from multiple possible sources
    if 'vital_signs_temperature_celsius' in df_safe.columns and df_safe['vital_signs_temperature_celsius'].notna().any():
        temp_col = 'vital_signs_temperature_celsius'
    else:
        # Fallback to a secondary temperature column if the primary is not available
        temp_col = 'max_skin_temp_celsius'

    # Define all columns required for calculations and their safe defaults
    required_cols = {
        'patient_id': 'object',
        'ai_followup_priority_score': np.nan,
        'min_spo2_pct': np.nan,
        temp_col: np.nan
    }
    
    # Ensure all required columns exist in the DataFrame, creating them with default values if not.
    for col, default in required_cols.items():
        if col not in df_safe:
            df_safe[col] = default

    # --- KPI Calculations ---
    # Wrap calculations in a try/except block for maximum robustness.
    try:
        # Total unique patients visited today
        visits = df_safe['patient_id'].nunique()
        
        # Count of patients with a high AI priority score
        prio_scores = convert_to_numeric(df_safe['ai_followup_priority_score']).fillna(0)
        high_prio_followups = df_safe[prio_scores >= _get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 80)]['patient_id'].nunique()

        # Count of patients with critically low SpO2 levels
        spo2_values = convert_to_numeric(df_safe['min_spo2_pct'])
        critical_spo2 = (spo2_values < _get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90)).sum()
        
        # Count of patients with a high fever
        temp_values = convert_to_numeric(df_safe[temp_col])
        high_fever = (temp_values >= _get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)).sum()

        return {
            "visits_count": int(visits),
            "high_ai_prio_followups_count": int(high_prio_followups),
            "critical_spo2_cases_identified_count": int(critical_spo2),
            "high_fever_cases_identified_count": int(high_fever),
        }
    except Exception as e:
        logger.error(f"Error in calculate_chw_daily_summary_metrics: {e}", exc_info=True)
        # Return a safe, empty structure in case of an unexpected calculation error.
        return {
            "visits_count": 0,
            "high_ai_prio_followups_count": 0,
            "critical_spo2_cases_identified_count": 0,
            "high_fever_cases_identified_count": 0,
        }
