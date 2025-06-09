# ssentinel_project_root/pages/chw_components/alert_generation.py
"""
SME FINAL VERSION: This is a DEPRECATED WRAPPER module.

This file is intentionally simple and acts as a bridge to the new, centralized
alerting engine. Its purpose is to maintain backward compatibility for older parts
of the application that have not yet been updated. It imports the new function
from `analytics.alerting` and exposes it under the old function name, while
logging a warning to developers. This file is complete and correct for its
deprecation purpose.
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime

# --- Module Setup ---
logger = logging.getLogger(__name__)


# --- Safe Import of the New Centralized Function ---
try:
    # This is the primary import from the new, correct location.
    from analytics.alerting import generate_chw_patient_alerts
except ImportError:
    # This fallback prevents the entire application from crashing if the new
    # `analytics.alerting` module is missing or has an error. It ensures
    # the app can still load, albeit with reduced functionality.
    def generate_chw_patient_alerts(*args, **kwargs) -> List[Dict[str, Any]]:
        logger.error(
            "CRITICAL FALLBACK: Could not import centralized 'generate_chw_patient_alerts' "
            "function from 'analytics.alerting'. Please ensure this file exists and is correct. "
            "Alert generation will be disabled."
        )
        return []


# --- Deprecated Public Function ---

def generate_chw_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    [DEPRECATED] This function is a wrapper for backward compatibility.
    It now calls the centralized alert generation engine. Please refactor all
    calls to use `analytics.alerting.generate_chw_patient_alerts` directly.
    """
    logger.warning(
        "Call to deprecated function 'generate_chw_alerts' in 'chw_components'. "
        "This function is now a wrapper. Please refactor calls to use the "
        "centralized 'generate_chw_patient_alerts' from 'analytics.alerting'."
    )
    
    # The call is safely forwarded to the new, centralized, and robust function.
    # The function signature is preserved for compatibility. The `chw_zone_context_str`
    # is ignored as the new function derives context directly from the data.
    return generate_chw_patient_alerts(
        patient_encounter_data_df=patient_encounter_data_df,
        for_date=for_date,
        max_alerts=max_alerts_to_return
    )
