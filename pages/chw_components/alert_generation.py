# ssentinel_project_root/pages/chw_components/alert_generation.py
"""
[DEPRECATED] This module is deprecated as of v4.1.0.
Its functionality has been moved to the centralized `CHWAlertGenerator` class
in `ssentinel_project_root/analytics/alerting.py`.
This wrapper is maintained for backward compatibility only.
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime

logger = logging.getLogger(__name__)

try:
    # Attempt to import the new, centralized alert generation function
    from analytics.alerting import generate_chw_patient_alerts
except ImportError:
    # Fallback if the new structure isn't available, to prevent crashing.
    def generate_chw_patient_alerts(*args, **kwargs) -> List[Dict[str, Any]]:
        logger.error("Could not import centralized generate_chw_patient_alerts function. Returning empty list.")
        return []


def generate_chw_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    [DEPRECATED] This function is a wrapper for backward compatibility.
    It now calls the centralized alert generation engine. Please refactor to use
    `analytics.alerting.generate_chw_patient_alerts` directly.
    """
    logger.warning(
        "Call to deprecated function 'generate_chw_alerts'. "
        "Please refactor to use 'analytics.alerting.generate_chw_patient_alerts'."
    )
    
    # The call is forwarded to the new, centralized, and robust function.
    # The function signature is preserved for compatibility.
    return generate_chw_patient_alerts(
        patient_encounter_data_df=patient_encounter_data_df,
        for_date=for_date,
        chw_zone_context_str=chw_zone_context_str,
        max_alerts_to_return=max_alerts_to_return
    )
