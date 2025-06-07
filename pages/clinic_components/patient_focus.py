# sentinel_project_root/pages/clinic_components/patient_focus.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

try:
    from config import settings
    from analytics.alerting import get_patient_alerts_for_clinic 
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in patient_focus.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)


def prepare_clinic_patient_focus_overview_data(
    # FIXED: Renamed the parameter to a simpler, consistent `filtered_health_df` to fix the NameError.
    filtered_health_df: Optional[pd.DataFrame],
    **kwargs
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and flagged patient cases.
    """
    module_log_prefix = "ClinicPatientFocusPrep"
    
    default_load_cols = ['period_start_date', 'condition', 'unique_patients_count']
    default_flagged_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
    
    output_data: Dict[str, Any] = {
        "patient_load_by_key_condition_df": pd.DataFrame(columns=default_load_cols),
        "flagged_patients_for_review_df": pd.DataFrame(columns=default_flagged_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df, pd.DataFrame) or filtered_health_df.empty:
        output_data["processing_notes"].append("No health data provided for patient focus analysis.")
        return output_data

    df_load_analysis = filtered_health_df.copy()
    
    required_cols = ['encounter_date', 'patient_id', 'condition']
    if not all(col in df_load_analysis.columns for col in required_cols):
        missing_cols = list(set(required_cols) - set(df_load_analysis.columns))
        output_data["processing_notes"].append(f"Required columns for patient load analysis are missing: {missing_cols}")
    else:
        df_load_analysis['encounter_date'] = pd.to_datetime(df_load_analysis['encounter_date'], errors='coerce')
        df_load_analysis.dropna(subset=required_cols, inplace=True)
        df_load_analysis = df_load_analysis[df_load_analysis['condition'] != "UnknownCondition"]

        if not df_load_analysis.empty:
            key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
            if key_conditions:
                aggregated_summaries = []
                for condition in key_conditions:
                    try:
                        mask = df_load_analysis['condition'].str.contains(re.escape(condition), case=False, na=False)
                        if mask.any():
                            df_cond = df_load_analysis[mask]
                            grouped = df_cond.groupby(pd.Grouper(key='encounter_date', freq='W-MON'))['patient_id'].nunique().reset_index()
                            grouped.rename(columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'}, inplace=True)
                            grouped['condition'] = condition
                            aggregated_summaries.append(grouped)
                    except Exception as e:
                        logger.error(f"Error aggregating load for condition '{condition}': {e}", exc_info=True)
                
                if aggregated_summaries:
                    final_load_df = pd.concat(aggregated_summaries, ignore_index=True)
                    output_data["patient_load_by_key_condition_df"] = final_load_df.reindex(columns=default_load_cols)
    
    # --- Flagged Patients for Review ---
    try:
        risk_moderate_thresh = float(_get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60))
        alerts_df = get_patient_alerts_for_clinic(
            health_df_period=filtered_health_df,
            risk_threshold_moderate=risk_moderate_thresh
        )
        if isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty:
            output_data["flagged_patients_for_review_df"] = alerts_df.reindex(columns=default_flagged_cols, fill_value=np.nan)
        else:
            output_data["processing_notes"].append("No patients were flagged for review in this period.")
    except Exception as e:
        logger.error(f"Error getting flagged patients: {e}", exc_info=True)
        output_data["processing_notes"].append("Error generating list of flagged patients.")

    logger.info(f"({module_log_prefix}) Patient focus data preparation complete.")
    return output_data
