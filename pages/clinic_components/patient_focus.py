# sentinel_project_root/pages/clinic_components/patient_focus.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.

import pandas as pd
import numpy as np
import logging
import re # For condition matching
from typing import Dict, Any, Optional, List

from config import settings
from analytics.alerting import get_patient_alerts_for_clinic # For flagged patients
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def prepare_clinic_patient_focus_overview_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    patient_load_time_aggregation_period: str = 'W-MON', # Weekly (Monday start)
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and flagged patient cases.
    """
    module_log_prefix = "ClinicPatientFocusPrep"
    logger.info(f"({module_log_prefix}) Preparing patient focus data. Period: {reporting_period_context_str}, Load Agg: {patient_load_time_aggregation_period}")

    default_load_cols = ['period_start_date', 'condition', 'unique_patients_count']
    default_flagged_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id'] # Simplified example
    
    output_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "patient_load_by_key_condition_df": pd.DataFrame(columns=default_load_cols),
        "flagged_patients_for_review_df": pd.DataFrame(columns=default_flagged_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note = "No health data for patient focus data preparation. Output will be empty."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["processing_notes"].append(note)
        return output_data

    df_focus_src = filtered_health_df_for_clinic_period.copy()

    # Standardize essential columns for patient load calculation
    load_calc_cols_cfg = {
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'patient_id': {"default": f"UPID_Focus_{reporting_period_context_str[:10]}", "type": str},
        'condition': {"default": "UnknownCondition", "type": str}
    }
    common_na_focus = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    na_regex_focus = r'^(?:' + '|'.join(re.escape(s) for s in common_na_focus if s) + r')$'

    for col, cfg in load_calc_cols_cfg.items():
        if col not in df_focus_src.columns: df_focus_src[col] = cfg["default"]
        if cfg["type"] == "datetime": df_focus_src[col] = pd.to_datetime(df_focus_src[col], errors='coerce')
        elif cfg["type"] == str:
            df_focus_src[col] = df_focus_src[col].astype(str).fillna(str(cfg["default"]))
            if any(common_na_focus): df_focus_src[col] = df_focus_src[col].replace(na_regex_focus, str(cfg["default"]), regex=True)
            df_focus_src[col] = df_focus_src[col].str.strip()
    
    df_focus_src.dropna(subset=['encounter_date', 'patient_id', 'condition'], inplace=True)
    if df_focus_src.empty:
        note = "No valid records with encounter_date, patient_id, & condition after cleaning for patient load analysis."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["processing_notes"].append(note)
        # Fall through to attempt flagged patient generation with original filtered_health_df...
    
    # Patient Load by Key Condition
    key_conditions_load = settings.KEY_CONDITIONS_FOR_ACTION
    if 'condition' in df_focus_src.columns and key_conditions_load and not df_focus_src.empty:
        load_summaries: List[pd.DataFrame] = []
        for cond_key in key_conditions_load:
            # Use regex for flexible, case-insensitive matching of the condition key
            cond_mask = df_focus_src['condition'].astype(str).str.contains(re.escape(cond_key), case=False, na=False, regex=True)
            df_one_cond = df_focus_src[cond_mask]
            if not df_one_cond.empty and 'patient_id' in df_one_cond.columns: # Ensure patient_id exists
                # Group by period and count unique patients for this condition
                grouped = df_one_cond.groupby(
                    pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation_period, label='left', closed='left')
                )['patient_id'].nunique().reset_index()
                grouped.rename(columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'}, inplace=True)
                grouped['condition'] = cond_key # Assign the specific condition name
                load_summaries.append(grouped)
        
        if load_summaries:
            output_data["patient_load_by_key_condition_df"] = pd.concat(load_summaries, ignore_index=True)
        else: output_data["processing_notes"].append("No patient load data aggregated for key conditions (empty after grouping).")
    elif df_focus_src.empty: pass # Already noted if empty due to cleaning
    else:
        missing_reason = ""
        if 'condition' not in df_focus_src.columns: missing_reason += "'condition' column missing. "
        if not key_conditions_load: missing_reason += "KEY_CONDITIONS_FOR_ACTION in config empty. "
        output_data["processing_notes"].append(f"Patient load by condition skipped. Reason: {missing_reason.strip()}")

    # Flagged Patient Cases for Clinical Review (uses original period-filtered df)
    try:
        df_flagged_patients = get_patient_alerts_for_clinic(
            health_df_period=filtered_health_df_for_clinic_period, # Original, less aggressively cleaned df
            risk_threshold_moderate=settings.RISK_SCORE_MODERATE_THRESHOLD,
            source_context=f"{module_log_prefix}/FlaggedPatients"
        )
    except Exception as e_flagged_calc:
        logger.error(f"({module_log_prefix}) Error getting flagged patients: {e_flagged_calc}", exc_info=True)
        df_flagged_patients = pd.DataFrame(columns=default_flagged_cols) # Ensure schema
        output_data["processing_notes"].append("Error generating list of flagged patients for review.")

    if isinstance(df_flagged_patients, pd.DataFrame) and not df_flagged_patients.empty:
        output_data["flagged_patients_for_review_df"] = df_flagged_patients
        logger.info(f"({module_log_prefix}) Identified {len(df_flagged_patients)} patient cases flagged for clinical review.")
    elif isinstance(df_flagged_patients, pd.DataFrame): # Empty DataFrame means no one flagged
        note_no_flagged = "No patient cases flagged for clinical review in this period based on criteria."
        logger.info(f"({module_log_prefix}) {note_no_flagged}")
        output_data["processing_notes"].append(note_no_flagged)
        # output_data["flagged_patients_for_review_df"] is already an empty DF with schema

    logger.info(f"({module_log_prefix}) Clinic patient focus data preparation complete. Notes: {len(output_data['processing_notes'])}")
    return output_data
