# sentinel_project_root/pages/clinic_components/patient_focus.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.
# Renamed from patient_focus_data_preparer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
# For flagged patients, use the alerting function from the analytics module
from analytics.alerting import get_patient_alerts_for_clinic
from data_processing.helpers import convert_to_numeric # Local import

logger = logging.getLogger(__name__)


def prepare_clinic_patient_focus_overview_data( # Renamed function
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame], # Health data already filtered for clinic & period
    reporting_period_context_str: str, # Renamed for clarity
    patient_load_time_aggregation_period: str = 'W-MON', # Renamed, default to weekly (Monday start)
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis (by key condition) and a list of
    flagged patient cases for clinical review on the clinic dashboard.

    Args:
        filtered_health_df_for_clinic_period: DataFrame of health records for the clinic and selected period.
        reporting_period_context_str: String describing the reporting period (for context).
        patient_load_time_aggregation_period: Aggregation period for patient load ('D', 'W-MON', 'M', etc.).

    Returns:
        Dict[str, Any]: A dictionary containing:
            "reporting_period": str
            "patient_load_by_key_condition_df": pd.DataFrame (period_start_date, condition, unique_patients_count)
            "flagged_patients_for_review_df": pd.DataFrame (from analytics.get_patient_alerts_for_clinic)
            "processing_notes": List[str]
    """
    module_log_prefix = "ClinicPatientFocusPrep" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Preparing patient focus data. Period: {reporting_period_context_str}, Load Agg: {patient_load_time_aggregation_period}")

    # Initialize output structure with defaults for DataFrames to ensure consistent schema
    default_patient_load_cols = ['period_start_date', 'condition', 'unique_patients_count']
    # Columns for flagged_patients_df should match output of analytics.get_patient_alerts_for_clinic
    default_flagged_patients_cols = [
        'patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score',
        'ai_risk_score', 'age', 'gender', 'zone_id', 'referred_to_facility_id',
        'min_spo2_pct', 'vital_signs_temperature_celsius'
    ]
    patient_focus_output_dict: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "patient_load_by_key_condition_df": pd.DataFrame(columns=default_patient_load_cols),
        "flagged_patients_for_review_df": pd.DataFrame(columns=default_flagged_patients_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note_msg = "No health data provided for patient focus data preparation. Output will be empty."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        patient_focus_output_dict["processing_notes"].append(note_msg)
        return patient_focus_output_dict

    df_focus_src_prepared = filtered_health_df_for_clinic_period.copy() # Work on a copy

    # --- Data Preparation and Validation ---
    # Ensure essential columns for this module's direct logic are present and correctly typed.
    # analytics.get_patient_alerts_for_clinic will do its own internal validation.
    essential_cols_for_load_calc = {
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'patient_id': {"default": f"UnknownPID_Focus_{reporting_period_context_str[:10]}", "type": str},
        'condition': {"default": "UnknownCondition", "type": str}
    }
    common_na_focus_prep = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown']

    for col, config_item_focus in essential_cols_for_load_calc.items():
        if col not in df_focus_src_prepared.columns:
            df_focus_src_prepared[col] = config_item_focus["default"]
        
        if config_item_focus["type"] == "datetime":
            df_focus_src_prepared[col] = pd.to_datetime(df_focus_src_prepared[col], errors='coerce')
        elif config_item_focus["type"] == str: # String columns
             df_focus_src_prepared[col] = df_focus_src_prepared[col].astype(str).fillna(str(config_item_focus["default"]))
             df_focus_src_prepared[col] = df_focus_src_prepared[col].replace(common_na_focus_prep, str(config_item_focus["default"]), regex=False).str.strip()
        # Numeric cols for load not directly used here, but underlying data may have them.

    # Drop rows if critical identifiers for load calculation are still missing/NaT after cleaning
    df_focus_src_prepared.dropna(subset=['encounter_date', 'patient_id', 'condition'], inplace=True)
    if df_focus_src_prepared.empty:
        note_msg = "No valid records with encounter_date, patient_id, & condition after cleaning for patient load analysis."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        patient_focus_output_dict["processing_notes"].append(note_msg)
        # Still try to get flagged patients if the main df had data before this specific dropna
        # The get_patient_alerts_for_clinic function will use filtered_health_df_for_clinic_period
    
    
    # --- 1. Patient Load by Key Condition ---
    key_conditions_for_load_analysis = settings.KEY_CONDITIONS_FOR_ACTION # List of condition names from config
    
    if 'condition' in df_focus_src_prepared.columns and key_conditions_for_load_analysis and not df_focus_src_prepared.empty:
        # Filter for encounters matching any of the key conditions (case-insensitive partial match)
        # This ensures if condition is "TB; Pneumonia", both are caught if in KEY_CONDITIONS_FOR_ACTION
        condition_regex_pattern_load = '|'.join([re.escape(cond_key) for cond_key in key_conditions_for_load_analysis])
        
        df_key_condition_encounters = df_focus_src_prepared[
            df_focus_src_prepared['condition'].astype(str).str.contains(condition_regex_pattern_load, case=False, na=False) &
            (df_focus_src_prepared['patient_id'].astype(str).str.lower() != f"unknownpid_focus_{reporting_period_context_str[:10]}".lower()) # Exclude generic PIDs
        ].copy()

        if not df_key_condition_encounters.empty:
            # For accurate patient load by condition when a patient has multiple key conditions listed in one encounter
            # (e.g. "TB; HIV-Positive"), we need to explode or assign primary.
            # Simpler: If 'condition' column can have "TB; HIV", this counts as one encounter under a combined label for now
            # or it relies on KEY_CONDITIONS_FOR_ACTION being specific enough.
            # A more robust way would be to iterate through KEY_CONDITIONS_FOR_ACTION and filter for each.
            
            load_summary_list = []
            for cond_key_iter in key_conditions_for_load_analysis:
                df_one_condition = df_key_condition_encounters[
                    df_key_condition_encounters['condition'].astype(str).str.contains(re.escape(cond_key_iter), case=False, na=False)
                ]
                if not df_one_condition.empty:
                    grouped_by_period = df_one_condition.groupby(
                        pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation_period, label='left', closed='left')
                    )['patient_id'].nunique().reset_index()
                    grouped_by_period.rename(columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'}, inplace=True)
                    grouped_by_period['condition'] = cond_key_iter # Add the condition name
                    load_summary_list.append(grouped_by_period)
            
            if load_summary_list:
                patient_load_df_result = pd.concat(load_summary_list, ignore_index=True)
                patient_focus_output_dict["patient_load_by_key_condition_df"] = patient_load_df_result
            else:
                patient_focus_output_dict["processing_notes"].append("No patient load data aggregated for key conditions in the period (empty after grouping).")
        else:
            patient_focus_output_dict["processing_notes"].append("No encounters found matching any key conditions for patient load analysis in the period.")
    elif df_focus_src_prepared.empty: # Already logged if due to cleaning
        pass
    else:
        missing_reason = ""
        if 'condition' not in df_focus_src_prepared.columns: missing_reason += "'condition' column missing. "
        if not key_conditions_for_load_analysis: missing_reason += "KEY_CONDITIONS_FOR_ACTION in config is empty. "
        patient_focus_output_dict["processing_notes"].append(f"Patient load by condition skipped. Reason: {missing_reason.strip()}")


    # --- 2. Flagged Patient Cases for Clinical Review ---
    # This uses the robust analytics.get_patient_alerts_for_clinic function.
    # It takes the *original* period-filtered DataFrame before the specific cleaning for load calc,
    # as it has its own internal data preparation.
    try:
        df_flagged_patients_result = get_patient_alerts_for_clinic(
            health_df_period=filtered_health_df_for_clinic_period, # Pass the broader period-filtered data
            risk_threshold_moderate=settings.RISK_SCORE_MODERATE_THRESHOLD, # Use configured threshold
            source_context=f"{module_log_prefix}/FlaggedPatientCases"
        )
    except Exception as e_flagged:
        logger.error(f"({module_log_prefix}) Error getting flagged patients for clinic review: {e_flagged}", exc_info=True)
        df_flagged_patients_result = pd.DataFrame(columns=default_flagged_patients_cols) # Empty with schema on error
        patient_focus_output_dict["processing_notes"].append("Error generating list of flagged patients for review.")


    if isinstance(df_flagged_patients_result, pd.DataFrame) and not df_flagged_patients_result.empty:
        patient_focus_output_dict["flagged_patients_for_review_df"] = df_flagged_patients_result
        logger.info(f"({module_log_prefix}) Identified {len(df_flagged_patients_result)} patient cases flagged for clinical review.")
    else: # Handles both None and empty DataFrame return from get_patient_alerts_for_clinic
        note_no_flagged = "No specific patient cases were flagged for clinical review in this period based on current criteria."
        logger.info(f"({module_log_prefix}) {note_no_flagged}")
        patient_focus_output_dict["processing_notes"].append(note_no_flagged)
        # Ensure flagged_patients_for_review_df is an empty DataFrame with expected schema if none found (already initialized)

    logger.info(f"({module_log_prefix}) Clinic patient focus data preparation complete. Notes: {len(patient_focus_output_dict['processing_notes'])}")
    return patient_focus_output_dict
