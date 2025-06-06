# sentinel_project_root/pages/clinic_components/patient_focus.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime # Added datetime for robust parsing

try:
    from config import settings
    # Assuming get_patient_alerts_for_clinic is defined and robust
    from analytics.alerting import get_patient_alerts_for_clinic 
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in patient_focus.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_FOCUS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_FOCUS_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_FOCUS if s) + r')$' if COMMON_NA_STRINGS_FOCUS else None

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


def _prepare_patient_focus_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    default_patient_id_prefix: str
) -> pd.DataFrame:
    """Prepares the DataFrame for patient focus analysis."""
    df_prepared = df.copy()
    for col_name, config in cols_config.items():
        default_value = config.get("default")
        target_type_str = config.get("type")

        if col_name not in df_prepared.columns:
            if col_name == 'patient_id': default_value = default_patient_id_prefix
            
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            elif isinstance(default_value, (list, dict)): 
                 df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value
        
        current_col_dtype = df_prepared[col_name].dtype
        if target_type_str in [float, "datetime"] and pd.api.types.is_object_dtype(current_col_dtype): # Int handled by convert_to_numeric
            if NA_REGEX_FOCUS_PATTERN:
                try:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_FOCUS_PATTERN, np.nan, regex=True)
                except Exception as e_regex:
                     logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float: # Includes int that might become float due to NaNs
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value)
            elif target_type_str == str:
                df_prepared[col_name] = df_prepared[col_name].fillna(str(default_value)).astype(str)
                if NA_REGEX_FOCUS_PATTERN:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_FOCUS_PATTERN, str(default_value), regex=True)
                df_prepared[col_name] = df_prepared[col_name].str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', default_patient_id_prefix).fillna(default_pid_prefix)
    return df_prepared


def prepare_clinic_patient_focus_overview_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str, # For logging and context
    patient_load_time_aggregation_period: str = 'W-MON', # Weekly, Monday start
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and flagged patient cases.
    """
    module_log_prefix = "ClinicPatientFocusPrep"
    logger.info(f"({module_log_prefix}) Preparing patient focus data. Period Context: {reporting_period_context_str}, Load Aggregation: {patient_load_time_aggregation_period}")

    # Define default column names for output DataFrames for schema consistency
    default_load_cols = ['period_start_date', 'condition', 'unique_patients_count']
    default_flagged_cols = [ # Example columns; should match output of get_patient_alerts_for_clinic
        'patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 
        'ai_risk_score', 'age', 'gender', 'zone_id'
    ]
    
    output_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "patient_load_by_key_condition_df": pd.DataFrame(columns=default_load_cols),
        "flagged_patients_for_review_df": pd.DataFrame(columns=default_flagged_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note = "No health data provided for patient focus data preparation. All outputs will be empty."
        logger.warning(f"({module_log_prefix}) {note}")
        output_data["processing_notes"].append(note)
        return output_data

    # --- Data Preparation for Patient Load ---
    # Use a copy for modifications specific to patient load calculation
    df_load_analysis_src = filtered_health_df_for_clinic_period.copy()
    
    pid_prefix_load = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15] # Unique prefix
    load_calc_cols_config = {
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'patient_id': {"default": f"UPID_Load_{pid_prefix_load}", "type": str},
        'condition': {"default": "UnknownCondition", "type": str}
    }
    df_load_analysis_prepared = _prepare_patient_focus_dataframe(
        df_load_analysis_src, load_calc_cols_config, 
        f"{module_log_prefix}/LoadPrep", f"UPID_Load_{pid_prefix_load}"
    )
    
    # Drop rows if essential columns for load analysis are NaT/empty after preparation
    df_load_analysis_prepared.dropna(subset=['encounter_date', 'patient_id', 'condition'], inplace=True)
    df_load_analysis_prepared = df_load_analysis_prepared[df_load_analysis_prepared['condition'] != "UnknownCondition"] # Filter out default unknown conditions

    if df_load_analysis_prepared.empty:
        note = "No valid records with encounter_date, patient_id, & condition after cleaning for patient load analysis."
        logger.warning(f"({module_log_prefix}) {note}")
        output_data["processing_notes"].append(note)
    else:
        # --- Patient Load by Key Condition ---
        key_conditions_for_load = _get_setting('KEY_CONDITIONS_FOR_ACTION', []) # List of strings
        if 'condition' in df_load_analysis_prepared.columns and key_conditions_for_load:
            aggregated_load_summaries: List[pd.DataFrame] = []
            for condition_key in key_conditions_for_load:
                if not isinstance(condition_key, str) or not condition_key.strip():
                    continue 
                try:
                    condition_mask = df_load_analysis_prepared['condition'].astype(str).str.contains(
                        re.escape(condition_key), case=False, na=False, regex=True
                    )
                    df_for_one_condition = df_load_analysis_prepared[condition_mask]
                    
                    if not df_for_one_condition.empty and 'patient_id' in df_for_one_condition.columns:
                        grouped_by_period = df_for_one_condition.groupby(
                            pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation_period, label='left', closed='left')
                        )['patient_id'].nunique().reset_index()
                        
                        grouped_by_period.rename(columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'}, inplace=True)
                        grouped_by_period['condition'] = condition_key
                        aggregated_load_summaries.append(grouped_by_period)
                except Exception as e_load_agg:
                    logger.error(f"({module_log_prefix}) Error aggregating load for condition '{condition_key}': {e_load_agg}", exc_info=True)
                    output_data["processing_notes"].append(f"Error processing patient load for condition: {condition_key}.")
            
            if aggregated_load_summaries:
                final_load_df = pd.concat(aggregated_load_summaries, ignore_index=True)
                # CORRECTED: Ensure the final concatenated DataFrame is assigned to the output.
                output_data["patient_load_by_key_condition_df"] = final_load_df[default_load_cols]
            else: output_data["processing_notes"].append("No patient load data aggregated for key conditions (empty after grouping or no key conditions processed).")
        elif df_load_analysis_prepared.empty: pass
        else:
            missing_reason_load = ""
            if 'condition' not in df_load_analysis_prepared.columns: missing_reason_load += "'condition' column missing. "
            if not key_conditions_for_load: missing_reason_load += "KEY_CONDITIONS_FOR_ACTION in config empty or not a list. "
            output_data["processing_notes"].append(f"Patient load by condition skipped. Reason: {missing_reason_load.strip()}")

    # --- Flagged Patient Cases for Clinical Review ---
    df_flagged_patients_final = pd.DataFrame(columns=default_flagged_cols) 
    try:
        risk_moderate_threshold = float(_get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60))
        
        alerts_df_from_component = get_patient_alerts_for_clinic(
            health_df_period=filtered_health_df_for_clinic_period.copy(),
            risk_threshold_moderate=risk_moderate_threshold,
            source_context=f"{module_log_prefix}/FlaggedPatients"
        )
        if isinstance(alerts_df_from_component, pd.DataFrame) and not alerts_df_from_component.empty:
            cols_to_use_flagged = [col for col in default_flagged_cols if col in alerts_df_from_component.columns]
            df_flagged_patients_final = alerts_df_from_component[cols_to_use_flagged]
            for col in default_flagged_cols:
                if col not in df_flagged_patients_final.columns:
                    df_flagged_patients_final[col] = np.nan
            df_flagged_patients_final = df_flagged_patients_final[default_flagged_cols]

            logger.info(f"({module_log_prefix}) Identified {len(df_flagged_patients_final)} patient cases flagged for clinical review.")
        elif isinstance(alerts_df_from_component, pd.DataFrame):
            note_no_flagged = "No patient cases flagged for clinical review in this period based on criteria by alerting component."
            logger.info(f"({module_log_prefix}) {note_no_flagged}")
            output_data["processing_notes"].append(note_no_flagged)
        else:
             note_bad_return = "Flagged patient component did not return a DataFrame."
             logger.warning(f"({module_log_prefix}) {note_bad_return}")
             output_data["processing_notes"].append(note_bad_return)

    except Exception as e_flagged_calc:
        logger.error(f"({module_log_prefix}) Error calling or processing output from get_patient_alerts_for_clinic: {e_flagged_calc}", exc_info=True)
        output_data["processing_notes"].append("Error generating list of flagged patients for review.")
    
    output_data["flagged_patients_for_review_df"] = df_flagged_patients_final

    logger.info(f"({module_log_prefix}) Clinic patient focus data preparation complete. Number of notes: {len(output_data['processing_notes'])}")
    return output_data
