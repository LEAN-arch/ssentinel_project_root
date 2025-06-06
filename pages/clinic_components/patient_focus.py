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

# Common NA strings for robust replacement
COMMON_NA_STRINGS_FOCUS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_FOCUS_PATTERN = r'^\s*$' + (r'|^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_FOCUS if s) + r')$' if any(COMMON_NA_STRINGS_FOCUS) else '')

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
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
            df_prepared[col_name] = default_value
        
        # Robust NA string and type conversion
        series = df_prepared[col_name]
        if target_type_str in [float, "datetime"] and pd.api.types.is_object_dtype(series.dtype):
            if NA_REGEX_FOCUS_PATTERN:
                series = series.replace(NA_REGEX_FOCUS_PATTERN, np.nan, regex=True)
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(series, errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(series, default_value=default_value)
            elif target_type_str == str:
                df_prepared[col_name] = series.fillna(str(default_value)).astype(str).str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting '{col_name}': {e_conv}. Using defaults.", exc_info=True)
            df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'].replace('', default_patient_id_prefix, inplace=True)
        df_prepared['patient_id'].fillna(default_patient_id_prefix, inplace=True)

    return df_prepared


def prepare_clinic_patient_focus_overview_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    patient_load_time_aggregation_period: str = 'W-MON',
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and flagged patient cases.
    """
    module_log_prefix = "ClinicPatientFocusPrep"
    logger.info(f"({module_log_prefix}) Preparing patient focus data for period: {reporting_period_context_str}")

    default_load_cols = ['period_start_date', 'condition', 'unique_patients_count']
    default_flagged_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
    
    output_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "patient_load_by_key_condition_df": pd.DataFrame(columns=default_load_cols),
        "flagged_patients_for_review_df": pd.DataFrame(columns=default_flagged_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note = "No health data provided for patient focus data preparation."
        logger.warning(f"({module_log_prefix}) {note}")
        output_data["processing_notes"].append(note)
        return output_data

    pid_prefix_load = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15]
    load_calc_cols_config = {
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'patient_id': {"default": f"UPID_Load_{pid_prefix_load}", "type": str},
        'condition': {"default": "UnknownCondition", "type": str}
    }
    df_load_analysis_prepared = _prepare_patient_focus_dataframe(
        filtered_health_df_for_clinic_period, load_calc_cols_config, 
        f"{module_log_prefix}/LoadPrep", f"UPID_Load_{pid_prefix_load}"
    )
    
    df_load_analysis_prepared.dropna(subset=['encounter_date', 'patient_id', 'condition'], inplace=True)
    df_load_analysis_prepared = df_load_analysis_prepared[df_load_analysis_prepared['condition'] != "UnknownCondition"]

    if not df_load_analysis_prepared.empty:
        key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
        if key_conditions:
            aggregated_summaries = []
            for condition in key_conditions:
                try:
                    mask = df_load_analysis_prepared['condition'].str.contains(re.escape(condition), case=False, na=False)
                    df_cond = df_load_analysis_prepared[mask]
                    if not df_cond.empty:
                        grouped = df_cond.groupby(pd.Grouper(key='encounter_date', freq=patient_load_time_aggregation_period))['patient_id'].nunique().reset_index()
                        grouped.rename(columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'}, inplace=True)
                        grouped['condition'] = condition
                        aggregated_summaries.append(grouped)
                except Exception as e:
                    logger.error(f"Error aggregating load for condition '{condition}': {e}", exc_info=True)
            
            if aggregated_summaries:
                final_load_df = pd.concat(aggregated_summaries, ignore_index=True)
                # CORRECTED: Ensure the final concatenated DataFrame is assigned to the output dictionary.
                output_data["patient_load_by_key_condition_df"] = final_load_df[default_load_cols]
    else:
        output_data["processing_notes"].append("No valid records for patient load analysis after cleaning.")

    try:
        risk_moderate_thresh = float(_get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60))
        alerts_df = get_patient_alerts_for_clinic(
            health_df_period=filtered_health_df_for_clinic_period,
            risk_threshold_moderate=risk_moderate_thresh,
            source_context=f"{module_log_prefix}/FlaggedPatients"
        )
        if isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty:
            output_data["flagged_patients_for_review_df"] = alerts_df.reindex(columns=default_flagged_cols, fill_value=np.nan)
        elif isinstance(alerts_df, pd.DataFrame):
            output_data["processing_notes"].append("No patients flagged for review in this period.")
    except Exception as e:
        logger.error(f"Error getting flagged patients: {e}", exc_info=True)
        output_data["processing_notes"].append("Error generating list of flagged patients.")

    logger.info(f"({module_log_prefix}) Patient focus data preparation complete. Notes: {len(output_data['processing_notes'])}")
    return output_data
