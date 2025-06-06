# sentinel_project_root/pages/chw_components/alert_generation.py
# Processes CHW daily data to generate structured patient alert information for Sentinel.

import pandas as pd
import numpy as np
import logging
import re 
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import date as date_type, datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
    from analytics.protocol_executor import execute_escalation_protocol # Ensure this is robust
except ImportError as e:
    # Basic logger if import fails early, critical for module function
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in alert_generation.py: {e}. Ensure paths and dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Pre-compile regex for NA values if common_na_values_alerts is static and frequently used
# However, since it's constructed inside the function per call, pre-compilation benefit is limited here.
COMMON_NA_VALUES_ALERTS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_VALUES_ALERTS if s) + r')$' if COMMON_NA_VALUES_ALERTS else None


def _prepare_alert_dataframe(
    df: pd.DataFrame,
    alert_cols_config: Dict[str, Dict[str, Any]],
    chw_zone_context_str: str,
    processing_date_str: str,
    log_prefix: str
) -> pd.DataFrame:
    """Prepares the DataFrame by ensuring columns exist and have correct types."""
    df_prepared = df.copy()

    for col_name, config in alert_cols_config.items():
        default_value = config["default"]
        target_type_str = config["type"]
        is_mutable_default = isinstance(default_value, (list, dict))

        if col_name not in df_prepared.columns:
            # logger.debug(f"({log_prefix}) Column '{col_name}' not found. Adding with default.")
            if is_mutable_default:
                # CORRECTED: Use apply with a lambda to create a unique copy for each row
                df_prepared[col_name] = pd.Series([default_value.copy() for _ in range(len(df_prepared))], index=df_prepared.index)
            elif target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            else:
                 df_prepared[col_name] = default_value
        elif is_mutable_default:
            # CORRECTED: If column exists, fill NaNs by applying the lambda function
            fill_mask = df_prepared[col_name].isnull()
            if fill_mask.any():
                df_prepared.loc[fill_mask, col_name] = df_prepared.loc[fill_mask, col_name].apply(lambda x: default_value.copy())

        
        # Replace common string NAs with np.nan before type conversion for numeric/datetime
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(df_prepared[col_name].dtype):
            if NA_REGEX_PATTERN:
                try:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_PATTERN, np.nan, regex=True)
                except Exception as e_regex_replace: # Catch potential regex errors with weird inputs
                    logger.warning(f"({log_prefix}) Regex NA replacement failed for column '{col_name}': {e_regex_replace}. Proceeding with original values for this column.")


        # Convert to target type
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value)
            elif target_type_str == int:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=int)
            elif target_type_str == str and not is_mutable_default: # Don't process mutables as strings here
                df_prepared[col_name] = df_prepared[col_name].astype(str).fillna(str(default_value))
                if NA_REGEX_PATTERN:
                     df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_PATTERN, str(default_value), regex=True)
                df_prepared[col_name] = df_prepared[col_name].str.strip()

        except Exception as e_type_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to type '{target_type_str}': {e_type_conv}. Using default values for this column.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    # Specific default for patient_id if it ends up as the placeholder after processing
    placeholder_pid = f"UnknownPID_CHWAlert_{processing_date_str}"
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', placeholder_pid).fillna(placeholder_pid)

    return df_prepared


def generate_chw_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime], 
    chw_zone_context_str: str, # Used as a fallback if zone_id is missing
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily encounter data to generate structured patient alerts.
    Prioritizes critical alerts and then sorts by a raw priority score.
    Integrates escalation protocol execution for certain critical alerts.
    """
    module_log_prefix = "CHWAlertGeneration"

    try:
        processing_date_dt = pd.to_datetime(for_date, errors='coerce')
        if pd.NaT is processing_date_dt: # Check if conversion resulted in NaT
             raise ValueError(f"'for_date' ({for_date}) could not be parsed to a valid date object.")
        processing_date = processing_date_dt.date() # Extract date part
    except Exception as e_date_parse:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ('{for_date}'): {e_date_parse}. Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date() # Fallback to current date
    
    processing_date_str = processing_date.isoformat()
    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {processing_date_str}, zone context: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided for date {processing_date_str}. No alerts will be generated.")
        return []

    # Define column configurations for preparation
    # Ensure default values are appropriate for their types
    alert_cols_config = {
        'patient_id': {"default": f"UnknownPID_CHWAlert_{processing_date_str}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": chw_zone_context_str if chw_zone_context_str else "UnknownZone", "type": str},
        'condition': {"default": "N/A", "type": str},
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int}, 
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "N/A", "type": str}
    }
    
    df_alert_src = _prepare_alert_dataframe(
        patient_encounter_data_df, alert_cols_config, 
        chw_zone_context_str, processing_date_str, module_log_prefix
    )

    # Determine primary temperature source after preparation
    temp_col_name_to_use = None
    if 'vital_signs_temperature_celsius' in df_alert_src and df_alert_src['vital_signs_temperature_celsius'].notna().any():
        temp_col_name_to_use = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_alert_src and df_alert_src['max_skin_temp_celsius'].notna().any():
        temp_col_name_to_use = 'max_skin_temp_celsius'
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' as primary temperature source for alerts.")


    alerts_buffer_list: List[Dict[str, Any]] = []
    for _, encounter_row in df_alert_src.iterrows():
        patient_id_val = str(encounter_row.get('patient_id', alert_cols_config['patient_id']['default']))
        
        # Ensure encounter_date from row is valid and matches processing_date for relevance
        row_encounter_date = encounter_row.get('encounter_date')
        effective_encounter_date_str = processing_date_str # Default to processing date context
        
        # This alert is for 'processing_date', but we can use encounter_row's date for context if it's from today
        if pd.notna(row_encounter_date) and isinstance(row_encounter_date, pd.Timestamp) and row_encounter_date.date() == processing_date:
            effective_encounter_date_str_for_context = row_encounter_date.strftime('%Y-%m-%d')
        else:
            effective_encounter_date_str_for_context = processing_date_str # Fallback to processing_date if row's date is different or invalid

        alert_context_info = (
            f"Cond: {encounter_row.get('condition', 'N/A')} | "
            f"Zone: {encounter_row.get('zone_id', 'N/A')} | "
            f"DataDate: {effective_encounter_date_str_for_context}"
        )

        # --- Alert Rules ---
        # Rule 1: Critical Low SpO2
        min_spo2_val = encounter_row.get('min_spo2_pct', np.nan)
        spo2_critical_low_thresh = settings.ALERT_SPO2_CRITICAL_LOW_PCT if hasattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT') else 90.0
        if pd.notna(min_spo2_val) and min_spo2_val < spo2_critical_low_thresh:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98.0 + max(0, spo2_critical_low_thresh - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": processing_date_str # Alert date is the processing date
            })
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", encounter_row.to_dict(), 
                                        additional_context={"SPO2_VALUE": min_spo2_val, "PATIENT_AGE": encounter_row.get('age')})
            continue # Prioritize: if SpO2 is critical, don't also generate a warning SpO2 for this row

        # Rule 2: Warning Low SpO2
        spo2_warning_low_thresh = settings.ALERT_SPO2_WARNING_LOW_PCT if hasattr(settings, 'ALERT_SPO2_WARNING_LOW_PCT') else 94.0
        if pd.notna(min_spo2_val) and min_spo2_val < spo2_warning_low_thresh: # No 'continue' if already caught by critical
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                "raw_priority_score": 75.0 + max(0, spo2_warning_low_thresh - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": processing_date_str
            })

        # Rule 3: High Fever
        current_temp_val = encounter_row.get(temp_col_name_to_use, np.nan) if temp_col_name_to_use else np.nan
        temp_high_fever_thresh = settings.ALERT_BODY_TEMP_HIGH_FEVER_C if hasattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C') else 39.0
        if pd.notna(current_temp_val) and current_temp_val >= temp_high_fever_thresh:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "High Fever",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                "raw_priority_score": 95.0 + max(0, (current_temp_val - temp_high_fever_thresh) * 2.0),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": processing_date_str
            })
            # execute_escalation_protocol("PATIENT_HIGH_FEVER_CRITICAL", encounter_row.to_dict(), ...)
            continue # Prioritize: if Temp is critical, don't also generate a warning Temp for this row

        # Rule 4: Moderate Fever
        temp_fever_thresh = settings.ALERT_BODY_TEMP_FEVER_C if hasattr(settings, 'ALERT_BODY_TEMP_FEVER_C') else 38.0
        if pd.notna(current_temp_val) and current_temp_val >= temp_fever_thresh:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Fever Present",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR",
                "raw_priority_score": 70.0 + max(0, current_temp_val - temp_fever_thresh),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": processing_date_str
            })

        # Rule 5: Fall Detected
        fall_detected_val = int(encounter_row.get('fall_detected_today', 0)) # Ensure int after prep
        if fall_detected_val > 0:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls recorded: {fall_detected_val}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92.0, 
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Fall(s) = {fall_detected_val}", "encounter_date": processing_date_str
            })
            execute_escalation_protocol("PATIENT_FALL_DETECTED", encounter_row.to_dict())

        # Rule 6: High AI Follow-up Priority Score
        ai_followup_score = encounter_row.get('ai_followup_priority_score', np.nan)
        prio_score_high_thresh = settings.FATIGUE_INDEX_HIGH_THRESHOLD if hasattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD') else 0.7 # Example
        if pd.notna(ai_followup_score) and ai_followup_score >= prio_score_high_thresh: # Using a setting for threshold
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.",
                "brief_details": f"AI Prio Score: {ai_followup_score:.0f}", "suggested_action_code": "ACTION_AI_REVIEW_FOLLOWUP",
                "raw_priority_score": min(90.0, ai_followup_score), 
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"AI Prio {ai_followup_score:.0f}", "encounter_date": processing_date_str
            })

        # Rule 7: High AI Risk Score (Informational, if no other CRITICAL/WARNING for this patient on this processing_date)
        ai_risk_score_val = encounter_row.get('ai_risk_score', np.nan)
        risk_score_high_thresh = settings.RISK_SCORE_HIGH_THRESHOLD if hasattr(settings, 'RISK_SCORE_HIGH_THRESHOLD') else 75.0 # Example
        if pd.notna(ai_risk_score_val) and ai_risk_score_val >= risk_score_high_thresh:
            # Check if a more severe alert for this patient on this processing_date is already in the buffer
            has_more_severe_alert_for_processing_date = any(
                a['patient_id'] == patient_id_val and a['encounter_date'] == processing_date_str and \
                a['alert_level'] in ["CRITICAL", "WARNING"]
                for a in alerts_buffer_list 
            )
            if not has_more_severe_alert_for_processing_date:
                alerts_buffer_list.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk: {ai_risk_score_val:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_ROUTINE",
                    "raw_priority_score": min(60.0, ai_risk_score_val), # Lower than warnings
                    "patient_id": patient_id_val, "context_info": alert_context_info,
                    "triggering_value": f"AI Risk {ai_risk_score_val:.0f}", "encounter_date": processing_date_str
                })
        
        # Rule 8: Pending Critical Referral
        referral_status_val = str(encounter_row.get('referral_status', 'Unknown')).lower() 
        if referral_status_val == 'pending':
            key_conditions_for_action = settings.KEY_CONDITIONS_FOR_ACTION if hasattr(settings, 'KEY_CONDITIONS_FOR_ACTION') else []
            is_key_condition_for_referral = any(
                re.escape(key_c).lower() in str(encounter_row.get('condition', '')).lower() for key_c in key_conditions_for_action
            )
            if is_key_condition_for_referral:
                alerts_buffer_list.append({
                    "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {encounter_row.get('condition', 'N/A')}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                    "raw_priority_score": 80.0, 
                    "patient_id": patient_id_val, "context_info": alert_context_info,
                    "triggering_value": "Pending Critical Referral", "encounter_date": processing_date_str
                })

    # Deduplicate alerts: For a given (patient_id, encounter_date (which is processing_date_str here)), keep only the highest priority alert.
    if not alerts_buffer_list:
        logger.info(f"({module_log_prefix}) No CHW patient alerts generated from the provided data for {processing_date_str}.")
        return []

    # Deduplication key is (patient_id, effective_encounter_date for the alert)
    # Since all alerts generated in this run are for 'processing_date_str', the encounter_date in the alert dict is uniform.
    alerts_deduplicated_map: Dict[str, Dict[str, Any]] = {} # Key: patient_id (since encounter_date is processing_date_str for all)
    
    for alert_item_current in alerts_buffer_list:
        patient_id_key = alert_item_current['patient_id'] # All alerts are for processing_date_str
        
        # If patient not seen yet for this processing_date, or current alert is higher priority
        if patient_id_key not in alerts_deduplicated_map or \
           alert_item_current['raw_priority_score'] > alerts_deduplicated_map[patient_id_key]['raw_priority_score']:
            alerts_deduplicated_map[patient_id_key] = alert_item_current
    
    final_alerts_sorted_list = sorted(
        list(alerts_deduplicated_map.values()), 
        key=lambda x_alert: (
            {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x_alert.get("alert_level", "INFO"), 3), # Sort by alert level
            -x_alert.get('raw_priority_score', 0.0) # Then by priority score descending
        )
    )
    
    num_final_alerts = len(final_alerts_sorted_list)
    logger.info(f"({module_log_prefix}) Generated {num_final_alerts} unique CHW patient alerts after deduplication for {processing_date_str}.")
    
    return final_alerts_sorted_list[:max_alerts_to_return]
