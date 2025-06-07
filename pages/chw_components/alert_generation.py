import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import date as date_type, datetime

# --- Logger Setup ---
# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Module Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from analytics.protocol_executor import execute_escalation_protocol
except ImportError as e:
    # Log a critical error if essential modules cannot be imported
    logger.error(f"Critical import error in alert_generation.py: {e}. Ensure paths and dependencies are correct.", exc_info=True)
    raise

# --- Constants ---
COMMON_NA_VALUES_ALERTS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])

# Reconstructed and valid regex pattern.
# This pattern matches any of the common NA values, surrounded by optional whitespace.
_pattern_parts = [re.escape(s) for s in COMMON_NA_VALUES_ALERTS if s]
NA_REGEX_PATTERN = (
    r'^\s*(?:' + '|'.join(_pattern_parts) + r')\s*$'
    if _pattern_parts
    else ''
)


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
            if is_mutable_default:
                df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            elif target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            else:
                 df_prepared[col_name] = default_value
        elif is_mutable_default:
            fill_mask = df_prepared[col_name].isnull()
            if fill_mask.any():
                df_prepared.loc[fill_mask, col_name] = df_prepared.loc[fill_mask, col_name].apply(
                    lambda x: default_value.copy() if pd.isna(x) else x
                )

        # Replace common string NAs with np.nan before type conversion for numeric/datetime
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(df_prepared[col_name].dtype):
            if NA_REGEX_PATTERN:
                try:
                    df_prepared[col_name].replace(NA_REGEX_PATTERN, np.nan, regex=True, inplace=True, case=False)
                except Exception as e_regex_replace:
                    logger.warning(f"({log_prefix}) Regex NA replacement failed for column '{col_name}': {e_regex_replace}. Proceeding with original values.")

        # Convert to target type
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=float)
            elif target_type_str == int:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=int)
            elif target_type_str == str and not is_mutable_default:
                series = df_prepared[col_name].fillna(str(default_value))
                df_prepared[col_name] = series.astype(str).str.strip()

        except Exception as e_type_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to type '{target_type_str}': {e_type_conv}. Using default values.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT:
                df_prepared[col_name] = pd.NaT
            else:
                df_prepared[col_name] = default_value
    
    placeholder_pid = f"UnknownPID_CHWAlert_{processing_date_str}"
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'].replace('', placeholder_pid, inplace=True)
        df_prepared['patient_id'].fillna(placeholder_pid, inplace=True)

    return df_prepared


def generate_chw_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context_str: str,  # Used as a fallback if zone_id is missing
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
        if pd.isna(processing_date_dt):
             raise ValueError(f"'for_date' ({for_date}) could not be parsed to a valid date object.")
        processing_date = processing_date_dt.date()
    except Exception as e_date_parse:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ('{for_date}'): {e_date_parse}. Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date()

    processing_date_str = processing_date.isoformat()
    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {processing_date_str}, zone context: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided for date {processing_date_str}. No alerts will be generated.")
        return []

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

    temp_col_name_to_use = None
    if 'vital_signs_temperature_celsius' in df_alert_src and df_alert_src['vital_signs_temperature_celsius'].notna().any():
        temp_col_name_to_use = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_alert_src and df_alert_src['max_skin_temp_celsius'].notna().any():
        temp_col_name_to_use = 'max_skin_temp_celsius'
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' as primary temperature source for alerts.")

    alerts_buffer_list: List[Dict[str, Any]] = []
    for _, encounter_row in df_alert_src.iterrows():
        patient_id_val = str(encounter_row.get('patient_id', alert_cols_config['patient_id']['default']))

        row_encounter_date = encounter_row.get('encounter_date')
        if pd.notna(row_encounter_date) and isinstance(row_encounter_date, pd.Timestamp) and row_encounter_date.date() == processing_date:
            effective_encounter_date_str_for_context = row_encounter_date.strftime('%Y-%m-%d')
        else:
            effective_encounter_date_str_for_context = processing_date_str

        alert_context_info = (
            f"Cond: {encounter_row.get('condition', 'N/A')} | "
            f"Zone: {encounter_row.get('zone_id', 'N/A')} | "
            f"DataDate: {effective_encounter_date_str_for_context}"
        )

        # --- Alert Rules ---
        min_spo2_val = encounter_row.get('min_spo2_pct', np.nan)
        spo2_critical_low_thresh = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90.0)
        if pd.notna(min_spo2_val) and min_spo2_val < spo2_critical_low_thresh:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98.0 + max(0, spo2_critical_low_thresh - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": processing_date_str
            })
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", encounter_row.to_dict(),
                                        additional_context={"SPO2_VALUE": min_spo2_val, "PATIENT_AGE": encounter_row.get('age')})
            continue

        spo2_warning_low_thresh = getattr(settings, 'ALERT_SPO2_WARNING_LOW_PCT', 94.0)
        if pd.notna(min_spo2_val) and min_spo2_val < spo2_warning_low_thresh:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                "raw_priority_score": 75.0 + max(0, spo2_warning_low_thresh - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": processing_date_str
            })

        current_temp_val = encounter_row.get(temp_col_name_to_use, np.nan) if temp_col_name_to_use else np.nan
        temp_high_fever_thresh = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
        if pd.notna(current_temp_val) and current_temp_val >= temp_high_fever_thresh:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "High Fever",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                "raw_priority_score": 95.0 + max(0, (current_temp_val - temp_high_fever_thresh) * 2.0),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": processing_date_str
            })
            continue

        temp_fever_thresh = getattr(settings, 'ALERT_BODY_TEMP_FEVER_C', 38.0)
        if pd.notna(current_temp_val) and current_temp_val >= temp_fever_thresh:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Fever Present",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR",
                "raw_priority_score": 70.0 + max(0, current_temp_val - temp_fever_thresh),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": processing_date_str
            })

        fall_detected_val = int(encounter_row.get('fall_detected_today', 0))
        if fall_detected_val > 0:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls recorded: {fall_detected_val}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92.0,
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Fall(s) = {fall_detected_val}", "encounter_date": processing_date_str
            })
            execute_escalation_protocol("PATIENT_FALL_DETECTED", encounter_row.to_dict())
            continue

        ai_followup_score = encounter_row.get('ai_followup_priority_score', np.nan)
        prio_score_high_thresh = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80.0)
        if pd.notna(ai_followup_score) and ai_followup_score >= prio_score_high_thresh:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.",
                "brief_details": f"AI Prio Score: {ai_followup_score:.0f}", "suggested_action_code": "ACTION_AI_REVIEW_FOLLOWUP",
                "raw_priority_score": min(90.0, ai_followup_score),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"AI Prio {ai_followup_score:.0f}", "encounter_date": processing_date_str
            })

        ai_risk_score_val = encounter_row.get('ai_risk_score', np.nan)
        risk_score_high_thresh = getattr(settings, 'RISK_SCORE_HIGH_THRESHOLD', 75.0)
        if pd.notna(ai_risk_score_val) and ai_risk_score_val >= risk_score_high_thresh:
            has_more_severe_alert_for_processing_date = any(
                a['patient_id'] == patient_id_val and a['encounter_date'] == processing_date_str and \
                a['alert_level'] in ["CRITICAL", "WARNING"]
                for a in alerts_buffer_list
            )
            if not has_more_severe_alert_for_processing_date:
                alerts_buffer_list.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk: {ai_risk_score_val:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_ROUTINE",
                    "raw_priority_score": min(60.0, ai_risk_score_val),
                    "patient_id": patient_id_val, "context_info": alert_context_info,
                    "triggering_value": f"AI Risk {ai_risk_score_val:.0f}", "encounter_date": processing_date_str
                })

        referral_status_val = str(encounter_row.get('referral_status', 'Unknown')).lower()
        if referral_status_val == 'pending':
            key_conditions_for_action = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
            is_key_condition_for_referral = any(
                key_c.lower() in str(encounter_row.get('condition', '')).lower() for key_c in key_conditions_for_action
            )
            if is_key_condition_for_referral:
                alerts_buffer_list.append({
                    "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {encounter_row.get('condition', 'N/A')}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                    "raw_priority_score": 80.0,
                    "patient_id": patient_id_val, "context_info": alert_context_info,
                    "triggering_value": "Pending Critical Referral", "encounter_date": processing_date_str
                })

    if not alerts_buffer_list:
        logger.info(f"({module_log_prefix}) No CHW patient alerts generated from the provided data for {processing_date_str}.")
        return []

    # Deduplicate alerts, keeping only the highest-priority alert for each patient
    alerts_deduplicated_map: Dict[str, Dict[str, Any]] = {}
    for alert_item_current in alerts_buffer_list:
        patient_id_key = alert_item_current['patient_id']

        if patient_id_key not in alerts_deduplicated_map or \
           alert_item_current['raw_priority_score'] > alerts_deduplicated_map[patient_id_key]['raw_priority_score']:
            alerts_deduplicated_map[patient_id_key] = alert_item_current

    # Sort the final list of unique alerts by level (Critical > Warning > Info) and then by priority score
    final_alerts_sorted_list = sorted(
        list(alerts_deduplicated_map.values()),
        key=lambda x_alert: (
            {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x_alert.get("alert_level", "INFO"), 3),
            -x_alert.get('raw_priority_score', 0.0)
        )
    )
    
    # FIXED: Removed duplicated block of code from here.
    num_final_alerts = len(final_alerts_sorted_list)
    logger.info(f"({module_log_prefix}) Generated {num_final_alerts} unique CHW patient alerts after deduplication for {processing_date_str}.")

    return final_alerts_sorted_list[:max_alerts_to_return]
