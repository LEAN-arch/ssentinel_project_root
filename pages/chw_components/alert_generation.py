# sentinel_project_root/pages/chw_components/alert_generation.py
# Processes CHW daily data to generate structured patient alert information for Sentinel.

import pandas as pd
import numpy as np
import logging
import re # For regex based NA string replacement
from typing import List, Dict, Any, Optional, Tuple
from datetime import date as date_type # For type hinting date objects

from config import settings
from data_processing.helpers import convert_to_numeric
from analytics.protocol_executor import execute_escalation_protocol

logger = logging.getLogger(__name__)


def generate_chw_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Any, 
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate structured patient alerts.
    Prioritizes critical alerts and then sorts by a raw priority score.
    Integrates escalation protocol execution for certain critical alerts.
    """
    module_log_prefix = "CHWAlertGeneration"

    try:
        processing_date = pd.to_datetime(for_date, errors='coerce').date()
        if pd.isna(processing_date): # Handle if for_date was unparseable
             raise ValueError("for_date could not be parsed to a valid date object.")
    except Exception as e_date_parse:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}): {e_date_parse}. Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date() # Fallback to current date
    processing_date_str = processing_date.isoformat()

    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {processing_date_str}, zone: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided for date {processing_date_str}. No alerts generated.")
        return []

    df_alert_src = patient_encounter_data_df.copy()

    alert_cols_config = {
        'patient_id': {"default": f"UnknownPID_CHWAlert_{processing_date_str}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": chw_zone_context_str or "UnknownZone", "type": str},
        'condition': {"default": "N/A", "type": str},
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int}, # Expect 0 or 1 after processing
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "N/A", "type": str}
    }
    common_na_values_alerts = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'] # Case-insensitive handled by regex

    for col_name, config_details in alert_cols_config.items():
        if col_name not in df_alert_src.columns:
            df_alert_src[col_name] = config_details["default"]
        
        current_col_dtype = df_alert_src[col_name].dtype
        if config_details["type"] in [float, int, "datetime"] and pd.api.types.is_object_dtype(current_col_dtype):
            # For object types that should be numeric/datetime, replace common NA strings with np.nan
            # Using regex for case-insensitive exact matches of NA strings
            na_regex_numeric = r'^(?:' + '|'.join(re.escape(s) for s in common_na_values_alerts if s) + r')$'
            if any(common_na_values_alerts): # Only if non-empty NA strings list
                 df_alert_src[col_name] = df_alert_src[col_name].replace(na_regex_numeric, np.nan, regex=True)
        
        if config_details["type"] == "datetime":
            df_alert_src[col_name] = pd.to_datetime(df_alert_src[col_name], errors='coerce')
        elif config_details["type"] == float:
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config_details["default"])
        elif config_details["type"] == int:
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config_details["default"], target_type=int)
        elif config_details["type"] == str:
            df_alert_src[col_name] = df_alert_src[col_name].astype(str).fillna(str(config_details["default"]))
            # For strings, replace common NA strings with the specific string default for this column
            na_regex_str = r'^(?:' + '|'.join(re.escape(s) for s in common_na_values_alerts if s) + r')$'
            if any(common_na_values_alerts):
                df_alert_src[col_name] = df_alert_src[col_name].replace(na_regex_str, str(config_details["default"]), regex=True)
            df_alert_src[col_name] = df_alert_src[col_name].str.strip()

    temp_col_name_to_use = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                                 if tc in df_alert_src.columns and df_alert_src[tc].notna().any()), None)
    if temp_col_name_to_use == 'max_skin_temp_celsius':
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' as primary temperature source for alerts.")

    alerts_buffer_list: List[Dict[str, Any]] = []
    for _, encounter_row in df_alert_src.iterrows():
        patient_id_val = str(encounter_row['patient_id'])
        
        row_encounter_date_obj = encounter_row['encounter_date']
        # Default to processing_date_str if row's date is invalid or does not match.
        # This ensures alerts are contextualized to the 'for_date'.
        effective_encounter_date_str = processing_date_str 
        if pd.notna(row_encounter_date_obj) and isinstance(row_encounter_date_obj, pd.Timestamp): # Check if it's a valid Timestamp
            if row_encounter_date_obj.date() == processing_date: # Focus on data relevant to processing_date
                 effective_encounter_date_str = row_encounter_date_obj.strftime('%Y-%m-%d')
            # Else, if data is from another date, alert is still for 'processing_date_str' review context
            # This assumes alerts are "what the CHW needs to know about today"

        alert_context_info = f"Cond: {encounter_row['condition']} | Zone: {encounter_row['zone_id']} | DataDate: {effective_encounter_date_str}"

        # Rule 1: Critical Low SpO2
        min_spo2_val = encounter_row['min_spo2_pct'] # Already numeric
        if pd.notna(min_spo2_val) and min_spo2_val < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98.0 + max(0, settings.ALERT_SPO2_CRITICAL_LOW_PCT - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": effective_encounter_date_str
            })
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", encounter_row.to_dict(), 
                                        additional_context={"SPO2_VALUE": min_spo2_val, "PATIENT_AGE": encounter_row.get('age')})
            continue # Prioritize critical SpO2 alert for this encounter_row

        # Rule 2: Warning Low SpO2 (if not already critical for this encounter_row)
        if pd.notna(min_spo2_val) and min_spo2_val < settings.ALERT_SPO2_WARNING_LOW_PCT:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                "raw_priority_score": 75.0 + max(0, settings.ALERT_SPO2_WARNING_LOW_PCT - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": effective_encounter_date_str
            })

        # Rule 3: High Fever
        current_temp_val = encounter_row.get(temp_col_name_to_use) if temp_col_name_to_use else np.nan
        if pd.notna(current_temp_val) and current_temp_val >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "High Fever",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                "raw_priority_score": 95.0 + max(0, (current_temp_val - settings.ALERT_BODY_TEMP_HIGH_FEVER_C) * 2.0),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": effective_encounter_date_str
            })
            # Consider execute_escalation_protocol("PATIENT_HIGH_FEVER_CRITICAL", ...)
            continue 

        # Rule 4: Moderate Fever (if not already high fever for this encounter_row)
        if pd.notna(current_temp_val) and current_temp_val >= settings.ALERT_BODY_TEMP_FEVER_C:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Fever Present",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR",
                "raw_priority_score": 70.0 + max(0, current_temp_val - settings.ALERT_BODY_TEMP_FEVER_C),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": effective_encounter_date_str
            })

        # Rule 5: Fall Detected
        fall_detected_val = encounter_row['fall_detected_today'] # Already int from prep
        if pd.notna(fall_detected_val) and fall_detected_val > 0:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls recorded: {int(fall_detected_val)}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92.0, # Fixed high score
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Fall(s) = {int(fall_detected_val)}", "encounter_date": effective_encounter_date_str
            })
            execute_escalation_protocol("PATIENT_FALL_DETECTED", encounter_row.to_dict())

        # Rule 6: High AI Follow-up Priority Score (Warning level)
        ai_followup_score = encounter_row['ai_followup_priority_score'] # Already numeric
        if pd.notna(ai_followup_score) and ai_followup_score >= settings.FATIGUE_INDEX_HIGH_THRESHOLD:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.",
                "brief_details": f"AI Prio Score: {ai_followup_score:.0f}", "suggested_action_code": "ACTION_AI_REVIEW_FOLLOWUP",
                "raw_priority_score": min(90.0, ai_followup_score), # Cap priority below criticals
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"AI Prio {ai_followup_score:.0f}", "encounter_date": effective_encounter_date_str
            })

        # Rule 7: High AI Risk Score (Informational, if no other CRITICAL/WARNING for this patient & date)
        ai_risk_score_val = encounter_row['ai_risk_score'] # Already numeric
        if pd.notna(ai_risk_score_val) and ai_risk_score_val >= settings.RISK_SCORE_HIGH_THRESHOLD:
            has_more_severe_alert_today = any(
                a['patient_id'] == patient_id_val and a['encounter_date'] == effective_encounter_date_str and \
                a['alert_level'] in ["CRITICAL", "WARNING"]
                for a in alerts_buffer_list # Check against already added alerts for THIS encounter_row processing cycle
            )
            if not has_more_severe_alert_today:
                alerts_buffer_list.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk: {ai_risk_score_val:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_ROUTINE",
                    "raw_priority_score": min(70.0, ai_risk_score_val), # Lower than warnings
                    "patient_id": patient_id_val, "context_info": alert_context_info,
                    "triggering_value": f"AI Risk {ai_risk_score_val:.0f}", "encounter_date": effective_encounter_date_str
                })
        
        # Rule 8: Pending Critical Referral (Warning level)
        referral_status_val = str(encounter_row['referral_status']).lower() # Already string from prep
        if referral_status_val == 'pending':
            is_key_condition_for_referral = any(
                re.escape(key_c).lower() in str(encounter_row['condition']).lower() for key_c in settings.KEY_CONDITIONS_FOR_ACTION
            ) # Use regex escape for safety if conditions have special chars
            if is_key_condition_for_referral:
                alerts_buffer_list.append({
                    "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {encounter_row['condition']}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                    "raw_priority_score": 80.0, # High warning
                    "patient_id": patient_id_val, "context_info": alert_context_info,
                    "triggering_value": "Pending Critical Referral", "encounter_date": effective_encounter_date_str
                })

    # Deduplicate alerts: For a given patient on a given encounter_date, keep only the highest priority alert.
    if alerts_buffer_list:
        alerts_deduplicated_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for alert_item_current in alerts_buffer_list:
            alert_dedup_key = (alert_item_current['patient_id'], alert_item_current['encounter_date'])
            
            if alert_dedup_key not in alerts_deduplicated_map or \
               alert_item_current['raw_priority_score'] > alerts_deduplicated_map[alert_dedup_key]['raw_priority_score']:
                alerts_deduplicated_map[alert_dedup_key] = alert_item_current
        
        final_alerts_sorted_list = sorted(
            list(alerts_deduplicated_map.values()), 
            key=lambda x_alert: (
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x_alert.get("alert_level", "INFO"), 3), # Sort by alert level first
                -x_alert.get('raw_priority_score', 0.0) # Then by priority score descending
            )
        )
        
        logger.info(f"({module_log_prefix}) Generated {len(final_alerts_sorted_list)} unique CHW patient alerts after deduplication for {processing_date_str}.")
        return final_alerts_sorted_list[:max_alerts_to_return]
    
    logger.info(f"({module_log_prefix}) No CHW patient alerts generated from the provided data for {processing_date_str}.")
    return []
