# sentinel_project_root/pages/chw_components/alert_generation.py
# Processes CHW daily data to generate structured patient alert information for Sentinel.
# Renamed from alert_generator.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import date # For type hinting and date operations

from config import settings # Use new settings module
from data_processing.helpers import convert_to_numeric # Local import
# Import protocol_executor from analytics module
from analytics.protocol_executor import execute_escalation_protocol

logger = logging.getLogger(__name__)


def generate_chw_alerts( # Renamed from generate_chw_patient_alerts_from_data
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Any, # Expects a date or date-like string for context
    chw_zone_context_str: str, # Zone context for these alerts
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts.
    Prioritizes critical alerts and then sorts by a raw priority score.
    Integrates escalation protocol execution for certain critical alerts.
    """
    module_log_prefix = "CHWAlertGeneration" # Renamed for clarity

    # Standardize for_date
    try:
        # Ensure for_date is a date object for consistent comparison and string formatting
        processing_date = pd.to_datetime(for_date).date() if for_date else pd.Timestamp('now').date()
    except Exception:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' provided ({for_date}). Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date() # Fallback to current date
    processing_date_str = processing_date.isoformat() # Use ISO format string for consistency

    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {processing_date_str}, zone: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided for date {processing_date_str}. No alerts will be generated.")
        return [] # Return empty list if no data

    df_alert_src = patient_encounter_data_df.copy()

    # Define expected columns and their configurations with defaults
    # Ensure defaults are of the type expected after processing to avoid downstream errors
    alert_cols_config = {
        'patient_id': {"default": f"UnknownPID_CHWAlert_{processing_date_str}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Will be converted to datetime
        'zone_id': {"default": chw_zone_context_str or "UnknownZone", "type": str},
        'condition': {"default": "N/A", "type": str}, # Using "N/A" consistently for unknown conditions
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float}, # Alternative temp source
        'fall_detected_today': {"default": 0, "type": int}, # Expect 0 or 1 after processing
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "N/A", "type": str}
        # Add any other columns from health_records that might be needed by escalation protocols
    }
    common_na_values_for_alerts = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown'] # Added 'unknown'

    # Prepare DataFrame: ensure columns exist, handle NaNs, and coerce types
    for col_name, config_details in alert_cols_config.items():
        if col_name not in df_alert_src.columns:
            logger.debug(f"({module_log_prefix}) Column '{col_name}' not found in input DataFrame, adding with default: {config_details['default']}")
            df_alert_src[col_name] = config_details["default"]
        
        # Standardize NA strings to actual NaN for numeric/datetime, or a consistent string for str types
        if config_details["type"] in [float, int, "datetime"]:
            if df_alert_src[col_name].dtype == 'object': # Only replace if it's object type currently
                 df_alert_src[col_name] = df_alert_src[col_name].replace(common_na_values_for_alerts, np.nan)
        
        # Type coercion
        if config_details["type"] == "datetime":
            df_alert_src[col_name] = pd.to_datetime(df_alert_src[col_name], errors='coerce')
        elif config_details["type"] == float:
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config_details["default"]) # Use helper
        elif config_details["type"] == int: # For flags like fall_detected_today
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config_details["default"], target_type=int) # Use helper
        elif config_details["type"] == str:
            df_alert_src[col_name] = df_alert_src[col_name].astype(str).fillna(str(config_details["default"]))
            # Replace common NA strings with the defined string default for this column
            df_alert_src[col_name] = df_alert_src[col_name].replace(common_na_values_for_alerts, str(config_details["default"]), regex=False)
            df_alert_src[col_name] = df_alert_src[col_name].str.strip() # Strip whitespace

    # Select temperature column: prioritize vital_signs_temperature_celsius
    temp_col_name_to_use = None
    if 'vital_signs_temperature_celsius' in df_alert_src.columns and df_alert_src['vital_signs_temperature_celsius'].notna().any():
        temp_col_name_to_use = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_alert_src.columns and df_alert_src['max_skin_temp_celsius'].notna().any():
        temp_col_name_to_use = 'max_skin_temp_celsius'
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' as primary temperature source for alerts.")


    alerts_buffer_list: List[Dict[str, Any]] = [] # Renamed for clarity

    # Iterate through rows to generate alerts based on rules
    # This ensures that each row (potential encounter) is evaluated independently.
    for _, encounter_row in df_alert_src.iterrows():
        patient_id_val = str(encounter_row['patient_id'])
        # Use the encounter_date from the row if valid, else use the processing_date_str for context
        # This is important if the df_alert_src contains data from multiple dates but we only care about 'processing_date' triggers
        encounter_date_for_alert = encounter_row['encounter_date']
        if pd.isna(encounter_date_for_alert) or encounter_date_for_alert.date() != processing_date:
            # If encounter date is not the processing date, this data point might be historical
            # For CHW daily alerts, we typically only care about *today's* new findings.
            # However, a pending referral from yesterday might still be relevant today.
            # For simplicity here, we'll use the row's date if valid, otherwise the processing_date.
            # More sophisticated logic could filter df_alert_src to only processing_date upfront.
            effective_encounter_date_str = processing_date_str if pd.isna(encounter_date_for_alert) else encounter_date_for_alert.strftime('%Y-%m-%d')
        else:
            effective_encounter_date_str = encounter_date_for_alert.strftime('%Y-%m-%d')
        
        alert_context_info = f"Cond: {encounter_row['condition']} | Zone: {encounter_row['zone_id']} | Date: {effective_encounter_date_str}"

        # Rule 1: Critical Low SpO2
        min_spo2_val = encounter_row['min_spo2_pct']
        if pd.notna(min_spo2_val) and min_spo2_val < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            alerts_buffer_list.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98.0 + max(0, settings.ALERT_SPO2_CRITICAL_LOW_PCT - min_spo2_val),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": effective_encounter_date_str
            })
            # Trigger escalation protocol
            execute_escalation_protocol(
                "PATIENT_CRITICAL_SPO2_LOW", 
                encounter_row.to_dict(), 
                additional_context={"SPO2_VALUE": min_spo2_val, "PATIENT_AGE": encounter_row.get('age')}
            )
            continue # Prioritize critical: if this alert is made, don't make a lesser SpO2 alert for same encounter

        # Rule 2: Warning Low SpO2
        if pd.notna(min_spo2_val) and min_spo2_val < settings.ALERT_SPO2_WARNING_LOW_PCT:
            # No need to check if already critical due to `continue` above
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
            # Potentially trigger escalation for very high fever if defined
            # execute_escalation_protocol("PATIENT_HIGH_FEVER_CRITICAL", encounter_row.to_dict(), {"TEMP_VALUE": current_temp_val})
            continue # Prioritize critical fever

        # Rule 4: Moderate Fever
        if pd.notna(current_temp_val) and current_temp_val >= settings.ALERT_BODY_TEMP_FEVER_C:
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "Fever Present",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR",
                "raw_priority_score": 70.0 + max(0, current_temp_val - settings.ALERT_BODY_TEMP_FEVER_C),
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": effective_encounter_date_str
            })

        # Rule 5: Fall Detected
        fall_detected_val = encounter_row['fall_detected_today'] # Already int after prep
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
        ai_followup_score = encounter_row['ai_followup_priority_score']
        if pd.notna(ai_followup_score) and ai_followup_score >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: # Using as general high prio threshold
            alerts_buffer_list.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.",
                "brief_details": f"AI Prio Score: {ai_followup_score:.0f}", "suggested_action_code": "ACTION_AI_REVIEW_FOLLOWUP",
                "raw_priority_score": min(90.0, ai_followup_score), # Cap priority from this rule below criticals
                "patient_id": patient_id_val, "context_info": alert_context_info,
                "triggering_value": f"AI Prio {ai_followup_score:.0f}", "encounter_date": effective_encounter_date_str
            })

        # Rule 7: High AI Risk Score (Informational, if no other CRITICAL/WARNING alert for this patient & date)
        ai_risk_score_val = encounter_row['ai_risk_score']
        if pd.notna(ai_risk_score_val) and ai_risk_score_val >= settings.RISK_SCORE_HIGH_THRESHOLD:
            # Check if a more severe alert already exists for this patient on this date in the buffer
            has_more_severe_alert_today = any(
                a['patient_id'] == patient_id_val and a['encounter_date'] == effective_encounter_date_str and \
                a['alert_level'] in ["CRITICAL", "WARNING"]
                for a in alerts_buffer_list
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
        referral_status_val = str(encounter_row['referral_status']).lower()
        if referral_status_val == 'pending':
            is_key_condition_for_referral = any(
                key_c.lower() in str(encounter_row['condition']).lower() for key_c in settings.KEY_CONDITIONS_FOR_ACTION
            )
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
            # Key for deduplication: patient_id and the specific encounter_date of the alert data
            alert_dedup_key = (alert_item_current['patient_id'], alert_item_current['encounter_date'])
            
            if alert_dedup_key not in alerts_deduplicated_map or \
               alert_item_current['raw_priority_score'] > alerts_deduplicated_map[alert_dedup_key]['raw_priority_score']:
                alerts_deduplicated_map[alert_dedup_key] = alert_item_current
        
        final_alerts_sorted_list = sorted(
            list(alerts_deduplicated_map.values()), 
            key=lambda x: (
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x.get("alert_level", "INFO"), 3), # Sort by alert level first
                -x.get('raw_priority_score', 0) # Then by priority score descending
            )
        )
        
        logger.info(f"({module_log_prefix}) Generated {len(final_alerts_sorted_list)} unique CHW patient alerts after deduplication for {processing_date_str}.")
        return final_alerts_sorted_list[:max_alerts_to_return] # Return top N alerts
    
    logger.info(f"({module_log_prefix}) No CHW patient alerts generated from the provided data for {processing_date_str}.")
    return []
