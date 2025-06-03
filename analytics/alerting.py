# sentinel_project_root/analytics/alerting.py
# Logic for generating alerts from health data for CHW and Clinic dashboards.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import date

from config import settings # Use the new settings module
from data_processing.helpers import convert_to_numeric # Local import
from .protocol_executor import execute_escalation_protocol # For triggering protocols on alert generation

logger = logging.getLogger(__name__)


def generate_chw_patient_alerts( # Renamed from generate_chw_patient_alerts_from_data
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Any, # Expects a date or date-like string for context
    chw_zone_context_str: str, # Zone context for these alerts
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts.
    Prioritizes critical alerts and then sorts by a raw priority score.
    """
    module_log_prefix = "CHWPatientAlertGen"

    try:
        processing_date = pd.to_datetime(for_date).date() if for_date else pd.Timestamp('now').date()
    except Exception:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}). Defaulting to current date.")
        processing_date = pd.Timestamp('now').date()
    processing_date_str = processing_date.isoformat()

    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {processing_date_str}, zone: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided for date {processing_date_str}.")
        return []

    df_alert_src = patient_encounter_data_df.copy()

    # Define expected columns and their configurations with defaults
    # Ensure defaults are of the type expected after processing to avoid downstream errors
    alert_cols_config = {
        'patient_id': {"default": "UnknownPID_Alert", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Will be converted to datetime
        'zone_id': {"default": chw_zone_context_str, "type": str},
        'condition': {"default": "N/A", "type": str}, # Using "N/A" consistently
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float}, # Alternative temp source
        'fall_detected_today': {"default": 0, "type": int}, # Expect 0 or 1 after processing
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "N/A", "type": str}
    }
    common_na_values = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL']

    # Prepare DataFrame: ensure columns exist, handle NaNs, and coerce types
    for col_name, config in alert_cols_config.items():
        if col_name not in df_alert_src.columns:
            logger.debug(f"({module_log_prefix}) Column '{col_name}' not found, adding with default: {config['default']}")
            df_alert_src[col_name] = config["default"]
        
        # Standardize NA strings to actual NaN for numeric/datetime, or a consistent string for str types
        if config["type"] in [float, int, "datetime"]:
            if df_alert_src[col_name].dtype == 'object': # Only replace if it's object type currently
                 df_alert_src[col_name] = df_alert_src[col_name].replace(common_na_values, np.nan)
        
        if config["type"] == "datetime":
            df_alert_src[col_name] = pd.to_datetime(df_alert_src[col_name], errors='coerce')
        elif config["type"] == float:
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config["default"])
        elif config["type"] == int: # For flags like fall_detected_today
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config["default"], target_type=int)
        elif config["type"] == str:
            df_alert_src[col_name] = df_alert_src[col_name].astype(str).fillna(str(config["default"]))
            df_alert_src[col_name] = df_alert_src[col_name].replace(common_na_values, str(config["default"]), regex=False)
            df_alert_src[col_name] = df_alert_src[col_name].str.strip()

    # Select temperature column: prioritize vital_signs_temperature_celsius
    temp_col_to_use = None
    if 'vital_signs_temperature_celsius' in df_alert_src.columns and df_alert_src['vital_signs_temperature_celsius'].notna().any():
        temp_col_to_use = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_alert_src.columns and df_alert_src['max_skin_temp_celsius'].notna().any():
        temp_col_to_use = 'max_skin_temp_celsius'
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' as primary temperature source due to missing/empty 'vital_signs_temperature_celsius'.")


    alerts_buffer: List[Dict[str, Any]] = []

    # Iterate through rows to generate alerts based on rules
    for _, row in df_alert_src.iterrows():
        patient_id_str = str(row['patient_id'])
        # Use the encounter_date from the row if valid and matches processing_date, else use processing_date_str for context
        row_encounter_date_str = row['encounter_date'].strftime('%Y-%m-%d') if pd.notna(row['encounter_date']) else processing_date_str
        
        # Context for the alert
        context_info_str = f"Cond: {row['condition']} | Zone: {row['zone_id']} | Date: {row_encounter_date_str}"

        # Rule 1: Critical Low SpO2
        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {row['min_spo2_pct']:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98 + max(0, settings.ALERT_SPO2_CRITICAL_LOW_PCT - row['min_spo2_pct']), # Higher deviation = higher score
                "patient_id": patient_id_str, "context_info": context_info_str,
                "triggering_value": f"SpO2 {row['min_spo2_pct']:.0f}%", "encounter_date": row_encounter_date_str
            })
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", row.to_dict(), {"SPO2_VALUE": row['min_spo2_pct']})


        # Rule 2: Warning Low SpO2 (if not already critical)
        elif pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < settings.ALERT_SPO2_WARNING_LOW_PCT:
            # Check if a critical SpO2 alert for this patient & date is already in buffer
            is_already_critical_spo2 = any(
                a['patient_id'] == patient_id_str and a['encounter_date'] == row_encounter_date_str and a['primary_reason'] == "Critical Low SpO2"
                for a in alerts_buffer
            )
            if not is_already_critical_spo2:
                alerts_buffer.append({
                    "alert_level": "WARNING", "primary_reason": "Low SpO2",
                    "brief_details": f"SpO2: {row['min_spo2_pct']:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                    "raw_priority_score": 75 + max(0, settings.ALERT_SPO2_WARNING_LOW_PCT - row['min_spo2_pct']),
                    "patient_id": patient_id_str, "context_info": context_info_str,
                    "triggering_value": f"SpO2 {row['min_spo2_pct']:.0f}%", "encounter_date": row_encounter_date_str
                })

        # Rule 3: High Fever
        if temp_col_to_use and pd.notna(row[temp_col_to_use]) and row[temp_col_to_use] >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "High Fever", # Changed from "Fever" to "High Fever"
                "brief_details": f"Temp: {row[temp_col_to_use]:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                "raw_priority_score": 95 + max(0, (row[temp_col_to_use] - settings.ALERT_BODY_TEMP_HIGH_FEVER_C) * 2),
                "patient_id": patient_id_str, "context_info": context_info_str,
                "triggering_value": f"Temp {row[temp_col_to_use]:.1f}°C", "encounter_date": row_encounter_date_str
            })
        
        # Rule 4: Moderate Fever (if not already high fever)
        elif temp_col_to_use and pd.notna(row[temp_col_to_use]) and row[temp_col_to_use] >= settings.ALERT_BODY_TEMP_FEVER_C:
            is_already_high_fever = any(
                a['patient_id'] == patient_id_str and a['encounter_date'] == row_encounter_date_str and a['primary_reason'] == "High Fever"
                for a in alerts_buffer
            )
            if not is_already_high_fever:
                alerts_buffer.append({
                    "alert_level": "WARNING", "primary_reason": "Fever Present",
                    "brief_details": f"Temp: {row[temp_col_to_use]:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR",
                    "raw_priority_score": 70 + max(0, row[temp_col_to_use] - settings.ALERT_BODY_TEMP_FEVER_C),
                    "patient_id": patient_id_str, "context_info": context_info_str,
                    "triggering_value": f"Temp {row[temp_col_to_use]:.1f}°C", "encounter_date": row_encounter_date_str
                })

        # Rule 5: Fall Detected
        if pd.notna(row['fall_detected_today']) and row['fall_detected_today'] > 0:
            alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls recorded: {int(row['fall_detected_today'])}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92, # Fixed high score for any fall
                "patient_id": patient_id_str, "context_info": context_info_str,
                "triggering_value": "Fall(s) > 0", "encounter_date": row_encounter_date_str
            })
            # Potentially trigger escalation protocol for fall
            execute_escalation_protocol("PATIENT_FALL_DETECTED", row.to_dict())


        # Rule 6: High AI Follow-up Priority Score (from AI engine)
        if pd.notna(row['ai_followup_priority_score']) and row['ai_followup_priority_score'] >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: # Re-using fatigue threshold as general high prio
            alerts_buffer.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.", # Renamed for clarity
                "brief_details": f"AI Prio Score: {row['ai_followup_priority_score']:.0f}", "suggested_action_code": "ACTION_AI_REVIEW_FOLLOWUP",
                "raw_priority_score": min(90, row['ai_followup_priority_score']), # Cap priority from this rule
                "patient_id": patient_id_str, "context_info": context_info_str,
                "triggering_value": f"AI Prio {row['ai_followup_priority_score']:.0f}", "encounter_date": row_encounter_date_str
            })

        # Rule 7: High AI Risk Score (Informational, if no other CRITICAL/WARNING alert for this patient on this day)
        if pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= settings.RISK_SCORE_HIGH_THRESHOLD:
            is_already_critical_or_warning = any(
                a['patient_id'] == patient_id_str and a['encounter_date'] == row_encounter_date_str and a['alert_level'] in ["CRITICAL", "WARNING"]
                for a in alerts_buffer
            )
            if not is_already_critical_or_warning:
                alerts_buffer.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk: {row['ai_risk_score']:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_ROUTINE",
                    "raw_priority_score": min(70, row['ai_risk_score']), # Lower priority than warnings
                    "patient_id": patient_id_str, "context_info": context_info_str,
                    "triggering_value": f"AI Risk {row['ai_risk_score']:.0f}", "encounter_date": row_encounter_date_str
                })
        
        # Rule 8: Pending Critical Referral
        if str(row['referral_status']).lower() == 'pending':
            is_key_condition_referral = any(
                key_c.lower() in str(row['condition']).lower() for key_c in settings.KEY_CONDITIONS_FOR_ACTION
            )
            if is_key_condition_referral:
                alerts_buffer.append({
                    "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                    "brief_details": f"For: {row['condition']}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                    "raw_priority_score": 80, # High warning
                    "patient_id": patient_id_str, "context_info": context_info_str,
                    "triggering_value": "Pending Critical Referral", "encounter_date": row_encounter_date_str
                })


    # Deduplicate alerts: For a given patient on a given encounter_date, keep only the highest priority alert.
    # The encounter_date here is the date associated with the data that triggered the alert.
    if alerts_buffer:
        alerts_deduplicated_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for alert_item in alerts_buffer:
            alert_key = (alert_item['patient_id'], alert_item['encounter_date']) # Use encounter_date from alert
            if alert_key not in alerts_deduplicated_map or \
               alert_item['raw_priority_score'] > alerts_deduplicated_map[alert_key]['raw_priority_score']:
                alerts_deduplicated_map[alert_key] = alert_item
        
        final_alerts_list = sorted(list(alerts_deduplicated_map.values()), key=lambda x: x['raw_priority_score'], reverse=True)
        
        logger.info(f"({module_log_prefix}) Generated {len(final_alerts_list)} unique CHW patient alerts after deduplication for {processing_date_str}.")
        return final_alerts_list[:max_alerts_to_return] # Return top N alerts
    
    logger.info(f"({module_log_prefix}) No CHW patient alerts generated from the provided data for {processing_date_str}.")
    return []


def get_patient_alerts_for_clinic(
    health_df_period: Optional[pd.DataFrame],
    risk_threshold_moderate: Optional[float] = None, # Allow override, else use config
    source_context: str = "ClinicPatientAlerts"
) -> pd.DataFrame:
    """
    Identifies patients needing clinical review based on AI risk, critical vitals, or other flags.
    This is a simplified version focusing on identifying patients, not generating detailed alert objects like CHW alerts.
    """
    logger.info(f"({source_context}) Generating patient alerts list for clinic review.")

    output_columns = [
        'patient_id', 'encounter_date', 'condition', 'Alert Reason',
        'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id',
        'referred_to_facility_id', 'min_spo2_pct', 'vital_signs_temperature_celsius' # Added vitals for context
    ]
    empty_alerts_df = pd.DataFrame(columns=output_columns)

    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) Health DataFrame for clinic alerts is empty or invalid.")
        return empty_alerts_df

    df = health_df_period.copy()
    
    # Define defaults and ensure columns exist (similar to CHW alert prep)
    cols_to_ensure = {
        'patient_id': "UnknownPID_ClinicAlert", 'encounter_date': pd.NaT, 'condition': "N/A",
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, 'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'referral_status': "Unknown",
        'age': np.nan, 'gender': "Unknown", 'zone_id': "UnknownZone",
        'referred_to_facility_id': "N/A"
    }
    common_na_values_clinic = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL']

    for col, default in cols_to_ensure.items():
        if col not in df.columns:
            df[col] = default
        if col == 'encounter_date':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif isinstance(default, (float, int)) or default is np.nan:
            if df[col].dtype == 'object': # Replace NAs if object type before numeric conversion
                 df[col] = df[col].replace(common_na_values_clinic, np.nan)
            df[col] = convert_to_numeric(df[col], default_value=default)
        else: # String columns
            df[col] = df[col].astype(str).fillna(str(default))
            df[col] = df[col].replace(common_na_values_clinic, str(default), regex=False)
            df[col] = df[col].str.strip()

    # Filter for most recent encounter per patient in the period if multiple exist
    if 'encounter_date' in df.columns and df['encounter_date'].notna().any():
        df_latest_encounters = df.sort_values('encounter_date').drop_duplicates(subset=['patient_id'], keep='last')
    else:
        df_latest_encounters = df.drop_duplicates(subset=['patient_id'], keep='last') # Fallback if no valid dates
        logger.warning(f"({source_context}) 'encounter_date' missing or all NaT for clinic alert patient selection. Using last record per patient_id.")


    alert_list_for_clinic = []
    
    # Use configured risk threshold or fallback
    actual_risk_threshold = risk_threshold_moderate if risk_threshold_moderate is not None else settings.RISK_SCORE_MODERATE_THRESHOLD

    for _, row in df_latest_encounters.iterrows():
        alert_reason_clinic = ""
        priority_score_clinic = row.get('ai_followup_priority_score', 0.0) # Start with AI prio
        if pd.isna(priority_score_clinic): priority_score_clinic = 0.0


        if pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= settings.RISK_SCORE_HIGH_THRESHOLD:
            alert_reason_clinic = f"High AI Risk ({row['ai_risk_score']:.0f})"
            priority_score_clinic = max(priority_score_clinic, row['ai_risk_score'])
        elif pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= actual_risk_threshold:
            alert_reason_clinic = f"Moderate AI Risk ({row['ai_risk_score']:.0f})"
            priority_score_clinic = max(priority_score_clinic, row['ai_risk_score'] * 0.8) # Moderate risk gets slightly lower base prio

        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            alert_reason_clinic += ("; " if alert_reason_clinic else "") + f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)"
            priority_score_clinic = max(priority_score_clinic, 95) # Very high priority
            # No protocol execution here, this is for clinic review list

        current_temp = row.get('vital_signs_temperature_celsius', np.nan)
        if pd.notna(current_temp) and current_temp >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            alert_reason_clinic += ("; " if alert_reason_clinic else "") + f"High Fever ({current_temp:.1f}°C)"
            priority_score_clinic = max(priority_score_clinic, 90)

        if str(row['referral_status']).lower() == 'pending' and \
           any(kc.lower() in str(row['condition']).lower() for kc in settings.KEY_CONDITIONS_FOR_ACTION):
            alert_reason_clinic += ("; " if alert_reason_clinic else "") + f"Pending Critical Referral ({row['condition']})"
            priority_score_clinic = max(priority_score_clinic, 85)
            
        if not alert_reason_clinic and pd.notna(row['ai_followup_priority_score']) and row['ai_followup_priority_score'] >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD:
            # If no other major alert, but AI followup prio is moderate/high, flag it.
            alert_reason_clinic = f"AI Follow-up Prio. ({row['ai_followup_priority_score']:.0f})"
            # Priority score is already set from ai_followup_priority_score

        if alert_reason_clinic: # Only add if there's a reason
            alert_list_for_clinic.append({
                'patient_id': str(row['patient_id']),
                'encounter_date': row['encounter_date'], # Keep as datetime for potential sorting
                'condition': str(row['condition']),
                'Alert Reason': alert_reason_clinic.strip("; "),
                'Priority Score': round(min(priority_score_clinic, 100),1),
                'ai_risk_score': row['ai_risk_score'] if pd.notna(row['ai_risk_score']) else np.nan,
                'age': row['age'] if pd.notna(row['age']) else np.nan,
                'gender': str(row['gender']),
                'zone_id': str(row['zone_id']),
                'referred_to_facility_id': str(row['referred_to_facility_id']),
                'min_spo2_pct': row['min_spo2_pct'] if pd.notna(row['min_spo2_pct']) else np.nan,
                'vital_signs_temperature_celsius': current_temp if pd.notna(current_temp) else np.nan
            })

    if not alert_list_for_clinic:
        logger.info(f"({source_context}) No patients met criteria for clinic alert/review list.")
        return empty_alerts_df

    alerts_df_for_clinic = pd.DataFrame(alert_list_for_clinic)
    # Sort by Priority Score (desc) then by encounter_date (desc)
    alerts_df_for_clinic.sort_values(by=['Priority Score', 'encounter_date'], ascending=[False, False], inplace=True)
    
    # Ensure all expected columns are present before returning
    for col in output_columns:
        if col not in alerts_df_for_clinic.columns:
            alerts_df_for_clinic[col] = np.nan # Add missing columns with NaN

    logger.info(f"({source_context}) Generated {len(alerts_df_for_clinic)} patient entries for clinic review list.")
    return alerts_df_for_clinic[output_columns].reset_index(drop=True) # Return with consistent column order
