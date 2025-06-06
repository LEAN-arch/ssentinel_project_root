# sentinel_project_root/analytics/alerting.py
# Logic for generating alerts from health data for CHW and Clinic dashboards.

import pandas as pd
import numpy as np
import logging
import re  # Added this import
from typing import List, Dict, Any, Optional, Tuple 
from datetime import date as date_type # For type hinting date objects

from config import settings
from data_processing.helpers import convert_to_numeric
from .protocol_executor import execute_escalation_protocol

logger = logging.getLogger(__name__)

def generate_chw_patient_alerts(
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
    module_log_prefix = "CHWPatientAlertGen"

    try:
        processing_date = pd.to_datetime(for_date, errors='coerce').date() if for_date else pd.Timestamp('now').date()
        if pd.isna(processing_date): # Handle if for_date was unparseable
            raise ValueError("for_date could not be parsed to a valid date.")
    except Exception as e_date:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}): {e_date}. Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date()
    processing_date_str = processing_date.isoformat()

    logger.info(f"({module_log_prefix}) Generating CHW patient alerts for date: {processing_date_str}, zone: {chw_zone_context_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({module_log_prefix}) No patient encounter data provided for date {processing_date_str}. No alerts generated.")
        return []

    df_alert_src = patient_encounter_data_df.copy()

    alert_cols_config = {
        'patient_id': {"default": f"UnknownPID_Alert_{processing_date_str}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": chw_zone_context_str or "UnknownZone", "type": str},
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
    common_na_values = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'] # Expanded list

    for col_name, config_details in alert_cols_config.items():
        if col_name not in df_alert_src.columns:
            df_alert_src[col_name] = config_details["default"]
        
        if config_details["type"] in [float, int, "datetime"] and df_alert_src[col_name].dtype == 'object':
            df_alert_src[col_name] = df_alert_src[col_name].replace(common_na_values, np.nan) # Case-insensitive replace handled by common_na_values
        
        if config_details["type"] == "datetime": 
            df_alert_src[col_name] = pd.to_datetime(df_alert_src[col_name], errors='coerce')
        elif config_details["type"] == float: 
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config_details["default"])
        elif config_details["type"] == int: 
            df_alert_src[col_name] = convert_to_numeric(df_alert_src[col_name], default_value=config_details["default"], target_type=int)
        elif config_details["type"] == str:
            df_alert_src[col_name] = df_alert_src[col_name].astype(str).fillna(str(config_details["default"]))
            # For strings, direct replace of common_na_values might be too broad if they are valid substrings.
            # Better to replace only if the entire string matches one of the NA values.
            # Using a regex with word boundaries for exact match of NA strings, case-insensitive.
            na_regex = r'^(?:' + '|'.join(re.escape(s) for s in common_na_values if s) + r')$' # Avoid empty string in regex if present
            if any(common_na_values): # Only if there are non-empty NA strings
                 df_alert_src[col_name] = df_alert_src[col_name].str.replace(na_regex, str(config_details["default"]), case=False, regex=True)
            df_alert_src[col_name] = df_alert_src[col_name].str.strip()


    temp_col_to_use = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                            if tc in df_alert_src.columns and df_alert_src[tc].notna().any()), None)
    if temp_col_to_use == 'max_skin_temp_celsius': 
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' as primary temperature source.")

    alerts_buffer: List[Dict[str, Any]] = []
    for _, encounter_row in df_alert_src.iterrows():
        patient_id_val = str(encounter_row['patient_id'])
        
        row_encounter_date = encounter_row['encounter_date']
        # Effective date for the alert context: if row's date is valid and matches processing_date, use it.
        # Else, use processing_date_str. This focuses alerts on "today's findings".
        if pd.notna(row_encounter_date) and row_encounter_date.date() == processing_date:
            effective_encounter_date_str = row_encounter_date.strftime('%Y-%m-%d')
        else:
            # If data is from a different date, or date is invalid, use processing_date for this alert context
            # This implies alerts are primarily for "today's" review of events that occurred today.
            # For alerts about historical data (e.g. pending referral from yesterday), logic might need adjustment
            # or `df_alert_src` should be pre-filtered more carefully.
            effective_encounter_date_str = processing_date_str
        
        context_info_str = f"Cond: {encounter_row['condition']} | Zone: {encounter_row['zone_id']} | Date: {effective_encounter_date_str}"

        # Rule 1: Critical Low SpO2
        min_spo2_val = encounter_row['min_spo2_pct']
        if pd.notna(min_spo2_val) and min_spo2_val < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT",
                "raw_priority_score": 98.0 + max(0, settings.ALERT_SPO2_CRITICAL_LOW_PCT - min_spo2_val),
                "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": effective_encounter_date_str
            })
            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", encounter_row.to_dict(), 
                                        additional_context={"SPO2_VALUE": min_spo2_val, "PATIENT_AGE": encounter_row.get('age')})
            continue # Prioritize critical SpO2 alert for this encounter

        # Rule 2: Warning Low SpO2 (if not already critical for this encounter)
        if pd.notna(min_spo2_val) and min_spo2_val < settings.ALERT_SPO2_WARNING_LOW_PCT:
            alerts_buffer.append({
                "alert_level": "WARNING", "primary_reason": "Low SpO2",
                "brief_details": f"SpO2: {min_spo2_val:.0f}%", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR",
                "raw_priority_score": 75.0 + max(0, settings.ALERT_SPO2_WARNING_LOW_PCT - min_spo2_val),
                "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": f"SpO2 {min_spo2_val:.0f}%", "encounter_date": effective_encounter_date_str
            })

        # Rule 3: High Fever
        current_temp_val = encounter_row.get(temp_col_to_use) if temp_col_to_use else np.nan
        if pd.notna(current_temp_val) and current_temp_val >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "High Fever",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT",
                "raw_priority_score": 95.0 + max(0, (current_temp_val - settings.ALERT_BODY_TEMP_HIGH_FEVER_C) * 2.0),
                "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": effective_encounter_date_str
            })
            # execute_escalation_protocol("PATIENT_HIGH_FEVER_CRITICAL", encounter_row.to_dict(), {"TEMP_VALUE": current_temp_val})
            continue # Prioritize critical fever

        # Rule 4: Moderate Fever (if not already high fever for this encounter)
        if pd.notna(current_temp_val) and current_temp_val >= settings.ALERT_BODY_TEMP_FEVER_C:
            alerts_buffer.append({
                "alert_level": "WARNING", "primary_reason": "Fever Present",
                "brief_details": f"Temp: {current_temp_val:.1f}°C", "suggested_action_code": "ACTION_FEVER_MONITOR",
                "raw_priority_score": 70.0 + max(0, current_temp_val - settings.ALERT_BODY_TEMP_FEVER_C),
                "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": f"Temp {current_temp_val:.1f}°C", "encounter_date": effective_encounter_date_str
            })

        # Rule 5: Fall Detected
        fall_detected_val = encounter_row['fall_detected_today']
        if pd.notna(fall_detected_val) and fall_detected_val > 0:
            alerts_buffer.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls recorded: {int(fall_detected_val)}", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT",
                "raw_priority_score": 92.0, "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": f"Fall(s) = {int(fall_detected_val)}", "encounter_date": effective_encounter_date_str
            })
            execute_escalation_protocol("PATIENT_FALL_DETECTED", encounter_row.to_dict())

        # Rule 6: High AI Follow-up Priority Score
        ai_followup_score = encounter_row['ai_followup_priority_score']
        if pd.notna(ai_followup_score) and ai_followup_score >= settings.FATIGUE_INDEX_HIGH_THRESHOLD:
            alerts_buffer.append({
                "alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio.",
                "brief_details": f"AI Prio Score: {ai_followup_score:.0f}", "suggested_action_code": "ACTION_AI_REVIEW_FOLLOWUP",
                "raw_priority_score": min(90.0, ai_followup_score),
                "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": f"AI Prio {ai_followup_score:.0f}", "encounter_date": effective_encounter_date_str
            })

        # Rule 7: High AI Risk Score (INFO if no other CRITICAL/WARNING for this patient-date)
        ai_risk_score_val = encounter_row['ai_risk_score']
        if pd.notna(ai_risk_score_val) and ai_risk_score_val >= settings.RISK_SCORE_HIGH_THRESHOLD:
            is_more_severe = any(
                a['patient_id'] == patient_id_val and a['encounter_date'] == effective_encounter_date_str and \
                a['alert_level'] in ["CRITICAL", "WARNING"] for a in alerts_buffer
            )
            if not is_more_severe:
                alerts_buffer.append({
                    "alert_level": "INFO", "primary_reason": "Elevated AI Risk Score",
                    "brief_details": f"AI Risk: {ai_risk_score_val:.0f}", "suggested_action_code": "ACTION_MONITOR_RISK_ROUTINE",
                    "raw_priority_score": min(70.0, ai_risk_score_val),
                    "patient_id": patient_id_val, "context_info": context_info_str,
                    "triggering_value": f"AI Risk {ai_risk_score_val:.0f}", "encounter_date": effective_encounter_date_str
                })
        
        # Rule 8: Pending Critical Referral
        if str(encounter_row['referral_status']).lower() == 'pending' and \
           any(kc.lower() in str(encounter_row['condition']).lower() for kc in settings.KEY_CONDITIONS_FOR_ACTION):
            alerts_buffer.append({
                "alert_level": "WARNING", "primary_reason": "Pending Critical Referral",
                "brief_details": f"For: {encounter_row['condition']}", "suggested_action_code": "ACTION_FOLLOWUP_REFERRAL_STATUS",
                "raw_priority_score": 80.0, "patient_id": patient_id_val, "context_info": context_info_str,
                "triggering_value": "Pending Critical Referral", "encounter_date": effective_encounter_date_str
            })

    if alerts_buffer:
        alerts_deduplicated_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for alert_item in alerts_buffer:
            alert_key = (alert_item['patient_id'], alert_item['encounter_date'])
            if alert_key not in alerts_deduplicated_map or \
               alert_item['raw_priority_score'] > alerts_deduplicated_map[alert_key]['raw_priority_score']:
                alerts_deduplicated_map[alert_key] = alert_item
        
        final_alerts_list = sorted(
            list(alerts_deduplicated_map.values()), 
            key=lambda x_alert: ({"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x_alert.get("alert_level", "INFO"), 3), 
                                 -x_alert.get('raw_priority_score', 0.0)) # Sort by level, then descending priority score
        )
        logger.info(f"({module_log_prefix}) Generated {len(final_alerts_list)} unique CHW patient alerts after deduplication for {processing_date_str}.")
        return final_alerts_list[:max_alerts_to_return]
    
    logger.info(f"({module_log_prefix}) No CHW patient alerts generated from the provided data for {processing_date_str}.")
    return []


def get_patient_alerts_for_clinic(
    health_df_period: Optional[pd.DataFrame],
    risk_threshold_moderate: Optional[float] = None,
    source_context: str = "ClinicPatientAlerts"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Generating patient alerts list for clinic review.")
    output_columns = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 
                      'ai_risk_score', 'age', 'gender', 'zone_id', 'referred_to_facility_id',
                      'min_spo2_pct', 'vital_signs_temperature_celsius']
    empty_alerts_df = pd.DataFrame(columns=output_columns)

    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) Health DataFrame for clinic alerts is empty or invalid.")
        return empty_alerts_df

    df = health_df_period.copy()
    cols_to_ensure = {
        'patient_id': "UnknownPID_ClinicAlert", 'encounter_date': pd.NaT, 'condition': "N/A",
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, 'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'referral_status': "Unknown",
        'age': np.nan, 'gender': "Unknown", 'zone_id': "UnknownZone",
        'referred_to_facility_id': "N/A"
    }
    common_na_values_clinic = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']

    for col, default in cols_to_ensure.items():
        if col not in df.columns: df[col] = default
        if col == 'encounter_date': df[col] = pd.to_datetime(df[col], errors='coerce')
        elif isinstance(default, (float, int)) or default is np.nan:
            if df[col].dtype == 'object': df[col] = df[col].replace(common_na_values_clinic, np.nan)
            df[col] = convert_to_numeric(df[col], default_value=default)
        else: # String columns
            df[col] = df[col].astype(str).fillna(str(default))
            df[col] = df[col].replace(common_na_values_clinic, str(default), regex=False).str.strip()

    if 'encounter_date' in df.columns and df['encounter_date'].notna().any():
        # Sort by encounter_date first (ascending so NaTs are first), then drop duplicates keeping last
        df_latest_encounters = df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['patient_id'], keep='last')
    else:
        df_latest_encounters = df.drop_duplicates(subset=['patient_id'], keep='last')
        logger.warning(f"({source_context}) 'encounter_date' missing or all NaT. Using last record per patient_id for clinic alerts.")

    alert_list_for_clinic: List[Dict[str, Any]] = []
    actual_risk_threshold = risk_threshold_moderate if risk_threshold_moderate is not None else settings.RISK_SCORE_MODERATE_THRESHOLD

    for _, row in df_latest_encounters.iterrows():
        alert_reason_clinic = ""
        priority_score_clinic = row.get('ai_followup_priority_score', 0.0)
        if pd.isna(priority_score_clinic): priority_score_clinic = 0.0

        if pd.notna(row['ai_risk_score']):
            if row['ai_risk_score'] >= settings.RISK_SCORE_HIGH_THRESHOLD:
                alert_reason_clinic = f"High AI Risk ({row['ai_risk_score']:.0f})"
                priority_score_clinic = max(priority_score_clinic, row['ai_risk_score'])
            elif row['ai_risk_score'] >= actual_risk_threshold:
                alert_reason_clinic = f"Moderate AI Risk ({row['ai_risk_score']:.0f})"
                priority_score_clinic = max(priority_score_clinic, row['ai_risk_score'] * 0.8)
        
        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            alert_reason_clinic += ("; " if alert_reason_clinic else "") + f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)"
            priority_score_clinic = max(priority_score_clinic, 95.0)

        current_temp = row.get('vital_signs_temperature_celsius', np.nan)
        if pd.notna(current_temp) and current_temp >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            alert_reason_clinic += ("; " if alert_reason_clinic else "") + f"High Fever ({current_temp:.1f}°C)"
            priority_score_clinic = max(priority_score_clinic, 90.0)

        if str(row['referral_status']).lower() == 'pending' and \
           any(kc.lower() in str(row['condition']).lower() for kc in settings.KEY_CONDITIONS_FOR_ACTION):
            alert_reason_clinic += ("; " if alert_reason_clinic else "") + f"Pending Critical Referral ({row['condition']})"
            priority_score_clinic = max(priority_score_clinic, 85.0)
            
        if not alert_reason_clinic and pd.notna(row['ai_followup_priority_score']) and \
           row['ai_followup_priority_score'] >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD:
            alert_reason_clinic = f"AI Follow-up Prio. ({row['ai_followup_priority_score']:.0f})"
        
        if alert_reason_clinic:
            alert_list_for_clinic.append({
                'patient_id': str(row['patient_id']), 'encounter_date': row['encounter_date'],
                'condition': str(row['condition']), 'Alert Reason': alert_reason_clinic.strip("; "),
                'Priority Score': round(min(priority_score_clinic, 100.0), 1),
                'ai_risk_score': row['ai_risk_score'], 'age': row['age'], 'gender': str(row['gender']),
                'zone_id': str(row['zone_id']), 'referred_to_facility_id': str(row['referred_to_facility_id']),
                'min_spo2_pct': row['min_spo2_pct'], 'vital_signs_temperature_celsius': current_temp
            })

    if not alert_list_for_clinic:
        logger.info(f"({source_context}) No patients met criteria for clinic alert/review list.")
        return empty_alerts_df

    alerts_df_for_clinic = pd.DataFrame(alert_list_for_clinic)
    alerts_df_for_clinic.sort_values(by=['Priority Score', 'encounter_date'], ascending=[False, False], inplace=True, na_position='last')
    
    for col in output_columns: # Ensure all expected columns are present before returning
        if col not in alerts_df_for_clinic.columns:
            alerts_df_for_clinic[col] = np.nan
            
    logger.info(f"({source_context}) Generated {len(alerts_df_for_clinic)} patient entries for clinic review list.")
    return alerts_df_for_clinic[output_columns].reset_index(drop=True)
        if col not in alerts_df_for_clinic.columns:
            alerts_df_for_clinic[col] = np.nan
            
    logger.info(f"({source_context}) Generated {len(alerts_df_for_clinic)} patient entries for clinic review list.")
    return alerts_df_for_clinic[output_columns].reset_index(drop=True)
