# sentinel_project_root/analytics/alerting.py
# Centralized logic for generating alerts from health data for all dashboards.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import standardize_missing_values
    from .protocol_executor import execute_escalation_protocol
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in alerting.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

# --- Column Configuration ---
# Centralizes the expected columns for alert generation for consistency.
ALERT_COLS_CONFIG_NUMERIC = {
    'age': np.nan, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
    'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
    'max_skin_temp_celsius': np.nan, 'fall_detected_today': 0
}
ALERT_COLS_CONFIG_STRING = {
    'patient_id': "UnknownPID_Alert", 'zone_id': "UnknownZone",
    'condition': "N/A", 'referral_status': "Unknown"
}


def generate_chw_patient_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts
    using a scalable, rule-based engine.
    """
    log_prefix = "CHWPatientAlertGen"

    try:
        processing_date = pd.to_datetime(for_date).date()
    except (ValueError, TypeError):
        logger.warning(f"({log_prefix}) Invalid 'for_date' ('{for_date}'). Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date()

    processing_date_str = processing_date.isoformat()
    logger.info(f"({log_prefix}) Generating CHW patient alerts for date: {processing_date_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        return []

    df = standardize_missing_values(
        patient_encounter_data_df, ALERT_COLS_CONFIG_STRING, ALERT_COLS_CONFIG_NUMERIC
    )
    if 'encounter_date' in df.columns:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        df = df[df['encounter_date'].dt.date == processing_date].copy()
    if df.empty: return []

    temp_col = 'vital_signs_temperature_celsius' if not df['vital_signs_temperature_celsius'].isnull().all() else 'max_skin_temp_celsius'

    # --- Rule-Based Alert Engine ---
    alert_rules = [
        {"level": "CRITICAL", "reason": "Critical Low SpO2", "field": "min_spo2_pct", "cond": lambda v: v < settings.Thresholds.SPO2_CRITICAL_LOW, "details": "SpO2: {v:.0f}%", "action": "ACTION_SPO2_MANAGE_URGENT", "prio": 98, "prio_factor": -1.0, "protocol": "PATIENT_CRITICAL_SPO2_LOW"},
        {"level": "CRITICAL", "reason": "High Fever", "field": temp_col, "cond": lambda v: v >= settings.Thresholds.BODY_TEMP_HIGH_FEVER, "details": "Temp: {v:.1f}°C", "action": "ACTION_FEVER_MANAGE_URGENT", "prio": 95, "prio_factor": 2.0, "protocol": "PATIENT_CRITICAL_HIGH_FEVER"},
        {"level": "CRITICAL", "reason": "Fall Detected", "field": "fall_detected_today", "cond": lambda v: v > 0, "details": "Falls recorded: {v:.0f}", "action": "ACTION_FALL_ASSESS_URGENT", "prio": 92, "prio_factor": 0},
        {"level": "WARNING", "reason": "Low SpO2", "field": "min_spo2_pct", "cond": lambda v: v < settings.Thresholds.SPO2_WARNING_LOW, "details": "SpO2: {v:.0f}%", "action": "ACTION_SPO2_RECHECK_MONITOR", "prio": 75, "prio_factor": -1.0},
        {"level": "WARNING", "reason": "Fever Present", "field": temp_col, "cond": lambda v: v >= settings.Thresholds.BODY_TEMP_FEVER, "details": "Temp: {v:.1f}°C", "action": "ACTION_FEVER_MONITOR", "prio": 70, "prio_factor": 1.0},
        {"level": "WARNING", "reason": "Pending Critical Referral", "field": "referral_status", "cond": lambda v, r: str(v).lower() == 'pending' and any(kc.lower() in str(r.get('condition')).lower() for kc in settings.Semantics.KEY_CONDITIONS_FOR_ACTION), "details": "For: {r[condition]}", "action": "ACTION_FOLLOWUP_REFERRAL_STATUS", "prio": 80, "prio_factor": 0},
        {"level": "INFO", "reason": "High AI Follow-up Prio.", "field": "ai_followup_priority_score", "cond": lambda v: v >= settings.Thresholds.FOLLOWUP_PRIORITY_HIGH, "details": "AI Prio Score: {v:.0f}", "action": "ACTION_AI_REVIEW_FOLLOWUP", "prio": min(70, v if pd.notna(v) else 70), "prio_factor": 0},
    ]

    alerts_buffer: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        # A single alert per patient row to avoid over-alerting from one encounter
        for rule in alert_rules:
            val = row.get(rule["field"])
            if pd.isna(val): continue
            
            try:
                # Pass row for context-aware conditions
                is_triggered = rule["cond"](val, row) if "r" in rule["cond"].__code__.co_varnames else rule["cond"](val)
            except Exception: continue

            if is_triggered:
                p_score = rule["prio"] + (float(val) * rule["prio_factor"])
                alerts_buffer.append({
                    "alert_level": rule["level"], "primary_reason": rule["reason"],
                    "brief_details": rule["details"].format(v=val, r=row), "suggested_action_code": rule["action"],
                    "raw_priority_score": np.clip(p_score, 0, 100), "patient_id": str(row['patient_id']),
                    "context_info": f"Cond: {row['condition']} | Zone: {row['zone_id']}", "triggering_value": f"{rule['field']}={val}",
                    "encounter_date": processing_date_str
                })
                if rule.get("protocol"):
                    execute_escalation_protocol(rule["protocol"], row.to_dict(), {"TRIGGERING_VALUE": val})
                break # IMPORTANT: prevents multiple alerts for the same patient from one pass

    if not alerts_buffer: return []

    # Deduplicate: Keep only the highest priority alert for each patient
    alerts_deduped = {a['patient_id']: a for a in sorted(alerts_buffer, key=lambda x: x['raw_priority_score'])}
    final_alerts = sorted(list(alerts_deduped.values()), key=lambda x: ({"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x["alert_level"], 3), -x['raw_priority_score']))
    
    logger.info(f"({log_prefix}) Generated {len(final_alerts)} unique CHW patient alerts for {processing_date_str}.")
    return final_alerts[:max_alerts_to_return]


def get_patient_alerts_for_clinic(
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicPatientAlerts"
) -> pd.DataFrame:
    """
    Generates a DataFrame of patients requiring clinical review based on risk and vitals.
    """
    logger.info(f"({source_context}) Generating patient alerts list for clinic review.")
    output_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return pd.DataFrame(columns=output_cols)

    df = standardize_missing_values(health_df_period, ALERT_COLS_CONFIG_STRING, ALERT_COLS_CONFIG_NUMERIC)
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    df_latest = df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['patient_id'], keep='last')

    alert_list = []
    for _, row in df_latest.iterrows():
        reasons, priority_score = [], 0.0
        
        # Rule-based reason aggregation
        if pd.notna(row['ai_risk_score']):
            if row['ai_risk_score'] >= settings.Thresholds.RISK_SCORE_HIGH:
                reasons.append(f"High AI Risk ({row['ai_risk_score']:.0f})")
                priority_score = max(priority_score, row['ai_risk_score'])
            elif row['ai_risk_score'] >= settings.Thresholds.RISK_SCORE_MODERATE:
                reasons.append(f"Mod. AI Risk ({row['ai_risk_score']:.0f})")
                priority_score = max(priority_score, row['ai_risk_score'] * 0.8)

        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < settings.Thresholds.SPO2_CRITICAL_LOW:
            reasons.append(f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)")
            priority_score = max(priority_score, 95.0)

        temp = row.get('vital_signs_temperature_celsius', np.nan)
        if pd.notna(temp) and temp >= settings.Thresholds.BODY_TEMP_HIGH_FEVER:
            reasons.append(f"High Fever ({temp:.1f}°C)")
            priority_score = max(priority_score, 90.0)

        if str(row['referral_status']).lower() == 'pending' and any(kc.lower() in str(row['condition']).lower() for kc in settings.Semantics.KEY_CONDITIONS_FOR_ACTION):
            reasons.append(f"Pending Critical Referral")
            priority_score = max(priority_score, 85.0)
            
        if reasons:
            row_dict = row.to_dict()
            row_dict['Alert Reason'] = "; ".join(reasons)
            row_dict['Priority Score'] = round(min(priority_score, 100.0), 1)
            alert_list.append(row_dict)

    if not alert_list:
        return pd.DataFrame(columns=output_cols)

    alerts_df = pd.DataFrame(alert_list)
    alerts_df.sort_values(by=['Priority Score', 'encounter_date'], ascending=[False, False], inplace=True)
    
    # Ensure all required output columns are present
    for col in output_cols:
        if col not in alerts_df.columns:
            alerts_df[col] = np.nan
            
    return alerts_df[output_cols].reset_index(drop=True)
