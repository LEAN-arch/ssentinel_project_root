# sentinel_project_root/pages/chw_components/alert_generation.py
# Processes CHW daily data to generate structured patient alert information for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric, standardize_missing_values
    from analytics.protocol_executor import execute_escalation_protocol
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in alert_generation.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

# --- Column Configuration ---
# Centralizes the expected columns and their properties for easier management.
ALERT_COLS_CONFIG = {
    'patient_id': {"default": "UnknownPID_Alert", "type": str},
    'encounter_date': {"default": pd.NaT, "type": "datetime"},
    'zone_id': {"default": "UnknownZone", "type": str},
    'condition': {"default": "N/A", "type": str},
    'age': {"default": np.nan, "type": float},
    'ai_risk_score': {"default": np.nan, "type": float},
    'ai_followup_priority_score': {"default": np.nan, "type": float},
    'min_spo2_pct': {"default": np.nan, "type": float},
    'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
    'max_skin_temp_celsius': {"default": np.nan, "type": float},
    'fall_detected_today': {"default": 0, "type": int},
    'referral_status': {"default": "Unknown", "type": str},
    'referral_reason': {"default": "N/A", "type": str},
    'blood_pressure_systolic': {"default": np.nan, "type": float},
    'blood_pressure_diastolic': {"default": np.nan, "type": float}
}


def _prepare_alert_dataframe(
    df: pd.DataFrame,
    processing_date_str: str,
    log_prefix: str
) -> pd.DataFrame:
    """Prepares the DataFrame by ensuring columns exist and have correct types."""
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=list(ALERT_COLS_CONFIG.keys()))

    # Use the robust standardize_missing_values helper
    numeric_defaults = {k: v['default'] for k, v in ALERT_COLS_CONFIG.items() if v['type'] in [int, float]}
    string_defaults = {k: v['default'] for k, v in ALERT_COLS_CONFIG.items() if v['type'] == str}
    
    # Dynamically update the default for patient_id with the processing date
    string_defaults['patient_id'] = f"UnknownPID_Alert_{processing_date_str}"
    
    df_prepared = standardize_missing_values(df, string_defaults, numeric_defaults)

    # Convert datetime columns
    date_cols = [k for k, v in ALERT_COLS_CONFIG.items() if v['type'] == 'datetime']
    for col in date_cols:
        if col in df_prepared.columns:
            df_prepared[col] = pd.to_datetime(df_prepared[col], errors='coerce')

    return df_prepared


def generate_chw_alerts(
    patient_encounter_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context_str: str,
    max_alerts_to_return: int = 15
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily encounter data to generate structured patient alerts
    using a scalable, rule-based engine.

    This function identifies critical and warning conditions, prioritizes them,
    and integrates with the escalation protocol system.
    """
    log_prefix = "CHWAlertGeneration"

    try:
        processing_date = pd.to_datetime(for_date).date()
    except (ValueError, TypeError):
        logger.warning(f"({log_prefix}) Invalid 'for_date' ('{for_date}'). Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date()
    
    processing_date_str = processing_date.isoformat()
    logger.info(f"({log_prefix}) Generating CHW patient alerts for date: {processing_date_str}")

    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        logger.warning(f"({log_prefix}) No patient encounter data provided for {processing_date_str}.")
        return []

    df_alert_src = _prepare_alert_dataframe(patient_encounter_data_df, processing_date_str, log_prefix)
    
    # Determine the best temperature column to use
    temp_col = 'vital_signs_temperature_celsius'
    if temp_col not in df_alert_src or df_alert_src[temp_col].isnull().all():
        temp_col = 'max_skin_temp_celsius'

    # --- Rule-Based Alert Engine ---
    alert_rules = [
        # Vital Sign Alerts (Highest Priority)
        {"level": "CRITICAL", "reason": "Critical Low SpO2", "field": "min_spo2_pct", "condition": lambda v: v < getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90), "details": "SpO2: {value:.0f}%", "action": "ACTION_SPO2_MANAGE_URGENT", "base_score": 98.0, "value_factor": -1.0, "protocol": "PATIENT_CRITICAL_SPO2_LOW"},
        {"level": "CRITICAL", "reason": "High Fever", "field": temp_col, "condition": lambda v: v >= getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5), "details": "Temp: {value:.1f}°C", "action": "ACTION_FEVER_MANAGE_URGENT", "base_score": 95.0, "value_factor": 2.0, "protocol": "PATIENT_CRITICAL_HIGH_FEVER"},
        {"level": "CRITICAL", "reason": "Fall Detected", "field": "fall_detected_today", "condition": lambda v: v > 0, "details": "Falls recorded: {value:.0f}", "action": "ACTION_FALL_ASSESS_URGENT", "base_score": 92.0, "value_factor": 0.0, "protocol": "PATIENT_FALL_DETECTED"},
        {"level": "WARNING", "reason": "Low SpO2", "field": "min_spo2_pct", "condition": lambda v: v < getattr(settings, 'ALERT_SPO2_WARNING_LOW_PCT', 94), "details": "SpO2: {value:.0f}%", "action": "ACTION_SPO2_RECHECK_MONITOR", "base_score": 75.0, "value_factor": -1.0},
        {"level": "WARNING", "reason": "Fever Present", "field": temp_col, "condition": lambda v: v >= getattr(settings, 'ALERT_BODY_TEMP_FEVER_C', 38.0), "details": "Temp: {value:.1f}°C", "action": "ACTION_FEVER_MONITOR", "base_score": 70.0, "value_factor": 1.0},
        
        # AI & Contextual Alerts
        {"level": "WARNING", "reason": "Pending Critical Referral", "field": "referral_status", "condition": lambda v, row: str(v).lower() == 'pending' and any(kc.lower() in str(row.get('condition', '')).lower() for kc in getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])), "details": "For: {row[condition]}", "action": "ACTION_FOLLOWUP_REFERRAL_STATUS", "base_score": 80.0, "value_factor": 0.0},
        {"level": "WARNING", "reason": "High AI Follow-up Prio.", "field": "ai_followup_priority_score", "condition": lambda v: v >= getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80), "details": "AI Prio Score: {value:.0f}", "action": "ACTION_AI_REVIEW_FOLLOWUP", "base_score": 70.0, "value_factor": 0.2},
        {"level": "INFO", "reason": "Elevated AI Risk Score", "field": "ai_risk_score", "condition": lambda v: v >= getattr(settings, 'RISK_SCORE_HIGH_THRESHOLD', 75), "details": "AI Risk: {value:.0f}", "action": "ACTION_MONITOR_RISK_ROUTINE", "base_score": 60.0, "value_factor": 0.15},
    ]

    alerts_buffer: List[Dict[str, Any]] = []
    for _, encounter_row in df_alert_src.iterrows():
        context_info = f"Cond: {encounter_row.get('condition', 'N/A')} | Zone: {encounter_row.get('zone_id', chw_zone_context_str)}"
        
        for rule in alert_rules:
            if rule['field'] not in encounter_row: continue
            
            value = encounter_row[rule['field']]
            if pd.isna(value): continue

            # Check condition
            try:
                # Pass the whole row for complex conditions
                if "row" in rule["condition"].__code__.co_varnames:
                    triggered = rule["condition"](value, encounter_row)
                else:
                    triggered = rule["condition"](value)
            except Exception:
                continue # Skip rule if condition lambda fails

            if triggered:
                # Calculate dynamic priority score
                priority_score = rule['base_score'] + (value * rule['value_factor'])
                
                # Format details string
                details_str = rule['details'].format(value=value, row=encounter_row)

                alert_record = {
                    "alert_level": rule["level"],
                    "primary_reason": rule["reason"],
                    "brief_details": details_str,
                    "suggested_action_code": rule["action"],
                    "raw_priority_score": np.clip(priority_score, 0, 100),
                    "patient_id": str(encounter_row['patient_id']),
                    "context_info": context_info,
                    "triggering_value": f"{rule['field']} = {value}",
                    "encounter_date": processing_date_str,
                }
                alerts_buffer.append(alert_record)

                # Execute escalation protocol if defined
                if rule.get("protocol"):
                    execute_escalation_protocol(
                        rule["protocol"],
                        encounter_row.to_dict(),
                        additional_context={"TRIGGERING_VALUE": value}
                    )
                # Break to ensure only highest-prio rule for a category (e.g., SpO2) triggers
                break
    
    # --- Deduplication & Sorting ---
    if not alerts_buffer:
        logger.info(f"({log_prefix}) No alerts generated from the provided data for {processing_date_str}.")
        return []

    # Keep only the highest priority alert for each patient for this date
    alerts_deduplicated_map: Dict[str, Dict[str, Any]] = {}
    for alert in alerts_buffer:
        patient_id_key = alert['patient_id']
        if patient_id_key not in alerts_deduplicated_map or \
           alert['raw_priority_score'] > alerts_deduplicated_map[patient_id_key]['raw_priority_score']:
            alerts_deduplicated_map[patient_id_key] = alert
    
    # Sort final list by level (Critical > Warning > Info) then by priority score descending
    final_alerts = sorted(
        list(alerts_deduplicated_map.values()),
        key=lambda x: (
            {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x.get("alert_level", "INFO"), 3),
            -x.get('raw_priority_score', 0.0)
        )
    )
    
    logger.info(f"({log_prefix}) Generated {len(final_alerts)} unique alerts for {processing_date_str}.")
    return final_alerts[:max_alerts_to_return]
