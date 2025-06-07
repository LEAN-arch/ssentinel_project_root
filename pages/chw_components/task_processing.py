# sentinel_project_root/pages/chw_components/task_processing.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.

import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import date as date_type, timedelta, datetime

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import standardize_missing_values
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in task_processing.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)


def _prepare_task_dataframe(
    df: pd.DataFrame,
    processing_date_str: str,
    log_prefix: str
) -> pd.DataFrame:
    """Prepares the source DataFrame for task generation."""
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    numeric_defaults = {
        'age': np.nan, 'ai_risk_score': 0.0, 'ai_followup_priority_score': 0.0,
        'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
        'max_skin_temp_celsius': np.nan, 'fall_detected_today': 0,
        'tb_contact_tracing_completed': 0, 'priority_score': 0.0,
    }
    string_defaults = {
        'patient_id': f"UPID_Task_{processing_date_str}", 'zone_id': "GeneralArea",
        'chw_id': "TeamDefaultCHW", 'condition': "N/A",
        'referral_status': "Unknown", 'referral_reason': "N/A",
        'medication_adherence_self_report': "Unknown",
    }
    
    df_prepared = standardize_missing_values(df, string_defaults, numeric_defaults)

    if 'encounter_date' in df_prepared.columns:
        df_prepared['encounter_date'] = pd.to_datetime(df_prepared['encounter_date'], errors='coerce')

    return df_prepared


def generate_chw_tasks(
    source_patient_data_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_id_context: Optional[str] = None,
    zone_context_str: Optional[str] = None,
    max_tasks_to_return: int = 20
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks using a scalable, rule-based engine.
    """
    log_prefix = "CHWTaskGenerator"

    try:
        target_date = pd.to_datetime(for_date).date()
    except (ValueError, TypeError):
        logger.warning(f"({log_prefix}) Invalid 'for_date' ('{for_date}'). Defaulting to current system date.")
        target_date = pd.Timestamp('now').date()

    target_date_iso = target_date.isoformat()
    logger.info(f"({log_prefix}) Generating CHW tasks for target date: {target_date_iso}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.warning(f"({log_prefix}) No patient data provided for task generation on {target_date_iso}.")
        return []

    df_source = _prepare_task_dataframe(source_patient_data_df, target_date_iso, log_prefix)
    df_today = df_source[df_source['encounter_date'].dt.date == target_date].copy()
    if df_today.empty:
        logger.info(f"({log_prefix}) No data for task generation on {target_date_iso} after date filtering.")
        return []

    df_today['sort_priority'] = df_today['ai_followup_priority_score'].fillna(0) + (df_today['ai_risk_score'].fillna(0) * 0.5)
    df_sorted = df_today.sort_values(by='sort_priority', ascending=False).drop(columns=['sort_priority'])

    temp_col = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_sorted and df_sorted[tc].notna().any()), None)

    TASK_RULES = [
        {"type": "URGENT_SPO2", "desc": "URGENT: Assess Critical Low SpO2 ({value:.0f}%)", "field": "min_spo2_pct", "condition": lambda v, r: v < settings.ALERT_SPO2_CRITICAL_LOW_PCT, "base_prio": 98.0, "prio_factor": -1.0, "due_days": 0},
        {"type": "URGENT_TEMP", "desc": "URGENT: Assess High Fever ({value:.1f}°C)", "field": temp_col, "condition": lambda v, r: v >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C, "base_prio": 95.0, "prio_factor": 2.0, "due_days": 0},
        {"type": "FALL_ASSESS", "desc": "Assess After Fall Detection", "field": "fall_detected_today", "condition": lambda v, r: v > 0, "base_prio": 92.0, "prio_factor": 0.0, "due_days": 0},
        {"type": "REFERRAL_TRACK", "desc": "Follow-up: Critical Referral for {row[condition]}", "field": "referral_status", "condition": lambda v, r: str(v).lower() == 'pending' and any(kc.lower() in str(r.get('condition')).lower() for kc in settings.KEY_CONDITIONS_FOR_ACTION), "base_prio": 88.0, "prio_factor": 0.0, "due_days": 1},
        {"type": "TB_TRACE", "desc": "Initiate/Continue TB Contact Tracing", "field": "condition", "condition": lambda v, r: re.search(r'\btb\b|tuberculosis', str(v), re.I) and not r.get('tb_contact_tracing_completed'), "base_prio": 85.0, "prio_factor": 0.0, "due_days": 1},
        {"type": "AI_PRIO_FOLLOWUP", "desc": "Priority Follow-up (AI Prio Score: {value:.0f})", "field": "ai_followup_priority_score", "condition": lambda v, r: v >= settings.FATIGUE_INDEX_HIGH_THRESHOLD, "base_prio": 78.0, "prio_factor": 0.15, "due_days": 1},
        {"type": "ADHERENCE_SUPPORT", "desc": "Support Medication Adherence (Reported Poor)", "field": "medication_adherence_self_report", "condition": lambda v, r: str(v).lower() == 'poor', "base_prio": 75.0, "prio_factor": 0.0, "due_days": 2},
        {"type": "ROUTINE_CHECK", "desc": "Routine Health Check (AI Prio: {value:.0f})", "field": "ai_followup_priority_score", "condition": lambda v, r: v >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD, "base_prio": 60.0, "prio_factor": 0.1, "due_days": 3},
    ]

    generated_tasks: List[Dict[str, Any]] = []
    processed_patient_task_types: Set[Tuple[str, str]] = set()

    for _, row in df_sorted.iterrows():
        patient_id = str(row['patient_id'])
        for rule in TASK_RULES:
            if (patient_id, rule["type"]) in processed_patient_task_types:
                continue

            if rule['field'] and rule['field'] in row:
                value = row[rule['field']]
                if pd.isna(value): continue
                
                try:
                    is_triggered = rule["condition"](value, row)
                except Exception:
                    continue

                if is_triggered:
                    priority = np.clip(rule["base_prio"] + (float(value) * rule["prio_factor"]), 0, 100)
                    due_date = target_date + timedelta(days=rule["due_days"])
                    
                    context_parts = [f"Cond: {row['condition']}"]
                    if pd.notna(row.get('age')): context_parts.append(f"Age: {row['age']:.0f}")
                    if pd.notna(row.get('ai_risk_score')): context_parts.append(f"AI Risk: {row['ai_risk_score']:.0f}")
                    if rule['field'] != 'condition' and isinstance(value, (int, float)):
                        context_parts.append(f"Trigger: {value:.1f}")
                    
                    generated_tasks.append({
                        "task_id": f"TSK_{patient_id}_{target_date_iso.replace('-', '')}_{len(generated_tasks)+1:03d}",
                        "patient_id": patient_id, "assigned_chw_id": str(row.get('chw_id', chw_id_context or 'TeamDefaultCHW')),
                        "zone_id": str(row.get('zone_id', zone_context_str or 'GeneralArea')), "task_type_code": rule["type"],
                        "task_description": rule["desc"].format(value=value, row=row), "priority_score": round(priority, 1),
                        "due_date": due_date.isoformat(), "status": "PENDING", "key_patient_context": " | ".join(context_parts),
                        "alert_source_info": f"Data from {target_date_iso}"
                    })
                    processed_patient_task_types.add((patient_id, rule["type"]))
                    break

    if not generated_tasks:
        logger.info(f"({log_prefix}) No tasks generated for {target_date_iso}.")
        return []

    final_tasks = sorted(generated_tasks, key=lambda x: x['priority_score'], reverse=True)
    logger.info(f"({log_prefix}) Generated {len(final_tasks)} CHW tasks for {target_date_iso}.")
    return final_tasks[:max_tasks_to_return]
