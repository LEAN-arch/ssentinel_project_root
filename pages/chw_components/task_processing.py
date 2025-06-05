# sentinel_project_root/pages/chw_components/task_processing.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.

import pandas as pd
import numpy as np
import logging
import re # For condition matching
from typing import List, Dict, Any, Optional, Set, Tuple # Added Set, Tuple
from datetime import date as date_type, timedelta # For due date calculations

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def generate_chw_tasks(
    source_patient_data_df: Optional[pd.DataFrame], 
    for_date: Any, 
    chw_id_context: Optional[str] = "TeamDefaultCHW", 
    zone_context_str: Optional[str] = "GeneralArea",  
    max_tasks_to_return_for_summary: int = 20
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data for a specific day.
    """
    module_log_prefix = "CHWTaskGenerator"

    try:
        task_gen_target_date = pd.to_datetime(for_date, errors='coerce').date()
        if pd.isna(task_gen_target_date): raise ValueError("Invalid for_date for task generation.")
    except Exception as e_date_task:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}): {e_date_task}. Defaulting to current system date.")
        task_gen_target_date = pd.Timestamp('now').date()
    task_gen_target_date_iso = task_gen_target_date.isoformat()

    logger.info(f"({module_log_prefix}) Generating CHW tasks for target date: {task_gen_target_date_iso}, CHW: {chw_id_context}, Zone: {zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.warning(f"({module_log_prefix}) No valid patient data for task generation on {task_gen_target_date_iso}.")
        return []

    df_task_src = source_patient_data_df.copy()
    
    # Assuming df_task_src is already filtered to the relevant day's findings.
    # If not, filter here:
    # if 'encounter_date' in df_task_src.columns:
    #     df_task_src['encounter_date_dt_task'] = pd.to_datetime(df_task_src['encounter_date'], errors='coerce')
    #     df_task_src = df_task_src[df_task_src['encounter_date_dt_task'].dt.date == task_gen_target_date]
    # if df_task_src.empty:
    #     logger.info(f"({module_log_prefix}) No data for task generation on {task_gen_target_date_iso} after date filter.")
    #     return []

    task_gen_cols_cfg = {
        'patient_id': {"default": f"UPID_Task_{task_gen_target_date_iso}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'zone_id': {"default": zone_context_str or "UZone", "type": str},
        'chw_id': {"default": chw_id_context or "Unassigned", "type": str},
        'condition': {"default": "N/A", "type": str}, 'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": 0.0, "type": float}, 'ai_followup_priority_score': {"default": 0.0, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float}, 'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float}, 'fall_detected_today': {"default": 0, "type": int},
        'referral_status': {"default": "Unknown", "type": str}, 'referral_reason': {"default": "N/A", "type": str},
        'medication_adherence_self_report': {"default": "Unknown", "type": str}, 'tb_contact_traced': {"default": 0, "type": int}
    }
    common_na_task = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    na_regex_task = r'^(?:' + '|'.join(re.escape(s) for s in common_na_task if s) + r')$'

    for col, cfg in task_gen_cols_cfg.items():
        if col not in df_task_src.columns: df_task_src[col] = cfg["default"]
        if cfg["type"] == "datetime": df_task_src[col] = pd.to_datetime(df_task_src[col], errors='coerce')
        elif cfg["type"] == float: df_task_src[col] = convert_to_numeric(df_task_src[col], default_value=cfg["default"])
        elif cfg["type"] == int: df_task_src[col] = convert_to_numeric(df_task_src[col], default_value=cfg["default"], target_type=int)
        elif cfg["type"] == str:
            df_task_src[col] = df_task_src[col].astype(str).fillna(str(cfg["default"]))
            if any(common_na_task): df_task_src[col] = df_task_src[col].replace(na_regex_task, str(cfg["default"]), regex=True)
            df_task_src[col] = df_task_src[col].str.strip()

    df_task_src['temp_sort_priority'] = df_task_src['ai_followup_priority_score'].fillna(0) + (df_task_src['ai_risk_score'].fillna(0) * 0.5)
    df_sorted_rules = df_task_src.sort_values(by='temp_sort_priority', ascending=False).drop(columns=['temp_sort_priority'])

    generated_tasks: List[Dict[str, Any]] = []
    processed_pat_task_types: Set[Tuple[str, str]] = set()
    temp_col_task = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_sorted_rules.columns and df_sorted_rules[tc].notna().any()), None)

    for _, row in df_sorted_rules.iterrows():
        pid = str(row['patient_id'])
        due_date = task_gen_target_date + timedelta(days=1) # Default due next day
        base_prio = row.get('ai_followup_priority_score', 0.0)
        if pd.isna(base_prio) or base_prio < 10:
            base_prio = max(base_prio if pd.notna(base_prio) else 0.0, row.get('ai_risk_score', 0.0) * 0.6)
        base_prio = max(10.0, base_prio)
        task_to_add: Optional[Dict[str, Any]] = None

        # Rule 1: Critical Vitals Follow-up
        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            ttc = "TASK_VISIT_VITALS_URGENT"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": f"URGENT: Assess Critical Low SpO2 ({row['min_spo2_pct']:.0f}%)", "prio": 98.0}; due_date = task_gen_target_date
        elif temp_col_task and pd.notna(row[temp_col_task]) and row[temp_col_task] >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            ttc = "TASK_VISIT_VITALS_URGENT"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": f"URGENT: Assess High Fever ({row[temp_col_task]:.1f}°C)", "prio": 95.0}; due_date = task_gen_target_date
        elif pd.notna(row['fall_detected_today']) and row['fall_detected_today'] > 0:
            ttc = "TASK_VISIT_FALL_ASSESS"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": f"Assess After Fall (Falls: {int(row['fall_detected_today'])})", "prio": 92.0}; due_date = task_gen_target_date
        
        # Rule 2: Pending Critical Referral Follow-up
        if not task_to_add and str(row['referral_status']).lower() == 'pending':
            is_key_ref = any(re.escape(kc).lower() in str(row['condition']).lower() for kc in settings.KEY_CONDITIONS_FOR_ACTION) or \
                         "urgent" in str(row.get('referral_reason','')).lower()
            if is_key_ref:
                ttc = "TASK_VISIT_REFERRAL_TRACK"
                if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": f"Follow-up: Critical Referral for {row['condition']}", "prio": 88.0}
        
        # Rule 3: High AI Follow-up Prio Task
        if not task_to_add and base_prio >= settings.FATIGUE_INDEX_HIGH_THRESHOLD:
            ttc = "TASK_VISIT_FOLLOWUP_AI"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": f"Priority Follow-up (AI Score: {base_prio:.0f})", "prio": base_prio}

        # Rule 4: Medication Adherence Support
        if not task_to_add and str(row['medication_adherence_self_report']).lower() == 'poor':
            ttc = "TASK_VISIT_ADHERENCE_SUPPORT"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": "Support Medication Adherence (Poor)", "prio": max(base_prio, 75.0)}
        
        # Rule 5: Pending TB Contact Tracing
        if not task_to_add and "tb" in str(row['condition']).lower() and pd.notna(row.get('tb_contact_traced')) and row['tb_contact_traced'] == 0:
            ttc = "TASK_TB_CONTACT_TRACE"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": "Initiate/Continue TB Contact Tracing", "prio": max(base_prio, 80.0)}
        
        # Rule 6: Default Routine Checkup
        if not task_to_add and base_prio >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD:
            ttc = "TASK_VISIT_ROUTINE_CHECK"
            if (pid, ttc) not in processed_pat_task_types: task_to_add = {"type": ttc, "desc": f"Routine Health Check (AI Prio: {base_prio:.0f})", "prio": base_prio}

        if task_to_add:
            ctx_parts = []
            if str(row['condition']) not in ["N/A", "UCond"]: ctx_parts.append(f"Cond: {row['condition']}")
            if pd.notna(row['age']): ctx_parts.append(f"Age: {row['age']:.0f}")
            if pd.notna(row['min_spo2_pct']): ctx_parts.append(f"SpO2: {row['min_spo2_pct']:.0f}%")
            if temp_col_task and pd.notna(row[temp_col_task]): ctx_parts.append(f"Temp: {row[temp_col_task]:.1f}°C")
            if pd.notna(row['ai_risk_score']): ctx_parts.append(f"AI Risk: {row['ai_risk_score']:.0f}")
            key_ctx_str = " | ".join(ctx_parts) if ctx_parts else "General Check"

            task_id_suffix = task_to_add['type'].split('_')[-1] if '_' in task_to_add['type'] else task_to_add['type']
            generated_tasks.append({
                "task_id": f"TSK_{pid}_{task_gen_target_date_iso.replace('-', '')}_{task_id_suffix}_{len(generated_tasks)}",
                "patient_id": pid, "assigned_chw_id": str(row.get('chw_id', chw_id_context)),
                "zone_id": str(row.get('zone_id', zone_context_str)), "task_type_code": task_to_add["type"],
                "task_description": f"{task_to_add['desc']} for Patient {pid}",
                "priority_score": round(min(task_to_add["prio"], 100.0), 1),
                "due_date": due_date.isoformat(), "status": "PENDING",
                "key_patient_context": key_ctx_str,
                "alert_source_info": f"Data from {(row.get('encounter_date') or task_gen_target_date).strftime('%Y-%m-%d')}"
            })
            processed_pat_task_types.add((pid, task_to_add["type"]))

    if generated_tasks:
        final_tasks = sorted(generated_tasks, key=lambda x: x['priority_score'], reverse=True)
        logger.info(f"({module_log_prefix}) Generated {len(final_tasks)} CHW tasks for {task_gen_target_date_iso}.")
        return final_tasks[:max_tasks_to_return_for_summary]
    
    logger.info(f"({module_log_prefix}) No tasks generated for {task_gen_target_date_iso}.")
    return []
