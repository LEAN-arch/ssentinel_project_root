# sentinel_project_root/pages/chw_components/task_processing.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.
# Renamed from task_processor.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import date, timedelta # For due date calculations

from config import settings # Use new settings module
from data_processing.helpers import convert_to_numeric # Local import

logger = logging.getLogger(__name__)


def generate_chw_tasks( # Renamed from generate_chw_prioritized_tasks
    source_patient_data_df: Optional[pd.DataFrame], # Data for the specific day/context
    for_date: Any, # The date these tasks are relevant FOR or DUE BY
    chw_id_context: Optional[str] = "TeamDefaultCHW", # Contextual CHW ID for assignment
    zone_context_str: Optional[str] = "GeneralArea",  # Contextual Zone ID
    max_tasks_to_return_for_summary: int = 20 # Limit for dashboard display
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data for a specific day.
    Tasks can include follow-ups for alerts, routine checks, medication adherence support, etc.
    """
    module_log_prefix = "CHWTaskGenerator" # Renamed for clarity

    # Standardize for_date (this is the date the tasks are being generated *for*)
    try:
        task_generation_target_date = pd.to_datetime(for_date).date() if for_date else pd.Timestamp('now').date()
    except Exception:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}). Defaulting to current system date.")
        task_generation_target_date = pd.Timestamp('now').date()

    logger.info(f"({module_log_prefix}) Generating CHW tasks for target date: {task_generation_target_date.isoformat()}, CHW: {chw_id_context}, Zone: {zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.warning(f"({module_log_prefix}) No valid patient data provided for task generation on {task_generation_target_date.isoformat()}.")
        return [] # Return empty list if no data

    df_task_src_raw = source_patient_data_df.copy()
    
    # Filter data to be relevant for task_generation_target_date if 'encounter_date' exists
    # Tasks are often generated based on *today's* findings or recent events.
    if 'encounter_date' in df_task_src_raw.columns:
        df_task_src_raw['encounter_date_dt'] = pd.to_datetime(df_task_src_raw['encounter_date'], errors='coerce')
        # For generating *today's* tasks, usually consider encounters from today or very recent past (e.g. yesterday for overdue items)
        # Here, we'll assume source_patient_data_df is ALREADY filtered to the relevant day for new task generation.
        # If it could contain older data, more complex filtering for "new findings today" vs "pending from past" would be needed.
        # For simplicity, this component will assume df_task_src_raw is for the 'for_date'.
        # df_task_src_raw = df_task_src_raw[df_task_src_raw['encounter_date_dt'].dt.date == task_generation_target_date]
        pass # Assuming input df is already for the target date's encounters/alerts

    if df_task_src_raw.empty:
        logger.info(f"({module_log_prefix}) No patient data rows relevant for task generation on {task_generation_target_date.isoformat()} after initial check.")
        return []

    # Define expected columns and their safe defaults for task generation logic
    task_gen_cols_config = {
        'patient_id': {"default": f"UnknownPID_TaskGen_{task_generation_target_date.isoformat()}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Date of the data point triggering task
        'zone_id': {"default": zone_context_str or "UnknownZone", "type": str},
        'chw_id': {"default": chw_id_context or "Unassigned", "type": str},
        'condition': {"default": "N/A", "type": str},
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": 0.0, "type": float}, # Default to 0 if not present for priority calc
        'ai_followup_priority_score': {"default": 0.0, "type": float}, # Default to 0
        # Columns that might directly trigger tasks or inform task details:
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int},
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "N/A", "type": str},
        'medication_adherence_self_report': {"default": "Unknown", "type": str},
        'tb_contact_traced': {"default": 0, "type": int} # 0=No/Pending, 1=Yes/Completed
    }
    common_na_values_task_gen = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown']

    df_for_task_rules = df_task_src_raw.copy() # Work on a copy
    for col_name, config in task_gen_cols_config.items():
        if col_name not in df_for_task_rules.columns:
            df_for_task_rules[col_name] = config["default"]
        
        if config["type"] == "datetime":
            df_for_task_rules[col_name] = pd.to_datetime(df_for_task_rules[col_name], errors='coerce')
        elif config["type"] == float:
            df_for_task_rules[col_name] = convert_to_numeric(df_for_task_rules[col_name], default_value=config["default"])
        elif config["type"] == int:
            df_for_task_rules[col_name] = convert_to_numeric(df_for_task_rules[col_name], default_value=config["default"], target_type=int)
        elif config["type"] == str:
            df_for_task_rules[col_name] = df_for_task_rules[col_name].astype(str).fillna(str(config["default"]))
            df_for_task_rules[col_name] = df_for_task_rules[col_name].replace(common_na_values_task_gen, str(config["default"]), regex=False).str.strip()

    # Prioritize records with higher AI follow-up scores or risk scores if follow-up score is low/missing
    # This helps process more "important" patients first for task generation.
    df_for_task_rules['temp_sort_priority'] = df_for_task_rules['ai_followup_priority_score'].fillna(0) + (df_for_task_rules['ai_risk_score'].fillna(0) * 0.5)
    df_sorted_for_rules = df_for_task_rules.sort_values(by='temp_sort_priority', ascending=False).drop(columns=['temp_sort_priority'])


    generated_tasks_list: List[Dict[str, Any]] = []
    processed_patient_task_types_today: set = set() # To avoid duplicate task types for same patient on same day

    # Determine temperature column to use
    temp_col_name_task = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_sorted_for_rules.columns and df_sorted_for_rules[tc].notna().any()), None)


    for _, patient_row_data in df_sorted_for_rules.iterrows():
        patient_id_val = str(patient_row_data['patient_id'])
        # Task due date is typically today or next day for CHW tasks generated from daily findings
        task_due_date = task_generation_target_date + timedelta(days=1) # Default due next day
        
        # Base priority: from AI follow-up score, fallback to AI risk, then default low
        base_priority_for_task = patient_row_data.get('ai_followup_priority_score', 0.0)
        if pd.isna(base_priority_for_task) or base_priority_for_task < 10: # If low/missing followup score, consider risk
            base_priority_for_task = max(base_priority_for_task if pd.notna(base_priority_for_task) else 0.0, 
                                         patient_row_data.get('ai_risk_score', 0.0) * 0.6) # Weight risk less than direct prio
        base_priority_for_task = max(10.0, base_priority_for_task) # Min base priority for any generated task


        task_details_to_add: Optional[Dict[str, Any]] = None

        # --- Task Generation Rules (Prioritized) ---

        # 1. Critical Vitals Follow-up (SpO2, High Fever, Fall)
        if pd.notna(patient_row_data['min_spo2_pct']) and patient_row_data['min_spo2_pct'] < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            task_type_code = "TASK_VISIT_VITALS_URGENT"
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                task_details_to_add = {"type": task_type_code, "desc": f"URGENT: Assess Critical Low SpO2 ({patient_row_data['min_spo2_pct']:.0f}%)", "prio": 98.0}
                task_due_date = task_generation_target_date # Same day urgency
        
        elif temp_col_name_task and pd.notna(patient_row_data[temp_col_name_task]) and patient_row_data[temp_col_name_task] >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            task_type_code = "TASK_VISIT_VITALS_URGENT"
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                task_details_to_add = {"type": task_type_code, "desc": f"URGENT: Assess High Fever ({patient_row_data[temp_col_name_task]:.1f}°C)", "prio": 95.0}
                task_due_date = task_generation_target_date # Same day
        
        elif pd.notna(patient_row_data['fall_detected_today']) and patient_row_data['fall_detected_today'] > 0:
            task_type_code = "TASK_VISIT_FALL_ASSESS"
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                task_details_to_add = {"type": task_type_code, "desc": f"Assess Patient After Fall Detection (Falls: {int(patient_row_data['fall_detected_today'])})", "prio": 92.0}
                task_due_date = task_generation_target_date # Same day

        # 2. Pending Critical Referral Follow-up
        if not task_details_to_add and str(patient_row_data['referral_status']).lower() == 'pending':
            is_key_cond_referral = any(
                kc.lower() in str(patient_row_data['condition']).lower() for kc in settings.KEY_CONDITIONS_FOR_ACTION
            ) or "urgent" in str(patient_row_data.get('referral_reason','')).lower() # Check reason too
            if is_key_cond_referral:
                task_type_code = "TASK_VISIT_REFERRAL_TRACK"
                if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                    task_details_to_add = {"type": task_type_code, "desc": f"Follow-up: Critical Referral for {patient_row_data['condition']}", "prio": 88.0}
        
        # 3. High AI Follow-up Priority Score Task
        if not task_details_to_add and base_priority_for_task >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: # Using general high prio threshold
            task_type_code = "TASK_VISIT_FOLLOWUP_AI"
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                task_details_to_add = {"type": task_type_code, "desc": f"Priority Follow-up (High AI Score: {base_priority_for_task:.0f})", "prio": base_priority_for_task}

        # 4. Medication Adherence Support
        if not task_details_to_add and str(patient_row_data['medication_adherence_self_report']).lower() == 'poor':
            task_type_code = "TASK_VISIT_ADHERENCE_SUPPORT"
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                task_details_to_add = {"type": task_type_code, "desc": "Support Medication Adherence (Reported Poor)", "prio": max(base_priority_for_task, 75.0)}
        
        # 5. Pending TB Contact Tracing (if TB is condition and contact not yet traced)
        if not task_details_to_add and \
           "tb" in str(patient_row_data['condition']).lower() and \
           pd.notna(patient_row_data.get('tb_contact_traced')) and patient_row_data['tb_contact_traced'] == 0:
            task_type_code = "TASK_TB_CONTACT_TRACE" # Specific task type
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                task_details_to_add = {"type": task_type_code, "desc": "Initiate/Continue TB Contact Tracing", "prio": max(base_priority_for_task, 80.0)}
        
        # (Add more rules here: e.g., routine wellness, maternal health schedule, etc.)

        # 6. Default Routine Checkup if no other specific task generated and some risk/moderate prio
        if not task_details_to_add and base_priority_for_task >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD: # Moderate prio threshold
            task_type_code = "TASK_VISIT_ROUTINE_CHECK"
            if (patient_id_val, task_type_code) not in processed_patient_task_types_today:
                 task_details_to_add = {"type": task_type_code, "desc": f"Routine Health Check (AI Prio: {base_priority_for_task:.0f})", "prio": base_priority_for_task}


        if task_details_to_add:
            # Construct patient context string for quick overview in task list
            context_parts = []
            if str(patient_row_data['condition']) not in ["N/A", "UnknownCondition"]: context_parts.append(f"Cond: {patient_row_data['condition']}")
            if pd.notna(patient_row_data['age']): context_parts.append(f"Age: {patient_row_data['age']:.0f}")
            if pd.notna(patient_row_data['min_spo2_pct']): context_parts.append(f"SpO2: {patient_row_data['min_spo2_pct']:.0f}%")
            if temp_col_name_task and pd.notna(patient_row_data[temp_col_name_task]): context_parts.append(f"Temp: {patient_row_data[temp_col_name_task]:.1f}°C")
            if pd.notna(patient_row_data['ai_risk_score']): context_parts.append(f"AI Risk: {patient_row_data['ai_risk_score']:.0f}")
            key_patient_context_str = " | ".join(context_parts) if context_parts else "General Check Required"

            task_record = {
                "task_id": f"TSK_{patient_id_val}_{task_generation_target_date.strftime('%Y%m%d')}_{task_details_to_add['type'].split('_')[-1]}_{len(generated_tasks_list)}",
                "patient_id": patient_id_val,
                "assigned_chw_id": str(patient_row_data.get('chw_id', chw_id_context)), # Assign to current CHW or default
                "zone_id": str(patient_row_data.get('zone_id', zone_context_str)),
                "task_type_code": task_details_to_add["type"],
                "task_description": f"{task_details_to_add['desc']} for Patient {patient_id_val}",
                "priority_score": round(min(task_details_to_add["prio"], 100.0), 1), # Cap priority at 100
                "due_date": task_due_date.isoformat(), # ISO format string
                "status": "PENDING", # Initial status
                "key_patient_context": key_patient_context_str,
                "alert_source_info": f"Data from {patient_row_data.get('encounter_date', task_generation_target_date).strftime('%Y-%m-%d')}"
            }
            generated_tasks_list.append(task_record)
            processed_patient_task_types_today.add((patient_id_val, task_details_to_add["type"]))


    # Sort all generated tasks by priority score (descending)
    if generated_tasks_list:
        final_tasks_sorted_list = sorted(generated_tasks_list, key=lambda x_task: x_task['priority_score'], reverse=True)
        logger.info(f"({module_log_prefix}) Generated {len(final_tasks_sorted_list)} unique CHW tasks for {task_generation_target_date.isoformat()}.")
        return final_tasks_sorted_list[:max_tasks_to_return_for_summary] # Return top N tasks
    
    logger.info(f"({module_log_prefix}) No tasks generated for {task_generation_target_date.isoformat()} based on current rules and data.")
    return []
