# sentinel_project_root/pages/chw_components/task_processing.py
# Processes CHW data to generate a prioritized list of tasks for Sentinel.

import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import date as date_type, timedelta, datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in task_processing.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_TASK = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_TASK_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_TASK if s) + r')$' if COMMON_NA_STRINGS_TASK else None


def _prepare_task_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    default_chw_id: str,
    default_zone_id: str,
    default_patient_id_prefix: str
) -> pd.DataFrame:
    """Prepares the DataFrame for task generation: ensures columns, types, and NA handling."""
    df_prepared = df.copy()

    for col_name, config in cols_config.items():
        default_value = config.get("default") # Use .get for safety
        target_type_str = config.get("type")

        if col_name not in df_prepared.columns:
            if col_name == 'chw_id': default_value = default_chw_id
            elif col_name == 'zone_id': default_value = default_zone_id
            elif col_name == 'patient_id': default_value = default_patient_id_prefix # Placeholder
            
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            elif isinstance(default_value, (list, dict)): 
                 df_prepared[col_name] = [default_value.copy() if isinstance(default_value, (list, dict)) else default_value for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value
        
        current_col_dtype = df_prepared[col_name].dtype
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(current_col_dtype):
            if NA_REGEX_TASK_PATTERN:
                try:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_TASK_PATTERN, np.nan, regex=True)
                except Exception as e_regex:
                    logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value)
            elif target_type_str == int:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=int)
            elif target_type_str == str:
                df_prepared[col_name] = df_prepared[col_name].fillna(str(default_value)).astype(str)
                if NA_REGEX_TASK_PATTERN:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_TASK_PATTERN, str(default_value), regex=True)
                df_prepared[col_name] = df_prepared[col_name].str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns: # Ensure patient_id is never empty string after processing
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', default_patient_id_prefix).fillna(default_patient_id_prefix)
    
    return df_prepared


def generate_chw_tasks(
    source_patient_data_df: Optional[pd.DataFrame], 
    for_date: Union[str, pd.Timestamp, date_type, datetime], 
    chw_id_context: Optional[str] = "TeamDefaultCHW", 
    zone_context_str: Optional[str] = "GeneralArea",  
    max_tasks_to_return_for_summary: int = 20
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks based on input patient data for a specific day.
    """
    module_log_prefix = "CHWTaskGenerator"

    try:
        task_gen_target_date_dt = pd.to_datetime(for_date, errors='coerce')
        if pd.NaT is task_gen_target_date_dt:
            raise ValueError(f"Invalid 'for_date' ({for_date}) for task generation.")
        task_gen_target_date = task_gen_target_date_dt.date()
    except Exception as e_date_task:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ('{for_date}'): {e_date_task}. Defaulting to current system date.", exc_info=True)
        task_gen_target_date = pd.Timestamp('now').date() # Fallback
    
    task_gen_target_date_iso = task_gen_target_date.isoformat()
    safe_chw_id_context = chw_id_context or "TeamDefaultCHW"
    safe_zone_context_str = zone_context_str or "GeneralArea"

    logger.info(f"({module_log_prefix}) Generating CHW tasks for target date: {task_gen_target_date_iso}, CHW: {safe_chw_id_context}, Zone: {safe_zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.warning(f"({module_log_prefix}) No valid patient data provided for task generation on {task_gen_target_date_iso}.")
        return []

    # Configuration for DataFrame preparation
    task_gen_cols_cfg = {
        'patient_id': {"default": f"UPID_Task_{task_gen_target_date_iso}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Will be filtered later
        'zone_id': {"default": safe_zone_context_str, "type": str},
        'chw_id': {"default": safe_chw_id_context, "type": str},
        'condition': {"default": "N/A", "type": str}, 
        'age': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": 0.0, "type": float}, 
        'ai_followup_priority_score': {"default": 0.0, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float}, 
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float}, 
        'fall_detected_today': {"default": 0, "type": int},
        'referral_status': {"default": "Unknown", "type": str}, 
        'referral_reason': {"default": "N/A", "type": str},
        'medication_adherence_self_report': {"default": "Unknown", "type": str}, 
        'tb_contact_traced': {"default": 0, "type": int} # 0 for No/Not yet, 1 for Yes
    }
    df_task_src = _prepare_task_dataframe(
        source_patient_data_df, task_gen_cols_cfg, module_log_prefix, 
        safe_chw_id_context, safe_zone_context_str, f"UPID_Task_{task_gen_target_date_iso}"
    )

    # Filter data to the target date for task generation if 'encounter_date' is present and valid
    # This assumes tasks are generated based on findings ON 'task_gen_target_date'
    if 'encounter_date' in df_task_src.columns and df_task_src['encounter_date'].notna().any():
        df_task_src = df_task_src[df_task_src['encounter_date'].dt.date == task_gen_target_date].copy() # Use .copy() to avoid SettingWithCopyWarning later
    
    if df_task_src.empty:
        logger.info(f"({module_log_prefix}) No data for task generation on {task_gen_target_date_iso} after date filtering or initial empty/invalid data.")
        return []

    # Pre-calculate a composite score for initial sorting to process higher-potential rows first
    df_task_src['temp_sort_priority'] = df_task_src['ai_followup_priority_score'].fillna(0.0) + \
                                        (df_task_src['ai_risk_score'].fillna(0.0) * 0.5) # Example weighting
    df_sorted_for_rules = df_task_src.sort_values(by='temp_sort_priority', ascending=False).drop(columns=['temp_sort_priority'])

    generated_tasks: List[Dict[str, Any]] = []
    # Set to keep track of (patient_id, task_type_code) to avoid duplicate task types for the same patient on the same day
    processed_patient_task_types: Set[Tuple[str, str]] = set() 

    # Determine which temperature column to use
    temp_col_to_use_for_tasks = None
    if 'vital_signs_temperature_celsius' in df_sorted_for_rules.columns and df_sorted_for_rules['vital_signs_temperature_celsius'].notna().any():
        temp_col_to_use_for_tasks = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_sorted_for_rules.columns and df_sorted_for_rules['max_skin_temp_celsius'].notna().any():
        temp_col_to_use_for_tasks = 'max_skin_temp_celsius'
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' for task temperature checks.")

    # Get setting values with fallbacks
    spo2_critical_thresh_task = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90.0)
    temp_high_fever_thresh_task = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0)
    prio_high_thresh_task = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 0.7) # Reusing this, consider specific task threshold
    prio_mod_thresh_task = getattr(settings, 'FATIGUE_INDEX_MODERATE_THRESHOLD', 0.5) # Reusing this
    key_conditions_for_action_task = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])


    for _, row_data in df_sorted_for_rules.iterrows():
        patient_id = str(row_data['patient_id'])
        # Default due date to next day, can be overridden by specific rules
        current_task_due_date = task_gen_target_date + timedelta(days=1) 
        
        # Calculate base priority for this patient, ensuring it's a float
        base_priority_score = float(row_data.get('ai_followup_priority_score', 0.0) or 0.0) # Ensure not NaN from .get()
        if base_priority_score < 10.0: # Boost if AI followup is too low but AI risk is somewhat high
             base_priority_score = max(base_priority_score, float(row_data.get('ai_risk_score', 0.0) or 0.0) * 0.6)
        base_priority_score = max(10.0, base_priority_score) # Ensure a minimum base priority

        task_details_to_add: Optional[Dict[str, Any]] = None # Stores details of the task if one is triggered

        # --- Rule Engine ---
        # Rule 1: Critical Vitals Follow-up (Highest Prio)
        current_min_spo2 = row_data.get('min_spo2_pct', np.nan)
        if pd.notna(current_min_spo2) and current_min_spo2 < spo2_critical_thresh_task:
            task_type_code = "TASK_VISIT_VITALS_URGENT_SPO2"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": f"URGENT: Assess Critical Low SpO2 ({current_min_spo2:.0f}%)", "prio": 98.0}
                current_task_due_date = task_gen_target_date # Urgent tasks due same day if possible/next
        
        current_temp = row_data.get(temp_col_to_use_for_tasks, np.nan) if temp_col_to_use_for_tasks else np.nan
        if not task_details_to_add and pd.notna(current_temp) and current_temp >= temp_high_fever_thresh_task:
            task_type_code = "TASK_VISIT_VITALS_URGENT_TEMP"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": f"URGENT: Assess High Fever ({current_temp:.1f}°C)", "prio": 95.0}
                current_task_due_date = task_gen_target_date
        
        current_falls = int(row_data.get('fall_detected_today', 0))
        if not task_details_to_add and current_falls > 0:
            task_type_code = "TASK_VISIT_FALL_ASSESS"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": f"Assess After Fall (Falls: {current_falls})", "prio": 92.0}
                current_task_due_date = task_gen_target_date
        
        # Rule 2: Pending Critical Referral Follow-up
        if not task_details_to_add and str(row_data.get('referral_status', '')).lower() == 'pending':
            condition_str = str(row_data.get('condition', '')).lower()
            referral_reason_str = str(row_data.get('referral_reason', '')).lower()
            is_critical_referral_condition = any(re.escape(kc).lower() in condition_str for kc in key_conditions_for_action_task)
            is_urgent_reason = "urgent" in referral_reason_str

            if is_critical_referral_condition or is_urgent_reason:
                task_type_code = "TASK_VISIT_REFERRAL_TRACK"
                if (patient_id, task_type_code) not in processed_patient_task_types:
                    task_details_to_add = {"type": task_type_code, "desc": f"Follow-up: Critical Referral for {row_data.get('condition', 'N/A')}", "prio": 88.0}
        
        # Rule 3: High AI Follow-up Priority Task
        if not task_details_to_add and base_priority_score >= prio_high_thresh_task:
            task_type_code = "TASK_VISIT_FOLLOWUP_AI"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": f"Priority Follow-up (AI Score: {base_priority_score:.0f})", "prio": base_priority_score}

        # Rule 4: Medication Adherence Support
        if not task_details_to_add and str(row_data.get('medication_adherence_self_report', '')).lower() == 'poor':
            task_type_code = "TASK_VISIT_ADHERENCE_SUPPORT"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": "Support Medication Adherence (Reported Poor)", "prio": max(base_priority_score, 75.0)}
        
        # Rule 5: Pending TB Contact Tracing
        # Assumes tb_contact_traced = 0 means not yet done/pending, 1 means done.
        if not task_details_to_add and \
           TB_PATTERN_EPI.search(str(row_data.get('condition', ''))) and \
           pd.notna(row_data.get('tb_contact_traced')) and int(row_data.get('tb_contact_traced', 1)) == 0: # Default to 1 (done) if missing
            task_type_code = "TASK_TB_CONTACT_TRACE"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": "Initiate/Continue TB Contact Tracing", "prio": max(base_priority_score, 80.0)}
        
        # Rule 6: Default Routine Checkup for Moderate AI Prio
        if not task_details_to_add and base_priority_score >= prio_mod_thresh_task:
            task_type_code = "TASK_VISIT_ROUTINE_CHECK"
            if (patient_id, task_type_code) not in processed_patient_task_types:
                task_details_to_add = {"type": task_type_code, "desc": f"Routine Health Check (AI Prio: {base_priority_score:.0f})", "prio": base_priority_score}

        # If a task was identified, format and add it
        if task_details_to_add:
            context_parts = []
            if str(row_data.get('condition', '')) not in ["N/A", task_gen_cols_cfg['condition']['default']]: 
                context_parts.append(f"Cond: {row_data['condition']}")
            if pd.notna(row_data.get('age')): context_parts.append(f"Age: {row_data['age']:.0f}")
            if pd.notna(current_min_spo2): context_parts.append(f"SpO2: {current_min_spo2:.0f}%")
            if temp_col_to_use_for_tasks and pd.notna(current_temp): context_parts.append(f"Temp: {current_temp:.1f}°C")
            if pd.notna(row_data.get('ai_risk_score')): context_parts.append(f"AI Risk: {row_data.get('ai_risk_score', 0.0):.0f}")
            key_context_summary = " | ".join(context_parts) if context_parts else "General Check"

            task_id_type_suffix = task_details_to_add['type'].split('_')[-1] if '_' in task_details_to_add['type'] else task_details_to_add['type'][:4]
            
            generated_tasks.append({
                "task_id": f"TSK_{patient_id}_{task_gen_target_date_iso.replace('-', '')}_{task_id_type_suffix}_{len(generated_tasks)+1}",
                "patient_id": patient_id, 
                "assigned_chw_id": str(row_data.get('chw_id', safe_chw_id_context)),
                "zone_id": str(row_data.get('zone_id', safe_zone_context_str)), 
                "task_type_code": task_details_to_add["type"],
                "task_description": f"{task_details_to_add['desc']} for Pt. {patient_id}", # Made Pt. explicit
                "priority_score": round(min(float(task_details_to_add["prio"]), 100.0), 1), # Ensure float for min
                "due_date": current_task_due_date.isoformat(), 
                "status": "PENDING",
                "key_patient_context": key_context_summary,
                "alert_source_info": f"Data from {(row_data.get('encounter_date').strftime('%Y-%m-%d') if pd.notna(row_data.get('encounter_date')) else task_gen_target_date_iso)}"
            })
            processed_patient_task_types.add((patient_id, task_details_to_add["type"])) # Mark this task type as processed for this patient

    if not generated_tasks:
        logger.info(f"({module_log_prefix}) No tasks generated for {task_gen_target_date_iso} based on current rules.")
        return []
    
    # Sort final tasks by priority score (descending)
    final_tasks_list = sorted(generated_tasks, key=lambda x: x['priority_score'], reverse=True)
    logger.info(f"({module_log_prefix}) Generated {len(final_tasks_list)} CHW tasks for {task_gen_target_date_iso}.")
    
    return final_tasks_list[:max_tasks_to_return_for_summary]
