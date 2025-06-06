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
    from data_processing.helpers import convert_to_numeric 
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__) 
    logger_init.error(f"Critical import error in task_processing.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

COMMON_NA_STRINGS_TASK = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown', '-'])
VALID_NA_FOR_REGEX_TASK = [s for s in COMMON_NA_STRINGS_TASK if s] 
NA_REGEX_TASK_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in VALID_NA_FOR_REGEX_TASK) + r')$' if VALID_NA_FOR_REGEX_TASK else None

# Pre-compile regex patterns used in rules
TB_CONDITION_PATTERN = re.compile(r"\btb\b|tuberculosis", re.IGNORECASE)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

def _prepare_task_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    default_chw_id: str,
    default_zone_id: str,
    default_patient_id_prefix: str
) -> pd.DataFrame:
    df_prepared = df.copy()
    logger.debug(f"({log_prefix}) Preparing task dataframe. Initial shape: {df_prepared.shape}, Columns: {df_prepared.columns.tolist()}")

    for col_name, config in cols_config.items():
        default_value = config.get("default")
        target_type_str = config.get("type")

        if col_name not in df_prepared.columns:
            if col_name == 'chw_id': default_value = default_chw_id
            elif col_name == 'zone_id': default_value = default_zone_id
            elif col_name == 'patient_id': default_value = default_patient_id_prefix
            
            logger.debug(f"({log_prefix}) Column '{col_name}' missing. Adding with default: '{default_value}' of type {type(default_value)}")
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT 
            elif isinstance(default_value, (list, dict)): 
                 df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value 
        
        current_col_dtype = df_prepared[col_name].dtype
        # Replace common string NAs with np.nan before type conversion for numeric/datetime
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(current_col_dtype):
            if NA_REGEX_TASK_PATTERN:
                try:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_TASK_PATTERN, np.nan, regex=True)
                except Exception as e_regex:
                     logger.warning(f"({log_prefix}) Regex NA replacement failed for column '{col_name}': {e_regex}. Proceeding.")
        
        # Type conversion
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=float)
            elif target_type_str == int:
                 # Ensure default_value for int is int, or handle np.nan carefully if target is nullable int
                 int_default = default_value if isinstance(default_value, int) else (0 if default_value is np.nan else default_value)
                 df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=int_default, target_type=int)
            elif target_type_str == str:
                df_prepared[col_name] = df_prepared[col_name].fillna(str(default_value)).astype(str)
                if NA_REGEX_TASK_PATTERN:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_TASK_PATTERN, str(default_value), regex=True)
                df_prepared[col_name] = df_prepared[col_name].str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Column filled with default.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', default_patient_id_prefix).fillna(default_patient_id_prefix).astype(str)
    
    logger.debug(f"({log_prefix}) Task dataframe preparation complete. Shape: {df_prepared.shape}. Dtypes:\n{df_prepared.dtypes}")
    return df_prepared


def generate_chw_tasks(
    source_patient_data_df: Optional[pd.DataFrame], 
    for_date: Union[str, pd.Timestamp, date_type, datetime], 
    chw_id_context: Optional[str] = None, 
    zone_context_str: Optional[str] = None,  
    max_tasks_to_return_for_summary: int = 20
) -> List[Dict[str, Any]]:
    module_log_prefix = "CHWTaskGenerator"

    try:
        task_gen_target_date_dt = pd.to_datetime(for_date, errors='coerce')
        if pd.NaT is task_gen_target_date_dt:
            raise ValueError(f"Invalid 'for_date' ('{for_date}') for task generation.")
        task_gen_target_date = task_gen_target_date_dt.date()
    except Exception as e_date_task:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ('{for_date}'): {e_date_task}. Defaulting to current system date.", exc_info=True)
        task_gen_target_date = pd.Timestamp('now').date() 
    
    task_gen_target_date_iso = task_gen_target_date.isoformat()
    safe_chw_id_context = chw_id_context if chw_id_context and chw_id_context.strip() else "TeamDefaultCHW"
    safe_zone_context_str = zone_context_str if zone_context_str and zone_context_str.strip() else "GeneralArea"
    default_pid_prefix_tasks = f"UPID_Task_{task_gen_target_date_iso.replace('-', '')}"

    logger.info(f"({module_log_prefix}) Generating CHW tasks for target date: {task_gen_target_date_iso}, CHW Context: {safe_chw_id_context}, Zone Context: {safe_zone_context_str}")

    if not isinstance(source_patient_data_df, pd.DataFrame) or source_patient_data_df.empty:
        logger.warning(f"({module_log_prefix}) No valid patient data provided for task generation on {task_gen_target_date_iso}.")
        return []

    task_gen_cols_cfg = {
        'patient_id': {"default": default_pid_prefix_tasks, "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, 
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
        'tb_contact_traced': {"default": 0, "type": int} 
    }
    df_task_src = _prepare_task_dataframe(
        source_patient_data_df, task_gen_cols_cfg, module_log_prefix, 
        safe_chw_id_context, safe_zone_context_str, default_pid_prefix_tasks
    )

    if 'encounter_date' in df_task_src.columns and df_task_src['encounter_date'].notna().any():
        df_task_src = df_task_src.loc[df_task_src['encounter_date'].dt.date == task_gen_target_date].copy()
    
    if df_task_src.empty:
        logger.info(f"({module_log_prefix}) No data for task generation on {task_gen_target_date_iso} after date filtering or preparation.")
        return []

    df_task_src['temp_sort_priority'] = df_task_src['ai_followup_priority_score'].fillna(0.0) + \
                                        (df_task_src['ai_risk_score'].fillna(0.0) * 0.5) 
    df_sorted_for_rules = df_task_src.sort_values(by='temp_sort_priority', ascending=False).drop(columns=['temp_sort_priority'])

    generated_tasks_list: List[Dict[str, Any]] = []
    processed_patient_task_types_set: Set[Tuple[str, str]] = set() 

    temp_col_to_use = None
    if 'vital_signs_temperature_celsius' in df_sorted_for_rules.columns and df_sorted_for_rules['vital_signs_temperature_celsius'].notna().any():
        temp_col_to_use = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in df_sorted_for_rules.columns and df_sorted_for_rules['max_skin_temp_celsius'].notna().any():
        temp_col_to_use = 'max_skin_temp_celsius'
        logger.debug(f"({module_log_prefix}) Using 'max_skin_temp_celsius' for temperature checks.")

    # Get setting values with robust fallbacks
    spo2_critical_thresh = float(_get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90.0))
    temp_high_fever_thresh = float(_get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0))
    # Assuming priority scores in data are 0-100, and thresholds in settings are 0-1 (like fatigue)
    # Adjust if settings provide 0-100 thresholds directly
    prio_high_thresh_raw = float(_get_setting('TASK_RULE_AI_PRIO_HIGH_THRESHOLD', _get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 0.7)))
    prio_high_thresh = prio_high_thresh_raw * 100 if prio_high_thresh_raw <=1 else prio_high_thresh_raw # Scale if it's 0-1
    
    prio_mod_thresh_raw = float(_get_setting('TASK_RULE_AI_PRIO_MODERATE_THRESHOLD', _get_setting('FATIGUE_INDEX_MODERATE_THRESHOLD', 0.5)))
    prio_mod_thresh = prio_mod_thresh_raw * 100 if prio_mod_thresh_raw <=1 else prio_mod_thresh_raw

    key_conditions_for_action_list = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])


    for _, row_data in df_sorted_for_rules.iterrows():
        patient_id_str = str(row_data.get('patient_id', default_pid_prefix_tasks))
        logger.debug(f"({module_log_prefix}) Evaluating tasks for patient: {patient_id_str}")
        
        current_task_due_date = task_gen_target_date + timedelta(days=_get_setting('DEFAULT_TASK_DUE_DAYS_OFFSET', 1))
        
        base_prio_val = float(row_data.get('ai_followup_priority_score', 0.0))
        ai_risk_val = float(row_data.get('ai_risk_score', 0.0))
        if base_prio_val < 10.0: base_prio_val = max(base_prio_val, ai_risk_val * 0.6)
        base_prio_val = max(10.0, base_prio_val) 

        triggered_task_details: Optional[Dict[str, Any]] = None

        # Rule 1: Critical Vitals
        min_spo2_val = row_data.get('min_spo2_pct') # Already float or NaN from prep
        if pd.notna(min_spo2_val) and min_spo2_val < spo2_critical_thresh:
            task_type = "TASK_VISIT_VITALS_URGENT_SPO2"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": f"URGENT: Assess Critical Low SpO2 ({min_spo2_val:.0f}%)", "prio": 98.0}
                current_task_due_date = task_gen_target_date 
        
        current_temp_val = row_data.get(temp_col_to_use) if temp_col_to_use else np.nan
        if not triggered_task_details and pd.notna(current_temp_val) and current_temp_val >= temp_high_fever_thresh:
            task_type = "TASK_VISIT_VITALS_URGENT_TEMP"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": f"URGENT: Assess High Fever ({current_temp_val:.1f}°C)", "prio": 95.0}
                current_task_due_date = task_gen_target_date
        
        falls_today_val = int(row_data.get('fall_detected_today', 0))
        if not triggered_task_details and falls_today_val > 0:
            task_type = "TASK_VISIT_FALL_ASSESS"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": f"Assess After Fall (Falls today: {falls_today_val})", "prio": 92.0}
                current_task_due_date = task_gen_target_date
        
        # Rule 2: Pending Critical Referral
        referral_status_str = str(row_data.get('referral_status', '')).lower().strip()
        if not triggered_task_details and referral_status_str == 'pending':
            condition_str_for_ref = str(row_data.get('condition', '')).lower()
            referral_reason_str_for_ref = str(row_data.get('referral_reason', '')).lower()
            is_key_cond_ref = any(re.escape(kc.lower()) in condition_str_for_ref for kc in key_conditions_for_action_list if kc)
            is_urgent_ref = "urgent" in referral_reason_str_for_ref

            if is_key_cond_ref or is_urgent_ref:
                task_type = "TASK_VISIT_REFERRAL_TRACK"
                if (patient_id_str, task_type) not in processed_patient_task_types_set:
                    triggered_task_details = {"type": task_type, "desc": f"Follow-up: Critical Referral for {row_data.get('condition', 'N/A')}", "prio": 88.0}
        
        # Rule 3: High AI Follow-up Priority
        if not triggered_task_details and base_prio_val >= prio_high_thresh:
            task_type = "TASK_VISIT_FOLLOWUP_AI"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": f"Priority Follow-up (AI Prio Score: {base_prio_val:.0f})", "prio": base_prio_val}

        # Rule 4: Medication Adherence
        med_adherence_str_val = str(row_data.get('medication_adherence_self_report', '')).lower().strip()
        if not triggered_task_details and med_adherence_str_val == 'poor':
            task_type = "TASK_VISIT_ADHERENCE_SUPPORT"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": "Support Medication Adherence (Reported Poor)", "prio": max(base_prio_val, 75.0)}
        
        # Rule 5: TB Contact Tracing
        tb_contact_traced_val_int = int(row_data.get('tb_contact_traced', 1)) # Default to 1 (traced) if missing/NaN
        if not triggered_task_details and \
           TB_CONDITION_PATTERN.search(str(row_data.get('condition', ''))) and \
           tb_contact_traced_val_int == 0: 
            task_type = "TASK_TB_CONTACT_TRACE"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": "Initiate/Continue TB Contact Tracing", "prio": max(base_prio_val, 80.0)}
        
        # Rule 6: Routine Checkup
        if not triggered_task_details and base_prio_val >= prio_mod_thresh:
            task_type = "TASK_VISIT_ROUTINE_CHECK"
            if (patient_id_str, task_type) not in processed_patient_task_types_set:
                triggered_task_details = {"type": task_type, "desc": f"Routine Health Check (AI Prio: {base_prio_val:.0f})", "prio": base_prio_val}

        if triggered_task_details:
            logger.debug(f"({module_log_prefix}) Triggered task '{triggered_task_details['type']}' for patient {patient_id_str} with prio {triggered_task_details['prio']:.1f}")
            context_parts = []
            if str(row_data.get('condition', 'N/A')) not in ["N/A", task_gen_cols_cfg['condition']['default']]: context_parts.append(f"Cond: {row_data['condition']}")
            if pd.notna(row_data.get('age')): context_parts.append(f"Age: {row_data['age']:.0f}")
            if pd.notna(min_spo2_val): context_parts.append(f"Last SpO2: {min_spo2_val:.0f}%")
            if temp_col_to_use and pd.notna(current_temp_val): context_parts.append(f"Last Temp: {current_temp_val:.1f}°C")
            if pd.notna(ai_risk_val): context_parts.append(f"AI Risk: {ai_risk_val:.0f}")
            key_ctx = " | ".join(context_parts) if context_parts else "General Assessment"

            task_type_short_suffix = triggered_task_details['type'].split('_')[-1][:4] if '_' in triggered_task_details['type'] else triggered_task_details['type'][:4]
            
            generated_tasks_list.append({
                "task_id": f"TSK_{patient_id_str}_{task_gen_target_date_iso.replace('-', '')}_{task_type_short_suffix.upper()}_{len(generated_tasks_list)+1:03d}",
                "patient_id": patient_id_str, 
                "assigned_chw_id": str(row_data.get('chw_id', safe_chw_id_context)),
                "zone_id": str(row_data.get('zone_id', safe_zone_context_str)), 
                "task_type_code": triggered_task_details["type"],
                "task_description": f"{triggered_task_details['desc']} for Pt. {patient_id_str}",
                "priority_score": round(min(float(triggered_task_details["prio"]), 100.0), 1),
                "due_date": current_task_due_date.isoformat(), 
                "status": "PENDING",
                "key_patient_context": key_ctx,
                "alert_source_info": f"Data from {(row_data.get('encounter_date').strftime('%Y-%m-%d') if pd.notna(row_data.get('encounter_date')) else task_gen_target_date_iso)}"
            })
            processed_patient_task_types_set.add((patient_id_str, triggered_task_details["type"])) 

    if not generated_tasks_list:
        logger.info(f"({module_log_prefix}) No tasks generated for {task_gen_target_date_iso} based on current rules and data.")
        return []
    
    final_tasks_to_return_sorted = sorted(generated_tasks_list, key=lambda x_task: x_task['priority_score'], reverse=True)
    num_total_generated = len(final_tasks_to_return_sorted)
    num_to_return = min(num_total_generated, max_tasks_to_return_for_summary)
    logger.info(f"({module_log_prefix}) Generated {num_total_generated} CHW tasks for {task_gen_target_date_iso}. Returning top {num_to_return}.")
    
    return final_tasks_to_return_sorted[:num_to_return]
