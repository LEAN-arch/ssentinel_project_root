# sentinel_project_root/pages/chw_components/summary_metrics.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, Union
from datetime import date as date_type, datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in summary_metrics.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_SUMMARY = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_SUMMARY_PATTERN = r'^\s*$' + (r'|^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_SUMMARY if s) + r')$' if any(COMMON_NA_STRINGS_SUMMARY) else '')


def _prepare_summary_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    target_date_iso_str: str # For default patient_id
) -> pd.DataFrame:
    """Prepares the DataFrame for summary metric calculation."""
    df_prepared = df.copy()

    for col_name, config in cols_config.items():
        default_value = config["default"]
        target_type_str = config["type"]

        if col_name not in df_prepared.columns:
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            elif isinstance(default_value, (list, dict)): 
                 df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value
        
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(df_prepared[col_name].dtype):
            if NA_REGEX_SUMMARY_PATTERN:
                try:
                    df_prepared[col_name].replace(NA_REGEX_SUMMARY_PATTERN, np.nan, regex=True, inplace=True)
                except Exception as e_regex:
                     logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=float)
            elif target_type_str == int:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=int)
            elif target_type_str == str:
                series = df_prepared[col_name].fillna(str(default_value))
                df_prepared[col_name] = series.astype(str).str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
    
    placeholder_pid = f"UPID_Sum_{target_date_iso_str}"
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'].replace('', placeholder_pid, inplace=True)
        df_prepared['patient_id'].fillna(placeholder_pid, inplace=True)

    return df_prepared


def calculate_chw_daily_summary_metrics(
    chw_daily_encounter_df: Optional[pd.DataFrame], 
    for_date: Union[str, pd.Timestamp, date_type, datetime], 
    chw_daily_kpi_input_data: Optional[Dict[str, Any]] = None, 
    source_context: str = "CHWDailySummaryMetrics"
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.
    Merges pre-calculated KPI data with metrics derived directly from daily encounter DataFrame.
    """
    try:
        target_processing_date_dt = pd.to_datetime(for_date, errors='coerce')
        if pd.NaT is target_processing_date_dt:
            raise ValueError(f"Invalid 'for_date' ({for_date}) for summary metrics.")
        target_processing_date = target_processing_date_dt.date()
    except Exception as e_date_parse_sum:
        logger.warning(f"({source_context}) Invalid 'for_date' ('{for_date}'): {e_date_parse_sum}. Defaulting to current system date.", exc_info=True)
        target_processing_date = pd.Timestamp('now').date()
    
    target_date_iso_str = target_processing_date.isoformat()
    logger.info(f"({source_context}) Calculating CHW daily summary metrics for date: {target_date_iso_str}")

    metrics_summary: Dict[str, Any] = {
        "date_of_activity": target_date_iso_str, "visits_count": 0, 
        "high_ai_prio_followups_count": 0, "avg_risk_of_visited_patients": np.nan, 
        "fever_cases_identified_count": 0, "high_fever_cases_identified_count": 0, 
        "critical_spo2_cases_identified_count": 0, "avg_steps_of_visited_patients": np.nan, 
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", 
        "worker_self_fatigue_index_today": np.nan
    }
    
    initial_metric_types = {k: type(v) for k, v in metrics_summary.items()}

    if isinstance(chw_daily_kpi_input_data, dict):
        logger.debug(f"({source_context}) Populating metrics from pre-calculated input data: {list(chw_daily_kpi_input_data.keys())}")
        for key, value in chw_daily_kpi_input_data.items():
            if key in metrics_summary and pd.notna(value):
                try:
                    # CORRECTED: Removed redundant/fragile nested casting. Delegated conversion fully
                    # to the robust `convert_to_numeric` helper by passing the target type.
                    expected_type = initial_metric_types.get(key)
                    if expected_type in [int, float]:
                        metrics_summary[key] = convert_to_numeric(
                            value,
                            default_value=metrics_summary[key],
                            target_type=expected_type
                        )
                    elif expected_type is str:
                        metrics_summary[key] = str(value)
                    else:
                        metrics_summary[key] = value
                except (ValueError, TypeError) as e_kpi_conv_sum:
                    logger.warning(f"({source_context}) Error converting pre-calc KPI '{key}' (value: '{value}', type: {type(value)}): {e_kpi_conv_sum}. Using default.")
                    
    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.info(f"({source_context}) No daily encounter DataFrame for {target_date_iso_str}. Metrics will rely on pre-calculated data or defaults.")
    else:
        df_enc_sum = chw_daily_encounter_df

        enc_cols_cfg_sum = {
            'patient_id': {"default": f"UPID_Sum_{target_date_iso_str}", "type": str},
            'encounter_date': {"default": pd.NaT, "type": "datetime"},
            'encounter_type': {"default": "UnknownEncounterType", "type": str},
            'ai_followup_priority_score': {"default": np.nan, "type": float},
            'ai_risk_score': {"default": np.nan, "type": float},
            'min_spo2_pct': {"default": np.nan, "type": float},
            'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
            'max_skin_temp_celsius': {"default": np.nan, "type": float},
            'avg_daily_steps': {"default": np.nan, "type": float},
            'fall_detected_today': {"default": 0, "type": int},
            'condition': {"default": "UnknownCondition", "type": str},
            'referral_status': {"default": "UnknownStatus", "type": str},
            'referral_reason': {"default": "N/A", "type": str}
        }
        df_enc_sum = _prepare_summary_dataframe(df_enc_sum, enc_cols_cfg_sum, source_context, target_date_iso_str)

        if 'encounter_date' in df_enc_sum.columns and df_enc_sum['encounter_date'].notna().any():
            df_enc_sum = df_enc_sum[df_enc_sum['encounter_date'].dt.date == target_processing_date]

        if df_enc_sum.empty:
            logger.info(f"({source_context}) No encounters for {target_date_iso_str} after date filtering.")
        else:
            patient_records_df = df_enc_sum[
                ~df_enc_sum.get('encounter_type', pd.Series(dtype=str)).astype(str).str.contains("WORKER_SELF_CHECK", case=False, na=False)
            ]

            if not patient_records_df.empty:
                metrics_summary["visits_count"] = patient_records_df['patient_id'].nunique()
                
                prio_score_col = 'ai_followup_priority_score'
                prio_high_thresh = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80)
                if prio_score_col in patient_records_df.columns:
                    metrics_summary["high_ai_prio_followups_count"] = patient_records_df[patient_records_df[prio_score_col] >= prio_high_thresh]['patient_id'].nunique()

                if 'ai_risk_score' in patient_records_df.columns:
                    unique_patient_risks = patient_records_df.drop_duplicates(subset=['patient_id'])['ai_risk_score'].dropna()
                    if not unique_patient_risks.empty:
                        metrics_summary["avg_risk_of_visited_patients"] = unique_patient_risks.mean()
                
                temp_col_to_use = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] 
                                        if tc in patient_records_df.columns and patient_records_df[tc].notna().any()), None)
                if temp_col_to_use:
                    temps = patient_records_df[temp_col_to_use]
                    fever_thresh = getattr(settings, 'ALERT_BODY_TEMP_FEVER_C', 38.0)
                    high_fever_thresh = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
                    metrics_summary["fever_cases_identified_count"] = patient_records_df[temps >= fever_thresh]['patient_id'].nunique()
                    metrics_summary["high_fever_cases_identified_count"] = patient_records_df[temps >= high_fever_thresh]['patient_id'].nunique()

                spo2_critical_thresh = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90.0)
                if 'min_spo2_pct' in patient_records_df.columns:
                    metrics_summary["critical_spo2_cases_identified_count"] = patient_records_df[patient_records_df['min_spo2_pct'] < spo2_critical_thresh]['patient_id'].nunique()

                if 'avg_daily_steps' in patient_records_df.columns:
                    unique_patient_steps = patient_records_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps'].dropna()
                    if not unique_patient_steps.empty:
                        metrics_summary["avg_steps_of_visited_patients"] = unique_patient_steps.mean()

                if 'fall_detected_today' in patient_records_df.columns:
                    metrics_summary["fall_events_among_visited_count"] = patient_records_df[patient_records_df['fall_detected_today'] > 0]['patient_id'].nunique()

                key_conds = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
                if key_conds and all(c in patient_records_df.columns for c in ['condition', 'referral_status']):
                    key_cond_pattern = '|'.join([re.escape(kc) for kc in key_conds])
                    crit_ref_mask = (
                        patient_records_df['referral_status'].str.lower() == 'pending'
                    ) & (
                        patient_records_df['condition'].str.contains(key_cond_pattern, case=False, na=False, regex=True)
                    )
                    metrics_summary["pending_critical_referrals_generated_today_count"] = patient_records_df[crit_ref_mask]['patient_id'].nunique()

        if pd.isna(metrics_summary.get("worker_self_fatigue_index_today")): 
            worker_self_checks_df = df_enc_sum[
                df_enc_sum.get('encounter_type', pd.Series(dtype=str)).str.contains("WORKER_SELF_CHECK", case=False, na=False)
            ]
            if not worker_self_checks_df.empty:
                fatigue_cols_to_check = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
                chosen_fatigue_metric_col = next((col for col in fatigue_cols_to_check if col in worker_self_checks_df.columns and worker_self_checks_df[col].notna().any()), None)
                if chosen_fatigue_metric_col:
                    fatigue_value = worker_self_checks_df[chosen_fatigue_metric_col].max()
                    if pd.notna(fatigue_value):
                        metrics_summary["worker_self_fatigue_index_today"] = float(fatigue_value)

    final_fatigue_score = metrics_summary.get("worker_self_fatigue_index_today")
    if pd.notna(final_fatigue_score):
        high_thresh = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80)
        mod_thresh = getattr(settings, 'FATIGUE_INDEX_MODERATE_THRESHOLD', 60)
        if final_fatigue_score >= high_thresh:
            metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif final_fatigue_score >= mod_thresh:
            metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary["worker_self_fatigue_level_code"] = "LOW"
    else:
        metrics_summary["worker_self_fatigue_level_code"] = "NOT_ASSESSED"

    float_metrics_to_round = {
        "avg_risk_of_visited_patients": 1, 
        "avg_steps_of_visited_patients": 0, 
        "worker_self_fatigue_index_today": 1
    }
    for metric_name, decimal_places in float_metrics_to_round.items():
        if pd.notna(metrics_summary.get(metric_name)):
            try:
                metrics_summary[metric_name] = round(float(metrics_summary[metric_name]), decimal_places)
            except (ValueError, TypeError) as e_round:
                logger.warning(f"({source_context}) Could not round metric '{metric_name}' (value: {metrics_summary[metric_name]}): {e_round}.")

    logger.info(f"({source_context}) CHW daily summary metrics calculated successfully for {target_date_iso_str}.")
    return metrics_summary
