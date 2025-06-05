# sentinel_project_root/pages/chw_components/summary_metrics.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.

import pandas as pd
import numpy as np
import logging
import re # For condition matching
from typing import Dict, Any, Optional
from datetime import date as date_type # For type hinting and date operations

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def calculate_chw_daily_summary_metrics(
    chw_daily_encounter_df: Optional[pd.DataFrame], 
    for_date: Any, 
    chw_daily_kpi_input_data: Optional[Dict[str, Any]] = None, 
    source_context: str = "CHWDailySummaryMetrics"
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.
    Merges pre-calculated KPI data with metrics derived directly from daily encounter DataFrame.
    """
    try:
        target_processing_date = pd.to_datetime(for_date, errors='coerce').date()
        if pd.isna(target_processing_date): raise ValueError("Invalid for_date for summary metrics.")
    except Exception as e_date_parse_sum:
        logger.warning(f"({source_context}) Invalid 'for_date' ({for_date}): {e_date_parse_sum}. Defaulting to current system date.")
        target_processing_date = pd.Timestamp('now').date()
    target_date_iso_str = target_processing_date.isoformat()

    logger.info(f"({source_context}) Calculating CHW daily summary metrics for date: {target_date_iso_str}")

    metrics_summary: Dict[str, Any] = {
        "date_of_activity": target_date_iso_str, "visits_count": 0, "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan, "fever_cases_identified_count": 0,
        "high_fever_cases_identified_count": 0, "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan, "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", "worker_self_fatigue_index_today": np.nan
    }

    if isinstance(chw_daily_kpi_input_data, dict):
        logger.debug(f"({source_context}) Populating metrics from pre-calculated input data.")
        for key, value in chw_daily_kpi_input_data.items():
            if key in metrics_summary:
                try:
                    if isinstance(metrics_summary[key], int) and pd.notna(value):
                        metrics_summary[key] = int(pd.to_numeric(value, errors='raise'))
                    elif isinstance(metrics_summary[key], float) and pd.notna(value):
                        metrics_summary[key] = float(pd.to_numeric(value, errors='raise'))
                    elif isinstance(metrics_summary[key], str) and pd.notna(value):
                        metrics_summary[key] = str(value)
                except (ValueError, TypeError) as e_kpi_conv_sum:
                    logger.warning(f"({source_context}) Error converting pre-calc KPI '{key}' (value: {value}): {e_kpi_conv_sum}. Using default.")

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.info(f"({source_context}) No daily encounter DataFrame for {target_date_iso_str}. Relying on pre-calc/defaults.")
        # Finalize fatigue level from pre-calculated index if DataFrame is empty
        if pd.notna(metrics_summary.get("worker_self_fatigue_index_today")):
            fatigue_score_pre = metrics_summary["worker_self_fatigue_index_today"]
            if fatigue_score_pre >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
            elif fatigue_score_pre >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD: metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
            else: metrics_summary["worker_self_fatigue_level_code"] = "LOW"
        return metrics_summary

    df_enc_sum = chw_daily_encounter_df.copy()
    if 'encounter_date' not in df_enc_sum.columns:
        logger.error(f"({source_context}) 'encounter_date' missing. Metrics for {target_date_iso_str} inaccurate."); return metrics_summary
    df_enc_sum['encounter_date'] = pd.to_datetime(df_enc_sum['encounter_date'], errors='coerce')
    df_enc_sum = df_enc_sum[df_enc_sum['encounter_date'].dt.date == target_processing_date]
    if df_enc_sum.empty: logger.info(f"({source_context}) No encounters for {target_date_iso_str} after date filtering."); return metrics_summary
    
    enc_cols_cfg_sum = {
        'patient_id': {"default": f"UPID_Sum_{target_date_iso_str}", "type": str},
        'encounter_type': {"default": "UType", "type": str},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float},
        'avg_daily_steps': {"default": np.nan, "type": float},
        'fall_detected_today': {"default": 0, "type": int},
        'condition': {"default": "UCond", "type": str},
        'referral_status': {"default": "UStat", "type": str},
        'referral_reason': {"default": "N/A", "type": str}
    }
    common_na_sum = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    na_regex_sum = r'^(?:' + '|'.join(re.escape(s) for s in common_na_sum if s) + r')$'

    for col, cfg in enc_cols_cfg_sum.items():
        if col not in df_enc_sum.columns: df_enc_sum[col] = cfg["default"]
        if cfg["type"] == float: df_enc_sum[col] = convert_to_numeric(df_enc_sum[col], default_value=cfg["default"])
        elif cfg["type"] == int: df_enc_sum[col] = convert_to_numeric(df_enc_sum[col], default_value=cfg["default"], target_type=int)
        elif cfg["type"] == str:
            df_enc_sum[col] = df_enc_sum[col].astype(str).fillna(str(cfg["default"]))
            if any(common_na_sum): df_enc_sum[col] = df_enc_sum[col].replace(na_regex_sum, str(cfg["default"]), regex=True)
            df_enc_sum[col] = df_enc_sum[col].str.strip()

    patient_records_df = df_enc_sum[~df_enc_sum['encounter_type'].astype(str).str.contains("WORKER_SELF", case=False, na=False)]
    if not patient_records_df.empty:
        if 'patient_id' in patient_records_df.columns: metrics_summary["visits_count"] = patient_records_df['patient_id'].nunique()
        if 'ai_followup_priority_score' in patient_records_df.columns:
            prio_s = patient_records_df['ai_followup_priority_score']
            metrics_summary["high_ai_prio_followups_count"] = patient_records_df[prio_s >= settings.FATIGUE_INDEX_HIGH_THRESHOLD]['patient_id'].nunique()
        if 'ai_risk_score' in patient_records_df.columns:
            risk_s = patient_records_df.drop_duplicates(subset=['patient_id'])['ai_risk_score']
            if risk_s.notna().any(): metrics_summary["avg_risk_of_visited_patients"] = risk_s.mean()
        
        temp_col_sum = next((c for c in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if c in patient_records_df.columns and patient_records_df[c].notna().any()), None)
        if temp_col_sum:
            temps_s = patient_records_df[temp_col_sum]
            metrics_summary["fever_cases_identified_count"] = patient_records_df[temps_s >= settings.ALERT_BODY_TEMP_FEVER_C]['patient_id'].nunique()
            metrics_summary["high_fever_cases_identified_count"] = patient_records_df[temps_s >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()
        if 'min_spo2_pct' in patient_records_df.columns:
            spo2_s = patient_records_df['min_spo2_pct']
            metrics_summary["critical_spo2_cases_identified_count"] = patient_records_df[spo2_s < settings.ALERT_SPO2_CRITICAL_LOW_PCT]['patient_id'].nunique()
        if 'avg_daily_steps' in patient_records_df.columns:
            steps_s = patient_records_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
            if steps_s.notna().any(): metrics_summary["avg_steps_of_visited_patients"] = steps_s.mean()
        if 'fall_detected_today' in patient_records_df.columns:
            metrics_summary["fall_events_among_visited_count"] = patient_records_df[patient_records_df['fall_detected_today'] > 0]['patient_id'].nunique()
        if 'condition' in patient_records_df.columns and 'referral_status' in patient_records_df.columns:
            crit_ref_mask = (patient_records_df['referral_status'].astype(str).str.lower() == 'pending') & \
                            (patient_records_df['condition'].astype(str).str.contains('|'.join(re.escape(kc) for kc in settings.KEY_CONDITIONS_FOR_ACTION), case=False, na=False, regex=True))
            metrics_summary["pending_critical_referrals_generated_today_count"] = patient_records_df[crit_ref_mask]['patient_id'].nunique()
    
    if pd.isna(metrics_summary["worker_self_fatigue_index_today"]): # Calculate if not pre-populated
        worker_self_checks = df_enc_sum[df_enc_sum['encounter_type'].astype(str).str.contains("WORKER_SELF_CHECK", case=False, na=False)]
        if not worker_self_checks.empty:
            fatigue_cols = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            fatigue_metric = next((c for c in fatigue_cols if c in worker_self_checks.columns and worker_self_checks[c].notna().any()), None)
            if fatigue_metric:
                fatigue_val = worker_self_checks[fatigue_metric].max()
                if pd.notna(fatigue_val): metrics_summary["worker_self_fatigue_index_today"] = float(fatigue_val)
    
    final_fatigue_score_sum = metrics_summary["worker_self_fatigue_index_today"]
    if pd.notna(final_fatigue_score_sum):
        if final_fatigue_score_sum >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif final_fatigue_score_sum >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD: metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else: metrics_summary["worker_self_fatigue_level_code"] = "LOW"
    else: metrics_summary["worker_self_fatigue_level_code"] = "NOT_ASSESSED"

    float_round_cfg = {"avg_risk_of_visited_patients": 1, "avg_steps_of_visited_patients": 0, "worker_self_fatigue_index_today": 1}
    for metric_key, places in float_round_cfg.items():
        if pd.notna(metrics_summary.get(metric_key)):
            try: metrics_summary[metric_key] = round(float(metrics_summary[metric_key]), places)
            except: logger.warning(f"({source_context}) Could not round metric '{metric_key}'.")

    logger.info(f"({source_context}) CHW daily summary metrics calculated for {target_date_iso_str}.")
    return metrics_summary
