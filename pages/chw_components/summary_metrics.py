# sentinel_project_root/pages/chw_components/summary_metrics.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.
# Renamed from summary_metrics_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import date # For type hinting and date operations

from config import settings # Use new settings module
from data_processing.helpers import convert_to_numeric # Local import

logger = logging.getLogger(__name__)


def calculate_chw_daily_summary_metrics(
    chw_daily_encounter_df: Optional[pd.DataFrame], # Data specific to the CHW for the target date
    for_date: Any, # Expects a date or date-like string for context
    chw_daily_kpi_input_data: Optional[Dict[str, Any]] = None, # Optional pre-calculated values
    source_context: str = "CHWDailySummaryMetrics" # Added for logging context
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.
    Merges pre-calculated KPI data with metrics derived directly from daily encounter DataFrame.
    Ensures robust handling of missing data and correct typing.

    Args:
        chw_daily_encounter_df: DataFrame of CHW's encounters for the specific 'for_date'.
                                Should already be filtered by CHW ID if applicable.
        for_date: The date these metrics apply to.
        chw_daily_kpi_input_data: Optional dict with pre-aggregated values that might
                                  come from a daily rollup process or another source.
                                  Keys should match those in `metrics_summary_output`.
        source_context: Logging context string.

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics.
    """
    # Standardize for_date
    try:
        target_processing_date = pd.to_datetime(for_date).date() if for_date else pd.Timestamp('now').date()
    except Exception:
        logger.warning(f"({source_context}) Invalid 'for_date' ({for_date}). Defaulting to current system date.")
        target_processing_date = pd.Timestamp('now').date()
    target_date_iso_str = target_processing_date.isoformat()

    logger.info(f"({source_context}) Calculating CHW daily summary metrics for date: {target_date_iso_str}")

    # Initialize metrics with defaults (ensure types are consistent with expected output)
    metrics_summary_output: Dict[str, Any] = {
        "date_of_activity": target_date_iso_str,
        "visits_count": 0, # int
        "high_ai_prio_followups_count": 0, # int
        "avg_risk_of_visited_patients": np.nan, # float
        "fever_cases_identified_count": 0, # int
        "high_fever_cases_identified_count": 0, # int
        "critical_spo2_cases_identified_count": 0, # int
        "avg_steps_of_visited_patients": np.nan, # float
        "fall_events_among_visited_count": 0, # int
        "pending_critical_referrals_generated_today_count": 0, # int
        "worker_self_fatigue_level_code": "NOT_ASSESSED", # str: LOW, MODERATE, HIGH, NOT_ASSESSED
        "worker_self_fatigue_index_today": np.nan # float (0-100 score)
    }

    # Populate from pre-calculated KPI input data first, if provided
    # This allows overriding defaults or providing externally calculated values.
    if isinstance(chw_daily_kpi_input_data, dict):
        logger.debug(f"({source_context}) Populating metrics from chw_daily_kpi_input_data.")
        for key, value in chw_daily_kpi_input_data.items():
            if key in metrics_summary_output:
                try:
                    if isinstance(metrics_summary_output[key], int) and pd.notna(value):
                        metrics_summary_output[key] = int(convert_to_numeric(pd.Series([value]), default_value=0).iloc[0])
                    elif isinstance(metrics_summary_output[key], float) and pd.notna(value): # Handles np.nan default
                        metrics_summary_output[key] = float(convert_to_numeric(pd.Series([value]), default_value=np.nan).iloc[0])
                    elif isinstance(metrics_summary_output[key], str) and pd.notna(value) : # For string types like fatigue_level_code
                        metrics_summary_output[key] = str(value)
                    # If value is None/NaN and key exists, it will keep the default (e.g., np.nan for float)
                except (ValueError, TypeError) as e_kpi_conv:
                    logger.warning(f"({source_context}) Error converting pre-calculated KPI '{key}' (value: {value}): {e_kpi_conv}. Using default.")
            else:
                logger.debug(f"({source_context}) Key '{key}' from input_data not in standard metrics_summary_output.")
    
    # Refine or calculate metrics using the daily encounter DataFrame
    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.info(f"({source_context}) No CHW daily encounter DataFrame provided for {target_date_iso_str}. Metrics will rely on pre_calculated_kpis or defaults.")
        # If no df, finalize fatigue level code based on index (if available from pre_calc)
        if pd.notna(metrics_summary_output["worker_self_fatigue_index_today"]):
             # Logic duplicated below, consider helper if this gets complex
            fatigue_score_val = metrics_summary_output["worker_self_fatigue_index_today"]
            if fatigue_score_val >= settings.FATIGUE_INDEX_HIGH_THRESHOLD:
                metrics_summary_output["worker_self_fatigue_level_code"] = "HIGH"
            elif fatigue_score_val >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD:
                metrics_summary_output["worker_self_fatigue_level_code"] = "MODERATE"
            else: # Covers < MODERATE and any NaNs if not caught by pd.notna earlier
                metrics_summary_output["worker_self_fatigue_level_code"] = "LOW" if pd.notna(fatigue_score_val) else "NOT_ASSESSED"
        return metrics_summary_output # Return based on defaults and pre_calculated_kpis

    df_enc = chw_daily_encounter_df.copy()
    
    # Ensure 'encounter_date' is datetime and filter for the target_date if not already done
    if 'encounter_date' in df_enc.columns:
        df_enc['encounter_date'] = pd.to_datetime(df_enc['encounter_date'], errors='coerce')
        df_enc = df_enc[df_enc['encounter_date'].dt.date == target_processing_date] # Strict filter
    else:
        logger.warning(f"({source_context}) 'encounter_date' column missing. Metrics for {target_date_iso_str} may be inaccurate.")
        return metrics_summary_output

    if df_enc.empty:
        logger.info(f"({source_context}) No CHW encounter data for {target_date_iso_str} after date filtering.")
        return metrics_summary_output

    # Define essential columns for calculations from encounters and their safe defaults
    encounter_cols_config = {
        'patient_id': {"default": f"UnknownPID_CHWSum_{target_date_iso_str}", "type": str},
        'encounter_type': {"default": "UnknownType", "type": str},
        'ai_followup_priority_score': {"default": np.nan, "type": float},
        'ai_risk_score': {"default": np.nan, "type": float},
        'min_spo2_pct': {"default": np.nan, "type": float},
        'vital_signs_temperature_celsius': {"default": np.nan, "type": float},
        'max_skin_temp_celsius': {"default": np.nan, "type": float}, # Alternative temp
        'avg_daily_steps': {"default": np.nan, "type": float}, # Patient steps
        'fall_detected_today': {"default": 0, "type": int}, # Patient fall flag
        'condition': {"default": "UnknownCondition", "type": str},
        'referral_status': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "N/A", "type": str}
    }
    common_na_summary_metrics = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown']

    for col, config in encounter_cols_config.items():
        if col not in df_enc.columns: df_enc[col] = config["default"]
        if config["type"] == float:
            df_enc[col] = convert_to_numeric(df_enc[col], default_value=config["default"])
        elif config["type"] == int:
            df_enc[col] = convert_to_numeric(df_enc[col], default_value=config["default"], target_type=int)
        elif config["type"] == str:
            df_enc[col] = df_enc[col].astype(str).fillna(str(config["default"]))
            df_enc[col] = df_enc[col].replace(common_na_summary_metrics, str(config["default"]), regex=False).str.strip()

    # Exclude worker self-checks for patient-specific metrics
    patient_records_for_day_df = df_enc[
        ~df_enc['encounter_type'].astype(str).str.contains("WORKER_SELF", case=False, na=False)
    ]

    if not patient_records_for_day_df.empty:
        metrics_summary_output["visits_count"] = patient_records_for_day_df['patient_id'].nunique()

        if 'ai_followup_priority_score' in patient_records_for_day_df.columns:
            high_prio_scores = patient_records_for_day_df['ai_followup_priority_score']
            metrics_summary_output["high_ai_prio_followups_count"] = patient_records_for_day_df[
                high_prio_scores >= settings.FATIGUE_INDEX_HIGH_THRESHOLD # Using this as general high prio
            ]['patient_id'].nunique()

        if 'ai_risk_score' in patient_records_for_day_df.columns:
            # Avg risk of unique patients visited
            unique_patient_risk_scores = patient_records_for_day_df.drop_duplicates(subset=['patient_id'])['ai_risk_score']
            if unique_patient_risk_scores.notna().any():
                metrics_summary_output["avg_risk_of_visited_patients"] = unique_patient_risk_scores.mean()

        temp_col_to_use_summary = next((c for c in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if c in patient_records_for_day_df.columns and patient_records_for_day_df[c].notna().any()), None)
        if temp_col_to_use_summary:
            temps_series = patient_records_for_day_df[temp_col_to_use_summary]
            metrics_summary_output["fever_cases_identified_count"] = patient_records_for_day_df[temps_series >= settings.ALERT_BODY_TEMP_FEVER_C]['patient_id'].nunique()
            metrics_summary_output["high_fever_cases_identified_count"] = patient_records_for_day_df[temps_series >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()

        if 'min_spo2_pct' in patient_records_for_day_df.columns:
            spo2_series = patient_records_for_day_df['min_spo2_pct']
            metrics_summary_output["critical_spo2_cases_identified_count"] = patient_records_for_day_df[
                spo2_series < settings.ALERT_SPO2_CRITICAL_LOW_PCT
            ]['patient_id'].nunique()

        if 'avg_daily_steps' in patient_records_for_day_df.columns:
            unique_patient_steps = patient_records_for_day_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
            if unique_patient_steps.notna().any():
                metrics_summary_output["avg_steps_of_visited_patients"] = unique_patient_steps.mean()

        if 'fall_detected_today' in patient_records_for_day_df.columns: # Already int
            metrics_summary_output["fall_events_among_visited_count"] = patient_records_for_day_df[
                patient_records_for_day_df['fall_detected_today'] > 0
            ]['patient_id'].nunique()

        if 'condition' in patient_records_for_day_df.columns and 'referral_status' in patient_records_for_day_df.columns:
            crit_referral_mask = (
                patient_records_for_day_df['referral_status'].astype(str).str.lower() == 'pending'
            ) & (
                patient_records_for_day_df['condition'].astype(str).str.contains('|'.join(settings.KEY_CONDITIONS_FOR_ACTION), case=False, na=False)
            )
            metrics_summary_output["pending_critical_referrals_generated_today_count"] = patient_records_for_day_df[crit_referral_mask]['patient_id'].nunique()
    
    # Derive worker fatigue index and level code (if not already set by pre_calculated_kpis)
    # This uses worker_self_check encounters from the full df_enc for the day
    if pd.isna(metrics_summary_output["worker_self_fatigue_index_today"]):
        worker_self_check_records = df_enc[df_enc['encounter_type'].astype(str).str.contains("WORKER_SELF_CHECK", case=False, na=False)]
        if not worker_self_check_records.empty:
            # Prioritize AI follow-up score if available for fatigue, then psych score, then stress level
            fatigue_metric_col_name = next((col for col in ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score'] 
                                            if col in worker_self_check_records.columns and worker_self_check_records[col].notna().any()), None)
            if fatigue_metric_col_name:
                fatigue_score_from_df = worker_self_check_records[fatigue_metric_col_name].max() # Max score if multiple checks
                if pd.notna(fatigue_score_from_df):
                    metrics_summary_output["worker_self_fatigue_index_today"] = float(fatigue_score_from_df)
            else:
                logger.debug(f"({source_context}) No suitable fatigue metric column found in WORKER_SELF_CHECK records for {target_date_iso_str}.")
    
    # Set fatigue level code based on the final fatigue index
    final_fatigue_score = metrics_summary_output["worker_self_fatigue_index_today"]
    if pd.notna(final_fatigue_score):
        if final_fatigue_score >= settings.FATIGUE_INDEX_HIGH_THRESHOLD:
            metrics_summary_output["worker_self_fatigue_level_code"] = "HIGH"
        elif final_fatigue_score >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD:
            metrics_summary_output["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary_output["worker_self_fatigue_level_code"] = "LOW"
    else: # If still NaN after all checks
         metrics_summary_output["worker_self_fatigue_level_code"] = "NOT_ASSESSED"


    # Final rounding for displayable float metrics
    float_metrics_to_round = {
        "avg_risk_of_visited_patients": 1,
        "avg_steps_of_visited_patients": 0,
        "worker_self_fatigue_index_today": 1
    }
    for metric_key, decimal_places in float_metrics_to_round.items():
        if pd.notna(metrics_summary_output.get(metric_key)):
            try:
                metrics_summary_output[metric_key] = round(float(metrics_summary_output[metric_key]), decimal_places)
            except (ValueError, TypeError):
                 logger.warning(f"({source_context}) Could not round metric '{metric_key}'. Value: {metrics_summary_output[metric_key]}")
                 # Keep as np.nan if it was already, or leave as is if unroundable string somehow
                 if not isinstance(metrics_summary_output[metric_key], (float,int)):
                     metrics_summary_output[metric_key] = np.nan


    logger.info(
        f"({source_context}) CHW daily summary metrics calculated for {target_date_iso_str}: "
        f"Visits={metrics_summary_output['visits_count']}, "
        f"AvgRisk={metrics_summary_output['avg_risk_of_visited_patients']:.1f if pd.notna(metrics_summary_output['avg_risk_of_visited_patients']) else 'N/A'}, "
        f"Fatigue={metrics_summary_output['worker_self_fatigue_level_code']}"
    )
    return metrics_summary_output
