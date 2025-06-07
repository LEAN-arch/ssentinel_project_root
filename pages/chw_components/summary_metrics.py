# sentinel_project_root/pages/chw_components/summary_metrics.py
# Calculates key summary metrics for a CHW's daily activity for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from datetime import date as date_type, datetime

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric, standardize_missing_values
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in summary_metrics.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)


def calculate_chw_daily_summary_metrics(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    source_context: str = "CHWDailySummary"
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.

    This enhanced function provides a holistic view of CHW performance, including patient
    interaction metrics, risk assessment summaries, and crucial operational indicators
    like CHW fatigue and device status.

    Returns:
        A dictionary containing key metrics such as:
        - `visits_count`: Number of unique patients visited.
        - `high_ai_prio_followups_count`: Patients with high AI follow-up scores.
        - `avg_risk_of_visited_patients`: Average AI risk score of visited patients.
        - `critical_spo2_cases_identified_count`: Patients with critically low SpO2.
        - `high_fever_cases_identified_count`: Patients with high fever.
        - `worker_self_fatigue_index_today`: Fatigue score from CHW self-assessment.
        - `chw_steps_today`: Step count from the CHW's device.
        - `device_battery_pct`: Last reported battery level of the CHW's device.
        - `avg_data_sync_latency_hours`: Time since last data sync.
    """
    try:
        target_date = pd.to_datetime(for_date).date()
    except (ValueError, TypeError):
        logger.warning(f"({source_context}) Invalid 'for_date' ('{for_date}'). Defaulting to current system date.")
        target_date = pd.Timestamp('now').date()

    target_date_iso = target_date.isoformat()
    logger.info(f"({source_context}) Calculating CHW daily summary for date: {target_date_iso}")

    metrics_summary: Dict[str, Any] = {
        "date_of_activity": target_date_iso,
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "critical_spo2_cases_identified_count": 0,
        "high_fever_cases_identified_count": 0,
        "worker_self_fatigue_index_today": np.nan,
        "chw_steps_today": 0,
        "device_battery_pct": np.nan,
        "avg_data_sync_latency_hours": np.nan,
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.info(f"({source_context}) No daily encounter data for {target_date_iso}. Returning default metrics.")
        return metrics_summary

    # --- Data Preparation ---
    numeric_defaults = {
        'ai_followup_priority_score': np.nan, 'ai_risk_score': np.nan, 'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'stress_level_score': np.nan, 'rapid_psychometric_distress_score': np.nan,
        'chw_daily_steps': np.nan, 'device_battery_level_pct': np.nan, 'data_sync_latency_hours': np.nan
    }
    string_defaults = {
        'patient_id': "UnknownPID", 'encounter_type': "UnknownEncounter",
        'chw_id': "UnknownCHW",
    }
    df = standardize_missing_values(chw_daily_encounter_df, string_defaults, numeric_defaults)

    # Separate CHW self-checks from patient encounters
    self_check_mask = df['encounter_type'].str.contains("WORKER_SELF_CHECK", case=False, na=False)
    worker_self_checks_df = df[self_check_mask]
    patient_records_df = df[~self_check_mask]

    # --- Calculations for Patient-Facing Metrics ---
    if not patient_records_df.empty:
        metrics_summary["visits_count"] = patient_records_df['patient_id'].nunique()

        prio_high_thresh = getattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD', 80)
        metrics_summary["high_ai_prio_followups_count"] = patient_records_df[
            patient_records_df['ai_followup_priority_score'] >= prio_high_thresh
        ]['patient_id'].nunique()

        unique_patient_risks = patient_records_df.drop_duplicates(subset=['patient_id'])['ai_risk_score'].dropna()
        if not unique_patient_risks.empty:
            metrics_summary["avg_risk_of_visited_patients"] = unique_patient_risks.mean()

        spo2_critical_thresh = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90)
        metrics_summary["critical_spo2_cases_identified_count"] = patient_records_df[
            patient_records_df['min_spo2_pct'] < spo2_critical_thresh
        ]['patient_id'].nunique()

        # Choose the best available temperature source
        temp_col = 'vital_signs_temperature_celsius'
        if df[temp_col].isnull().all() and 'max_skin_temp_celsius' in df.columns:
            temp_col = 'max_skin_temp_celsius'
        
        high_fever_thresh = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
        metrics_summary["high_fever_cases_identified_count"] = patient_records_df[
            patient_records_df[temp_col] >= high_fever_thresh
        ]['patient_id'].nunique()

    # --- Calculations for CHW-Specific Metrics ---
    if not worker_self_checks_df.empty:
        # Use the first available fatigue metric from the self-check record
        fatigue_cols = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
        for col in fatigue_cols:
            if col in worker_self_checks_df.columns and worker_self_checks_df[col].notna().any():
                metrics_summary["worker_self_fatigue_index_today"] = worker_self_checks_df[col].max()
                break # Stop after finding the first valid fatigue metric

        if 'chw_daily_steps' in worker_self_checks_df.columns and worker_self_checks_df['chw_daily_steps'].notna().any():
            metrics_summary["chw_steps_today"] = worker_self_checks_df['chw_daily_steps'].max()
        
        if 'device_battery_level_pct' in worker_self_checks_df.columns and worker_self_checks_df['device_battery_level_pct'].notna().any():
            metrics_summary["device_battery_pct"] = worker_self_checks_df['device_battery_level_pct'].iloc[-1] # Last reported value

        if 'data_sync_latency_hours' in worker_self_checks_df.columns and worker_self_checks_df['data_sync_latency_hours'].notna().any():
            metrics_summary["avg_data_sync_latency_hours"] = worker_self_checks_df['data_sync_latency_hours'].mean()

    # --- Final Formatting ---
    float_metrics_to_round = {
        "avg_risk_of_visited_patients": 1,
        "worker_self_fatigue_index_today": 1,
        "avg_data_sync_latency_hours": 1,
        "device_battery_pct": 0,
    }
    for metric, places in float_metrics_to_round.items():
        if pd.notna(metrics_summary.get(metric)):
            metrics_summary[metric] = round(float(metrics_summary[metric]), places)

    logger.info(f"({source_context}) Daily summary metrics calculated successfully for {target_date_iso}.")
    return metrics_summary
