# sentinel_project_root/data_processing/aggregation.py
# Functions for aggregating data to compute KPIs and summaries for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
import re # For condition matching
from typing import Dict, Any, Optional, Union, Callable, List
from datetime import date as date_type # for type hinting

from config import settings
from .helpers import convert_to_numeric # Removed hash_dataframe_safe as st.cache_data is removed

logger = logging.getLogger(__name__)

# Removed @st.cache_data decorators. Caching is now handled at the UI/page level.

def get_trend_data(
    df: Optional[pd.DataFrame],
    value_col: str,
    date_col: str = 'encounter_date',
    period: str = 'D', # Default to Daily aggregation
    agg_func: Union[str, Callable[[Any], Any]] = 'mean',
    filter_col: Optional[str] = None,
    filter_val: Optional[Any] = None,
    source_context: str = "TrendCalculator"
) -> pd.Series:
    """
    Calculates a trend (time series) for a given value column, aggregated by period.
    Handles missing data and ensures date column is properly formatted.
    """
    logger.debug(f"({source_context}) Generating trend for '{value_col}' by '{period}' period. Agg func: {agg_func}")

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"({source_context}) Input DataFrame is empty or invalid for trend calculation of '{value_col}'.")
        return pd.Series(dtype='float64')

    df_trend = df.copy() # Work on a copy

    # Validate essential columns
    if date_col not in df_trend.columns:
        logger.error(f"({source_context}) Date column '{date_col}' not found in DataFrame for trend of '{value_col}'.")
        return pd.Series(dtype='float64')
    if value_col not in df_trend.columns:
        logger.error(f"({source_context}) Value column '{value_col}' not found in DataFrame for trend calculation.")
        return pd.Series(dtype='float64')

    # Ensure date column is datetime
    try:
        df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    except Exception as e:
        logger.error(f"({source_context}) Could not convert date_col '{date_col}' to datetime: {e}", exc_info=True)
        return pd.Series(dtype='float64')

    # Drop rows where date is NaT after conversion (value column NaNs handled later or by agg func)
    df_trend.dropna(subset=[date_col], inplace=True) # Only drop if date itself is bad

    # Apply optional filter
    if filter_col and filter_val is not None:
        if filter_col in df_trend.columns:
            df_trend = df_trend[df_trend[filter_col] == filter_val]
        else:
            logger.warning(f"({source_context}) Filter column '{filter_col}' not found. Trend calculated without this filter.")

    if df_trend.empty:
        logger.info(f"({source_context}) DataFrame became empty after date cleaning or filtering for trend of '{value_col}'.")
        return pd.Series(dtype='float64')

    try:
        # Convert value_col to numeric if agg_func is a common numeric one
        numeric_agg_functions = ['mean', 'sum', 'median', 'std', 'var', 'min', 'max']
        if isinstance(agg_func, str) and agg_func in numeric_agg_functions:
            df_trend[value_col] = convert_to_numeric(df_trend[value_col], default_value=np.nan) # Convert, default non-numeric to NaN
            df_trend.dropna(subset=[value_col], inplace=True) # Drop rows where value became NaN if agg needs numeric
            if df_trend.empty:
                 logger.info(f"({source_context}) DataFrame empty after numeric conversion of '{value_col}' for trend.")
                 return pd.Series(dtype='float64')

        # Perform resampling and aggregation
        trend_series = df_trend.set_index(date_col)[value_col].resample(period).agg(agg_func)

        # For count-based aggregations, fill NaNs with 0 as no occurrences means count is 0
        count_based_agg_functions = ['count', 'nunique', 'size']
        if isinstance(agg_func, str) and agg_func in count_based_agg_functions:
            trend_series = trend_series.fillna(0).astype(int) # Ensure integer type for counts

        logger.debug(f"({source_context}) Trend for '{value_col}' generated with {len(trend_series)} data points.")
        return trend_series
    except Exception as e:
        logger.error(f"({source_context}) Error generating trend for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype='float64')


def get_overall_kpis(
    health_df: Optional[pd.DataFrame],
    date_filter_start: Optional[Any] = None,
    date_filter_end: Optional[Any] = None,
    source_context: str = "GlobalKPIs"
) -> Dict[str, Any]:
    """
    Calculates overall key performance indicators from health data for a given period.
    """
    logger.info(f"({source_context}) Calculating overall KPIs for period: {date_filter_start} to {date_filter_end}")

    kpis: Dict[str, Any] = {
        "total_patients_period": 0, "avg_patient_ai_risk_period": np.nan,
        "malaria_rdt_positive_rate_period": np.nan, "key_supply_stockout_alerts_period": 0,
        "total_encounters_period": 0
    }
    for condition_name in settings.KEY_CONDITIONS_FOR_ACTION:
        kpi_key = f"active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
        kpis[kpi_key] = 0

    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        logger.warning(f"({source_context}) Health DataFrame is empty or invalid. Returning default KPIs.")
        return kpis

    df = health_df.copy()
    if 'encounter_date' not in df.columns:
        logger.error(f"({source_context}) 'encounter_date' column missing. KPIs may be inaccurate.")
        return kpis # Cannot proceed reliably
    
    try:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        if date_filter_start:
            start_dt = pd.to_datetime(date_filter_start, errors='coerce').normalize()
            if pd.notna(start_dt): df = df[df['encounter_date'] >= start_dt]
        if date_filter_end:
            end_dt = pd.to_datetime(date_filter_end, errors='coerce').normalize()
            if pd.notna(end_dt): df = df[df['encounter_date'] <= end_dt]
    except Exception as e_date_filter:
        logger.warning(f"({source_context}) Error applying date filters: {e_date_filter}. Proceeding with potentially unfiltered data for KPIs.", exc_info=True)


    if df.empty:
        logger.info(f"({source_context}) No data remains after date filtering. Returning default KPIs.")
        return kpis

    # Calculate KPIs
    if 'patient_id' in df.columns: kpis["total_patients_period"] = df['patient_id'].nunique()
    if 'encounter_id' in df.columns: kpis["total_encounters_period"] = df['encounter_id'].nunique()

    if 'ai_risk_score' in df.columns:
        risk_scores_series = convert_to_numeric(df['ai_risk_score'], default_value=np.nan)
        if risk_scores_series.notna().any():
            kpis["avg_patient_ai_risk_period"] = risk_scores_series.mean()

    if 'condition' in df.columns and 'patient_id' in df.columns:
        for condition_name_iter in settings.KEY_CONDITIONS_FOR_ACTION:
            kpi_key_dyn = f"active_{condition_name_iter.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
            # Case-insensitive partial match for condition
            condition_mask_iter = df['condition'].astype(str).str.contains(condition_name_iter, case=False, na=False, regex=True) # Use regex for robustness
            kpis[kpi_key_dyn] = df.loc[condition_mask_iter, 'patient_id'].nunique() if condition_mask_iter.any() else 0

    # Malaria RDT Positivity Rate
    if 'test_type' in df.columns and 'test_result' in df.columns:
        malaria_tests_df = df[df['test_type'].astype(str).str.contains("RDT-Malaria", case=False, na=False)]
        if not malaria_tests_df.empty:
            conclusive_malaria_tests = malaria_tests_df[
                ~malaria_tests_df['test_result'].astype(str).str.lower().isin(['pending', 'rejected', 'unknown', 'n/a', 'indeterminate'])
            ]
            if not conclusive_malaria_tests.empty:
                positive_malaria_tests_count = conclusive_malaria_tests[
                    conclusive_malaria_tests['test_result'].astype(str).str.lower() == 'positive'
                ].shape[0]
                kpis["malaria_rdt_positive_rate_period"] = (positive_malaria_tests_count / len(conclusive_malaria_tests)) * 100
    
    # Key Supply Stockout Alerts
    supply_related_cols = ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date', 'zone_id'] # Added zone_id for stock context
    if all(col in df.columns for col in supply_related_cols):
        latest_stock_df = df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['item', 'zone_id'], keep='last').copy() # Use .copy()
        latest_stock_df['consumption_rate_per_day'] = convert_to_numeric(latest_stock_df['consumption_rate_per_day'], default_value=0.001)
        latest_stock_df.loc[latest_stock_df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001 # Prevent DivByZero
        latest_stock_df['item_stock_agg_zone'] = convert_to_numeric(latest_stock_df['item_stock_agg_zone'], default_value=0.0)
        latest_stock_df['days_of_supply'] = latest_stock_df['item_stock_agg_zone'] / latest_stock_df['consumption_rate_per_day']
        
        key_drugs_df = latest_stock_df[
            latest_stock_df['item'].astype(str).str.contains('|'.join(re.escape(s) for s in settings.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False, regex=True)
        ]
        if not key_drugs_df.empty:
            kpis['key_supply_stockout_alerts_period'] = key_drugs_df[
                key_drugs_df['days_of_supply'] < settings.CRITICAL_SUPPLY_DAYS_REMAINING
            ]['item'].nunique() # Count unique items critically low
            
    logger.info(f"({source_context}) Overall KPIs calculated: {kpis}")
    return kpis


def get_chw_summary_kpis(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    for_date: Any,
    chw_daily_kpi_input_data: Optional[Dict[str, Any]] = None,
    source_context: str = "CHWSummaryKPIs"
) -> Dict[str, Any]:
    """Calculates CHW daily summary metrics, merging pre-calculated values if provided."""
    try:
        target_date = pd.to_datetime(for_date, errors='coerce').date()
        if pd.isna(target_date): raise ValueError("Invalid for_date")
    except Exception:
        logger.error(f"({source_context}) Invalid date '{for_date}'. Defaulting to today.")
        target_date = pd.Timestamp('now').date()
    target_date_iso = target_date.isoformat()

    logger.info(f"({source_context}) Calculating CHW summary KPIs for: {target_date_iso}")
    summary = {
        "date_of_activity": target_date_iso, "visits_count": 0, "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan, "fever_cases_identified_count": 0,
        "high_fever_cases_identified_count": 0, "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan, "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", "worker_self_fatigue_index_today": np.nan
    }
    if isinstance(chw_daily_kpi_input_data, dict): # Populate from pre-calculated data first
        for key, value in chw_daily_kpi_input_data.items():
            if key in summary:
                try: # Attempt robust type conversion
                    if isinstance(summary[key], int) and pd.notna(value): summary[key] = int(pd.to_numeric(value, errors='coerce'))
                    elif isinstance(summary[key], float) and pd.notna(value): summary[key] = float(pd.to_numeric(value, errors='coerce'))
                    elif isinstance(summary[key], str) and pd.notna(value): summary[key] = str(value)
                except (ValueError, TypeError): logger.warning(f"({source_context}) Error converting pre-calc KPI '{key}'.")

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.info(f"({source_context}) No daily encounter DataFrame for {target_date_iso}. Relying on pre-calc/defaults.")
        # Finalize fatigue level from pre-calculated index if DataFrame is empty
        if pd.notna(summary["worker_self_fatigue_index_today"]):
            fatigue_score = summary["worker_self_fatigue_index_today"]
            if fatigue_score >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: summary["worker_self_fatigue_level_code"] = "HIGH"
            elif fatigue_score >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD: summary["worker_self_fatigue_level_code"] = "MODERATE"
            else: summary["worker_self_fatigue_level_code"] = "LOW"
        return summary

    df = chw_daily_encounter_df.copy()
    if 'encounter_date' not in df.columns:
        logger.error(f"({source_context}) 'encounter_date' missing in daily data. Metrics for {target_date_iso} inaccurate."); return summary
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    df = df[df['encounter_date'].dt.date == target_date]
    if df.empty: logger.info(f"({source_context}) No encounters for {target_date_iso} after filtering."); return summary
    
    # Define and standardize essential columns for calculations
    enc_cols_config = { # Renamed from encounter_cols_config to avoid outer scope conflict
        'patient_id': {"default": f"UPID_CHWSum_{target_date_iso}", "type": str},
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
    common_na = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    for col, cfg in enc_cols_config.items():
        if col not in df.columns: df[col] = cfg["default"]
        if cfg["type"] == float: df[col] = convert_to_numeric(df[col], default_value=cfg["default"])
        elif cfg["type"] == int: df[col] = convert_to_numeric(df[col], default_value=cfg["default"], target_type=int)
        elif cfg["type"] == str:
            df[col] = df[col].astype(str).fillna(str(cfg["default"]))
            na_regex = r'^(?:' + '|'.join(re.escape(s) for s in common_na if s) + r')$'
            if any(common_na): df[col] = df[col].str.replace(na_regex, str(cfg["default"]), case=False, regex=True)
            df[col] = df[col].str.strip()

    patient_df = df[~df['encounter_type'].astype(str).str.contains("WORKER_SELF", case=False, na=False)]
    if not patient_df.empty:
        if 'patient_id' in patient_df.columns: summary["visits_count"] = patient_df['patient_id'].nunique()
        if 'ai_followup_priority_score' in patient_df.columns:
            prio_scores = patient_df['ai_followup_priority_score'] # Already numeric from prep
            summary["high_ai_prio_followups_count"] = patient_df[prio_scores >= settings.FATIGUE_INDEX_HIGH_THRESHOLD]['patient_id'].nunique()
        if 'ai_risk_score' in patient_df.columns:
            risk_scores = patient_df.drop_duplicates(subset=['patient_id'])['ai_risk_score'] # Already numeric
            if risk_scores.notna().any(): summary["avg_risk_of_visited_patients"] = risk_scores.mean()
        
        temp_col = next((c for c in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if c in patient_df.columns and patient_df[c].notna().any()), None)
        if temp_col:
            temps = patient_df[temp_col] # Already numeric
            summary["fever_cases_identified_count"] = patient_df[temps >= settings.ALERT_BODY_TEMP_FEVER_C]['patient_id'].nunique()
            summary["high_fever_cases_identified_count"] = patient_df[temps >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()
        if 'min_spo2_pct' in patient_df.columns:
            spo2_values = patient_df['min_spo2_pct'] # Already numeric
            summary["critical_spo2_cases_identified_count"] = patient_df[spo2_values < settings.ALERT_SPO2_CRITICAL_LOW_PCT]['patient_id'].nunique()
        if 'avg_daily_steps' in patient_df.columns:
            steps_values = patient_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
            if steps_values.notna().any(): summary["avg_steps_of_visited_patients"] = steps_values.mean()
        if 'fall_detected_today' in patient_df.columns:
            summary["fall_events_among_visited_count"] = patient_df[patient_df['fall_detected_today'] > 0]['patient_id'].nunique()
        if 'condition' in patient_df.columns and 'referral_status' in patient_df.columns:
            crit_referral_mask = (patient_df['referral_status'].astype(str).str.lower() == 'pending') & \
                                 (patient_df['condition'].astype(str).str.contains('|'.join(re.escape(kc) for kc in settings.KEY_CONDITIONS_FOR_ACTION), case=False, na=False, regex=True))
            summary["pending_critical_referrals_generated_today_count"] = patient_df[crit_referral_mask]['patient_id'].nunique()
    
    # Worker fatigue: use full df for the day (df), not just patient_df
    # Only calculate if not already populated from chw_daily_kpi_input_data
    if pd.isna(summary["worker_self_fatigue_index_today"]):
        worker_self_checks_df = df[df['encounter_type'].astype(str).str.contains("WORKER_SELF_CHECK", case=False, na=False)]
        if not worker_self_checks_df.empty:
            fatigue_score_col_options = ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score']
            fatigue_metric_col = next((col for col in fatigue_score_col_options if col in worker_self_checks_df.columns and worker_self_checks_df[col].notna().any()), None)
            if fatigue_metric_col:
                fatigue_score_val = worker_self_checks_df[fatigue_metric_col].max() # Max score if multiple checks
                if pd.notna(fatigue_score_val): summary["worker_self_fatigue_index_today"] = float(fatigue_score_val)

    # Set fatigue level code based on the final fatigue index (either from pre-calc or df)
    final_fatigue_score = summary["worker_self_fatigue_index_today"]
    if pd.notna(final_fatigue_score):
        if final_fatigue_score >= settings.FATIGUE_INDEX_HIGH_THRESHOLD: summary["worker_self_fatigue_level_code"] = "HIGH"
        elif final_fatigue_score >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD: summary["worker_self_fatigue_level_code"] = "MODERATE"
        else: summary["worker_self_fatigue_level_code"] = "LOW"
    else: summary["worker_self_fatigue_level_code"] = "NOT_ASSESSED" # Remains this if score is NaN

    # Final rounding for displayable float metrics
    float_metrics_round_config = {"avg_risk_of_visited_patients": 1, "avg_steps_of_visited_patients": 0, "worker_self_fatigue_index_today": 1}
    for metric_key_round, places in float_metrics_round_config.items():
        if pd.notna(summary.get(metric_key_round)):
            try: summary[metric_key_round] = round(float(summary[metric_key_round]), places)
            except (ValueError, TypeError): logger.warning(f"({source_context}) Could not round metric '{metric_key_round}'.")

    logger.info(f"({source_context}) CHW Daily Summary KPIs calculated for {target_date_iso}.")
    return summary


def get_clinic_summary_kpis(
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicSummaryKPIs"
) -> Dict[str, Any]:
    """Calculates key summary KPIs for clinic operations over a period."""
    logger.info(f"({source_context}) Calculating clinic summary KPIs.")
    summary = {"overall_avg_test_turnaround_conclusive_days": np.nan, "perc_critical_tests_tat_met": 0.0,
               "total_pending_critical_tests_patients": 0, "sample_rejection_rate_perc": 0.0,
               "key_drug_stockouts_count": 0, "test_summary_details": {}} # Initialize sub-dict
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) Health DataFrame for clinic summary is empty or invalid.")
        return summary
    df = health_df_period.copy()
    # Standardize relevant columns
    cols_std_clinic = { # Renamed to avoid conflict
        'test_type': "UnknownTest", 'test_result': "UnknownResult", 'test_turnaround_days': np.nan, 
        'sample_status': "UnknownStatus", 'patient_id': "UnknownPID", 'item': "UnknownItem",
        'item_stock_agg_zone': 0.0, 'consumption_rate_per_day': 0.001, 'encounter_date': pd.NaT,
        'zone_id': "UnknownZone" # Added for stock context if needed
    }
    common_na_clinic = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    for col, default_val_clinic in cols_std_clinic.items(): # Renamed iter var
        if col not in df.columns: df[col] = default_val_clinic
        if col == 'encounter_date': df[col] = pd.to_datetime(df[col], errors='coerce')
        elif isinstance(default_val_clinic, (float, int)) or default_val_clinic is np.nan :
            df[col] = convert_to_numeric(df[col], default_value=default_val_clinic)
        else: # String columns
            df[col] = df[col].astype(str).fillna(str(default_val_clinic))
            na_regex_clinic = r'^(?:' + '|'.join(re.escape(s) for s in common_na_clinic if s) + r')$'
            if any(common_na_clinic): df[col] = df[col].str.replace(na_regex_clinic, str(default_val_clinic), case=False, regex=True)
            df[col] = df[col].str.strip()

    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    conclusive_tests_df = df[~df['test_result'].astype(str).str.lower().isin(['pending', 'rejected', 'unknownresult', 'indeterminate', 'n/a'])]
    if not conclusive_tests_df.empty and 'test_turnaround_days' in conclusive_tests_df.columns and \
       conclusive_tests_df['test_turnaround_days'].notna().any():
        summary["overall_avg_test_turnaround_conclusive_days"] = conclusive_tests_df['test_turnaround_days'].mean()

    # 2. Percentage of CRITICAL Tests Meeting TAT Target & Total Pending Critical Tests
    critical_test_keys = settings.CRITICAL_TESTS
    df_critical_tests = df[df['test_type'].isin(critical_test_keys)]
    if not df_critical_tests.empty:
        df_critical_conclusive = df_critical_tests[~df_critical_tests['test_result'].astype(str).str.lower().isin(['pending', 'rejected', 'unknownresult', 'indeterminate'])]
        if not df_critical_conclusive.empty and 'test_turnaround_days' in df_critical_conclusive.columns:
            met_tat_count = 0
            for _, row_crit in df_critical_conclusive.iterrows():
                test_config = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_crit['test_type'], {})
                target_tat = test_config.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS)
                if pd.notna(row_crit['test_turnaround_days']) and row_crit['test_turnaround_days'] <= target_tat:
                    met_tat_count += 1
            summary["perc_critical_tests_tat_met"] = (met_tat_count / len(df_critical_conclusive)) * 100 if len(df_critical_conclusive) > 0 else 0.0
        
        if 'patient_id' in df_critical_tests.columns:
            summary["total_pending_critical_tests_patients"] = df_critical_tests[
                df_critical_tests['test_result'].astype(str).str.lower() == 'pending'
            ]['patient_id'].nunique()

    # 3. Sample Rejection Rate (%)
    unique_test_event_col_clinic = 'encounter_id' if 'encounter_id' in df.columns else 'patient_id'
    df_with_sample_status = df[~df['sample_status'].astype(str).str.lower().isin(['unknownstatus', 'n/a', ''])]
    if not df_with_sample_status.empty and unique_test_event_col_clinic in df_with_sample_status.columns:
        total_samples_processed_unique = df_with_sample_status[unique_test_event_col_clinic].nunique()
        rejected_samples_unique = df_with_sample_status[
            df_with_sample_status['sample_status'].astype(str).str.lower() == 'rejected'
        ][unique_test_event_col_clinic].nunique()
        if total_samples_processed_unique > 0:
            summary["sample_rejection_rate_perc"] = (rejected_samples_unique / total_samples_processed_unique) * 100
    
    # 4. Key Drug Stockouts Count
    if 'encounter_date' in df.columns and df['encounter_date'].notna().any() and 'zone_id' in df.columns: # zone_id for context
        latest_stock_df_clinic = df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['item', 'zone_id'], keep='last').copy()
        latest_stock_df_clinic.loc[latest_stock_df_clinic['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001
        latest_stock_df_clinic['days_of_supply'] = latest_stock_df_clinic['item_stock_agg_zone'] / latest_stock_df_clinic['consumption_rate_per_day']
        
        key_drugs_stock_df_clinic = latest_stock_df_clinic[
            latest_stock_df_clinic['item'].astype(str).str.contains('|'.join(re.escape(s_drug) for s_drug in settings.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False, regex=True)
        ]
        if not key_drugs_stock_df_clinic.empty:
            summary["key_drug_stockouts_count"] = key_drugs_stock_df_clinic[
                key_drugs_stock_df_clinic['days_of_supply'] < settings.CRITICAL_SUPPLY_DAYS_REMAINING
            ]['item'].nunique()

    # 5. Detailed breakdown per test type
    test_details_map_clinic = {} # Renamed to avoid outer scope conflict
    for test_key_orig_clinic, test_config_props_clinic in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        test_display_name_clinic = test_config_props_clinic.get("display_name", test_key_orig_clinic)
        df_specific_test_clinic = df[df['test_type'] == test_key_orig_clinic]
        
        current_test_details: Dict[str, Any] = {
            "positive_rate_perc": 0.0, "avg_tat_days": np.nan, "perc_met_tat_target": 0.0,
            "pending_count_patients": 0, "rejected_count_patients": 0, "total_conclusive_tests": 0
        }
        if not df_specific_test_clinic.empty:
            df_st_conclusive_clinic = df_specific_test_clinic[~df_specific_test_clinic['test_result'].astype(str).str.lower().isin(['pending', 'rejected', 'unknownresult', 'indeterminate'])]
            current_test_details["total_conclusive_tests"] = len(df_st_conclusive_clinic)

            if not df_st_conclusive_clinic.empty:
                pos_count_clinic = df_st_conclusive_clinic[df_st_conclusive_clinic['test_result'].astype(str).str.lower() == 'positive'].shape[0]
                current_test_details["positive_rate_perc"] = (pos_count_clinic / len(df_st_conclusive_clinic)) * 100
                if 'test_turnaround_days' in df_st_conclusive_clinic.columns and df_st_conclusive_clinic['test_turnaround_days'].notna().any():
                    current_test_details["avg_tat_days"] = df_st_conclusive_clinic['test_turnaround_days'].mean()
                
                met_tat_specific_count_clinic = 0
                target_tat_specific_clinic = test_config_props_clinic.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS)
                for _, row_st_clinic in df_st_conclusive_clinic.iterrows():
                    if pd.notna(row_st_clinic.get('test_turnaround_days')) and row_st_clinic['test_turnaround_days'] <= target_tat_specific_clinic:
                        met_tat_specific_count_clinic +=1
                current_test_details["perc_met_tat_target"] = (met_tat_specific_count_clinic / len(df_st_conclusive_clinic)) * 100 if len(df_st_conclusive_clinic) > 0 else 0.0
            
            if 'patient_id' in df_specific_test_clinic.columns:
                 current_test_details["pending_count_patients"] = df_specific_test_clinic[df_specific_test_clinic['test_result'].astype(str).str.lower() == 'pending']['patient_id'].nunique()
                 current_test_details["rejected_count_patients"] = df_specific_test_clinic[df_specific_test_clinic['sample_status'].astype(str).str.lower() == 'rejected']['patient_id'].nunique()
        test_details_map_clinic[test_display_name_clinic] = current_test_details
    summary["test_summary_details"] = test_details_map_clinic
    
    logger.info(f"({source_context}) Clinic Summary KPIs calculated: {list(summary.keys())[:5]}...") # Log snippet
    return summary


def get_clinic_environmental_summary_kpis(
    iot_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicEnvSummaryKPIs"
) -> Dict[str, Any]:
    """Calculates summary KPIs for clinic environmental data."""
    logger.info(f"({source_context}) Calculating clinic environmental summary KPIs.")
    summary = {
        "avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count": 0, "rooms_co2_high_alert_latest_count":0, # Added high count
        "avg_pm25_overall_ugm3": np.nan, "rooms_pm25_very_high_alert_latest_count": 0, "rooms_pm25_high_alert_latest_count":0, # Added high count
        "avg_waiting_room_occupancy_overall_persons": np.nan, "waiting_room_high_occupancy_alert_latest_flag": False,
        "avg_noise_overall_dba": np.nan, "rooms_noise_high_alert_latest_count": 0,
        "avg_temp_overall_celsius": np.nan, "avg_humidity_overall_rh": np.nan,
        "latest_readings_timestamp": None
    }
    if not isinstance(iot_df_period, pd.DataFrame) or iot_df_period.empty:
        logger.warning(f"({source_context}) IoT DataFrame for environmental summary is empty or invalid.")
        return summary
    df = iot_df_period.copy()
    if 'timestamp' not in df.columns:
        logger.warning(f"({source_context}) 'timestamp' column missing in IoT data. KPIs may be unreliable.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: logger.info(f"({source_context}) No IoT data with valid timestamps."); return summary
        summary["latest_readings_timestamp"] = df['timestamp'].max().isoformat() if pd.notna(df['timestamp'].max()) else None

    env_num_cols = ['avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy']
    for col in env_num_cols:
        if col in df.columns: df[col] = convert_to_numeric(df[col], default_value=np.nan)
        else: df[col] = np.nan # Ensure column exists

    if df['avg_co2_ppm'].notna().any(): summary["avg_co2_overall_ppm"] = df['avg_co2_ppm'].mean()
    if df['avg_pm25'].notna().any(): summary["avg_pm25_overall_ugm3"] = df['avg_pm25'].mean()
    if df['avg_noise_db'].notna().any(): summary["avg_noise_overall_dba"] = df['avg_noise_db'].mean()
    if df['avg_temp_celsius'].notna().any(): summary["avg_temp_overall_celsius"] = df['avg_temp_celsius'].mean()
    if df['avg_humidity_rh'].notna().any(): summary["avg_humidity_overall_rh"] = df['avg_humidity_rh'].mean()

    if 'clinic_id' in df.columns and 'room_name' in df.columns and 'timestamp' in df.columns:
        latest_readings_room = df.sort_values('timestamp', na_position='first').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        if not latest_readings_room.empty:
            if latest_readings_room['avg_co2_ppm'].notna().any():
                summary["rooms_co2_very_high_alert_latest_count"] = latest_readings_room[latest_readings_room['avg_co2_ppm'] > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM].shape[0]
                summary["rooms_co2_high_alert_latest_count"] = latest_readings_room[latest_readings_room['avg_co2_ppm'] > settings.ALERT_AMBIENT_CO2_HIGH_PPM].shape[0]
            if latest_readings_room['avg_pm25'].notna().any():
                summary["rooms_pm25_very_high_alert_latest_count"] = latest_readings_room[latest_readings_room['avg_pm25'] > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3].shape[0]
                summary["rooms_pm25_high_alert_latest_count"] = latest_readings_room[latest_readings_room['avg_pm25'] > settings.ALERT_AMBIENT_PM25_HIGH_UGM3].shape[0]
            if latest_readings_room['avg_noise_db'].notna().any():
                summary["rooms_noise_high_alert_latest_count"] = latest_readings_room[latest_readings_room['avg_noise_db'] > settings.ALERT_AMBIENT_NOISE_HIGH_DBA].shape[0]
            
            waiting_rooms_latest = latest_readings_room[latest_readings_room.get('room_name', pd.Series(dtype=str)).astype(str).str.contains("Waiting", case=False, na=False)]
            if not waiting_rooms_latest.empty and 'waiting_room_occupancy' in waiting_rooms_latest.columns and \
               waiting_rooms_latest['waiting_room_occupancy'].notna().any():
                summary["avg_waiting_room_occupancy_overall_persons"] = waiting_rooms_latest['waiting_room_occupancy'].mean()
                summary["waiting_room_high_occupancy_alert_latest_flag"] = (waiting_rooms_latest['waiting_room_occupancy'] > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX).any()
    logger.info(f"({source_context}) Clinic Environmental Summary KPIs calculated.")
    return summary


def get_district_summary_kpis(
    enriched_zone_df: Optional[pd.DataFrame],
    source_context: str = "DistrictKPIs"
) -> Dict[str, Any]:
    """Calculates district-wide summary KPIs from enriched zone data."""
    logger.info(f"({source_context}) Calculating district summary KPIs.")
    kpis: Dict[str, Any] = {
        "total_zones_in_df": 0, "total_population_district": 0.0,
        "population_weighted_avg_ai_risk_score": np.nan, "zones_meeting_high_risk_criteria_count": 0,
        "district_avg_facility_coverage_score": np.nan, "district_overall_key_disease_prevalence_per_1000": np.nan,
        "district_population_weighted_avg_steps": np.nan, "district_avg_clinic_co2_ppm": np.nan
    }
    for cond_name in settings.KEY_CONDITIONS_FOR_ACTION:
        kpi_key = f"district_total_active_{cond_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        kpis[kpi_key] = 0

    if not isinstance(enriched_zone_df, pd.DataFrame) or enriched_zone_df.empty:
        logger.warning(f"({source_context}) Enriched zone DataFrame is empty or invalid.")
        return kpis
    df = enriched_zone_df.copy()

    if 'zone_id' in df.columns: kpis["total_zones_in_df"] = df['zone_id'].nunique()
    else: kpis["total_zones_in_df"] = len(df); logger.warning(f"({source_context}) 'zone_id' missing, using df length.")

    total_district_population = 0.0
    if 'population' in df.columns:
        df['population'] = convert_to_numeric(df['population'], default_value=0.0)
        total_district_population = df['population'].sum()
        kpis["total_population_district"] = total_district_population
    else: logger.warning(f"({source_context}) 'population' column missing. Population-weighted KPIs will be affected.")

    # Helper for weighted average calculation
    def weighted_avg(df_calc: pd.DataFrame, value_col: str, weight_col: str, total_weight: float) -> float:
        if value_col not in df_calc.columns or weight_col not in df_calc.columns or total_weight <= 0:
            return convert_to_numeric(df_calc.get(value_col, pd.Series(dtype=float)), np.nan).mean() if value_col in df_calc.columns else np.nan
        
        df_calc[value_col] = convert_to_numeric(df_calc[value_col], default_value=np.nan)
        # fillna(0) for value if weight exists, or only use rows where both value and weight are valid
        weighted_sum = (df_calc[value_col].fillna(0) * df_calc[weight_col]).sum() # Fill value with 0 if weight exists
        return weighted_sum / total_weight

    kpis["population_weighted_avg_ai_risk_score"] = weighted_avg(df, 'avg_risk_score', 'population', total_district_population)
    if 'avg_risk_score' in df.columns:
        kpis["zones_meeting_high_risk_criteria_count"] = df[convert_to_numeric(df['avg_risk_score'], np.nan) >= settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE].shape[0]
    kpis["district_avg_facility_coverage_score"] = weighted_avg(df, 'facility_coverage_score', 'population', total_district_population)

    total_key_infections = 0
    for cond_name in settings.KEY_CONDITIONS_FOR_ACTION:
        cond_col = f"active_{cond_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        kpi_key_dist = f"district_total_active_{cond_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        if cond_col in df.columns:
            df[cond_col] = convert_to_numeric(df[cond_col], default_value=0)
            total_cases = df[cond_col].sum()
            kpis[kpi_key_dist] = total_cases
            total_key_infections += total_cases
        else: kpis[kpi_key_dist] = 0
    
    if total_district_population > 0:
        kpis["district_overall_key_disease_prevalence_per_1000"] = (total_key_infections / total_district_population) * 1000
    
    # For avg_daily_steps_zone, fillna with a reasonable default like 60% of target before weighting, or calculate weighted avg only on non-NaNs.
    # Here, just pass to weighted_avg which will use mean if total_weight is 0 or NaN handling is done by convert_to_numeric.
    if 'avg_daily_steps_zone' in df.columns:
         kpis["district_population_weighted_avg_steps"] = weighted_avg(df, 'avg_daily_steps_zone', 'population', total_district_population)
    
    if 'zone_avg_co2' in df.columns: # This column from enrichment, simple mean of zonal means
        kpis["district_avg_clinic_co2_ppm"] = convert_to_numeric(df['zone_avg_co2'], np.nan).mean()

    logger.info(f"({source_context}) District Summary KPIs calculated.")
    return kpis
