# sentinel_project_root/data_processing/aggregation.py
# Functions for aggregating data to compute KPIs and summaries for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Callable, List

from config import settings # Use the new settings module
from .helpers import convert_to_numeric # Local import from helpers

logger = logging.getLogger(__name__)

# --- Common Aggregation Utilities ---

def get_trend_data(
    df: Optional[pd.DataFrame],
    value_col: str,
    date_col: str = 'encounter_date', # Default date column
    period: str = 'D', # Default to Daily aggregation
    agg_func: Union[str, Callable[[Any], Any]] = 'mean', # Default aggregation function
    filter_col: Optional[str] = None, # Optional column to filter on before aggregation
    filter_val: Optional[Any] = None,   # Optional value for the filter_col
    source_context: str = "TrendCalculator"
) -> pd.Series:
    """
    Calculates a trend (time series) for a given value column, aggregated by period.
    Handles missing data and ensures date column is properly formatted.
    """
    logger.debug(f"({source_context}) Generating trend for '{value_col}' by '{period}' period. Agg func: {agg_func}")

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"({source_context}) Input DataFrame is empty or invalid for trend calculation of '{value_col}'. Returning empty Series.")
        return pd.Series(dtype='float64')

    df_trend = df.copy()

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
        logger.error(f"({source_context}) Could not convert date_col '{date_col}' to datetime: {e}")
        return pd.Series(dtype='float64')

    # Drop rows where date or value is NaT/NaN after conversion
    df_trend.dropna(subset=[date_col, value_col], inplace=True)

    if filter_col and filter_val is not None:
        if filter_col in df_trend.columns:
            df_trend = df_trend[df_trend[filter_col] == filter_val]
        else:
            logger.warning(f"({source_context}) Filter column '{filter_col}' not found. Trend calculated without this filter.")

    if df_trend.empty:
        logger.info(f"({source_context}) DataFrame became empty after date/value cleaning or filtering for trend of '{value_col}'.")
        return pd.Series(dtype='float64')

    try:
        # Convert value_col to numeric if agg_func is a common numeric one
        if isinstance(agg_func, str) and agg_func in ['mean', 'sum', 'median', 'std', 'var', 'min', 'max']:
            df_trend[value_col] = convert_to_numeric(df_trend[value_col], default_value=np.nan)
            df_trend.dropna(subset=[value_col], inplace=True) # Drop if value became NaN and agg needs numeric
            if df_trend.empty and agg_func not in ['count', 'nunique']: # count/nunique can work on non-numeric if that's intended
                 logger.info(f"({source_context}) DataFrame empty after numeric conversion of '{value_col}' for trend.")
                 return pd.Series(dtype='float64')


        trend_series = df_trend.set_index(date_col)[value_col].resample(period).agg(agg_func)

        # For count-based aggregations, fill NaNs with 0 as no occurrences means count is 0
        if isinstance(agg_func, str) and agg_func in ['count', 'nunique', 'size']:
            trend_series = trend_series.fillna(0).astype(int) # Ensure integer type for counts

        logger.debug(f"({source_context}) Trend for '{value_col}' generated with {len(trend_series)} data points.")
        return trend_series
    except Exception as e:
        logger.error(f"({source_context}) Error generating trend for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype='float64')


# --- KPI and Summary Functions ---

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
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

    # Initialize KPIs with default values
    kpis: Dict[str, Any] = {
        "total_patients_period": 0,
        "avg_patient_ai_risk_period": np.nan,
        "malaria_rdt_positive_rate_period": np.nan,
        "key_supply_stockout_alerts_period": 0,
        "total_encounters_period": 0
    }
    # Dynamically add keys for each condition defined in settings
    for condition_name in settings.KEY_CONDITIONS_FOR_ACTION:
        kpi_key = f"active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
        kpis[kpi_key] = 0

    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        logger.warning(f"({source_context}) Health DataFrame is empty or invalid. Returning default KPIs.")
        return kpis

    df = health_df.copy()

    # Ensure 'encounter_date' is datetime and filter by date range
    if 'encounter_date' in df.columns:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        if date_filter_start:
            try:
                start_dt = pd.to_datetime(date_filter_start).normalize()
                df = df[df['encounter_date'] >= start_dt]
            except Exception as e:
                logger.warning(f"({source_context}) Invalid start_date_filter '{date_filter_start}': {e}")
        if date_filter_end:
            try:
                end_dt = pd.to_datetime(date_filter_end).normalize()
                df = df[df['encounter_date'] <= end_dt]
            except Exception as e:
                logger.warning(f"({source_context}) Invalid end_date_filter '{date_filter_end}': {e}")
    else:
        logger.warning(f"({source_context}) 'encounter_date' column missing. KPIs may be inaccurate.")
        return kpis # Cannot proceed without encounter_date for period filtering

    if df.empty:
        logger.info(f"({source_context}) No data remains after date filtering. Returning default KPIs.")
        return kpis

    # Calculate KPIs
    if 'patient_id' in df.columns:
        kpis["total_patients_period"] = df['patient_id'].nunique()
    if 'encounter_id' in df.columns:
        kpis["total_encounters_period"] = df['encounter_id'].nunique()

    if 'ai_risk_score' in df.columns:
        risk_scores = convert_to_numeric(df['ai_risk_score'])
        if risk_scores.notna().any():
            kpis["avg_patient_ai_risk_period"] = risk_scores.mean()

    if 'condition' in df.columns:
        for condition_name in settings.KEY_CONDITIONS_FOR_ACTION:
            kpi_key = f"active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
            # Case-insensitive matching for condition
            condition_mask = df['condition'].astype(str).str.contains(condition_name, case=False, na=False)
            if 'patient_id' in df.columns:
                kpis[kpi_key] = df.loc[condition_mask, 'patient_id'].nunique() if condition_mask.any() else 0

    # Malaria RDT Positivity Rate
    if 'test_type' in df.columns and 'test_result' in df.columns:
        malaria_tests_df = df[df['test_type'].astype(str).str.contains("RDT-Malaria", case=False, na=False)]
        if not malaria_tests_df.empty:
            conclusive_malaria_tests = malaria_tests_df[
                ~malaria_tests_df['test_result'].astype(str).str.lower().isin(['pending', 'rejected', 'unknown', 'n/a', 'indeterminate'])
            ]
            if not conclusive_malaria_tests.empty:
                positive_malaria_tests = conclusive_malaria_tests[
                    conclusive_malaria_tests['test_result'].astype(str).str.lower() == 'positive'
                ].shape[0]
                kpis["malaria_rdt_positive_rate_period"] = (positive_malaria_tests / len(conclusive_malaria_tests)) * 100
            else:
                logger.debug(f"({source_context}) No conclusive Malaria RDT results in period.")
        else:
            logger.debug(f"({source_context}) No Malaria RDT tests found in period.")

    # Key Supply Stockout Alerts
    if all(col in df.columns for col in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']):
        # Get the latest stock status for each item within the period
        latest_stock_df = df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
        
        latest_stock_df['consumption_rate_per_day'] = convert_to_numeric(latest_stock_df['consumption_rate_per_day'], default_value=0.001)
        # Ensure consumption rate is not zero to avoid division by zero
        latest_stock_df.loc[latest_stock_df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001
        
        latest_stock_df['item_stock_agg_zone'] = convert_to_numeric(latest_stock_df['item_stock_agg_zone'], default_value=0.0)
        latest_stock_df['days_of_supply'] = latest_stock_df['item_stock_agg_zone'] / latest_stock_df['consumption_rate_per_day']

        key_drugs_df = latest_stock_df[
            latest_stock_df['item'].astype(str).str.contains('|'.join(settings.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)
        ]
        if not key_drugs_df.empty:
            kpis['key_supply_stockout_alerts_period'] = key_drugs_df[
                key_drugs_df['days_of_supply'] < settings.CRITICAL_SUPPLY_DAYS_REMAINING
            ]['item'].nunique() # Count unique items that are critically low
        else:
            logger.debug(f"({source_context}) No key drugs found in latest stock data.")
            
    logger.info(f"({source_context}) Overall KPIs calculated: {kpis}")
    return kpis


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_chw_summary_kpis( # Renamed for clarity
    health_df_daily: Optional[pd.DataFrame],
    for_date: Any, # Expects a date or date-like string
    source_context: str = "CHWSummaryKPIs"
) -> Dict[str, Any]:
    """
    Calculates CHW daily summary metrics.
    `for_date` is primarily for context and ensuring data is for that day.
    """
    try:
        target_date = pd.to_datetime(for_date).date()
    except Exception:
        logger.error(f"({source_context}) Invalid date '{for_date}' provided for CHW summary. Defaulting to today.")
        target_date = pd.Timestamp('now').date()

    logger.info(f"({source_context}) Calculating CHW summary KPIs for date: {target_date}")

    summary = {
        "date_of_activity": target_date.isoformat(),
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "fever_cases_identified_count": 0,
        "high_fever_cases_identified_count": 0,
        "critical_spo2_cases_identified_count": 0,
        "avg_steps_of_visited_patients": np.nan,
        "fall_events_among_visited_count": 0,
        "pending_critical_referrals_generated_today_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED", # LOW, MODERATE, HIGH
        "worker_self_fatigue_index_today": np.nan
    }

    if not isinstance(health_df_daily, pd.DataFrame) or health_df_daily.empty:
        logger.warning(f"({source_context}) Health DataFrame for CHW daily summary is empty or invalid.")
        return summary

    df = health_df_daily.copy()
    
    # Ensure 'encounter_date' is datetime and filter for the target_date
    if 'encounter_date' in df.columns:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        df = df[df['encounter_date'].dt.date == target_date]
    else:
        logger.warning(f"({source_context}) 'encounter_date' column missing. CHW KPIs may be inaccurate.")
        return summary # Cannot proceed without encounter_date for daily filtering

    if df.empty:
        logger.info(f"({source_context}) No CHW data for date {target_date}. Returning default summary.")
        return summary

    # Exclude worker self-checks for patient-related metrics
    patient_records_df = df[~df.get('encounter_type', pd.Series(dtype=str)).astype(str).str.contains("WORKER_SELF", case=False, na=False)]

    if not patient_records_df.empty:
        if 'patient_id' in patient_records_df.columns:
            summary["visits_count"] = patient_records_df['patient_id'].nunique()

        if 'ai_followup_priority_score' in patient_records_df.columns:
            prio_scores = convert_to_numeric(patient_records_df['ai_followup_priority_score'])
            summary["high_ai_prio_followups_count"] = patient_records_df[
                prio_scores >= settings.FATIGUE_INDEX_HIGH_THRESHOLD # Using this as a general high priority threshold
            ]['patient_id'].nunique()

        if 'ai_risk_score' in patient_records_df.columns:
            risk_scores = convert_to_numeric(patient_records_df.drop_duplicates(subset=['patient_id'])['ai_risk_score'])
            if risk_scores.notna().any():
                summary["avg_risk_of_visited_patients"] = risk_scores.mean()

        temp_col_name = next((col for col in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if col in patient_records_df.columns and patient_records_df[col].notna().any()), None)
        if temp_col_name:
            temps = convert_to_numeric(patient_records_df[temp_col_name])
            summary["fever_cases_identified_count"] = patient_records_df[temps >= settings.ALERT_BODY_TEMP_FEVER_C]['patient_id'].nunique()
            summary["high_fever_cases_identified_count"] = patient_records_df[temps >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()

        if 'min_spo2_pct' in patient_records_df.columns:
            spo2_values = convert_to_numeric(patient_records_df['min_spo2_pct'])
            summary["critical_spo2_cases_identified_count"] = patient_records_df[
                spo2_values < settings.ALERT_SPO2_CRITICAL_LOW_PCT
            ]['patient_id'].nunique()

        if 'avg_daily_steps' in patient_records_df.columns:
            steps_values = convert_to_numeric(patient_records_df.drop_duplicates(subset=['patient_id'])['avg_daily_steps'])
            if steps_values.notna().any():
                summary["avg_steps_of_visited_patients"] = steps_values.mean()
        
        if 'fall_detected_today' in patient_records_df.columns:
            fall_values = convert_to_numeric(patient_records_df['fall_detected_today'], default_value=0)
            summary["fall_events_among_visited_count"] = patient_records_df[fall_values > 0]['patient_id'].nunique()

        if 'condition' in patient_records_df.columns and 'referral_status' in patient_records_df.columns:
            critical_referral_mask = (
                patient_records_df['referral_status'].astype(str).str.lower() == 'pending'
            ) & (
                patient_records_df['condition'].astype(str).str.contains('|'.join(settings.KEY_CONDITIONS_FOR_ACTION), case=False, na=False)
            )
            summary["pending_critical_referrals_generated_today_count"] = patient_records_df[critical_referral_mask]['patient_id'].nunique()
    else:
        logger.info(f"({source_context}) No patient-specific encounters found for CHW on {target_date}.")


    # Worker fatigue (from any encounter_type on that day for that CHW, if data is per CHW)
    worker_self_checks_df = df[df.get('encounter_type', pd.Series(dtype=str)).astype(str).str.contains("WORKER_SELF_CHECK", case=False, na=False)]
    if not worker_self_checks_df.empty:
        fatigue_score_col = next((col for col in ['ai_followup_priority_score', 'rapid_psychometric_distress_score', 'stress_level_score'] if col in worker_self_checks_df.columns and worker_self_checks_df[col].notna().any()), None)
        if fatigue_score_col:
            fatigue_val = convert_to_numeric(worker_self_checks_df[fatigue_score_col]).max() # Max fatigue score if multiple self-checks
            if pd.notna(fatigue_val):
                summary["worker_self_fatigue_index_today"] = fatigue_val
                if fatigue_val >= settings.FATIGUE_INDEX_HIGH_THRESHOLD:
                    summary["worker_self_fatigue_level_code"] = "HIGH"
                elif fatigue_val >= settings.FATIGUE_INDEX_MODERATE_THRESHOLD:
                    summary["worker_self_fatigue_level_code"] = "MODERATE"
                else:
                    summary["worker_self_fatigue_level_code"] = "LOW"
        else:
             logger.debug(f"({source_context}) No fatigue score column found in worker self-checks.")
    else:
        logger.debug(f"({source_context}) No worker self-check records found for fatigue assessment on {target_date}.")

    logger.info(f"({source_context}) CHW Daily Summary KPIs calculated for {target_date}: {summary}")
    return summary


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_summary_kpis( # Renamed for clarity
    health_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicSummaryKPIs"
) -> Dict[str, Any]:
    """
    Calculates key summary KPIs for clinic operations over a period.
    """
    logger.info(f"({source_context}) Calculating clinic summary KPIs.")
    summary = {
        "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests_patients": 0,
        "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0,
        "test_summary_details": {} # Detailed breakdown per test type
    }

    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.warning(f"({source_context}) Health DataFrame for clinic summary is empty or invalid.")
        return summary

    df = health_df_period.copy()

    # Standardize relevant columns (ensure they exist and have consistent missing value representation)
    cols_to_standardize = {
        'test_type': "UnknownTest", 'test_result': "UnknownResult",
        'test_turnaround_days': np.nan, 'sample_status': "UnknownStatus",
        'patient_id': "UnknownPID", 'item': "UnknownItem",
        'item_stock_agg_zone': 0.0, 'consumption_rate_per_day': 0.001,
        'encounter_date': pd.NaT # Needed for latest stock
    }
    for col, default_val in cols_to_standardize.items():
        if col not in df.columns:
            df[col] = default_val
        if col == 'encounter_date':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif isinstance(default_val, (float, int)) or default_val is np.nan :
            df[col] = convert_to_numeric(df[col], default_value=default_val)
        else: # String columns
            df[col] = df[col].astype(str).fillna(str(default_val))
            df[col] = df[col].replace(['nan', 'None', 'N/A', '', ' '], str(default_val), regex=False)


    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    conclusive_tests_df = df[
        ~df['test_result'].str.lower().isin(['pending', 'rejected', 'unknownresult', 'indeterminate', 'n/a'])
    ]
    if not conclusive_tests_df.empty and conclusive_tests_df['test_turnaround_days'].notna().any():
        summary["overall_avg_test_turnaround_conclusive_days"] = conclusive_tests_df['test_turnaround_days'].mean()

    # 2. Percentage of CRITICAL Tests Meeting TAT Target & Total Pending Critical Tests
    critical_test_keys = settings.CRITICAL_TESTS
    df_critical_tests = df[df['test_type'].isin(critical_test_keys)]

    if not df_critical_tests.empty:
        df_critical_conclusive = df_critical_tests[
            ~df_critical_tests['test_result'].str.lower().isin(['pending', 'rejected', 'unknownresult', 'indeterminate'])
        ]
        if not df_critical_conclusive.empty:
            met_tat_count = 0
            for _, row in df_critical_conclusive.iterrows():
                test_config = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(row['test_type'], {})
                target_tat = test_config.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS)
                if pd.notna(row['test_turnaround_days']) and row['test_turnaround_days'] <= target_tat:
                    met_tat_count += 1
            summary["perc_critical_tests_tat_met"] = (met_tat_count / len(df_critical_conclusive)) * 100
        
        summary["total_pending_critical_tests_patients"] = df_critical_tests[
            df_critical_tests['test_result'].str.lower() == 'pending'
        ]['patient_id'].nunique()

    # 3. Sample Rejection Rate (%)
    # Use encounter_id if available for uniqueness of test events, else patient_id if context implies one test per patient per relevant event
    unique_test_event_col = 'encounter_id' if 'encounter_id' in df.columns else 'patient_id'
    
    # All samples for which a status was recorded (excluding truly unknown/not applicable status)
    df_with_sample_status = df[~df['sample_status'].str.lower().isin(['unknownstatus', 'n/a', ''])]
    if not df_with_sample_status.empty:
        total_samples_processed_unique_events = df_with_sample_status[unique_test_event_col].nunique()
        rejected_samples_unique_events = df_with_sample_status[
            df_with_sample_status['sample_status'].str.lower() == 'rejected'
        ][unique_test_event_col].nunique()

        if total_samples_processed_unique_events > 0:
            summary["sample_rejection_rate_perc"] = (rejected_samples_unique_events / total_samples_processed_unique_events) * 100
    
    # 4. Key Drug Stockouts Count
    if 'encounter_date' in df.columns and df['encounter_date'].notna().any():
        latest_stock_df = df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # Assuming zone_id context for stock
        latest_stock_df.loc[latest_stock_df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001 # Avoid DivByZero
        latest_stock_df['days_of_supply'] = latest_stock_df['item_stock_agg_zone'] / latest_stock_df['consumption_rate_per_day']
        
        key_drugs_stock_df = latest_stock_df[
            latest_stock_df['item'].astype(str).str.contains('|'.join(settings.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)
        ]
        if not key_drugs_stock_df.empty:
            summary["key_drug_stockouts_count"] = key_drugs_stock_df[
                key_drugs_stock_df['days_of_supply'] < settings.CRITICAL_SUPPLY_DAYS_REMAINING
            ]['item'].nunique()


    # 5. Detailed breakdown per test type
    test_details_map = {}
    for test_key_orig, test_config_props in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        test_display_name = test_config_props.get("display_name", test_key_orig)
        df_specific_test = df[df['test_type'] == test_key_orig]
        
        if not df_specific_test.empty:
            df_st_conclusive = df_specific_test[
                ~df_specific_test['test_result'].str.lower().isin(['pending', 'rejected', 'unknownresult', 'indeterminate'])
            ]
            
            pos_rate = 0.0
            if not df_st_conclusive.empty:
                pos_count = df_st_conclusive[df_st_conclusive['test_result'].str.lower() == 'positive'].shape[0]
                pos_rate = (pos_count / len(df_st_conclusive)) * 100

            avg_tat_val = df_st_conclusive['test_turnaround_days'].mean() if not df_st_conclusive.empty and df_st_conclusive['test_turnaround_days'].notna().any() else np.nan
            
            met_tat_specific_count = 0
            if not df_st_conclusive.empty:
                target_tat_specific = test_config_props.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS)
                for _, row_st in df_st_conclusive.iterrows():
                    if pd.notna(row_st['test_turnaround_days']) and row_st['test_turnaround_days'] <= target_tat_specific:
                        met_tat_specific_count +=1
            perc_met_tat_specific = (met_tat_specific_count / len(df_st_conclusive)) * 100 if not df_st_conclusive.empty else 0.0

            test_details_map[test_display_name] = {
                "positive_rate_perc": pos_rate,
                "avg_tat_days": avg_tat_val,
                "perc_met_tat_target": perc_met_tat_specific,
                "pending_count_patients": df_specific_test[df_specific_test['test_result'].str.lower() == 'pending']['patient_id'].nunique(),
                "rejected_count_patients": df_specific_test[df_specific_test['sample_status'].str.lower() == 'rejected']['patient_id'].nunique(),
                "total_conclusive_tests": len(df_st_conclusive)
            }
        else: # No data for this specific test type
            test_details_map[test_display_name] = {
                "positive_rate_perc": 0.0, "avg_tat_days": np.nan, "perc_met_tat_target": 0.0,
                "pending_count_patients": 0, "rejected_count_patients": 0, "total_conclusive_tests": 0
            }
    summary["test_summary_details"] = test_details_map
    
    logger.info(f"({source_context}) Clinic Summary KPIs calculated: {list(summary.keys())}")
    return summary


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_environmental_summary_kpis( # Renamed for clarity
    iot_df_period: Optional[pd.DataFrame],
    source_context: str = "ClinicEnvSummaryKPIs"
) -> Dict[str, Any]:
    """
    Calculates summary KPIs for clinic environmental data.
    """
    logger.info(f"({source_context}) Calculating clinic environmental summary KPIs.")
    summary = {
        "avg_co2_overall_ppm": np.nan,
        "rooms_co2_very_high_alert_latest_count": 0,
        "avg_pm25_overall_ugm3": np.nan,
        "rooms_pm25_very_high_alert_latest_count": 0,
        "avg_waiting_room_occupancy_overall_persons": np.nan,
        "waiting_room_high_occupancy_alert_latest_flag": False,
        "avg_noise_overall_dba": np.nan,
        "rooms_noise_high_alert_latest_count": 0,
        "avg_temp_overall_celsius": np.nan,
        "avg_humidity_overall_rh": np.nan,
        "latest_readings_timestamp": None # Timestamp of the latest reading used for "latest" alerts
    }

    if not isinstance(iot_df_period, pd.DataFrame) or iot_df_period.empty:
        logger.warning(f"({source_context}) IoT DataFrame for environmental summary is empty or invalid.")
        return summary

    df = iot_df_period.copy()

    # Ensure 'timestamp' is datetime and other relevant columns are numeric
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
            logger.info(f"({source_context}) No IoT data with valid timestamps. Returning default summary.")
            return summary
        summary["latest_readings_timestamp"] = df['timestamp'].max() # Overall latest timestamp in period
    else:
        logger.warning(f"({source_context}) 'timestamp' column missing in IoT data. Some KPIs will be unreliable.")
        # Cannot determine "latest" without timestamp, but can still calculate overall averages if other cols exist.

    env_numeric_cols = ['avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy']
    for col in env_numeric_cols:
        if col in df.columns:
            df[col] = convert_to_numeric(df[col], default_value=np.nan)

    # Overall averages for the period
    if df['avg_co2_ppm'].notna().any(): summary["avg_co2_overall_ppm"] = df['avg_co2_ppm'].mean()
    if df['avg_pm25'].notna().any(): summary["avg_pm25_overall_ugm3"] = df['avg_pm25'].mean()
    if df['avg_noise_db'].notna().any(): summary["avg_noise_overall_dba"] = df['avg_noise_db'].mean()
    if df['avg_temp_celsius'].notna().any(): summary["avg_temp_overall_celsius"] = df['avg_temp_celsius'].mean()
    if df['avg_humidity_rh'].notna().any(): summary["avg_humidity_overall_rh"] = df['avg_humidity_rh'].mean()

    # "Latest" alerts based on the last reading for each room within the period
    if 'clinic_id' in df.columns and 'room_name' in df.columns and 'timestamp' in df.columns:
        # Get the last (latest timestamp) record for each unique room
        latest_readings_per_room_df = df.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')

        if not latest_readings_per_room_df.empty:
            if latest_readings_per_room_df['avg_co2_ppm'].notna().any():
                summary["rooms_co2_very_high_alert_latest_count"] = latest_readings_per_room_df[
                    latest_readings_per_room_df['avg_co2_ppm'] > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM
                ].shape[0]
            
            if latest_readings_per_room_df['avg_pm25'].notna().any():
                summary["rooms_pm25_very_high_alert_latest_count"] = latest_readings_per_room_df[
                    latest_readings_per_room_df['avg_pm25'] > settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3
                ].shape[0]

            if latest_readings_per_room_df['avg_noise_db'].notna().any():
                summary["rooms_noise_high_alert_latest_count"] = latest_readings_per_room_df[
                    latest_readings_per_room_df['avg_noise_db'] > settings.ALERT_AMBIENT_NOISE_HIGH_DBA
                ].shape[0]

            # Waiting room specific metrics from these latest readings
            waiting_rooms_latest_df = latest_readings_per_room_df[
                latest_readings_per_room_df.get('room_name', pd.Series(dtype=str)).astype(str).str.contains("Waiting", case=False, na=False)
            ]
            if not waiting_rooms_latest_df.empty and 'waiting_room_occupancy' in waiting_rooms_latest_df.columns and waiting_rooms_latest_df['waiting_room_occupancy'].notna().any():
                summary["avg_waiting_room_occupancy_overall_persons"] = waiting_rooms_latest_df['waiting_room_occupancy'].mean() # Avg across latest readings of waiting rooms
                summary["waiting_room_high_occupancy_alert_latest_flag"] = (
                    waiting_rooms_latest_df['waiting_room_occupancy'] > settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX
                ).any()
    else:
        logger.warning(f"({source_context}) Missing 'clinic_id', 'room_name', or 'timestamp' for latest environmental alerts calculation.")

    logger.info(f"({source_context}) Clinic Environmental Summary KPIs calculated: {list(summary.keys())}")
    return summary


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_district_summary_kpis(
    enriched_zone_df: Optional[pd.DataFrame], # Expects DataFrame with 'geometry_obj' and other aggregates
    source_context: str = "DistrictKPIs"
) -> Dict[str, Any]:
    """
    Calculates district-wide summary KPIs from enriched zone data.
    Enriched zone_df should be the output of `enrich_zone_geodata_with_health_aggregates`.
    """
    logger.info(f"({source_context}) Calculating district summary KPIs.")
    
    # Initialize KPIs
    kpis: Dict[str, Any] = {
        "total_zones_in_df": 0,
        "total_population_district": 0.0,
        "population_weighted_avg_ai_risk_score": np.nan,
        "zones_meeting_high_risk_criteria_count": 0,
        "district_avg_facility_coverage_score": np.nan, # Population weighted if possible
        "district_overall_key_disease_prevalence_per_1000": np.nan,
        "district_population_weighted_avg_steps": np.nan,
        "district_avg_clinic_co2_ppm": np.nan # Avg of zonal means
    }
    # Dynamically add keys for each condition
    for condition_name in settings.KEY_CONDITIONS_FOR_ACTION:
        kpi_key = f"district_total_active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        kpis[kpi_key] = 0

    if not isinstance(enriched_zone_df, pd.DataFrame) or enriched_zone_df.empty:
        logger.warning(f"({source_context}) Enriched zone DataFrame is empty or invalid.")
        return kpis

    df = enriched_zone_df.copy()

    # Ensure 'zone_id' and 'population' exist and population is numeric
    if 'zone_id' in df.columns:
        kpis["total_zones_in_df"] = df['zone_id'].nunique()
    else:
        kpis["total_zones_in_df"] = len(df) # Fallback if no zone_id
        logger.warning(f"({source_context}) 'zone_id' missing, using df length for total_zones.")

    if 'population' in df.columns:
        df['population'] = convert_to_numeric(df['population'], default_value=0.0)
        total_district_population = df['population'].sum()
        kpis["total_population_district"] = total_district_population
    else:
        logger.warning(f"({source_context}) 'population' column missing. Population-weighted KPIs will be NaN or simple averages.")
        total_district_population = 0.0 # Cannot calculate weighted averages

    # Calculate Population-Weighted Average AI Risk Score
    if 'avg_risk_score' in df.columns and total_district_population > 0:
        df['avg_risk_score'] = convert_to_numeric(df['avg_risk_score'], default_value=np.nan)
        weighted_risk_sum = (df['avg_risk_score'].fillna(0) * df['population']).sum() # fillna(0) for risk if pop exists
        kpis["population_weighted_avg_ai_risk_score"] = weighted_risk_sum / total_district_population
    elif 'avg_risk_score' in df.columns: # Population is 0 or missing, use simple mean
        kpis["population_weighted_avg_ai_risk_score"] = convert_to_numeric(df['avg_risk_score']).mean()


    # Zones Meeting High Risk Criteria
    if 'avg_risk_score' in df.columns:
        kpis["zones_meeting_high_risk_criteria_count"] = df[
            convert_to_numeric(df['avg_risk_score']) >= settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE
        ].shape[0]

    # District Average Facility Coverage Score (Population Weighted)
    if 'facility_coverage_score' in df.columns and total_district_population > 0:
        df['facility_coverage_score'] = convert_to_numeric(df['facility_coverage_score'], default_value=np.nan)
        weighted_coverage_sum = (df['facility_coverage_score'].fillna(0) * df['population']).sum()
        kpis["district_avg_facility_coverage_score"] = weighted_coverage_sum / total_district_population
    elif 'facility_coverage_score' in df.columns:
        kpis["district_avg_facility_coverage_score"] = convert_to_numeric(df['facility_coverage_score']).mean()


    # Total Active Cases for Key Conditions & Overall Prevalence
    total_key_infections_district = 0
    for condition_name in settings.KEY_CONDITIONS_FOR_ACTION:
        condition_col_name = f"active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        kpi_key = f"district_total_active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        if condition_col_name in df.columns:
            df[condition_col_name] = convert_to_numeric(df[condition_col_name], default_value=0)
            total_cases_for_condition = df[condition_col_name].sum()
            kpis[kpi_key] = total_cases_for_condition
            total_key_infections_district += total_cases_for_condition
        else:
            logger.debug(f"({source_context}) Metric column '{condition_col_name}' not found in enriched zone data.")
            kpis[kpi_key] = 0 # Ensure key exists

    if total_district_population > 0:
        kpis["district_overall_key_disease_prevalence_per_1000"] = (total_key_infections_district / total_district_population) * 1000
    else:
        kpis["district_overall_key_disease_prevalence_per_1000"] = np.nan

    # District Population-Weighted Average Daily Steps
    if 'avg_daily_steps_zone' in df.columns and total_district_population > 0:
        df['avg_daily_steps_zone'] = convert_to_numeric(df['avg_daily_steps_zone'], default_value=np.nan)
        # Fill NaN steps with a moderate default before weighting if population exists for that zone
        # Or, only consider zones with step data for the weighted average. For simplicity, use fillna.
        weighted_steps_sum = (df['avg_daily_steps_zone'].fillna(settings.TARGET_DAILY_STEPS * 0.6) * df['population']).sum()
        kpis["district_population_weighted_avg_steps"] = weighted_steps_sum / total_district_population
    elif 'avg_daily_steps_zone' in df.columns:
        kpis["district_population_weighted_avg_steps"] = convert_to_numeric(df['avg_daily_steps_zone']).mean()


    # District Average Clinic CO2 (Average of Zonal Means)
    if 'zone_avg_co2' in df.columns: # This column should come from enrichment
        kpis["district_avg_clinic_co2_ppm"] = convert_to_numeric(df['zone_avg_co2']).mean()

    logger.info(f"({source_context}) District Summary KPIs calculated: {list(kpis.keys())}")
    return kpis
