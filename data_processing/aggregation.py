# sentinel_project_root/data_processing/aggregation.py
"""
A collection of robust, high-performance, and encapsulated components for
aggregating data to compute KPIs and summaries for Sentinel dashboards.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Callable
import streamlit as st
import re

try:
    from config import settings
    from .helpers import convert_to_numeric
except ImportError as e:
    logging.critical(f"Critical import error in aggregation.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Hashing Function for Streamlit Caching ---
def hash_dataframe_safe(df: pd.DataFrame) -> int:
    """A robust hashing function for pandas DataFrames suitable for st.cache_data."""
    return pd.util.hash_pandas_object(df, index=True).sum()

# --- Generic Trend Calculation Utility ---
def get_trend_data(df: Optional[pd.DataFrame], value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: Union[str, Callable] = 'mean') -> pd.Series:
    """Calculates a time-series trend for a given column, aggregated by a specified period."""
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=np.float64)
    df_trend = df[[date_col, value_col]].copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend[value_col] = convert_to_numeric(df_trend[value_col])
    df_trend.dropna(subset=[date_col, value_col], inplace=True)
    if df_trend.empty: return pd.Series(dtype=np.float64)

    try:
        trend_series = df_trend.set_index(date_col)[value_col].resample(period).agg(agg_func)
        if isinstance(agg_func, str) and agg_func in ['count', 'size', 'nunique']:
            trend_series = trend_series.fillna(0).astype(int)
        return trend_series
    except Exception as e:
        logger.error(f"Error generating trend for '{value_col}': {e}", exc_info=True)
        return pd.Series(dtype=np.float64)

# --- Preparer Classes for KPI Calculation ---

class ClinicKPIPreparer:
    """Encapsulates all logic for calculating clinic-level summary KPIs."""
    def __init__(self, health_df: pd.DataFrame):
        self.df = health_df.copy() if isinstance(health_df, pd.DataFrame) else pd.DataFrame()
        self.summary: Dict[str, Any] = {"overall_avg_test_turnaround_conclusive_days": np.nan, "perc_critical_tests_tat_met": 0.0, "total_pending_critical_tests_patients": 0, "sample_rejection_rate_perc": 0.0, "key_drug_stockouts_count": 0, "test_summary_details": {}}

    def _calculate_testing_kpis(self):
        if not all(c in self.df.columns for c in ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'patient_id']): return
        conclusive = self.df[~self.df['test_result'].str.lower().isin(['pending', 'rejected', 'unknown'])]
        if not conclusive.empty: self.summary["overall_avg_test_turnaround_conclusive_days"] = conclusive['test_turnaround_days'].mean()
        
        critical_keys = [k for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() if isinstance(v, dict) and v.get("critical")]
        critical_df = self.df[self.df['test_type'].isin(critical_keys)]
        if not critical_df.empty:
            conclusive_crit = critical_df[~critical_df['test_result'].str.lower().isin(['pending', 'rejected'])]
            if not conclusive_crit.empty:
                target_map = {k: v.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS) for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items()}
                conclusive_crit['target_tat'] = conclusive_crit['test_type'].map(target_map)
                self.summary["perc_critical_tests_tat_met"] = (conclusive_crit['test_turnaround_days'] <= conclusive_crit['target_tat']).mean() * 100
            self.summary["total_pending_critical_tests_patients"] = critical_df[critical_df['test_result'].str.lower() == 'pending']['patient_id'].nunique()
        
        valid_status = self.df[self.df['sample_status'].notna() & ~self.df['sample_status'].str.lower().isin(['unknown', ''])]
        if not valid_status.empty and valid_status['patient_id'].nunique() > 0:
            rejected = valid_status[valid_status['sample_status'].str.lower() == 'rejected by lab']['patient_id'].nunique()
            self.summary["sample_rejection_rate_perc"] = (rejected / valid_status['patient_id'].nunique()) * 100

    def _calculate_supply_kpis(self):
        if not all(c in self.df.columns for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date', 'zone_id']): return
        latest = self.df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
        latest['consumption_rate_per_day'] = latest['consumption_rate_per_day'].replace(0, 0.001)
        latest['days_of_supply'] = latest['item_stock_agg_zone'] / latest['consumption_rate_per_day']
        key_drugs = latest[latest['item'].str.contains('|'.join(settings.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        if not key_drugs.empty:
            self.summary["key_drug_stockouts_count"] = key_drugs[key_drugs['days_of_supply'] < settings.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()

    def _calculate_test_breakdown(self):
        if not all(c in self.df.columns for c in ['test_type', 'test_result', 'patient_id', 'sample_status', 'test_turnaround_days']): return
        
        tests_df = self.df[self.df['test_type'].isin(settings.KEY_TEST_TYPES_FOR_ANALYSIS.keys())].copy()
        if tests_df.empty: return
        
        tests_df['is_positive'] = (tests_df['test_result'].str.lower() == 'positive')
        tests_df['is_pending'] = (tests_df['test_result'].str.lower() == 'pending')
        tests_df['is_rejected'] = (tests_df['sample_status'].str.lower() == 'rejected by lab')
        target_map = {k: v.get('target_tat_days', settings.TARGET_TEST_TURNAROUND_DAYS) for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items()}
        tests_df['tat_met'] = tests_df['test_turnaround_days'] <= tests_df['test_type'].map(target_map)
        
        breakdown = tests_df.groupby('test_type').agg(pending_count_patients=('is_pending', 'sum'), rejected_count_patients=('is_rejected', 'sum')).fillna(0)
        
        conclusive_df = tests_df[~tests_df['test_result'].str.lower().isin(['pending', 'rejected'])]
        if not conclusive_df.empty:
            conclusive_agg = conclusive_df.groupby('test_type').agg(positive_rate_perc=('is_positive', lambda x: x.mean() * 100), avg_tat_days=('test_turnaround_days', 'mean'), perc_met_tat_target=('tat_met', lambda x: x.mean() * 100))
            breakdown = breakdown.join(conclusive_agg, how='left')
        
        # Use display names from settings as keys in the final dictionary
        display_name_map = {k: v.get("display_name", k) for k, v in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items()}
        breakdown.rename(index=display_name_map, inplace=True)
        self.summary['test_summary_details'] = breakdown.fillna(0).to_dict('index')

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return self.summary
        self._calculate_testing_kpis(); self._calculate_supply_kpis(); self._calculate_test_breakdown()
        return self.summary

class ClinicEnvKPIPreparer:
    """Encapsulates logic for calculating environmental summary KPIs."""
    def __init__(self, iot_df: pd.DataFrame):
        self.df = iot_df.copy() if isinstance(iot_df, pd.DataFrame) else pd.DataFrame()
        self.summary: Dict[str, Any] = {"avg_co2_overall_ppm": np.nan, "avg_pm25_overall_ugm3": np.nan, "avg_waiting_room_occupancy_overall_persons": np.nan, "rooms_noise_high_alert_latest_count": 0}

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return self.summary
        for col in ['avg_co2_ppm', 'avg_pm25', 'avg_noise_db', 'waiting_room_occupancy']:
            if col in self.df.columns: self.df[col] = convert_to_numeric(self.df[col])
        
        self.summary['avg_co2_overall_ppm'] = self.df.get('avg_co2_ppm', pd.Series(dtype=float)).mean()
        self.summary['avg_pm25_overall_ugm3'] = self.df.get('avg_pm25', pd.Series(dtype=float)).mean()
        self.summary['avg_waiting_room_occupancy_overall_persons'] = self.df.get('waiting_room_occupancy', pd.Series(dtype=float)).mean()

        if 'timestamp' in self.df.columns and 'room_name' in self.df.columns:
            latest = self.df.sort_values('timestamp').drop_duplicates('room_name', keep='last')
            self.summary['rooms_noise_high_alert_latest_count'] = (latest.get('avg_noise_db', pd.Series(dtype=float)) > settings.ALERT_AMBIENT_NOISE_HIGH_DBA).sum()
        return self.summary

class CHWDailySummaryPreparer:
    """Encapsulates logic for calculating CHW daily summary metrics."""
    def __init__(self, health_df: pd.DataFrame):
        self.df = health_df.copy() if isinstance(health_df, pd.DataFrame) else pd.DataFrame()
        self.summary = {"visits_count": 0, "high_ai_prio_followups_count": 0, "fever_cases_identified_count": 0, "pending_critical_referrals_generated_today_count": 0, "avg_risk_of_visited_patients": np.nan}

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return self.summary
        patients = self.df[~self.df.get('encounter_type', pd.Series(dtype=str)).str.contains("WORKER_SELF", na=False)]
        if patients.empty: return self.summary

        self.summary["visits_count"] = patients['patient_id'].nunique()
        self.summary["avg_risk_of_visited_patients"] = patients.drop_duplicates('patient_id')['ai_risk_score'].mean()
        
        temp_col = next((c for c in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if c in patients), None)
        if temp_col: self.summary["fever_cases_identified_count"] = (patients[temp_col] >= settings.ALERT_BODY_TEMP_FEVER_C).sum()
        
        prio_scores = convert_to_numeric(patients['ai_followup_priority_score'])
        self.summary["high_ai_prio_followups_count"] = (prio_scores >= settings.FATIGUE_INDEX_HIGH_THRESHOLD).sum()

        is_pending = patients['referral_status'].str.lower() == 'pending'
        is_critical = patients['condition'].str.contains('|'.join(settings.KEY_CONDITIONS_FOR_ACTION), na=False)
        self.summary["pending_critical_referrals_generated_today_count"] = patients[is_pending & is_critical]['patient_id'].nunique()
        return self.summary

class DistrictKPIPreparer:
    """Encapsulates logic for calculating district-wide summary KPIs from enriched zone data."""
    def __init__(self, enriched_zone_df: pd.DataFrame):
        self.df = enriched_zone_df.copy() if isinstance(enriched_zone_df, pd.DataFrame) else pd.DataFrame()
        self.summary = {"total_zones_in_df": 0, "total_population_district": 0, "population_weighted_avg_ai_risk_score": np.nan, "zones_meeting_high_risk_criteria_count": 0}

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return self.summary
        self.df['population'] = convert_to_numeric(self.df.get('population'), default_value=0.0)
        self.df['avg_risk_score'] = convert_to_numeric(self.df.get('avg_risk_score'))
        
        self.summary["total_zones_in_df"] = self.df['zone_id'].nunique()
        total_pop = self.df['population'].sum()
        self.summary["total_population_district"] = total_pop
        
        if total_pop > 0:
            self.summary["population_weighted_avg_ai_risk_score"] = (self.df['avg_risk_score'].fillna(0) * self.df['population']).sum() / total_pop
        else:
            self.summary["population_weighted_avg_ai_risk_score"] = self.df['avg_risk_score'].mean()
        
        self.summary["zones_meeting_high_risk_criteria_count"] = (self.df['avg_risk_score'] >= settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE).sum()
        
        for cond in settings.KEY_CONDITIONS_FOR_ACTION:
            col = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond.lower())}_cases"
            self.summary[f"district_total_{col}"] = convert_to_numeric(self.df.get(col)).sum()
        return self.summary

# --- Public Factory Functions ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_summary_kpis(health_df_period: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    """Factory to calculate all summary KPIs for a clinic over a period."""
    return ClinicKPIPreparer(health_df_period).prepare()

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_environmental_summary_kpis(iot_df_period: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    """Factory to calculate environmental summary KPIs for a clinic."""
    return ClinicEnvKPIPreparer(iot_df_period).prepare()

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_chw_summary_kpis(health_df_daily: Optional[pd.DataFrame], for_date: Any, source_context: str = "") -> Dict[str, Any]:
    """Factory to calculate CHW daily summary metrics for a specific date."""
    if not isinstance(health_df_daily, pd.DataFrame) or health_df_daily.empty: return CHWDailySummaryPreparer(pd.DataFrame()).summary
    target_date = pd.to_datetime(for_date).date()
    daily_df = health_df_daily[pd.to_datetime(health_df_daily['encounter_date']).dt.date == target_date]
    return CHWDailySummaryPreparer(daily_df).prepare()

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_district_summary_kpis(enriched_zone_df: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    """Factory to calculate district-wide summary KPIs from enriched zone data."""
    return DistrictKPIPreparer(enriched_zone_df).prepare()
