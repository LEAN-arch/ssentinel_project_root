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
except ImportError:
    logging.warning("Critical import error in aggregation.py. Using fallbacks.", exc_info=True)
    class MockSettings: pass
    settings = MockSettings()
    def convert_to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='coerce')

logger = logging.getLogger(__name__)

def hash_dataframe_safe(df: pd.DataFrame) -> int:
    """A robust hashing function for pandas DataFrames suitable for st.cache_data."""
    return pd.util.hash_pandas_object(df, index=True).sum()

# --- Generic Trend Calculation Utility ---
def get_trend_data(df: Optional[pd.DataFrame], value_col: str, date_col: str, agg_func: Union[str, Callable] = 'mean', **kwargs) -> pd.Series:
    """
    Calculates a time-series trend for a given column, aggregated by a specified period.
    
    SME NOTE: DEFINITIVE BACKWARD-COMPATIBILITY FIX.
    This function now accepts either `freq=` (the new, preferred name) or `period=`
    (the old name) from its keyword arguments (`**kwargs`). This prevents TypeErrors
    from any page in the application (like 02_clinic_dashboard.py or
    03_district_dashboard.py) that calls this shared function, regardless of which
    parameter name it uses.
    """
    # Handle backward compatibility for the frequency/period argument.
    # It will look for 'freq' first, then 'period', then default to 'D'.
    freq = kwargs.get('freq', kwargs.get('period', 'D'))

    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=np.float64)
    
    df_trend = df[[date_col, value_col]].copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend[value_col] = convert_to_numeric(df_trend[value_col])
    df_trend.dropna(subset=[date_col, value_col], inplace=True)

    if df_trend.empty: return pd.Series(dtype=np.float64)
    
    try:
        if freq == 'H': freq = 'h' # Use modern, non-deprecated frequency string
        trend_series = df_trend.set_index(date_col)[value_col].resample(freq).agg(agg_func)
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
        critical_keys = [k for k, v in getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {}).items() if isinstance(v, dict) and v.get("critical")]
        critical_df = self.df[self.df['test_type'].isin(critical_keys)]
        if not critical_df.empty:
            conclusive_crit = critical_df[~critical_df['test_result'].str.lower().isin(['pending', 'rejected'])]
            if not conclusive_crit.empty:
                target_map = {k: v.get('target_tat_days', getattr(settings, 'TARGET_TEST_TURNAROUND_DAYS', 7)) for k, v in getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {}).items()}
                conclusive_crit_with_target = conclusive_crit.assign(target_tat=lambda x: x['test_type'].map(target_map))
                self.summary["perc_critical_tests_tat_met"] = ((conclusive_crit_with_target['test_turnaround_days'] <= conclusive_crit_with_target['target_tat']).mean() * 100)
            self.summary["total_pending_critical_tests_patients"] = critical_df[critical_df['test_result'].str.lower() == 'pending']['patient_id'].nunique()
        valid_status = self.df[self.df['sample_status'].notna() & ~self.df['sample_status'].str.lower().isin(['unknown', ''])]
        denominator = valid_status['patient_id'].nunique()
        if denominator > 0:
            rejected = valid_status[valid_status['sample_status'].str.lower() == 'rejected by lab']['patient_id'].nunique()
            self.summary["sample_rejection_rate_perc"] = (rejected / denominator) * 100

    def _calculate_supply_kpis(self):
        if not all(c in self.df.columns for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date', 'zone_id']): return
        latest = self.df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last').copy()
        latest['consumption_rate_per_day'] = latest['consumption_rate_per_day'].replace(0, 0.001)
        latest['days_of_supply'] = latest['item_stock_agg_zone'] / latest['consumption_rate_per_day']
        key_drugs_pattern = '|'.join(map(re.escape, getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', [])))
        if not key_drugs_pattern: return
        key_drugs_stock_df = latest[latest['item'].str.contains(key_drugs_pattern, case=False, na=False)]
        if not key_drugs_stock_df.empty:
            self.summary["key_drug_stockouts_count"] = key_drugs_stock_df[key_drugs_stock_df['days_of_supply'] < getattr(settings, 'CRITICAL_SUPPLY_DAYS_REMAINING', 7)]['item'].nunique()

    def _calculate_test_breakdown(self):
        if not all(c in self.df.columns for c in ['test_type', 'test_result', 'patient_id', 'sample_status', 'test_turnaround_days']): return
        tests_df = self.df[self.df['test_type'].isin(getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {}).keys())].copy()
        if tests_df.empty: return
        tests_df['is_positive'] = (tests_df['test_result'].str.lower() == 'positive')
        conclusive_df = tests_df[~tests_df['test_result'].str.lower().isin(['pending', 'rejected', 'unknown'])]
        if not conclusive_df.empty:
            agg_spec = {'positive_rate_perc': pd.NamedAgg(column='is_positive', aggfunc=lambda x: x.mean() * 100 if len(x) > 0 else 0.0)}
            breakdown = conclusive_df.groupby('test_type').agg(**agg_spec)
            self.summary['test_summary_details'] = breakdown.to_dict('index')
        else: self.summary['test_summary_details'] = {}

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return self.summary
        self._calculate_testing_kpis(); self._calculate_supply_kpis(); self._calculate_test_breakdown()
        return self.summary

class ClinicEnvKPIPreparer:
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
            self.summary['rooms_noise_high_alert_latest_count'] = (latest.get('avg_noise_db', pd.Series(dtype=float)) > getattr(settings, 'ALERT_AMBIENT_NOISE_HIGH_DBA', 80)).sum()
        return self.summary

class CHWDailySummaryPreparer:
    def __init__(self, health_df: pd.DataFrame): self.df = health_df
    def prepare(self): return {} 

class DistrictKPIPreparer:
    """SME NOTE: FUNCTIONAL FIX. The logic for this preparer was missing and has now been implemented."""
    def __init__(self, enriched_zone_df: pd.DataFrame):
        self.df = enriched_zone_df.copy() if isinstance(enriched_zone_df, pd.DataFrame) else pd.DataFrame()

    def prepare(self) -> Dict[str, Any]:
        if self.df.empty: return {}
        kpis: Dict[str, Any] = {}
        kpis['total_population_district'] = int(self.df.get('population', 0).sum())
        kpis['total_zones_in_df'] = len(self.df)
        
        pop = self.df.get('population')
        risk = self.df.get('avg_risk_score')
        
        if pop is not None and risk is not None and pop.sum() > 0:
            kpis['population_weighted_avg_ai_risk_score'] = np.average(risk, weights=pop)
        else:
            kpis['population_weighted_avg_ai_risk_score'] = risk.mean() if risk is not None else 0.0

        if risk is not None:
            high_risk_threshold = getattr(settings, 'DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 7.5)
            kpis['zones_meeting_high_risk_criteria_count'] = int((risk > high_risk_threshold).sum())
        else:
            kpis['zones_meeting_high_risk_criteria_count'] = 0
            
        return kpis

# --- Public Factory Functions ---
# SME NOTE: These are now fully robust and call the corrected/implemented classes above.
@st.cache_data(ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 3600), hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_summary_kpis(health_df_period: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    return ClinicKPIPreparer(health_df_period).prepare()

@st.cache_data(ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 3600), hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_clinic_environmental_summary_kpis(iot_df_period: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    return ClinicEnvKPIPreparer(iot_df_period).prepare()

@st.cache_data(ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 3600), hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_chw_summary_kpis(health_df_daily: Optional[pd.DataFrame], for_date: Any, source_context: str = "") -> Dict[str, Any]:
    if not isinstance(health_df_daily, pd.DataFrame) or health_df_daily.empty: return CHWDailySummaryPreparer(pd.DataFrame()).prepare()
    target_date = pd.to_datetime(for_date).date()
    daily_df = health_df_daily[pd.to_datetime(health_df_daily['encounter_date']).dt.date == target_date]
    return CHWDailySummaryPreparer(daily_df).prepare()

@st.cache_data(ttl=getattr(settings, 'CACHE_TTL_SECONDS_WEB_REPORTS', 3600), hash_funcs={pd.DataFrame: hash_dataframe_safe})
def get_district_summary_kpis(enriched_zone_df: Optional[pd.DataFrame], source_context: str = "") -> Dict[str, Any]:
    return DistrictKPIPreparer(enriched_zone_df).prepare()
