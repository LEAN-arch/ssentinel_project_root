# sentinel_project_root/data_processing/aggregation.py
# Functions for aggregating data to compute KPIs and summaries for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, Union, Callable, List
from datetime import date as date_type, datetime 

try:
    from config import settings
    from .helpers import convert_to_numeric 
except ImportError as e:
    # Fallback settings remain the same...
    # ...

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

def get_trend_data(
    df: Optional[pd.DataFrame], value_col: str, date_col: str = 'encounter_date',
    period: str = 'D', agg_func: Union[str, Callable[[Any], Any]] = 'mean',
    filter_col: Optional[str] = None, filter_val: Optional[Any] = None,
    source_context: str = "TrendCalculator"
) -> pd.Series:
    # Function implementation remains the same
    # ...

def get_district_summary_kpis(
    enriched_zone_df: Optional[pd.DataFrame],
    source_context: str = "DistrictKPIs"
) -> Dict[str, Any]:
    logger.info(f"({source_context}) Calculating district-level summary KPIs.")
    
    kpis: Dict[str, Any] = {
        "total_zones_in_df": 0, "total_population_district": 0.0,
        "population_weighted_avg_ai_risk_score": np.nan,
        "zones_meeting_high_risk_criteria_count": 0,
        "district_avg_facility_coverage_score": np.nan,
        "district_overall_key_disease_prevalence_per_1000": np.nan,
        "top_active_condition_name": "N/A", # New KPI
        "top_active_condition_count": 0,      # New KPI
    }
    
    if not isinstance(enriched_zone_df, pd.DataFrame) or enriched_zone_df.empty:
        logger.warning(f"({source_context}) Enriched zone DataFrame is empty. Cannot calculate district KPIs.")
        return kpis

    df = enriched_zone_df.copy()
    kpis["total_zones_in_df"] = df['zone_id'].nunique()
    
    # Calculate population-based metrics
    df['population'] = convert_to_numeric(df['population'], 0.0)
    total_pop = df['population'].sum()
    kpis["total_population_district"] = total_pop
    
    if total_pop > 0:
        df['risk_score'] = convert_to_numeric(df.get('avg_risk_score'), np.nan)
        kpis["population_weighted_avg_ai_risk_score"] = (df['risk_score'] * df['population']).sum() / total_pop if df['risk_score'].notna().any() else np.nan
        
        df['facility_coverage_score'] = convert_to_numeric(df.get('facility_coverage_score'), np.nan)
        kpis["district_avg_facility_coverage_score"] = (df['facility_coverage_score'] * df['population']).sum() / total_pop if df['facility_coverage_score'].notna().any() else np.nan
        
    # High-Risk Zones
    high_risk_thresh = _get_setting('DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 70)
    kpis["zones_meeting_high_risk_criteria_count"] = df[df.get('avg_risk_score', pd.Series(dtype=float)) >= high_risk_thresh].shape[0]

    # CORRECTED: Loop to calculate individual and then find the top condition
    active_case_counts: Dict[str, int] = {}
    total_key_infections = 0
    for cond_name in _get_setting('KEY_CONDITIONS_FOR_ACTION', []):
        col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_name.lower().strip())}_cases"
        if col_name in df.columns:
            count = int(df[col_name].sum())
            kpis[f"district_total_{col_name}"] = count
            active_case_counts[cond_name.replace("(Severe)", "").strip()] = count
            total_key_infections += count

    if total_pop > 0:
        kpis["district_overall_key_disease_prevalence_per_1000"] = (total_key_infections / total_pop) * 1000
        
    # New logic to find the top condition by case count
    if active_case_counts:
        top_condition = max(active_case_counts, key=active_case_counts.get)
        kpis["top_active_condition_name"] = top_condition
        kpis["top_active_condition_count"] = active_case_counts[top_condition]

    return kpis
    df = health_df_period.copy()
    
    # Prepare test-related columns
    test_cols = ['test_type', 'test_result', 'test_turnaround_days', 'patient_id', 'sample_status']
    for col in test_cols:
        if col not in df.columns: df[col] = "Unknown" if col in ['test_type', 'test_result', 'sample_status'] else np.nan
    df['test_turnaround_days'] = convert_to_numeric(df['test_turnaround_days'], np.nan)

    # --- Calculations for Main KPIs ---
    non_conclusive_results = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', ['pending', 'rejected'])
    critical_tests = _get_setting('CRITICAL_TESTS', [])
    
    df_conclusive = df[~df['test_result'].astype(str).str.lower().isin(non_conclusive_results)]
    
    if not df_conclusive.empty and 'test_turnaround_days' in df_conclusive and df_conclusive['test_turnaround_days'].notna().any():
        kpis["overall_avg_test_turnaround_conclusive_days"] = df_conclusive['test_turnaround_days'].mean()

    df_critical_conclusive = df_conclusive[df_conclusive['test_type'].isin(critical_tests)]
    if not df_critical_conclusive.empty:
        met_tat_count = 0
        key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
        for _, row in df_critical_conclusive.iterrows():
            target_tat = key_test_configs.get(row['test_type'], {}).get('target_tat_days', _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2))
            if pd.notna(row['test_turnaround_days']) and row['test_turnaround_days'] <= target_tat:
                met_tat_count += 1
        kpis["perc_critical_tests_tat_met"] = (met_tat_count / len(df_critical_conclusive)) * 100 if len(df_critical_conclusive) > 0 else 0.0

    df_pending_critical = df[(df['test_result'].astype(str).str.lower() == 'pending') & (df['test_type'].isin(critical_tests))]
    if 'patient_id' in df_pending_critical.columns:
        kpis["total_pending_critical_tests_patients"] = df_pending_critical['patient_id'].nunique()

    total_samples = df[df['sample_status'] != 'Unknown']['sample_status'].count()
    rejected_samples = df[df['sample_status'].astype(str).str.lower() == 'rejected by lab']['sample_status'].count()
    if total_samples > 0:
        kpis["sample_rejection_rate_perc"] = (rejected_samples / total_samples) * 100
        
    # --- Detailed Summary for Each Test Type ---
    test_details = {}
    if 'test_type' in df.columns:
        for test_name, group_df in df.groupby('test_type'):
            if test_name == 'Unknown': continue
            display_name = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {}).get(test_name, {}).get('display_name', test_name)
            group_conclusive = group_df[~group_df['test_result'].astype(str).str.lower().isin(non_conclusive_results)]
            
            pos_rate = np.nan
            if not group_conclusive.empty:
                positives = group_conclusive[group_conclusive['test_result'].astype(str).str.lower() == 'positive'].shape[0]
                pos_rate = (positives / len(group_conclusive)) * 100 if len(group_conclusive) > 0 else 0.0

            test_details[display_name] = {
                "positive_rate_perc": pos_rate,
                "avg_tat_days": group_conclusive['test_turnaround_days'].mean(),
                "total_conclusive_tests": len(group_conclusive),
                "pending_count_patients": group_df[group_df['test_result'].astype(str).str.lower() == 'pending']['patient_id'].nunique()
            }
    kpis["test_summary_details"] = test_details

    # --- Stockout Calculation ---
    key_drugs = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if key_drugs:
        drug_pattern = '|'.join(key_drugs)
        df_drugs = df[df['item'].str.contains(drug_pattern, case=False, na=False)].copy()
        if not df_drugs.empty and all(c in df_drugs for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day']):
            df_drugs['days_of_supply'] = df_drugs['item_stock_agg_zone'] / df_drugs['consumption_rate_per_day'].replace(0, np.nan)
            latest_stock = df_drugs.sort_values('encounter_date').drop_duplicates(subset='item', keep='last')
            stockout_count = latest_stock[latest_stock['days_of_supply'] < _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)].shape[0]
            kpis['key_drug_stockouts_count'] = stockout_count

    logger.info(f"({source_context}) Clinic KPIs calculated successfully.")
    return kpis

# --- Other Placeholder Functions ---
def get_chw_summary_kpis(*args, **kwargs) -> Dict[str, Any]: return {}
def get_clinic_environmental_summary_kpis(*args, **kwargs) -> Dict[str, Any]: return {}
def get_district_summary_kpis(*args, **kwargs) -> Dict[str, Any]: return {}
