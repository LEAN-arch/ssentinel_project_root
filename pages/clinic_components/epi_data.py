# sentinel_project_root/pages/clinic_components/epi_data.py
# Calculates clinic-level epidemiological data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
import re # For regex based symptom/condition matching
from typing import Dict, Any, Optional, List

from config import settings
from data_processing.aggregation import get_trend_data
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def calculate_clinic_epidemiological_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    condition_filter_for_demographics: str = "All Conditions",
    num_top_symptoms_to_trend: int = 5
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.
    """
    module_log_prefix = "ClinicEpiDataCalc"
    logger.info(f"({module_log_prefix}) Calculating clinic epi data. Period: {reporting_period_context_str}, Demo Cond: {condition_filter_for_demographics}")

    default_symptom_trend_cols = ['week_start_date', 'symptom', 'count']
    default_demo_age_cols = ['Age Group', 'Patient Count']
    default_demo_gender_cols = ['Gender', 'Patient Count']
    default_referral_cols = ['Stage', 'Count']

    epi_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "symptom_trends_weekly_top_n_df": pd.DataFrame(columns=default_symptom_trend_cols),
        "key_test_positivity_trends": {},
        "demographics_by_condition_data": {
            "age_distribution_df": pd.DataFrame(columns=default_demo_age_cols),
            "gender_distribution_df": pd.DataFrame(columns=default_demo_gender_cols),
            "condition_analyzed": condition_filter_for_demographics
        },
        "referral_funnel_summary_df": pd.DataFrame(columns=default_referral_cols),
        "calculation_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        msg = "No health data for clinic epi analysis. Calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}"); epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    df_epi_src = filtered_health_df_clinic_period.copy()

    # Data Preparation
    if 'encounter_date' not in df_epi_src.columns:
        msg = "'encounter_date' missing. Critical for epi calcs."
        logger.error(f"({module_log_prefix}) {msg}"); epi_data_output["calculation_notes"].append(msg); return epi_data_output
    try:
        df_epi_src['encounter_date'] = pd.to_datetime(df_epi_src['encounter_date'], errors='coerce')
        df_epi_src.dropna(subset=['encounter_date'], inplace=True)
    except Exception as e_date_epi:
        logger.error(f"({module_log_prefix}) Error converting 'encounter_date': {e_date_epi}")
        epi_data_output["calculation_notes"].append("Error processing encounter dates."); return epi_data_output
    if df_epi_src.empty:
        msg = "No valid encounter dates after cleaning."; logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg); return epi_data_output

    epi_cols_cfg = {
        'patient_id': {"default": f"UPID_Epi_{reporting_period_context_str[:10]}", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str}, 'condition': {"default": "UCond", "type": str},
        'test_type': {"default": "UTest", "type": str}, 'test_result': {"default": "URes", "type": str},
        'age': {"default": np.nan, "type": float}, 'gender': {"default": "Ungen", "type": str},
        'referral_status': {"default": "UStat", "type": str}, 'referral_outcome': {"default": "UOut", "type": str},
        'encounter_id': {"default": f"UEID_Epi_{reporting_period_context_str[:10]}", "type": str}
    }
    common_na_epi = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    na_regex_epi = r'^(?:' + '|'.join(re.escape(s) for s in common_na_epi if s) + r')$'

    for col, cfg in epi_cols_cfg.items():
        if col not in df_epi_src.columns: df_epi_src[col] = cfg["default"]
        if cfg["type"] == float: df_epi_src[col] = convert_to_numeric(df_epi_src[col], default_value=cfg["default"])
        elif cfg["type"] == str:
            df_epi_src[col] = df_epi_src[col].astype(str).fillna(str(cfg["default"]))
            if any(common_na_epi): df_epi_src[col] = df_epi_src[col].replace(na_regex_epi, str(cfg["default"]), regex=True)
            df_epi_src[col] = df_epi_src[col].str.strip()
    
    # Symptom Trends
    if 'patient_reported_symptoms' in df_epi_src.columns and df_epi_src['patient_reported_symptoms'].str.strip().astype(bool).any():
        df_symptoms_base = df_epi_src[['encounter_date', 'patient_reported_symptoms']].copy()
        non_info_symptoms = ["unknown", "n/a", "none", "", " ", "no symptoms", "asymptomatic", "well", "routine", "follow up"]
        df_symptoms_base = df_symptoms_base[~df_symptoms_base['patient_reported_symptoms'].astype(str).str.lower().isin(non_info_symptoms)]
        df_symptoms_base.dropna(subset=['patient_reported_symptoms'], inplace=True)
        if not df_symptoms_base.empty:
            symptoms_exploded = df_symptoms_base.assign(symptom=df_symptoms_base['patient_reported_symptoms'].str.split(r'[;,|]')).explode('symptom')
            symptoms_exploded['symptom'] = symptoms_exploded['symptom'].astype(str).str.strip().str.title()
            symptoms_exploded = symptoms_exploded[symptoms_exploded['symptom'] != ''].dropna(subset=['symptom'])
            if not symptoms_exploded.empty:
                top_symptoms = symptoms_exploded['symptom'].value_counts().nlargest(num_top_symptoms_to_trend).index.tolist()
                df_top_symptoms = symptoms_exploded[symptoms_exploded['symptom'].isin(top_symptoms)]
                if not df_top_symptoms.empty:
                    weekly_counts = df_top_symptoms.groupby([pd.Grouper(key='encounter_date', freq='W-MON', label='left', closed='left'), 'symptom']).size().reset_index(name='count')
                    weekly_counts.rename(columns={'encounter_date': 'week_start_date'}, inplace=True)
                    epi_data_output["symptom_trends_weekly_top_n_df"] = weekly_counts
                else: epi_data_output["calculation_notes"].append(f"Not enough diverse symptom data after filtering for top {num_top_symptoms_to_trend}.")
            else: epi_data_output["calculation_notes"].append("No valid individual symptoms after cleaning/exploding.")
        else: epi_data_output["calculation_notes"].append("No actionable symptoms data after filtering non-informative entries.")
    else: epi_data_output["calculation_notes"].append("'patient_reported_symptoms' missing or empty. Symptom trends skipped.")

    # Test Positivity Trends
    test_pos_trends: Dict[str, pd.Series] = {}
    conclusive_mask = ~df_epi_src.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(["pending", "rejected", "ures", "indeterminate", "n/a", ""])
    df_conclusive = df_epi_src[conclusive_mask].copy()
    if not df_conclusive.empty:
        for test_key, test_cfg in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items():
            test_disp_name = test_cfg.get("display_name", test_key)
            df_spec_test_concl = df_conclusive[df_conclusive['test_type'] == test_key]
            if not df_spec_test_concl.empty:
                df_spec_test_concl['is_positive'] = (df_spec_test_concl['test_result'].astype(str).str.lower() == 'positive')
                pos_rate_trend = get_trend_data(df=df_spec_test_concl, value_col='is_positive', date_col='encounter_date', period='W-MON', agg_func='mean', source_context=f"{module_log_prefix}/PosTrend/{test_disp_name}")
                if isinstance(pos_rate_trend, pd.Series) and not pos_rate_trend.empty: test_pos_trends[test_disp_name] = (pos_rate_trend * 100).round(1)
                else: epi_data_output["calculation_notes"].append(f"No weekly positivity trend for {test_disp_name} (empty series).")
            else: epi_data_output["calculation_notes"].append(f"No conclusive test data for '{test_disp_name}'.")
    else: epi_data_output["calculation_notes"].append("No conclusive test results for positivity trends.")
    epi_data_output["key_test_positivity_trends"] = test_pos_trends

    # Demographic Breakdown
    demo_output = epi_data_output["demographics_by_condition_data"]
    df_demo = df_epi_src.copy()
    if condition_filter_for_demographics != "All Conditions":
        df_demo = df_epi_src[df_epi_src['condition'].astype(str).str.contains(re.escape(condition_filter_for_demographics), case=False, na=False, regex=True)]
    if not df_demo.empty and 'patient_id' in df_demo.columns:
        df_unique_patients_demo = df_demo.drop_duplicates(subset=['patient_id'])
        if not df_unique_patients_demo.empty:
            if 'age' in df_unique_patients_demo.columns and df_unique_patients_demo['age'].notna().any():
                age_bins_cfg = [0, settings.AGE_THRESHOLD_LOW, settings.AGE_THRESHOLD_MODERATE, settings.AGE_THRESHOLD_HIGH, settings.AGE_THRESHOLD_VERY_HIGH, np.inf]
                age_lbls_cfg = [f'0-{settings.AGE_THRESHOLD_LOW-1}', f'{settings.AGE_THRESHOLD_LOW}-{settings.AGE_THRESHOLD_MODERATE-1}', f'{settings.AGE_THRESHOLD_MODERATE}-{settings.AGE_THRESHOLD_HIGH-1}', f'{settings.AGE_THRESHOLD_HIGH}-{settings.AGE_THRESHOLD_VERY_HIGH-1}', f'{settings.AGE_THRESHOLD_VERY_HIGH}+']
                age_dist_df = pd.cut(df_unique_patients_demo['age'], bins=age_bins_cfg, labels=age_lbls_cfg, right=False, include_lowest=True).value_counts().sort_index().reset_index()
                age_dist_df.columns = default_demo_age_cols; demo_output["age_distribution_df"] = age_dist_df
            if 'gender' in df_unique_patients_demo.columns and df_unique_patients_demo['gender'].notna().any():
                gender_map_norm = lambda g: "Male" if str(g).lower() in ['m', 'male'] else ("Female" if str(g).lower() in ['f', 'female'] else "Other/Unknown")
                gender_dist_df = df_unique_patients_demo['gender'].apply(gender_map_norm).value_counts().reset_index()
                gender_dist_df.columns = default_demo_gender_cols; demo_output["gender_distribution_df"] = gender_dist_df[gender_dist_df['Gender'].isin(["Male", "Female"])] # Filter for M/F
        else: epi_data_output["calculation_notes"].append(f"No unique patients for condition '{condition_filter_for_demographics}' for demographics.")
    elif 'patient_id' not in df_epi_src.columns: epi_data_output["calculation_notes"].append("'patient_id' missing for demographic breakdown.")
    else: epi_data_output["calculation_notes"].append(f"No data for condition '{condition_filter_for_demographics}' for demographics.")

    # Referral Funnel
    if 'referral_status' in df_epi_src.columns and 'encounter_id' in df_epi_src.columns:
        df_referrals = df_epi_src[df_epi_src['referral_status'].astype(str).str.lower() != 'ustat'].copy() # Exclude default unknown
        if not df_referrals.empty:
            total_refs = df_referrals['encounter_id'].nunique()
            pos_outcomes = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended', 'admitted', 'treatment complete']
            pos_concluded_refs = 0
            if 'referral_outcome' in df_referrals.columns:
                pos_concluded_refs = df_referrals[df_referrals['referral_outcome'].astype(str).str.lower().isin(pos_outcomes)]['encounter_id'].nunique()
            pending_refs = df_referrals[df_referrals['referral_status'].astype(str).str.lower() == 'pending']['encounter_id'].nunique()
            funnel_data = [{'Stage': 'Total Referrals Made/Active (in Period)', 'Count': total_refs},
                           {'Stage': 'Concluded Positively (Outcome Known)', 'Count': pos_concluded_refs},
                           {'Stage': 'Currently Pending (Status "Pending")', 'Count': pending_refs}]
            funnel_df = pd.DataFrame(funnel_data)
            epi_data_output["referral_funnel_summary_df"] = funnel_df[funnel_df['Count'] > 0].reset_index(drop=True)
        else: epi_data_output["calculation_notes"].append("No actionable referral records for funnel analysis.")
    else: epi_data_output["calculation_notes"].append("Referral status or encounter ID missing, skipping referral funnel.")
    
    logger.info(f"({module_log_prefix}) Clinic epi data calculation finished. Notes: {len(epi_data_output['calculation_notes'])}")
    return epi_data_output
