# sentinel_project_root/pages/clinic_components/epi_data.py
# Calculates clinic-level epidemiological data for Sentinel Health Co-Pilot.
# Renamed from epi_data_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
from data_processing.aggregation import get_trend_data # For time-series trends
from data_processing.helpers import convert_to_numeric # For data cleaning

logger = logging.getLogger(__name__)


def calculate_clinic_epidemiological_data( # Renamed function
    filtered_health_df_clinic_period: Optional[pd.DataFrame], # Data already filtered for clinic and period
    reporting_period_context_str: str, # Renamed for clarity
    condition_filter_for_demographics: str = "All Conditions", # Renamed, default for demographics tab
    num_top_symptoms_to_trend: int = 5 # Reduced default for clearer visualization
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.
    Includes symptom trends, test positivity trends, demographic breakdowns, and referral funnel.
    """
    module_log_prefix = "ClinicEpiDataCalc"
    logger.info(f"({module_log_prefix}) Calculating clinic epi data. Period: {reporting_period_context_str}, Demo Cond: {condition_filter_for_demographics}")

    # Initialize output structure with defaults, especially for DataFrames for consistent schema
    epi_data_output_dict: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "symptom_trends_weekly_top_n_df": pd.DataFrame(columns=['week_start_date', 'symptom', 'count']),
        "key_test_positivity_trends": {},       # Dict: {test_display_name: pd.Series(pos_rate_pct)}
        "demographics_by_condition_data": {     # Nested dict for demographics
            "age_distribution_df": pd.DataFrame(columns=['Age Group', 'Patient Count']),
            "gender_distribution_df": pd.DataFrame(columns=['Gender', 'Patient Count']),
            "condition_analyzed": condition_filter_for_demographics # Store which condition was used
        },
        "referral_funnel_summary_df": pd.DataFrame(columns=['Stage', 'Count']),
        "calculation_notes": []                 # List to store notes on data availability/issues
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        msg = "No health data provided for clinic epidemiological analysis. Calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output_dict["calculation_notes"].append(msg)
        return epi_data_output_dict

    df_epi_src_cleaned = filtered_health_df_clinic_period.copy() # Work on a copy

    # --- Data Preparation and Validation ---
    # 1. Ensure 'encounter_date' exists and is valid datetime
    if 'encounter_date' not in df_epi_src_cleaned.columns:
        msg = "'encounter_date' column missing. Critical for epidemiological calculations."
        logger.error(f"({module_log_prefix}) {msg}")
        epi_data_output_dict["calculation_notes"].append(msg); return epi_data_output_dict
        
    try:
        df_epi_src_cleaned['encounter_date'] = pd.to_datetime(df_epi_src_cleaned['encounter_date'], errors='coerce')
        df_epi_src_cleaned.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed
    except Exception as e_date_conv_epi:
        logger.error(f"({module_log_prefix}) Error converting 'encounter_date' column for epi calcs: {e_date_conv_epi}")
        epi_data_output_dict["calculation_notes"].append("Error processing encounter dates for epi analysis.")
        return epi_data_output_dict

    if df_epi_src_cleaned.empty:
        msg = "No valid encounter dates found in data after cleaning for epi calculations."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output_dict["calculation_notes"].append(msg); return epi_data_output_dict

    # 2. Ensure other essential columns exist with safe defaults
    essential_cols_config_epi = {
        'patient_id': {"default": f"UnknownPID_Epi_{reporting_period_context_str[:10]}", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str}, # Default to empty string
        'condition': {"default": "UnknownCondition", "type": str},
        'test_type': {"default": "UnknownTest", "type": str},
        'test_result': {"default": "UnknownResult", "type": str},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_status': {"default": "Unknown", "type": str},
        'referral_outcome': {"default": "Unknown", "type": str},
        'encounter_id': {"default": f"UnknownEncID_Epi_{reporting_period_context_str[:10]}", "type": str}
    }
    common_na_values_epi_calc = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown']

    for col, config_item_epi in essential_cols_config_epi.items():
        if col not in df_epi_src_cleaned.columns:
            df_epi_src_cleaned[col] = config_item_epi["default"]
        
        if config_item_epi["type"] == float:
            df_epi_src_cleaned[col] = convert_to_numeric(df_epi_src_cleaned[col], default_value=config_item_epi["default"])
        elif config_item_epi["type"] == str:
            df_epi_src_cleaned[col] = df_epi_src_cleaned[col].astype(str).fillna(str(config_item_epi["default"]))
            df_epi_src_cleaned[col] = df_epi_src_cleaned[col].replace(common_na_values_epi_calc, str(config_item_epi["default"]), regex=False).str.strip()

    # --- 1. Symptom Trends (Weekly Top N Symptoms) ---
    if 'patient_reported_symptoms' in df_epi_src_cleaned.columns and \
       df_epi_src_cleaned['patient_reported_symptoms'].str.strip().astype(bool).any(): # Check if any non-empty symptom strings
        
        df_symptoms_analysis_base = df_epi_src_cleaned[['encounter_date', 'patient_reported_symptoms']].copy()
        # Filter out non-informative symptom entries before splitting/exploding
        non_informative_symptoms_list_lower = ["unknown", "n/a", "none", "", " ", "no symptoms", "asymptomatic", "well", "routine", "follow up"]
        df_symptoms_analysis_base = df_symptoms_analysis_base[
            ~df_symptoms_analysis_base['patient_reported_symptoms'].astype(str).str.lower().isin(non_informative_symptoms_list_lower)
        ]
        df_symptoms_analysis_base.dropna(subset=['patient_reported_symptoms'], inplace=True)


        if not df_symptoms_analysis_base.empty:
            # Explode symptoms (assuming semicolon delimiter, make configurable if needed)
            # And standardize: strip whitespace, title case each symptom
            symptoms_exploded_for_trend = df_symptoms_analysis_base.assign(
                symptom_single=df_symptoms_analysis_base['patient_reported_symptoms'].str.split(r'[;,|]') # Split by common delimiters
            ).explode('symptom_single')
            symptoms_exploded_for_trend['symptom_single'] = symptoms_exploded_for_trend['symptom_single'].astype(str).str.strip().str.title()
            symptoms_exploded_for_trend.dropna(subset=['symptom_single'], inplace=True)
            symptoms_exploded_for_trend = symptoms_exploded_for_trend[symptoms_exploded_for_trend['symptom_single'] != ''] # Remove empty strings post-split

            if not symptoms_exploded_for_trend.empty:
                # Determine top N most frequent symptoms in the period
                top_n_symptom_names = symptoms_exploded_for_trend['symptom_single'].value_counts().nlargest(num_top_symptoms_to_trend).index.tolist()
                
                df_top_symptoms_filtered_for_trend = symptoms_exploded_for_trend[symptoms_exploded_for_trend['symptom_single'].isin(top_n_symptom_names)]

                if not df_top_symptoms_filtered_for_trend.empty:
                    # Group by week (starting Monday) and symptom to count occurrences
                    df_weekly_symptom_counts_result = df_top_symptoms_filtered_for_trend.groupby(
                        [pd.Grouper(key='encounter_date', freq='W-MON', label='left', closed='left'), 'symptom_single']
                    ).size().reset_index(name='count')
                    df_weekly_symptom_counts_result.rename(columns={'encounter_date': 'week_start_date', 'symptom_single':'symptom'}, inplace=True)
                    epi_data_output_dict["symptom_trends_weekly_top_n_df"] = df_weekly_symptom_counts_result
                else:
                    epi_data_output_dict["calculation_notes"].append(f"Not enough diverse symptom data after filtering for top {num_top_symptoms_to_trend} symptoms.")
            else:
                epi_data_output_dict["calculation_notes"].append("No valid individual symptoms found after cleaning/exploding for trend analysis.")
        else:
            epi_data_output_dict["calculation_notes"].append("No actionable patient-reported symptoms data found for trends after filtering non-informative entries.")
    else:
        epi_data_output_dict["calculation_notes"].append("'patient_reported_symptoms' column missing or empty. Symptom trends skipped.")


    # --- 2. Test Positivity Rate Trends (Weekly for Key Tests) ---
    test_positivity_trends_map: Dict[str, pd.Series] = {}
    # Define conclusive results (excluding pending, rejected, etc.) to ensure accurate rate calculation
    conclusive_test_results_mask = ~df_epi_src_cleaned.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(
        ["pending", "rejected", "unknownresult", "indeterminate", "n/a", ""] # 'rejected sample' might be 'rejected'
    )
    df_conclusive_tests_for_positivity = df_epi_src_cleaned[conclusive_test_results_mask].copy()

    if not df_conclusive_tests_for_positivity.empty:
        for test_original_config_key, test_config_details in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items():
            test_display_name_output = test_config_details.get("display_name", test_original_config_key)
            
            # Filter for the specific test type using the original key from config
            df_specific_test_conclusive = df_conclusive_tests_for_positivity[
                df_conclusive_tests_for_positivity['test_type'] == test_original_config_key
            ]
            
            if not df_specific_test_conclusive.empty:
                # Create a boolean flag for positive results
                df_specific_test_conclusive['is_positive_flag'] = (
                    df_specific_test_conclusive['test_result'].astype(str).str.lower() == 'positive'
                )
                
                # Use get_trend_data: mean of boolean (0/1) gives the proportion (rate)
                weekly_positivity_rate_series = get_trend_data(
                    df=df_specific_test_conclusive,
                    value_col='is_positive_flag',
                    date_col='encounter_date',
                    period='W-MON', # Weekly, starting Monday
                    agg_func='mean', 
                    source_context=f"{module_log_prefix}/PositivityTrend/{test_display_name_output}"
                )
                if isinstance(weekly_positivity_rate_series, pd.Series) and not weekly_positivity_rate_series.empty:
                    test_positivity_trends_map[test_display_name_output] = (weekly_positivity_rate_series * 100).round(1) # Store as percentage
                else:
                    epi_data_output_dict["calculation_notes"].append(f"No aggregated weekly positivity trend data for {test_display_name_output} (empty series).")
            else:
                epi_data_output_dict["calculation_notes"].append(f"No conclusive test data found for '{test_display_name_output}' in the period for positivity trend.")
    else:
        epi_data_output_dict["calculation_notes"].append("No conclusive test results found in the period for any test positivity trends.")
    epi_data_output_dict["key_test_positivity_trends"] = test_positivity_trends_map


    # --- 3. Demographic Breakdown for Selected Condition ---
    demographics_output_sub_dict = epi_data_output_dict["demographics_by_condition_data"] # Reference for easier access
    
    df_for_demographics_analysis = df_epi_src_cleaned.copy()
    if condition_filter_for_demographics != "All Conditions": # If a specific condition is chosen
        df_for_demographics_analysis = df_epi_src_cleaned[
            df_epi_src_cleaned['condition'].astype(str).str.contains(condition_filter_for_demographics, case=False, na=False)
        ]

    if not df_for_demographics_analysis.empty and 'patient_id' in df_for_demographics_analysis.columns:
        # Use unique patients for demographic breakdown
        df_unique_patients_for_demographics = df_for_demographics_analysis.drop_duplicates(subset=['patient_id'])
        
        if not df_unique_patients_for_demographics.empty:
            # Age breakdown
            if 'age' in df_unique_patients_for_demographics.columns and df_unique_patients_for_demographics['age'].notna().any():
                age_bins_config = [0, settings.AGE_THRESHOLD_LOW, settings.AGE_THRESHOLD_MODERATE, 
                                   settings.AGE_THRESHOLD_HIGH, settings.AGE_THRESHOLD_VERY_HIGH, np.inf]
                age_labels_config = [f'0-{settings.AGE_THRESHOLD_LOW-1}', f'{settings.AGE_THRESHOLD_LOW}-{settings.AGE_THRESHOLD_MODERATE-1}', 
                                     f'{settings.AGE_THRESHOLD_MODERATE}-{settings.AGE_THRESHOLD_HIGH-1}', 
                                     f'{settings.AGE_THRESHOLD_HIGH}-{settings.AGE_THRESHOLD_VERY_HIGH-1}', 
                                     f'{settings.AGE_THRESHOLD_VERY_HIGH}+']
                
                temp_age_df_for_cut = df_unique_patients_for_demographics.copy() # Avoid SettingWithCopyWarning
                temp_age_df_for_cut['age_group_display_val'] = pd.cut(
                    temp_age_df_for_cut['age'], # Already numeric from prep
                    bins=age_bins_config, labels=age_labels_config, right=False, include_lowest=True
                )
                age_distribution_df_result = temp_age_df_for_cut['age_group_display_val'].value_counts().sort_index().reset_index()
                age_distribution_df_result.columns = ['Age Group', 'Patient Count']
                demographics_output_sub_dict["age_distribution_df"] = age_distribution_df_result
            
            # Gender breakdown
            if 'gender' in df_unique_patients_for_demographics.columns and df_unique_patients_for_demographics['gender'].notna().any():
                temp_gender_df_for_norm = df_unique_patients_for_demographics.copy()
                # Normalize gender values for consistent grouping
                gender_map_normalize = lambda g_val_str: "Male" if str(g_val_str).strip().lower() in ['m', 'male'] else \
                                                          "Female" if str(g_val_str).strip().lower() in ['f', 'female'] else "Other/Unknown"
                temp_gender_df_for_norm['gender_normalized_display_val'] = temp_gender_df_for_norm['gender'].apply(gender_map_normalize)
                
                gender_distribution_df_result = temp_gender_df_for_norm[ # Filter for common categories if desired
                    temp_gender_df_for_norm['gender_normalized_display_val'].isin(["Male", "Female"])
                ]['gender_normalized_display_val'].value_counts().reset_index()
                gender_distribution_df_result.columns = ['Gender', 'Patient Count']
                demographics_output_sub_dict["gender_distribution_df"] = gender_distribution_df_result
        else:
            epi_data_output_dict["calculation_notes"].append(f"No unique patients found for condition '{condition_filter_for_demographics}' for demographic breakdown.")
    elif 'patient_id' not in df_epi_src_cleaned.columns:
        epi_data_output_dict["calculation_notes"].append("'patient_id' column missing from source data for demographic breakdown.")
    else: # df_for_demographics_analysis was empty
        epi_data_output_dict["calculation_notes"].append(f"No patient data found matching condition '{condition_filter_for_demographics}' for demographic analysis.")


    # --- 4. Referral Funnel Analysis (Simplified) ---
    if 'referral_status' in df_epi_src_cleaned.columns and 'encounter_id' in df_epi_src_cleaned.columns:
        # Consider referrals made or active within the period
        df_referrals_active_in_period = df_epi_src_cleaned[
            df_epi_src_cleaned['referral_status'].astype(str).str.lower() != 'unknown' # Exclude completely unknown
        ].copy()

        if not df_referrals_active_in_period.empty:
            total_referral_events_in_period = df_referrals_active_in_period['encounter_id'].nunique() # Unique encounters involving a referral
            
            # Example of positively concluded outcomes (can be expanded based on actual data values)
            positively_concluded_outcomes_list_lower = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended', 'admitted', 'treatment complete']
            
            # Count referrals with a known positive outcome (using referral_outcome column)
            count_positively_concluded_referrals = 0
            if 'referral_outcome' in df_referrals_active_in_period.columns:
                count_positively_concluded_referrals = df_referrals_active_in_period[
                    df_referrals_active_in_period['referral_outcome'].astype(str).str.lower().isin(positively_concluded_outcomes_list_lower)
                ]['encounter_id'].nunique()
            
            count_referrals_still_pending = df_referrals_active_in_period[
                df_referrals_active_in_period['referral_status'].astype(str).str.lower() == 'pending'
            ]['encounter_id'].nunique()
            
            referral_funnel_data_list = [
                {'Stage': 'Total Referrals Made/Active (in Period)', 'Count': total_referral_events_in_period},
                {'Stage': 'Concluded Positively (Outcome Known)', 'Count': count_positively_concluded_referrals},
                {'Stage': 'Currently Pending (Status "Pending")', 'Count': count_referrals_still_pending},
            ]
            funnel_summary_df_result = pd.DataFrame(referral_funnel_data_list)
            # Only show stages with non-zero counts for cleaner display in UI
            epi_data_output_dict["referral_funnel_summary_df"] = funnel_summary_df_result[funnel_summary_df_result['Count'] > 0].reset_index(drop=True)
        else:
            epi_data_output_dict["calculation_notes"].append("No actionable referral records found for funnel analysis in the period.")
    else:
        epi_data_output_dict["calculation_notes"].append("Referral status or encounter ID data missing, skipping referral funnel analysis.")
    
    logger.info(f"({module_log_prefix}) Clinic epi data calculation finished. Notes recorded: {len(epi_data_output_dict['calculation_notes'])}")
    return epi_data_output_dict
