# sentinel_project_root/pages/clinic_components/epi_data.py
# Calculates clinic-level epidemiological data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

try:
    from config import settings
    from data_processing.aggregation import get_trend_data # Ensure this is robust
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in epi_data.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_EPI = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_EPI_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_EPI if s) + r')$' if COMMON_NA_STRINGS_EPI else None

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


def _prepare_epi_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    default_patient_id_prefix: str # Used if patient_id is missing
) -> pd.DataFrame:
    """Prepares the DataFrame for epi signal extraction."""
    df_prepared = df.copy()
    for col_name, config in cols_config.items():
        default_value = config.get("default")
        target_type_str = config.get("type")

        if col_name not in df_prepared.columns:
            if col_name == 'patient_id': default_value = default_patient_id_prefix
            
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            elif isinstance(default_value, (list, dict)): 
                 df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value
        
        current_col_dtype = df_prepared[col_name].dtype
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(current_col_dtype):
            if NA_REGEX_EPI_PATTERN:
                try:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_EPI_PATTERN, np.nan, regex=True)
                except Exception as e_regex:
                     logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value)
            # No explicit int conversion here, convert_to_numeric handles float mostly for epi metrics
            elif target_type_str == str:
                df_prepared[col_name] = df_prepared[col_name].fillna(str(default_value)).astype(str)
                if NA_REGEX_EPI_PATTERN:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_EPI_PATTERN, str(default_value), regex=True)
                df_prepared[col_name] = df_prepared[col_name].str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', default_patient_id_prefix).fillna(default_patient_id_prefix)
    return df_prepared


def calculate_clinic_epidemiological_data(
    filtered_health_df_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str, # For logging and context
    condition_filter_for_demographics: str = "All Conditions", # Filter for demographic breakdown
    num_top_symptoms_to_trend: int = 5 # Configurable number of top symptoms
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.
    """
    module_log_prefix = "ClinicEpiDataCalc"
    
    # Robust date parsing for reporting_period_context_str if it's used for date ops
    # For now, assuming it's just a string label. If dates are needed from it, parse carefully.
    logger.info(f"({module_log_prefix}) Calculating clinic epi data. Period Context: {reporting_period_context_str}, Demo Cond: {condition_filter_for_demographics}")

    # Initialize output with default empty structures and correct column names
    default_symptom_trend_cols = ['week_start_date', 'symptom', 'count']
    default_demo_age_cols = ['Age Group', 'Patient Count']
    default_demo_gender_cols = ['Gender', 'Patient Count']
    default_referral_cols = ['Stage', 'Count']

    epi_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "symptom_trends_weekly_top_n_df": pd.DataFrame(columns=default_symptom_trend_cols),
        "key_test_positivity_trends": {}, # Dict of {test_name: pd.Series}
        "demographics_by_condition_data": {
            "age_distribution_df": pd.DataFrame(columns=default_demo_age_cols),
            "gender_distribution_df": pd.DataFrame(columns=default_demo_gender_cols),
            "condition_analyzed": condition_filter_for_demographics # Store which condition was used
        },
        "referral_funnel_summary_df": pd.DataFrame(columns=default_referral_cols),
        "calculation_notes": []
    }

    if not isinstance(filtered_health_df_clinic_period, pd.DataFrame) or filtered_health_df_clinic_period.empty:
        msg = "No health data provided for clinic epidemiological analysis. All calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    df_epi_src_raw = filtered_health_df_clinic_period.copy() # Work on a copy

    # --- Data Preparation ---
    # Ensure encounter_date is present and valid first, as it's critical for filtering
    date_col = 'encounter_date'
    if date_col not in df_epi_src_raw.columns:
        msg = f"Critical: '{date_col}' column missing. Cannot perform epidemiological calculations."
        logger.error(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_epi_src_raw[date_col]):
            df_epi_src_raw[date_col] = pd.to_datetime(df_epi_src_raw[date_col], errors='coerce')
        if df_epi_src_raw[date_col].dt.tz is not None:
            df_epi_src_raw[date_col] = df_epi_src_raw[date_col].dt.tz_localize(None)
        df_epi_src_raw.dropna(subset=[date_col], inplace=True) # Remove rows with invalid dates
    except Exception as e_date_epi:
        logger.error(f"({module_log_prefix}) Error processing '{date_col}': {e_date_epi}", exc_info=True)
        epi_data_output["calculation_notes"].append(f"Error processing '{date_col}'. Calculations may be incomplete.")
        return epi_data_output # Stop if date processing fails fundamentally
    
    if df_epi_src_raw.empty:
        msg = "No records with valid encounter dates after cleaning. Epi calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    # Define configuration for other essential columns and prepare the DataFrame
    # Using reporting_period_context_str for default PID uniqueness if needed
    pid_prefix = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15] # Create a somewhat unique prefix
    epi_cols_cfg = {
        'patient_id': {"default": f"UPID_Epi_{pid_prefix}", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str}, 
        'condition': {"default": "UnknownCondition", "type": str},
        'test_type': {"default": "UnknownTest", "type": str}, 
        'test_result': {"default": "UnknownResult", "type": str},
        'age': {"default": np.nan, "type": float}, 
        'gender': {"default": "UnknownGender", "type": str},
        'referral_status': {"default": "UnknownStatus", "type": str}, 
        'referral_outcome': {"default": "UnknownOutcome", "type": str},
        'encounter_id': {"default": f"UEID_Epi_{pid_prefix}", "type": str} # If encounter_id is used as unique event
    }
    df_epi_src = _prepare_epi_dataframe(df_epi_src_raw, epi_cols_cfg, module_log_prefix, f"UPID_Epi_{pid_prefix}")
    
    # --- Symptom Trends ---
    symptoms_col = 'patient_reported_symptoms'
    if symptoms_col in df_epi_src.columns and df_epi_src[symptoms_col].astype(str).str.strip().astype(bool).any():
        try:
            df_symptoms_base = df_epi_src[[date_col, symptoms_col]].dropna(subset=[symptoms_col]).copy()
            # Define non-informative symptoms, make configurable via settings if needed
            non_info_symptoms_list = _get_setting('NON_INFORMATIVE_SYMPTOMS', 
                ["unknown", "n/a", "none", "", " ", "no symptoms", "asymptomatic", "well", "routine", "follow up", "negative", "clear"]
            )
            # Filter out non-informative symptoms (case-insensitive)
            df_symptoms_base = df_symptoms_base[
                ~df_symptoms_base[symptoms_col].astype(str).str.lower().str.strip().isin(non_info_symptoms_list)
            ]
            
            if not df_symptoms_base.empty:
                # Split multiple symptoms (assuming comma, semi-colon, or pipe separated) and explode
                symptoms_exploded_df = df_symptoms_base.assign(
                    symptom=df_symptoms_base[symptoms_col].str.split(r'[;,|]')
                ).explode('symptom')
                
                symptoms_exploded_df['symptom'] = symptoms_exploded_df['symptom'].astype(str).str.strip().str.title() # Clean and title case
                symptoms_exploded_df = symptoms_exploded_df[symptoms_exploded_df['symptom'] != ''] # Remove empty strings after split
                
                if not symptoms_exploded_df.empty:
                    top_n_symptoms_val = _get_setting('NUM_TOP_SYMPTOMS_TO_TREND_CLINIC', num_top_symptoms_to_trend)
                    top_symptoms_list = symptoms_exploded_df['symptom'].value_counts().nlargest(top_n_symptoms_val).index.tolist()
                    
                    if top_symptoms_list:
                        df_top_symptoms_data = symptoms_exploded_df[symptoms_exploded_df['symptom'].isin(top_symptoms_list)]
                        if not df_top_symptoms_data.empty:
                            # Aggregate weekly counts for these top symptoms
                            weekly_symptom_counts = df_top_symptoms_data.groupby([
                                pd.Grouper(key=date_col, freq='W-MON', label='left', closed='left'), 
                                'symptom'
                            ]).size().reset_index(name='count')
                            weekly_symptom_counts.rename(columns={date_col: 'week_start_date'}, inplace=True)
                            epi_data_output["symptom_trends_weekly_top_n_df"] = weekly_symptom_counts
                        else: epi_data_output["calculation_notes"].append(f"No data for top {top_n_symptoms_val} symptoms after filtering.")
                    else: epi_data_output["calculation_notes"].append("No top symptoms identified from the data.")
                else: epi_data_output["calculation_notes"].append("No valid individual symptoms after cleaning/exploding reported symptoms.")
            else: epi_data_output["calculation_notes"].append("No actionable symptoms data after filtering non-informative entries.")
        except Exception as e_symp_trend:
            logger.error(f"({module_log_prefix}) Error processing symptom trends: {e_symp_trend}", exc_info=True)
            epi_data_output["calculation_notes"].append("Failed to generate symptom trends.")
    else: epi_data_output["calculation_notes"].append(f"'{symptoms_col}' column missing or empty. Symptom trends skipped.")

    # --- Test Positivity Trends ---
    key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {}) # Expects a dict like {"TestInternalName": {"display_name": "Pretty Name"}}
    test_positivity_trends_map: Dict[str, pd.Series] = {}
    
    non_conclusive_test_results = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', ["pending", "rejected", "unknownresult", "indeterminate", "n/a", "", "invalid", "error"])
    
    # Ensure 'test_result' and 'test_type' columns exist
    if 'test_result' in df_epi_src.columns and 'test_type' in df_epi_src.columns and isinstance(key_test_configs, dict):
        df_conclusive_tests = df_epi_src[
            ~df_epi_src['test_result'].astype(str).str.lower().str.strip().isin(non_conclusive_test_results)
        ].copy() # Work with conclusive tests

        if not df_conclusive_tests.empty:
            for internal_test_name, test_config_details in key_test_configs.items():
                display_test_name = test_config_details.get("display_name", internal_test_name)
                df_specific_conclusive_test = df_conclusive_tests[df_conclusive_tests['test_type'] == internal_test_name]
                
                if not df_specific_conclusive_test.empty:
                    try:
                        # Ensure 'is_positive' is boolean for mean calculation
                        df_specific_conclusive_test['is_positive'] = (
                            df_specific_conclusive_test['test_result'].astype(str).str.lower().str.strip() == 'positive'
                        ).astype(float) # Convert True/False to 1.0/0.0 for mean

                        positivity_rate_trend_series = get_trend_data(
                            df=df_specific_conclusive_test, 
                            value_col='is_positive', 
                            date_col=date_col, 
                            period='W-MON', # Weekly, starting Monday
                            agg_func='mean', # Mean of 1s and 0s gives proportion
                            source_context=f"{module_log_prefix}/PositivityTrend/{display_test_name}"
                        )
                        if isinstance(positivity_rate_trend_series, pd.Series) and not positivity_rate_trend_series.empty:
                            test_positivity_trends_map[display_test_name] = (positivity_rate_trend_series * 100).round(1) # Convert to percentage
                        else: epi_data_output["calculation_notes"].append(f"No weekly positivity trend data for {display_test_name} (empty series or calculation error).")
                    except Exception as e_pos_trend:
                        logger.error(f"({module_log_prefix}) Error calculating positivity trend for {display_test_name}: {e_pos_trend}", exc_info=True)
                        epi_data_output["calculation_notes"].append(f"Failed to generate positivity trend for {display_test_name}.")
                else: epi_data_output["calculation_notes"].append(f"No conclusive test data found for '{display_test_name}'.")
        else: epi_data_output["calculation_notes"].append("No conclusive test results in the period for positivity trend calculation.")
    else: epi_data_output["calculation_notes"].append("'test_result' or 'test_type' columns missing, or KEY_TEST_TYPES_FOR_ANALYSIS not configured. Test positivity trends skipped.")
    epi_data_output["key_test_positivity_trends"] = test_positivity_trends_map

    # --- Demographic Breakdown by Condition ---
    demographics_output_dict = epi_data_output["demographics_by_condition_data"]
    df_for_demographics = df_epi_src.copy() # Use the prepared df_epi_src
    
    if condition_filter_for_demographics != "All Conditions":
        if 'condition' in df_for_demographics.columns:
            try:
                # Use regex for robust partial matching of the condition filter
                df_for_demographics = df_for_demographics[
                    df_for_demographics['condition'].astype(str).str.contains(re.escape(condition_filter_for_demographics), case=False, na=False, regex=True)
                ]
            except Exception as e_cond_filter_demo:
                logger.warning(f"({module_log_prefix}) Error applying condition filter '{condition_filter_for_demographics}' for demographics: {e_cond_filter_demo}. Using all data.")
                # Fallback to all data if filter fails, or keep df_for_demographics as is
        else:
            epi_data_output["calculation_notes"].append(f"'condition' column missing. Cannot filter demographics by '{condition_filter_for_demographics}'. Analyzing all conditions.")

    if not df_for_demographics.empty and 'patient_id' in df_for_demographics.columns:
        # Consider only unique patients for demographic breakdown
        df_unique_patients_for_demographics = df_for_demographics.drop_duplicates(subset=['patient_id'])
        
        if not df_unique_patients_for_demographics.empty:
            # Age Distribution
            if 'age' in df_unique_patients_for_demographics.columns and df_unique_patients_for_demographics['age'].notna().any():
                try:
                    # Get age thresholds from settings with robust fallbacks
                    age_bins = [0, 
                                _get_setting('AGE_THRESHOLD_LOW', 5), 
                                _get_setting('AGE_THRESHOLD_MODERATE', 18), 
                                _get_setting('AGE_THRESHOLD_HIGH', 60), 
                                _get_setting('AGE_THRESHOLD_VERY_HIGH', 75), 
                                np.inf]
                    age_labels = [f'0-{age_bins[1]-1}', 
                                  f'{age_bins[1]}-{age_bins[2]-1}', 
                                  f'{age_bins[2]}-{age_bins[3]-1}', 
                                  f'{age_bins[3]}-{age_bins[4]-1}', 
                                  f'{age_bins[4]}+']
                    
                    age_data_for_cut = convert_to_numeric(df_unique_patients_for_demographics['age'], np.nan).dropna()
                    if not age_data_for_cut.empty:
                        age_distribution_series = pd.cut(age_data_for_cut, bins=age_bins, labels=age_labels, right=False, include_lowest=True).value_counts().sort_index()
                        demographics_output_dict["age_distribution_df"] = age_distribution_series.reset_index().rename(columns={'index': default_demo_age_cols[0], 'age': default_demo_age_cols[1]})
                except Exception as e_age_dist:
                     logger.error(f"({module_log_prefix}) Error calculating age distribution: {e_age_dist}", exc_info=True)
                     epi_data_output["calculation_notes"].append("Failed to calculate age distribution.")
            
            # Gender Distribution
            if 'gender' in df_unique_patients_for_demographics.columns and df_unique_patients_for_demographics['gender'].notna().any():
                try:
                    def map_gender_robust(g: Any) -> str:
                        g_str = str(g).lower().strip()
                        if g_str in ['m', 'male']: return "Male"
                        if g_str in ['f', 'female']: return "Female"
                        return "Other/Unknown" # Group others
                    
                    gender_distribution_series = df_unique_patients_for_demographics['gender'].apply(map_gender_robust).value_counts()
                    gender_df = gender_distribution_series.reset_index().rename(columns={'index': default_demo_gender_cols[0], 'gender': default_demo_gender_cols[1]})
                    # Optionally filter to only show Male/Female or include Other/Unknown
                    demographics_output_dict["gender_distribution_df"] = gender_df[gender_df[default_demo_gender_cols[0]].isin(["Male", "Female"])]
                except Exception as e_gender_dist:
                    logger.error(f"({module_log_prefix}) Error calculating gender distribution: {e_gender_dist}", exc_info=True)
                    epi_data_output["calculation_notes"].append("Failed to calculate gender distribution.")
        else: epi_data_output["calculation_notes"].append(f"No unique patients found after filtering for condition '{condition_filter_for_demographics}'. Demographics skipped.")
    elif 'patient_id' not in df_epi_src.columns: epi_data_output["calculation_notes"].append("'patient_id' column missing. Demographic breakdown skipped.")
    else: epi_data_output["calculation_notes"].append(f"No data after filtering for condition '{condition_filter_for_demographics}'. Demographics skipped.")


    # --- Referral Funnel Summary ---
    # This requires unique encounter/referral event IDs for accurate counting. Using 'encounter_id' as proxy.
    referral_status_col = 'referral_status'
    referral_outcome_col = 'referral_outcome'
    unique_event_id_col = 'encounter_id' # Or a specific referral_id if available

    if referral_status_col in df_epi_src.columns and unique_event_id_col in df_epi_src.columns:
        # Consider only records that represent a referral event (e.g., referral_status is not default/unknown)
        df_referral_events = df_epi_src[
            ~df_epi_src[referral_status_col].astype(str).str.lower().isin(["unknownstatus", "ustat", "n/a", ""])
        ].copy()

        if not df_referral_events.empty:
            total_referrals_made = df_referral_events[unique_event_id_col].nunique()
            
            positive_outcome_strings = _get_setting('POSITIVE_REFERRAL_OUTCOMES', 
                ['completed', 'service provided', 'attended consult', 'attended followup', 'attended', 'admitted', 'treatment complete']
            )
            positively_concluded_referrals = 0
            if referral_outcome_col in df_referral_events.columns:
                positively_concluded_referrals = df_referral_events[
                    df_referral_events[referral_outcome_col].astype(str).str.lower().isin(positive_outcome_strings)
                ][unique_event_id_col].nunique()
            
            pending_referrals_count = df_referral_events[
                df_referral_events[referral_status_col].astype(str).str.lower() == 'pending'
            ][unique_event_id_col].nunique()

            funnel_data_list = [
                {'Stage': 'Total Referrals Active/Made (in Period)', 'Count': total_referrals_made},
                {'Stage': 'Concluded Positively (Outcome Known)', 'Count': positively_concluded_referrals},
                {'Stage': 'Currently Pending (Status "Pending")', 'Count': pending_referrals_count}
            ]
            funnel_summary_df = pd.DataFrame(funnel_data_list)
            # Filter out stages with zero count for cleaner display, unless all are zero
            if not (funnel_summary_df['Count'] == 0).all():
                epi_data_output["referral_funnel_summary_df"] = funnel_summary_df[funnel_summary_df['Count'] > 0].reset_index(drop=True)
            else:
                epi_data_output["referral_funnel_summary_df"] = funnel_summary_df # Show all zeros if that's the case
        else: epi_data_output["calculation_notes"].append("No actionable referral records found for funnel analysis.")
    else: epi_data_output["calculation_notes"].append(f"'{referral_status_col}' or '{unique_event_id_col}' columns missing. Referral funnel skipped.")
    
    logger.info(f"({module_log_prefix}) Clinic epi data calculation finished. Number of notes: {len(epi_data_output['calculation_notes'])}")
    return epi_data_output
