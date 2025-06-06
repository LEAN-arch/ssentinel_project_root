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
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in epi_data.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_EPI = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_EPI_PATTERN = r'^\s*$' + (r'|^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_EPI if s) + r')$' if any(COMMON_NA_STRINGS_EPI) else '')

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


def _prepare_epi_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    default_patient_id_prefix: str
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
                    df_prepared[col_name].replace(NA_REGEX_EPI_PATTERN, np.nan, regex=True, inplace=True)
                except Exception as e_regex:
                     logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=float)
            elif target_type_str == str:
                series = df_prepared[col_name].fillna(str(default_value))
                df_prepared[col_name] = series.astype(str).str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        # CORRECTED: Use the correct variable name passed into the function.
        df_prepared['patient_id'].replace('', default_patient_id_prefix, inplace=True)
        df_prepared['patient_id'].fillna(default_patient_id_prefix, inplace=True)
    return df_prepared


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
    logger.info(f"({module_log_prefix}) Calculating clinic epi data. Period Context: {reporting_period_context_str}, Demo Cond: {condition_filter_for_demographics}")

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
        msg = "No health data provided for clinic epidemiological analysis. All calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    df_epi_src_raw = filtered_health_df_clinic_period.copy()

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
        df_epi_src_raw.dropna(subset=[date_col], inplace=True)
    except Exception as e_date_epi:
        logger.error(f"({module_log_prefix}) Error processing '{date_col}': {e_date_epi}", exc_info=True)
        epi_data_output["calculation_notes"].append(f"Error processing '{date_col}'. Calculations may be incomplete.")
        return epi_data_output
    
    if df_epi_src_raw.empty:
        msg = "No records with valid encounter dates after cleaning. Epi calculations skipped."
        logger.warning(f"({module_log_prefix}) {msg}")
        epi_data_output["calculation_notes"].append(msg)
        return epi_data_output

    pid_prefix = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15]
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
        'encounter_id': {"default": f"UEID_Epi_{pid_prefix}", "type": str}
    }
    df_epi_src = _prepare_epi_dataframe(df_epi_src_raw, epi_cols_cfg, module_log_prefix, f"UPID_Epi_{pid_prefix}")
    
    symptoms_col = 'patient_reported_symptoms'
    if symptoms_col in df_epi_src.columns and df_epi_src[symptoms_col].astype(str).str.strip().astype(bool).any():
        try:
            df_symptoms_base = df_epi_src[[date_col, symptoms_col]].dropna(subset=[symptoms_col]).copy()
            non_info_symptoms_list = _get_setting('NON_INFORMATIVE_SYMPTOMS', 
                ["unknown", "n/a", "none", "", "no symptoms", "asymptomatic", "well", "routine", "follow up"]
            )
            df_symptoms_base = df_symptoms_base[~df_symptoms_base[symptoms_col].str.lower().str.strip().isin(non_info_symptoms_list)]
            
            if not df_symptoms_base.empty:
                symptoms_exploded_df = df_symptoms_base.assign(symptom=df_symptoms_base[symptoms_col].str.split(r'[;,|]')).explode('symptom')
                symptoms_exploded_df['symptom'] = symptoms_exploded_df['symptom'].str.strip().str.title()
                symptoms_exploded_df = symptoms_exploded_df[symptoms_exploded_df['symptom'] != '']
                
                if not symptoms_exploded_df.empty:
                    top_n = _get_setting('NUM_TOP_SYMPTOMS_TO_TREND_CLINIC', num_top_symptoms_to_trend)
                    top_symptoms = symptoms_exploded_df['symptom'].value_counts().nlargest(top_n).index
                    
                    if not top_symptoms.empty:
                        df_top_symptoms = symptoms_exploded_df[symptoms_exploded_df['symptom'].isin(top_symptoms)]
                        weekly_counts = df_top_symptoms.groupby([pd.Grouper(key=date_col, freq='W-MON', label='left', closed='left'), 'symptom']).size().reset_index(name='count')
                        weekly_counts.rename(columns={date_col: 'week_start_date'}, inplace=True)
                        epi_data_output["symptom_trends_weekly_top_n_df"] = weekly_counts
        except Exception as e_symp_trend:
            logger.error(f"({module_log_prefix}) Error processing symptom trends: {e_symp_trend}", exc_info=True)
            epi_data_output["calculation_notes"].append("Failed to generate symptom trends.")

    key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
    test_positivity_trends_map: Dict[str, pd.Series] = {}
    non_conclusive = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', ["pending", "rejected", "unknown"])
    
    if 'test_result' in df_epi_src.columns and 'test_type' in df_epi_src.columns and isinstance(key_test_configs, dict):
        df_conclusive = df_epi_src[~df_epi_src['test_result'].str.lower().isin(non_conclusive)].copy()

        if not df_conclusive.empty:
            for internal_name, config in key_test_configs.items():
                display_name = config.get("display_name", internal_name)
                df_test = df_conclusive[df_conclusive['test_type'] == internal_name]
                if not df_test.empty:
                    try:
                        df_test['is_positive'] = (df_test['test_result'].str.lower() == 'positive').astype(float)
                        trend = get_trend_data(df=df_test, value_col='is_positive', date_col=date_col, period='W-MON', agg_func='mean', source_context=f"{module_log_prefix}/{display_name}")
                        if not trend.empty:
                            test_positivity_trends_map[display_name] = (trend * 100).round(1)
                    except Exception as e: logger.error(f"Error calculating positivity for {display_name}: {e}")
    epi_data_output["key_test_positivity_trends"] = test_positivity_trends_map

    df_for_demographics = df_epi_src.drop_duplicates(subset=['patient_id']) if 'patient_id' in df_epi_src.columns else pd.DataFrame()
    if not df_for_demographics.empty:
        if 'age' in df_for_demographics.columns:
            try:
                age_bins = [0, 5, 18, 60, 75, np.inf]
                age_labels = ['0-4', '5-17', '18-59', '60-74', '75+']
                age_series = pd.cut(df_for_demographics['age'].dropna(), bins=age_bins, labels=age_labels, right=False).value_counts().sort_index()
                epi_data_output["demographics_by_condition_data"]["age_distribution_df"] = age_series.reset_index().rename(columns={'index': default_demo_age_cols[0], 'age': default_demo_age_cols[1]})
            except Exception as e: logger.error(f"Error calculating age distribution: {e}")
        
        if 'gender' in df_for_demographics.columns:
            try:
                gender_series = df_for_demographics['gender'].str.title().value_counts()
                gender_df = gender_series.reset_index().rename(columns={'index': default_demo_gender_cols[0], 'gender': default_demo_gender_cols[1]})
                epi_data_output["demographics_by_condition_data"]["gender_distribution_df"] = gender_df[gender_df[default_demo_gender_cols[0]].isin(["Male", "Female"])]
            except Exception as e: logger.error(f"Error calculating gender distribution: {e}")
    
    if 'referral_status' in df_epi_src.columns:
        referral_df = df_epi_src.dropna(subset=['referral_status'])
        if not referral_df.empty:
            total_referrals = referral_df['encounter_id'].nunique()
            completed = referral_df[referral_df['referral_status'].str.lower() == 'completed']['encounter_id'].nunique()
            pending = referral_df[referral_df['referral_status'].str.lower() == 'pending']['encounter_id'].nunique()
            funnel_data = [{'Stage': 'Total Referrals', 'Count': total_referrals}, {'Stage': 'Completed', 'Count': completed}, {'Stage': 'Pending', 'Count': pending}]
            epi_data_output["referral_funnel_summary_df"] = pd.DataFrame(funnel_data)

    logger.info(f"({module_log_prefix}) Clinic epi data calculation finished. Notes: {len(epi_data_output['calculation_notes'])}")
    return epi_data_output
