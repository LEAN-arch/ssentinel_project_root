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


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)


def _calculate_demographics(df_unique_patients: pd.DataFrame, log_prefix: str) -> Dict[str, pd.DataFrame]:
    """Helper to calculate age and gender distributions."""
    demographics_output = {"age_distribution_df": pd.DataFrame(), "gender_distribution_df": pd.DataFrame()}
    if 'age' in df_unique_patients:
        age_bins = [0, 5, 18, 60, 75, np.inf]; age_labels = ['0-4', '5-17', '18-59', '60-74', '75+']
        age_df = pd.cut(df_unique_patients['age'].dropna(), bins=age_bins, labels=age_labels, right=False).value_counts().sort_index().reset_index()
        age_df.columns = ['Age Group', 'Patient Count']; demographics_output["age_distribution_df"] = age_df
    if 'gender' in df_unique_patients:
        gender_df = df_unique_patients['gender'].str.title().value_counts().reset_index(); gender_df.columns = ['Gender', 'Patient Count']
        demographics_output["gender_distribution_df"] = gender_df[gender_df['Gender'].isin(["Male", "Female"])]
    return demographics_output


def calculate_clinic_epidemiological_data(
    filtered_health_df: Optional[pd.DataFrame],
    **kwargs
) -> Dict[str, Any]:
    """Calculates various epidemiological data sets for a clinic over a specified period."""
    module_log_prefix = "ClinicEpiDataCalc"
    epi_data_output: Dict[str, Any] = {
        "symptom_trends_weekly_top_n_df": pd.DataFrame(),
        "key_test_positivity_trends": {},
        "demographics_by_condition_data": {},
        "calculation_notes": []
    }

    if not isinstance(filtered_health_df, pd.DataFrame) or filtered_health_df.empty:
        epi_data_output["calculation_notes"].append("No health data provided.")
        return epi_data_output

    df_epi = filtered_health_df.copy()
    date_col = 'encounter_date'
    if date_col not in df_epi.columns:
        epi_data_output["calculation_notes"].append(f"'{date_col}' column missing.")
        return epi_data_output
    
    df_epi['encounter_date'] = pd.to_datetime(df_epi['encounter_date'], errors='coerce').dt.tz_localize(None)
    df_epi.dropna(subset=[date_col], inplace=True)
    
    # --- Symptom Trends ---
    symptoms_col = 'patient_reported_symptoms'
    if symptoms_col in df_epi.columns and df_epi[symptoms_col].astype(str).str.strip().any():
        non_info_symptoms = _get_setting('NON_INFORMATIVE_SYMPTOMS', ["unknown", "n/a", "none", ""])
        df_symptoms = df_epi[[date_col, symptoms_col]].copy()
        df_symptoms = df_symptoms[~df_symptoms[symptoms_col].str.lower().isin(non_info_symptoms)]
        if not df_symptoms.empty:
            exploded = df_symptoms.assign(symptom=df_symptoms[symptoms_col].str.split(r'[;,|]')).explode('symptom')
            exploded['symptom'] = exploded['symptom'].str.strip().str.title()
            top_symptoms = exploded['symptom'].value_counts().nlargest(5).index
            df_top = exploded[exploded['symptom'].isin(top_symptoms)]
            weekly_counts = df_top.groupby([pd.Grouper(key=date_col, freq='W-MON'), 'symptom']).size().reset_index(name='count')
            weekly_counts.rename(columns={date_col: 'week_start_date'}, inplace=True)
            epi_data_output["symptom_trends_weekly_top_n_df"] = weekly_counts

    # --- Test Positivity Trends ---
    key_tests = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
    if 'test_result' in df_epi.columns and 'test_type' in df_epi.columns and key_tests:
        non_conclusive = _get_setting('NON_CONCLUSIVE_TEST_RESULTS', ["pending", "rejected"])
        df_conclusive = df_epi[~df_epi['test_result'].str.lower().isin(non_conclusive)].copy()
        if not df_conclusive.empty:
            positivity_trends = {}
            for internal_name, config in key_tests.items():
                if not isinstance(config, dict): continue
                # FIXED: Create an explicit copy of the slice to avoid the SettingWithCopyWarning
                df_test = df_conclusive[df_conclusive['test_type'] == internal_name].copy()
                if not df_test.empty:
                    df_test['is_positive'] = (df_test['test_result'].str.lower() == 'positive').astype(float)
                    trend = get_trend_data(df=df_test, value_col='is_positive', date_col=date_col, period='W-MON', agg_func='mean')
                    if not trend.empty:
                        positivity_trends[config.get("display_name", internal_name)] = (trend * 100).round(1)
            epi_data_output["key_test_positivity_trends"] = positivity_trends

    # --- Demographic Breakdown ---
    if 'patient_id' in df_epi.columns:
        df_unique_patients = df_epi.drop_duplicates(subset=['patient_id'])
        if not df_unique_patients.empty:
            epi_data_output["demographics_by_condition_data"] = _calculate_demographics(df_unique_patients, module_log_prefix)
            
    return epi_data_output
