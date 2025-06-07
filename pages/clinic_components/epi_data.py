# sentinel_project_root/pages/clinic_components/epi_data.py
# Calculates clinic-level epidemiological data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

try:
    from config import settings
    from data_processing.aggregation import get_trend_data
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in epi_data.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# --- Constants for Clarity and Maintainability ---
COL_ENCOUNTER_DATE = 'encounter_date'
COL_SYMPTOMS = 'patient_reported_symptoms'
COL_PATIENT_ID = 'patient_id'
COL_AGE = 'age'
COL_GENDER = 'gender'
COL_TEST_TYPE = 'test_type'
COL_TEST_RESULT = 'test_result'

class ClinicEpiDataCalculator:
    """
    Encapsulates all logic for calculating epidemiological data for the clinic dashboard.
    This class structure improves maintainability, testability, and clarity.
    """
    def __init__(self, filtered_health_df: Optional[pd.DataFrame]):
        self.df_epi = self._validate_and_prepare_df(filtered_health_df)
        self.notes: List[str] = []
        self.module_log_prefix = self.__class__.__name__

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        """Helper to safely get attributes from settings."""
        return getattr(settings, attr_name, default_value)

    def _validate_and_prepare_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Validates the input DataFrame and performs initial cleaning."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        
        df_copy = df.copy()
        if COL_ENCOUNTER_DATE not in df_copy.columns:
            logger.error(f"'{COL_ENCOUNTER_DATE}' column missing from health data.")
            return pd.DataFrame()
        
        df_copy[COL_ENCOUNTER_DATE] = pd.to_datetime(df_copy[COL_ENCOUNTER_DATE], errors='coerce').dt.tz_localize(None)
        df_copy.dropna(subset=[COL_ENCOUNTER_DATE], inplace=True)
        return df_copy

    def _calculate_symptom_trends(self) -> pd.DataFrame:
        """Extracts and calculates weekly trends for the top N reported symptoms."""
        if COL_SYMPTOMS not in self.df_epi.columns or self.df_epi[COL_SYMPTOMS].astype(str).str.strip().eq('').all():
            return pd.DataFrame()
        
        non_info_symptoms = self._get_setting('NON_INFORMATIVE_SYMPTOMS', ["unknown", "n/a", "none", ""])
        df_symptoms = self.df_epi[[COL_ENCOUNTER_DATE, COL_SYMPTOMS]].dropna(subset=[COL_SYMPTOMS])
        df_symptoms = df_symptoms[~df_symptoms[COL_SYMPTOMS].str.lower().isin(non_info_symptoms)]
        
        if df_symptoms.empty:
            return pd.DataFrame()
            
        exploded = df_symptoms.assign(symptom=df_symptoms[COL_SYMPTOMS].str.split(r'[;,|]')).explode('symptom')
        exploded['symptom'] = exploded['symptom'].str.strip().str.title()
        exploded = exploded[exploded['symptom'] != '']
        
        top_n = self._get_setting('EPI_TOP_N_SYMPTOMS', 5)
        top_symptoms = exploded['symptom'].value_counts().nlargest(top_n).index
        df_top = exploded[exploded['symptom'].isin(top_symptoms)]
        
        weekly_counts = df_top.groupby([pd.Grouper(key=COL_ENCOUNTER_DATE, freq='W-MON'), 'symptom']).size().reset_index(name='count')
        return weekly_counts.rename(columns={COL_ENCOUNTER_DATE: 'week_start_date'})

    def _calculate_test_positivity(self) -> Dict[str, pd.Series]:
        """Calculates weekly positivity trends for key tests in a single, efficient pass."""
        required_cols = [COL_TEST_RESULT, COL_TEST_TYPE, COL_ENCOUNTER_DATE]
        if not all(c in self.df_epi.columns for c in required_cols):
            return {}

        key_tests = self._get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
        if not key_tests:
            return {}

        non_conclusive = self._get_setting('NON_CONCLUSIVE_TEST_RESULTS', ["pending", "rejected"])
        df_conclusive = self.df_epi[~self.df_epi[COL_TEST_RESULT].str.lower().isin(non_conclusive)]
        
        # Calculate positivity for all tests present
        df_conclusive['is_positive'] = (df_conclusive[COL_TEST_RESULT].str.lower() == 'positive').astype(float)
        
        # Efficiently calculate weekly mean for all tests at once
        weekly_positivity = df_conclusive.groupby([
            pd.Grouper(key=COL_ENCOUNTER_DATE, freq='W-MON'), COL_TEST_TYPE
        ])['is_positive'].mean().unstack(COL_TEST_TYPE)

        # Build the final output dictionary by selecting the key tests
        positivity_trends = {}
        for internal_name, config in key_tests.items():
            if not isinstance(config, dict) or internal_name not in weekly_positivity.columns:
                continue
            trend = weekly_positivity[internal_name].dropna()
            if not trend.empty:
                display_name = config.get("display_name", internal_name)
                positivity_trends[display_name] = (trend * 100).round(1)
        
        return positivity_trends

    def _calculate_demographics(self, df_unique_patients: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculates age and gender distributions from a dataframe of unique patients."""
        demographics = {"age_distribution_df": pd.DataFrame(), "gender_distribution_df": pd.DataFrame()}
        
        if COL_AGE in df_unique_patients.columns:
            age_bins = [0, 5, 18, 60, 75, np.inf]
            age_labels = ['0-4', '5-17', '18-59', '60-74', '75+']
            age_df = pd.cut(df_unique_patients[COL_AGE].dropna(), bins=age_bins, labels=age_labels, right=False).value_counts().sort_index().reset_index()
            age_df.columns = ['Age Group', 'Patient Count']
            demographics["age_distribution_df"] = age_df
        
        if COL_GENDER in df_unique_patients.columns:
            gender_df = df_unique_patients[COL_GENDER].str.title().value_counts().reset_index()
            gender_df.columns = ['Gender', 'Patient Count']
            demographics["gender_distribution_df"] = gender_df[gender_df['Gender'].isin(["Male", "Female"])]
            
        return demographics

    def calculate(self) -> Dict[str, Any]:
        """Orchestrates all epidemiological calculations."""
        if self.df_epi.empty:
            self.notes.append("No valid health data provided.")
            return {"calculation_notes": self.notes}

        output: Dict[str, Any] = {
            "symptom_trends_weekly_top_n_df": self._calculate_symptom_trends(),
            "key_test_positivity_trends": self._calculate_test_positivity(),
            "demographics_by_condition_data": {},
            "calculation_notes": self.notes
        }

        if COL_PATIENT_ID in self.df_epi.columns:
            df_unique_patients = self.df_epi.drop_duplicates(subset=[COL_PATIENT_ID])
            if not df_unique_patients.empty:
                output["demographics_by_condition_data"] = self._calculate_demographics(df_unique_patients)
        
        return output

def calculate_clinic_epidemiological_data(
    filtered_health_df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for a clinic over a specified period.
    This factory function instantiates and runs the main calculator class.
    """
    calculator = ClinicEpiDataCalculator(filtered_health_df)
    return calculator.calculate()
