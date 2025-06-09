# sentinel_project_root/pages/clinic_components/epi_data.py
# SME-EVALUATED AND REVISED VERSION
# This version significantly improves performance by replacing looped aggregations
# with vectorized, pandas-native groupby operations for both positivity and demographics.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.aggregation import get_trend_data
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.critical(f"Fatal import error in epi_data.py: {e}. Check project structure and dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Constants for Clarity and Maintainability ---
COL_DATE = 'encounter_date'
COL_SYMPTOMS = 'patient_reported_symptoms'
COL_TEST_TYPE = 'test_type'
COL_TEST_RESULT = 'test_result'
COL_PATIENT_ID = 'patient_id'
COL_DIAGNOSIS = 'diagnosis'


class ClinicEpidemiologyPreparer:
    """
    Encapsulates all logic for calculating epidemiological metrics for a clinic.

    This class processes a period-filtered health DataFrame to generate structured
    data for symptom trends, test positivity rates, and demographic breakdowns.
    It is performance-optimized to use vectorized operations where possible.
    """

    def __init__(self, filtered_health_df: Optional[pd.DataFrame], reporting_period_str: str):
        """
        Initializes the preparer with health data for a specific period.

        Args:
            filtered_health_df: A DataFrame of health data pre-filtered for the report period.
            reporting_period_str: A human-readable string describing the report period.
        """
        self.df = filtered_health_df.copy() if isinstance(filtered_health_df, pd.DataFrame) and not filtered_health_df.empty else pd.DataFrame()
        self.reporting_period = reporting_period_str
        self.notes: List[str] = []
        self._validate_and_prepare_input_df()

    @staticmethod
    def _get_setting(attr_name: str, default_value: Any) -> Any:
        """Safely retrieves a configuration value from the global settings object."""
        return getattr(settings, attr_name, default_value)

    def _validate_and_prepare_input_df(self):
        """Performs essential validation and cleaning of the input DataFrame."""
        if self.df.empty:
            self.notes.append("No health data was provided for epidemiological analysis.")
            return

        if COL_DATE not in self.df.columns:
            self.notes.append(f"Data Integrity Issue: The required '{COL_DATE}' column is missing.")
            self.df = pd.DataFrame()  # Invalidate DataFrame to halt further processing.
            return

        self.df[COL_DATE] = pd.to_datetime(self.df[COL_DATE], errors='coerce')
        self.df.dropna(subset=[COL_DATE], inplace=True)
        if self.df.empty:
            self.notes.append("No records with valid encounter dates were found after cleaning.")

    def _prepare_symptom_trends(self) -> pd.DataFrame:
        """
        Analyzes and aggregates the top N patient-reported symptoms weekly.

        Returns:
            A DataFrame with columns ['week_start_date', 'symptom', 'count'].
        """
        if self.df.empty or COL_SYMPTOMS not in self.df.columns:
            return pd.DataFrame()

        try:
            symptoms_df = self.df[[COL_DATE, COL_SYMPTOMS]].dropna(subset=[COL_SYMPTOMS]).copy()
            symptoms_df[COL_SYMPTOMS] = symptoms_df[COL_SYMPTOMS].astype(str).str.strip()
            symptoms_df = symptoms_df[symptoms_df[COL_SYMPTOMS] != '']

            if symptoms_df.empty:
                self.notes.append("Symptom data column is present but contains no usable text.")
                return pd.DataFrame()

            exploded = symptoms_df.assign(symptom=symptoms_df[COL_SYMPTOMS].str.split(r'[;,|]')).explode('symptom')
            exploded['symptom'] = exploded['symptom'].str.strip().str.title()

            non_info = [s.lower() for s in self._get_setting('NON_INFORMATIVE_SYMPTOMS', ['none', 'n/a'])]
            exploded = exploded[~exploded['symptom'].str.lower().isin(non_info)]
            if exploded.empty: return pd.DataFrame()

            top_n = self._get_setting('EPI_TOP_N_SYMPTOMS', 5)
            top_symptoms = exploded['symptom'].value_counts().nlargest(top_n).index
            df_top = exploded[exploded['symptom'].isin(top_symptoms)]

            weekly_counts = df_top.groupby([pd.Grouper(key=COL_DATE, freq='W-MON'), 'symptom']).size().reset_index(name='count')
            return weekly_counts.rename(columns={COL_DATE: 'week_start_date'})
        except Exception as e:
            logger.error(f"Failed to process symptom trends: {e}", exc_info=True)
            self.notes.append("An error occurred during symptom trend analysis.")
            return pd.DataFrame()

    def _prepare_positivity_rates(self) -> Dict[str, pd.Series]:
        """
        Calculates weekly test positivity rates using a single vectorized operation.

        Returns:
            A dictionary mapping test display names to their positivity trend Series.
        """
        required_cols = [COL_TEST_TYPE, COL_TEST_RESULT, COL_DATE]
        if self.df.empty or not all(col in self.df.columns for col in required_cols):
            return {}

        try:
            key_tests_config = self._get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
            if not key_tests_config: return {}
            
            key_test_names = list(key_tests_config.keys())
            non_conclusive = [s.lower() for s in self._get_setting('NON_CONCLUSIVE_TEST_RESULTS', [])]
            
            # Filter once to a working DataFrame for efficiency
            df_tests = self.df[self.df[COL_TEST_TYPE].isin(key_test_names)].copy()
            df_tests = df_tests[~df_tests[COL_TEST_RESULT].str.lower().isin(non_conclusive)]
            if df_tests.empty: return {}

            df_tests['is_positive'] = (df_tests[COL_TEST_RESULT].str.lower() == 'positive').astype(float)
            
            # --- PERFORMANCE REFACTOR: Single GroupBy for all tests ---
            # Group by week and test type, calculate mean positivity, then unstack
            all_trends = df_tests.groupby(
                [pd.Grouper(key=COL_DATE, freq='W-MON'), COL_TEST_TYPE]
            )['is_positive'].mean().unstack(level=COL_TEST_TYPE)

            # Convert to dictionary of Series and apply formatting
            positivity_trends = {}
            for internal_name, trend_series in all_trends.items():
                display_name = key_tests_config.get(internal_name, {}).get("display_name", internal_name)
                positivity_trends[display_name] = (trend_series.dropna() * 100).round(1)

            return positivity_trends
        except Exception as e:
            logger.error(f"Failed to calculate test positivity rates: {e}", exc_info=True)
            self.notes.append("An error occurred during test positivity analysis.")
            return {}

    def _prepare_demographics_by_condition(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculates age/gender breakdowns for top conditions using vectorized aggregations.

        Returns:
            A nested dictionary: {condition: {"age_distribution_df": df, "gender_distribution_df": df}}.
        """
        required_cols = [COL_PATIENT_ID, COL_DIAGNOSIS, 'age', 'gender']
        if self.df.empty or not all(col in self.df.columns for col in required_cols):
            self.notes.append("Demographic analysis skipped due to missing columns (patient_id, diagnosis, age, gender).")
            return {}
        
        try:
            top_n = self._get_setting('EPI_TOP_N_CONDITIONS_FOR_DEMOGRAPHICS', 5)
            top_conditions = self.df[COL_DIAGNOSIS].dropna().value_counts().nlargest(top_n).index
            if top_conditions.empty: return {}
            
            df_focus = self.df[self.df[COL_DIAGNOSIS].isin(top_conditions)].copy()

            # --- PERFORMANCE REFACTOR: Vectorized aggregations ---
            # 1. Prepare binning columns
            age_bins = self._get_setting('EPI_DEMOGRAPHICS_AGE_BINS', [0, 5, 18, 45, 65, np.inf])
            age_labels = self._get_setting('EPI_DEMOGRAPHICS_AGE_LABELS', ['0-4', '5-17', '18-44', '45-64', '65+'])
            df_focus['age_group'] = pd.cut(df_focus['age'].dropna(), bins=age_bins, labels=age_labels, right=False)
            df_focus['gender_simple'] = df_focus['gender'].str.title().where(df_focus['gender'].str.title().isin(['Male', 'Female']))

            # 2. Perform aggregations using nunique to count unique patients
            age_counts = df_focus.groupby([COL_DIAGNOSIS, 'age_group'])[COL_PATIENT_ID].nunique().reset_index(name='Patient Count')
            gender_counts = df_focus.groupby([COL_DIAGNOSIS, 'gender_simple'])[COL_PATIENT_ID].nunique().reset_index(name='Patient Count')
            
            # 3. Structure the output dictionary
            demographics = {}
            for condition in top_conditions:
                age_df = age_counts[age_counts[COL_DIAGNOSIS] == condition][['age_group', 'Patient Count']].rename(columns={'age_group': 'Age Group'})
                gender_df = gender_counts[gender_counts[COL_DIAGNOSIS] == condition][['gender_simple', 'Patient Count']].rename(columns={'gender_simple': 'Gender'})
                
                demographics[condition] = {
                    "age_distribution_df": age_df.sort_values(by="Age Group").reset_index(drop=True),
                    "gender_distribution_df": gender_df.dropna().reset_index(drop=True)
                }
            return demographics
        except Exception as e:
            logger.error(f"Failed to calculate demographics by condition: {e}", exc_info=True)
            self.notes.append("An error occurred during demographic analysis.")
            return {}

    def prepare(self) -> Dict[str, Any]:
        """
        Orchestrates all epidemiological calculations and returns a consolidated dictionary.
        
        Returns:
            A dictionary containing all prepared data components and any processing notes.
        """
        logger.info(f"Starting epidemiological data preparation for period: {self.reporting_period}")

        if self.df.empty:
            logger.warning("Epidemiological preparation skipped as input DataFrame is empty.")
            return {
                "symptom_trends_weekly_top_n_df": pd.DataFrame(),
                "key_test_positivity_trends": {},
                "demographics_by_condition_data": {},
                "calculation_notes": self.notes
            }

        return {
            "symptom_trends_weekly_top_n_df": self._prepare_symptom_trends(),
            "key_test_positivity_trends": self._prepare_positivity_rates(),
            "demographics_by_condition_data": self._prepare_demographics_by_condition(),
            "calculation_notes": list(set(self.notes)) # Ensure unique notes
        }


def calculate_clinic_epidemiological_data(
    filtered_health_df: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Public factory function to instantiate and run the ClinicEpidemiologyPreparer.
    
    Args:
        filtered_health_df: A DataFrame of health data for the reporting period.
        reporting_period_context_str: A string describing the period (e.g., "Last 30 Days").
        **kwargs: Catches any unused keyword arguments for forward compatibility.

    Returns:
        A dictionary containing structured epidemiological data for the UI.
    """
    preparer = ClinicEpidemiologyPreparer(filtered_health_df, reporting_period_context_str)
    return preparer.prepare()
