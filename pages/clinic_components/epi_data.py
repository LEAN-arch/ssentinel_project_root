# sentinel_project_root/pages/clinic_components/epi_data.py
# Prepares epidemiological data (symptoms, positivity, demographics) for the Sentinel Clinic Dashboard.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.aggregation import get_trend_data
except ImportError as e:
    # Use a basic logger for critical import errors, as the full config may not be available.
    logging.basicConfig(level=logging.ERROR)
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.critical(f"Fatal import error in epi_data.py: {e}. Check project structure and dependencies.", exc_info=True)
    # Re-raise to prevent the application from starting in a broken state.
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

    This class takes a raw, period-filtered health DataFrame and processes it to
    generate structured data for symptom trends, test positivity rates, and
    demographic breakdowns, ready for UI consumption. It is designed to be
    resilient to missing data and configurable via the central settings file.
    """

    def __init__(self, filtered_health_df: Optional[pd.DataFrame], reporting_period_str: str):
        """Initializes the preparer with health data for a specific period."""
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
            self.notes.append("No health data was provided for analysis.")
            return

        if COL_DATE not in self.df.columns:
            self.notes.append(f"Data Integrity Issue: The required '{COL_DATE}' column is missing.")
            self.df = pd.DataFrame()  # Invalidate DataFrame to halt further processing.
            return

        # Standardize date column for reliable, timezone-naive processing.
        self.df[COL_DATE] = pd.to_datetime(self.df[COL_DATE], errors='coerce')
        self.df.dropna(subset=[COL_DATE], inplace=True)
        if self.df.empty:
            self.notes.append("No records with valid encounter dates were found after cleaning.")

    def _prepare_symptom_trends(self) -> pd.DataFrame:
        """Analyzes and aggregates the top N patient-reported symptoms weekly."""
        if self.df.empty or COL_SYMPTOMS not in self.df.columns:
            return pd.DataFrame()

        try:
            symptoms_df = self.df[[COL_DATE, COL_SYMPTOMS]].dropna(subset=[COL_SYMPTOMS]).copy()
            symptoms_df[COL_SYMPTOMS] = symptoms_df[COL_SYMPTOMS].astype(str).str.strip()
            symptoms_df = symptoms_df[symptoms_df[COL_SYMPTOMS] != '']

            if symptoms_df.empty:
                self.notes.append("Symptom data is present but contains no meaningful text.")
                return pd.DataFrame()

            # Explode multi-symptom strings (e.g., "Fever;Cough") into individual rows.
            exploded = symptoms_df.assign(symptom=symptoms_df[COL_SYMPTOMS].str.split(r'[;,|]')).explode('symptom')
            exploded['symptom'] = exploded['symptom'].str.strip().str.title()

            # Filter out non-informative symptoms (e.g., "None", "N/A") defined in settings.
            non_info = [s.lower() for s in self._get_setting('NON_INFORMATIVE_SYMPTOMS', [])]
            exploded = exploded[~exploded['symptom'].str.lower().isin(non_info)]
            if exploded.empty: return pd.DataFrame()

            top_n = self._get_setting('EPI_TOP_N_SYMPTOMS', 5)
            top_symptoms = exploded['symptom'].value_counts().nlargest(top_n).index
            df_top = exploded[exploded['symptom'].isin(top_symptoms)]

            # Aggregate their counts weekly, aligning to Mondays.
            weekly_counts = df_top.groupby([pd.Grouper(key=COL_DATE, freq='W-MON'), 'symptom']).size().reset_index(name='count')
            return weekly_counts.rename(columns={COL_DATE: 'week_start_date'})
        except Exception as e:
            logger.error(f"Failed to process symptom trends: {e}", exc_info=True)
            self.notes.append("An error occurred during symptom trend analysis.")
            return pd.DataFrame()

    def _prepare_positivity_rates(self) -> Dict[str, pd.Series]:
        """Calculates weekly test positivity rates for key configured tests."""
        positivity_trends = {}
        required_cols = [COL_TEST_TYPE, COL_TEST_RESULT]
        if self.df.empty or not all(col in self.df.columns for col in required_cols):
            return positivity_trends

        try:
            key_tests = self._get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
            non_conclusive = [s.lower() for s in self._get_setting('NON_CONCLUSIVE_TEST_RESULTS', [])]
            
            df_conclusive = self.df[~self.df[COL_TEST_RESULT].str.lower().isin(non_conclusive)].copy()
            if df_conclusive.empty: return positivity_trends

            df_conclusive['is_positive'] = (df_conclusive[COL_TEST_RESULT].str.lower() == 'positive').astype(float)

            for internal_name, config in key_tests.items():
                if not isinstance(config, dict): continue
                
                df_test = df_conclusive[df_conclusive[COL_TEST_TYPE] == internal_name]
                if df_test.empty: continue

                trend = get_trend_data(df=df_test, value_col='is_positive', date_col=COL_DATE, period='W-MON', agg_func='mean')
                if not trend.empty:
                    positivity_trends[config.get("display_name", internal_name)] = (trend * 100).round(1)

            return positivity_trends
        except Exception as e:
            logger.error(f"Failed to calculate test positivity rates: {e}", exc_info=True)
            self.notes.append("An error occurred during test positivity analysis.")
            return {}

    def _prepare_demographics_by_condition(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Provides age and gender demographic breakdowns for each major diagnosis."""
        demographics = {}
        required_cols = [COL_PATIENT_ID, COL_DIAGNOSIS, 'age', 'gender']
        if self.df.empty or not all(col in self.df.columns for col in required_cols):
            self.notes.append("Demographic analysis skipped due to missing columns (patient_id, diagnosis, age, gender).")
            return demographics
        
        try:
            top_n = self._get_setting('EPI_TOP_N_CONDITIONS_FOR_DEMOGRAPHICS', 5)
            top_conditions = self.df[COL_DIAGNOSIS].dropna().value_counts().nlargest(top_n).index
            df_focus = self.df[self.df[COL_DIAGNOSIS].isin(top_conditions)]

            age_bins = self._get_setting('EPI_DEMOGRAPHICS_AGE_BINS', [0, 5, 18, 45, 65, np.inf])
            age_labels = self._get_setting('EPI_DEMOGRAPHICS_AGE_LABELS', ['0-4', '5-17', '18-44', '45-64', '65+'])

            for condition, group_df in df_focus.groupby(COL_DIAGNOSIS):
                # We count unique patients to avoid double-counting from multiple visits.
                unique_patients = group_df.drop_duplicates(subset=[COL_PATIENT_ID])
                
                age_counts = pd.cut(unique_patients['age'].dropna(), bins=age_bins, labels=age_labels, right=False).value_counts().sort_index()
                age_df = age_counts.reset_index(name='Patient Count').rename(columns={'index': 'Age Group'})
                
                gender_counts = unique_patients['gender'].str.title().value_counts()
                gender_df = gender_counts.reset_index(name='Patient Count').rename(columns={'index': 'Gender'})
                gender_df = gender_df[gender_df['Gender'].isin(['Male', 'Female'])]

                demographics[condition] = { "age_distribution_df": age_df, "gender_distribution_df": gender_df }
            return demographics
        except Exception as e:
            logger.error(f"Failed to calculate demographics by condition: {e}", exc_info=True)
            self.notes.append("An error occurred during demographic analysis.")
            return {}

    def prepare(self) -> Dict[str, Any]:
        """Orchestrates all epidemiological calculations and returns a consolidated dictionary."""
        logger.info(f"Starting epidemiological data preparation for period: {self.reporting_period}")

        if self.df.empty:
            logger.warning("Epidemiological preparation skipped as input DataFrame is empty.")
            return {
                "symptom_trends_weekly_top_n_df": pd.DataFrame(),
                "key_test_positivity_trends": {},
                "demographics_by_condition_data": {},
                "calculation_notes": self.notes
            }

        output = {
            "symptom_trends_weekly_top_n_df": self._prepare_symptom_trends(),
            "key_test_positivity_trends": self._prepare_positivity_rates(),
            "demographics_by_condition_data": self._prepare_demographics_by_condition(),
            "calculation_notes": self.notes,
        }
        
        logger.info("Epidemiological data preparation complete.")
        return output


def calculate_clinic_epidemiological_data(
    filtered_health_df: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    **kwargs # Captures any unused keyword arguments for forward compatibility.
) -> Dict[str, Any]:
    """
    Public factory function to instantiate and run the ClinicEpidemiologyPreparer.

    This serves as the clean, public-facing entry point for other modules,
    abstracting away the internal class implementation.
    """
    preparer = ClinicEpidemiologyPreparer(filtered_health_df, reporting_period_context_str)
    return preparer.prepare()
