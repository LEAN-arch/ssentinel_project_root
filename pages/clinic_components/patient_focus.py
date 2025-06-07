# sentinel_project_root/pages/clinic_components/patient_focus.py
# Prepares data for clinic patient load and flagged patient cases for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    from analytics.alerting import get_patient_alerts_for_clinic
except ImportError as e:
    # Use a basic logger for critical import errors, as the full config may not be available.
    logging.basicConfig(level=logging.ERROR)
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.critical(f"Fatal import error in patient_focus.py: {e}. Check project structure and dependencies.", exc_info=True)
    # Re-raise to prevent the application from starting in a broken state.
    raise

logger = logging.getLogger(__name__)

# --- Constants for Clarity and Maintainability ---
COL_DATE = 'encounter_date'
COL_PATIENT_ID = 'patient_id'
COL_CONDITION = 'condition'
# Define output schemas as constants for predictability and easier maintenance.
FLAGGED_PATIENT_COLS = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
PATIENT_LOAD_COLS = ['period_start_date', 'condition', 'unique_patients_count']


class PatientFocusPreparer:
    """
    Encapsulates all logic for calculating patient focus metrics.

    This class processes a filtered health DataFrame to generate structured data
    for patient load trends and a list of high-priority patients for review,
    ensuring all data is clean and ready for direct consumption by the UI.
    """

    def __init__(self, filtered_health_df: Optional[pd.DataFrame], reporting_period_str: str):
        """Initializes the preparer with health data for a specific period."""
        self.df = filtered_health_df.copy() if isinstance(filtered_health_df, pd.DataFrame) and not filtered_health_df.empty else pd.DataFrame()
        self.reporting_period = reporting_period_str
        self.notes: List[str] = []
        self._validate_and_prepare_input_df()

    def _validate_and_prepare_input_df(self):
        """Performs essential validation, cleaning, and type conversion on the input DataFrame."""
        if self.df.empty:
            self.notes.append("No health data was provided for patient focus analysis.")
            return

        required_cols = [COL_DATE, COL_PATIENT_ID, COL_CONDITION]
        if not all(col in self.df.columns for col in required_cols):
            missing = sorted(list(set(required_cols) - set(self.df.columns)))
            self.notes.append(f"Data Integrity Issue: Required columns are missing: {missing}.")
            self.df = pd.DataFrame()  # Invalidate to halt further processing.
            return

        # Standardize date column for reliable processing.
        self.df[COL_DATE] = pd.to_datetime(self.df[COL_DATE], errors='coerce')
        self.df.dropna(subset=required_cols, inplace=True)
        if self.df.empty:
            self.notes.append("No valid records remain after cleaning for patient focus analysis.")

    def _prepare_patient_load(self) -> pd.DataFrame:
        """Calculates the weekly unique patient load for key, configured conditions."""
        if self.df.empty:
            return pd.DataFrame(columns=PATIENT_LOAD_COLS)

        try:
            key_conditions = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
            if not key_conditions:
                self.notes.append("Configuration missing: No key conditions are defined for patient load analysis.")
                return pd.DataFrame(columns=PATIENT_LOAD_COLS)

            # This vectorized approach is highly performant compared to looping through conditions.
            df_focus = self.df[self.df[COL_CONDITION].isin(key_conditions)]
            if df_focus.empty:
                return pd.DataFrame(columns=PATIENT_LOAD_COLS)

            # A single groupby calculates unique patient counts for all relevant conditions at once.
            patient_load = df_focus.groupby([
                pd.Grouper(key=COL_DATE, freq='W-MON'),
                COL_CONDITION
            ])[COL_PATIENT_ID].nunique().reset_index()

            patient_load.rename(columns={
                COL_DATE: 'period_start_date',
                COL_PATIENT_ID: 'unique_patients_count'
            }, inplace=True)

            return patient_load[PATIENT_LOAD_COLS]  # Enforce consistent column order
        except Exception as e:
            logger.error(f"Failed to calculate patient load: {e}", exc_info=True)
            self.notes.append("An error occurred during patient load analysis.")
            return pd.DataFrame(columns=PATIENT_LOAD_COLS)

    def _prepare_flagged_patients(self) -> pd.DataFrame:
        """Identifies and retrieves a pre-sorted list of high-priority patients for clinical review."""
        if self.df.empty:
            return pd.DataFrame(columns=FLAGGED_PATIENT_COLS)

        try:
            risk_threshold = float(getattr(settings, 'RISK_SCORE_MODERATE_THRESHOLD', 60.0))
            
            alerts_df = get_patient_alerts_for_clinic(
                health_df_period=self.df,
                risk_threshold_moderate=risk_threshold
            )
            
            if isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty:
                # Pre-sorting here ensures the UI receives data ready for immediate display.
                sorted_alerts = alerts_df.sort_values(by='Priority Score', ascending=False)
                
                # Reindex ensures the DataFrame has a predictable schema, even if the source changes.
                return sorted_alerts.reindex(columns=FLAGGED_PATIENT_COLS, fill_value=np.nan)
            
            self.notes.append("No patients were flagged for review in this period based on current criteria.")
            return pd.DataFrame(columns=FLAGGED_PATIENT_COLS)
        except Exception as e:
            logger.error(f"Error getting flagged patients: {e}", exc_info=True)
            self.notes.append("An error occurred while generating the list of flagged patients.")
            return pd.DataFrame(columns=FLAGGED_PATIENT_COLS)

    def prepare(self) -> Dict[str, Any]:
        """
        Orchestrates all patient focus calculations and returns a consolidated dictionary.
        
        Returns:
            A dictionary containing:
            - 'patient_load_by_key_condition_df': DataFrame with weekly patient counts.
            - 'flagged_patients_for_review_df': DataFrame with high-priority patients, sorted by score.
            - 'processing_notes': A list of informational or error messages.
        """
        logger.info(f"Starting patient focus data preparation for period: '{self.reporting_period}'")

        # The internal methods are already robust to an empty self.df, so we can call them directly.
        # The empty check in init handles adding the primary processing note.
        output = {
            "patient_load_by_key_condition_df": self._prepare_patient_load(),
            "flagged_patients_for_review_df": self._prepare_flagged_patients(),
            "processing_notes": self.notes,
        }
        
        logger.info("Patient focus data preparation complete.")
        return output


def prepare_clinic_patient_focus_overview_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    **kwargs # Captures any unused keyword arguments for forward compatibility.
) -> Dict[str, Any]:
    """

    Public factory function to instantiate and run the PatientFocusPreparer.

    This serves as the clean, public-facing entry point for other modules,
    abstracting away the internal class implementation.
    """
    preparer = PatientFocusPreparer(filtered_health_df_for_clinic_period, reporting_period_context_str)
    return preparer.prepare()
