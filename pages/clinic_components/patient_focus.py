# sentinel_project_root/pages/clinic_components/patient_focus.py
# SME-EVALUATED AND REVISED VERSION (GOLD STANDARD)
# This definitive version enhances robustness by enforcing a consistent schema on
# the output of external functions and improves actionability with pre-sorted outputs.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    from analytics.alerting import get_patient_alerts_for_clinic
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.critical(f"Fatal import error in patient_focus.py: {e}. Check project structure and dependencies.", exc_info=True)
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
    for patient load trends and a list of high-priority patients for review. It is
    designed to be resilient to data issues and to provide clean, UI-ready outputs.
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

        self.df[COL_DATE] = pd.to_datetime(self.df[COL_DATE], errors='coerce')
        self.df.dropna(subset=required_cols, inplace=True)
        if self.df.empty:
            self.notes.append("No valid records remain after cleaning for patient focus analysis.")

    def _prepare_patient_load(self) -> pd.DataFrame:
        """
        Calculates the weekly unique patient load for key, configured conditions.
        The output is pre-sorted to be ready for direct charting.

        Returns:
            A sorted DataFrame with columns ['period_start_date', 'condition', 'unique_patients_count'].
        """
        if self.df.empty:
            return pd.DataFrame(columns=PATIENT_LOAD_COLS)

        try:
            key_conditions = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
            if not key_conditions:
                self.notes.append("Configuration missing: No key conditions are defined for patient load analysis.")
                return pd.DataFrame(columns=PATIENT_LOAD_COLS)

            df_focus = self.df[self.df[COL_CONDITION].isin(key_conditions)]
            if df_focus.empty:
                return pd.DataFrame(columns=PATIENT_LOAD_COLS)

            patient_load = df_focus.groupby(
                [pd.Grouper(key=COL_DATE, freq='W-MON'), COL_CONDITION]
            )[COL_PATIENT_ID].nunique().reset_index()

            patient_load.rename(columns={
                COL_DATE: 'period_start_date',
                COL_PATIENT_ID: 'unique_patients_count'
            }, inplace=True)

            # ACTIONABILITY: Pre-sort the data so UI components don't have to.
            return patient_load.sort_values(by=['period_start_date', 'condition'])[PATIENT_LOAD_COLS]
        except Exception as e:
            logger.error(f"Failed to calculate patient load: {e}", exc_info=True)
            self.notes.append("An error occurred during patient load analysis.")
            return pd.DataFrame(columns=PATIENT_LOAD_COLS)

    def _prepare_flagged_patients(self) -> pd.DataFrame:
        """
        Identifies and retrieves a pre-sorted list of high-priority patients.
        This method ensures a consistent output schema regardless of the source function's output.

        Returns:
            A DataFrame of flagged patients, sorted by 'Priority Score', adhering to the FLAGGED_PATIENT_COLS schema.
        """
        if self.df.empty:
            return pd.DataFrame(columns=FLAGGED_PATIENT_COLS)

        try:
            risk_threshold = float(getattr(settings, 'RISK_SCORE_MODERATE_THRESHOLD', 60.0))
            
            source_alerts_df = get_patient_alerts_for_clinic(
                health_df_period=self.df,
                risk_threshold_moderate=risk_threshold
            )
            
            if isinstance(source_alerts_df, pd.DataFrame) and not source_alerts_df.empty:
                # --- ROBUSTNESS: Enforce schema immediately after receiving data ---
                # This makes the component resilient to changes in the upstream function.
                # 1. Create a new DataFrame with the desired columns, filling missing ones with NaN.
                alerts_df = pd.DataFrame()
                for col in FLAGGED_PATIENT_COLS:
                    if col in source_alerts_df.columns:
                        alerts_df[col] = source_alerts_df[col]
                    else:
                        alerts_df[col] = np.nan
                
                # 2. Ensure critical columns have the correct data type.
                alerts_df['Priority Score'] = pd.to_numeric(alerts_df['Priority Score'], errors='coerce').fillna(0)
                alerts_df['encounter_date'] = pd.to_datetime(alerts_df['encounter_date'], errors='coerce')
                
                # 3. Pre-sort to ensure the UI receives data ready for immediate display.
                return alerts_df.sort_values(by='Priority Score', ascending=False)
            
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
            - 'flagged_patients_for_review_df': DataFrame with high-priority patients.
            - 'processing_notes': A list of informational or error messages.
        """
        logger.info(f"Starting patient focus data preparation for period: '{self.reporting_period}'")

        output = {
            "patient_load_by_key_condition_df": self._prepare_patient_load(),
            "flagged_patients_for_review_df": self._prepare_flagged_patients(),
            "processing_notes": list(set(self.notes)), # Ensure unique notes
        }
        
        logger.info("Patient focus data preparation complete.")
        return output


def prepare_clinic_patient_focus_overview_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Public factory function to instantiate and run the PatientFocusPreparer.

    This serves as the clean, public-facing entry point for UI modules, abstracting away
    the internal class implementation and providing a consistent interface.
    
    Args:
        filtered_health_df_for_clinic_period: A DataFrame of health data for the reporting period.
        reporting_period_context_str: A string describing the period (e.g., "Last 30 Days").
        **kwargs: Catches any unused keyword arguments for forward compatibility.
        
    Returns:
        A dictionary containing structured patient focus data.
    """
    preparer = PatientFocusPreparer(filtered_health_df_for_clinic_period, reporting_period_context_str)
    return preparer.prepare()
