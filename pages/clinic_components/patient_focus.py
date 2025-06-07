# sentinel_project_root/pages/clinic_components/patient_focus.py
# Prepares patient-centric data for the Clinic Console.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

try:
    from config import settings
    from analytics.alerting import ClinicPatientAlerts
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in patient_focus.py: {e}. Check paths/dependencies.")
    raise

logger = logging.getLogger(__name__)

# --- Constants ---
COL_DIAGNOSIS = 'diagnosis'
COL_ENCOUNTER_DATE = 'encounter_date'
COL_PATIENT_ID = 'patient_id'

class ClinicPatientFocusPreparer:
    """
    Orchestrates the preparation of patient-focused data, including load by
    condition and a list of patients flagged for review.
    """
    def __init__(self, filtered_health_df: pd.DataFrame):
        self.df = filtered_health_df
        self.notes: List[str] = []
        self.module_log_prefix = self.__class__.__name__

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _prepare_patient_load_by_condition(self) -> pd.DataFrame:
        """
        Efficiently calculates weekly patient load for key conditions using vectorized operations.
        """
        key_conditions = self._get_setting('PATIENT_FOCUS_KEY_CONDITIONS', [])
        if not key_conditions or COL_DIAGNOSIS not in self.df.columns:
            return pd.DataFrame()

        # Create a temporary mapping column for conditions
        condition_map = {cond.lower(): cond.title() for cond in key_conditions}
        
        # Vectorized search for all conditions at once
        # This creates a column where each cell is a list of matching key conditions
        df_temp = self.df[[COL_ENCOUNTER_DATE, COL_PATIENT_ID, COL_DIAGNOSIS]].dropna()
        df_temp['matched_conditions'] = df_temp[COL_DIAGNOSIS].str.lower().apply(
            lambda dx: [condition_map[cond] for cond in condition_map if cond in dx]
        )

        # Explode the DataFrame to have one row per patient per matched condition
        exploded_df = df_temp[df_temp['matched_conditions'].map(len) > 0].explode('matched_conditions')

        if exploded_df.empty:
            return pd.DataFrame()
            
        # Perform a single, efficient groupby operation
        weekly_patients = exploded_df.groupby([
            pd.Grouper(key=COL_ENCOUNTER_DATE, freq='W-MON'),
            'matched_conditions'
        ])[COL_PATIENT_ID].nunique().reset_index()

        weekly_patients.rename(columns={
            'matched_conditions': 'condition',
            COL_PATIENT_ID: 'unique_patients_count',
            COL_ENCOUNTER_DATE: 'period_start_date'
        }, inplace=True)

        return weekly_patients

    def _prepare_flagged_patients_list(self) -> pd.DataFrame:
        """
        Generates and formats a list of patients flagged for clinical review.
        """
        try:
            # This class is a dependency; we just call its public interface.
            alert_generator = ClinicPatientAlerts(health_data_df=self.df, reporting_period_df=self.df)
            flagged_df = alert_generator.generate_patient_review_list()
            
            if flagged_df.empty:
                return pd.DataFrame()

            # Robustly map available columns to desired display names
            column_map = {
                'patient_id': "Patient ID",
                'age': 'Age',
                'gender': "Gender",
                'ai_risk_score': "Risk Score",
                'alert_reason': 'Reason for Flag',
                'last_encounter_date': "Last Visit",
                'diagnosis': "Last Diagnosis",
                'key_vitals': "Key Vitals"
            }
            
            # Select only the columns that exist in the source df
            final_cols = {k: v for k, v in column_map.items() if k in flagged_df.columns}
            
            return flagged_df[final_cols.keys()].rename(columns=final_cols)

        except Exception as e:
            logger.error(f"({self.module_log_prefix}) Failed to generate flagged patient list: {e}", exc_info=True)
            self.notes.append("Error generating flagged patient list.")
            return pd.DataFrame()

    def prepare(self) -> Dict[str, Any]:
        """Orchestrates all patient focus data preparation."""
        if self.df.empty:
            self.notes.append("Health data is empty.")
            return {
                "patient_load_by_key_condition_df": pd.DataFrame(),
                "flagged_patients_for_review_df": pd.DataFrame(),
                "processing_notes": self.notes
            }

        output = {
            "patient_load_by_key_condition_df": self._prepare_patient_load_by_condition(),
            "flagged_patients_for_review_df": self._prepare_flagged_patients_list(),
            "processing_notes": self.notes
        }
        
        logger.info(f"({self.module_log_prefix}) Patient focus data preparation complete.")
        return output

def prepare_clinic_patient_focus_overview_data(
    filtered_health_df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Factory function to prepare an overview of patient-focused data.
    """
    preparer = ClinicPatientFocusPreparer(filtered_health_df or pd.DataFrame())
    return preparer.prepare()
