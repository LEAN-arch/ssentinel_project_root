# sentinel_project_root/analytics/risk_prediction.py
"""
Contains the Sentinel Risk Prediction Model.
Calculates a composite risk score based on various health and environmental factors.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in risk_prediction.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    A rule-based model to calculate a composite health risk score for individuals.
    """
    def __init__(self):
        self.module_log_prefix = self.__class__.__name__
        self.base_scores = self._get_setting('RISK_MODEL_BASE_SCORES', {})
        self.factors = self._get_setting('RISK_MODEL_FACTORS', {})
        logger.info(f"({self.module_log_prefix}) Initialized with configured base scores and factors.")

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct numeric type."""
        clean_df = df.copy()
        # Define all columns the model might use and their default neutral values
        required_cols = {
            'age': 30, 'chronic_condition_flag': 0, 'min_spo2_pct': 98,
            'vital_signs_temperature_celsius': 37.0, 'signs_of_fatigue_observed_flag': 0,
            'rapid_psychometric_distress_score': 0, 'medication_adherence_self_report': "Good",
        }
        for col, default in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default
            # Ensure columns are numeric for calculation
            if col != 'medication_adherence_self_report':
                 clean_df[col] = convert_to_numeric(clean_df[col], default_value=default)
        return clean_df

    def calculate_risk_scores(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates risk scores for each record in the DataFrame.
        This is the primary public method for this class.
        """
        if not isinstance(health_df, pd.DataFrame) or health_df.empty:
            if 'ai_risk_score' not in health_df.columns:
                health_df['ai_risk_score'] = np.nan
            return health_df

        df = self._prepare_data(health_df)
        
        # Start with a base score
        risk_scores = pd.Series(self.base_scores.get('default', 5.0), index=df.index)

        # Apply age-based adjustments
        age_factor = self.factors.get('age', {})
        if age_factor:
            risk_scores += np.where(df['age'] > age_factor.get('elderly_threshold', 65), age_factor.get('elderly_modifier', 15), 0)
            risk_scores += np.where(df['age'] < age_factor.get('pediatric_threshold', 5), age_factor.get('pediatric_modifier', 10), 0)

        # Apply SpO2 adjustments
        spo2_factor = self.factors.get('spo2', {})
        if spo2_factor:
            risk_scores += np.where(df['min_spo2_pct'] < spo2_factor.get('low_threshold', 92), spo2_factor.get('low_modifier', 20), 0)

        # Apply temperature adjustments
        temp_factor = self.factors.get('temperature', {})
        if temp_factor:
            risk_scores += np.where(df['vital_signs_temperature_celsius'] > temp_factor.get('high_threshold', 38.0), temp_factor.get('high_modifier', 15), 0)

        # Apply other flag-based modifiers
        risk_scores += df['chronic_condition_flag'] * self.factors.get('chronic_condition_modifier', 20)
        risk_scores += df['signs_of_fatigue_observed_flag'] * self.factors.get('fatigue_modifier', 10)
        
        # Finalize scores
        final_scores = risk_scores.clip(0, 100).round(1)
        logger.info(f"({self.module_log_prefix}) Bulk risk scores calculated for {len(df)} records.")
        
        df['ai_risk_score'] = final_scores
        return df
