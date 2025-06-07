# ssentinel_project_root/analytics/risk_prediction.py
"""
Contains the Sentinel Patient Risk Prediction model.
Calculates a composite risk score based on various health factors using
efficient, vectorized, and configurable rules.
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
    This model is designed to be fully configurable via settings.py.
    """
    def __init__(self):
        """Initializes the model, loading its parameters from the central settings file."""
        self.module_log_prefix = self.__class__.__name__
        self.base_scores = self._get_setting('RISK_MODEL_BASE_SCORES', {'default': 5.0})
        self.factors = self._get_setting('RISK_MODEL_FACTORS', {})
        logger.info(f"({self.module_log_prefix}) Initialized with configured base scores and factors.")

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        """Safely retrieves a value from settings, falling back to a default."""
        return getattr(settings, attr_name, default_value)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct numeric type."""
        df_copy = df.copy()
        
        # Define all columns the model might use and their default "no-risk" values.
        # This makes the model robust to incomplete input data.
        required_cols = {
            'age': 30, 'chronic_condition_flag': 0, 'min_spo2_pct': 98,
            'vital_signs_temperature_celsius': 37.0, 'signs_of_fatigue_observed_flag': 0,
        }
        for col, default in required_cols.items():
            if col not in df_copy.columns:
                df_copy[col] = default
            # Ensure columns are numeric for calculation
            df_copy[col] = convert_to_numeric(df_copy[col], default_value=default)
            
        return df_copy

    def calculate_risk_scores(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates risk scores for each record in the DataFrame using vectorized operations.
        
        Args:
            health_df: The input pandas DataFrame.
            
        Returns:
            The DataFrame with an added 'ai_risk_score' column.
        """
        if not isinstance(health_df, pd.DataFrame):
            return pd.DataFrame()
        if health_df.empty:
            health_df['ai_risk_score'] = pd.Series(dtype=float)
            return health_df

        df = self._prepare_data(health_df)
        
        # --- Vectorized Score Calculation ---
        
        # Start with a base score defined in settings
        risk_scores = pd.Series(self.base_scores.get('default', 5.0), index=df.index)

        # Apply age-based adjustments from settings
        age_factor = self.factors.get('age', {})
        if age_factor:
            risk_scores += np.where(df['age'] >= age_factor.get('elderly_threshold', 65), age_factor.get('elderly_modifier', 15), 0)
            risk_scores += np.where(df['age'] <= age_factor.get('pediatric_threshold', 5), age_factor.get('pediatric_modifier', 10), 0)

        # Apply SpO2 adjustments from settings
        spo2_factor = self.factors.get('spo2', {})
        if spo2_factor:
            risk_scores += np.where(df['min_spo2_pct'] < spo2_factor.get('low_threshold', 92), spo2_factor.get('low_modifier', 20), 0)

        # Apply temperature adjustments from settings
        temp_factor = self.factors.get('temperature', {})
        if temp_factor:
            risk_scores += np.where(df['vital_signs_temperature_celsius'] > temp_factor.get('high_threshold', 38.0), temp_factor.get('high_modifier', 15), 0)

        # Apply other flag-based modifiers from settings
        risk_scores += df['chronic_condition_flag'] * self.factors.get('chronic_condition_modifier', 20)
        risk_scores += df['signs_of_fatigue_observed_flag'] * self.factors.get('fatigue_modifier', 10)
        
        # Finalize scores by clipping to the 0-100 range and rounding
        final_scores = risk_scores.clip(0, 100).round(1)
        
        logger.info(f"({self.module_log_prefix}) Generated {len(final_scores)} AI risk scores.")
        
        # Add or update the column in the original DataFrame structure
        health_df['ai_risk_score'] = final_scores
        return health_df
