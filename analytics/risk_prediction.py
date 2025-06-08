# sentinel_project_root/analytics/risk_prediction.py
"""
Contains the Sentinel AI Risk Prediction model.
Calculates a patient's risk score using a declarative, data-driven rules
engine with efficient, vectorized operations.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

try:
    from config import settings
    from data_processing.helpers import data_cleaner
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in risk_prediction.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    A rule-based model to simulate a patient risk score based on clinical and
    contextual data. Higher scores indicate a higher risk of adverse outcomes.
    """
    def __init__(self):
        """Initializes the model with weights from the settings file."""
        self.weights = getattr(settings, 'RISK_MODEL_WEIGHTS', {})

    def _get_rules_config(self) -> List[Dict[str, Any]]:
        """
        Defines the declarative rule set for calculating the risk score.
        Each rule has a condition and the points to add if met.
        """
        return [
            {"condition": lambda df: df['min_spo2_pct'] < 92, "points": self.weights.get('spo2_low_points', 25)},
            {"condition": lambda df: df['vital_signs_temperature_celsius'] >= 38.0, "points": self.weights.get('fever_points', 15)},
            {"condition": lambda df: df['vital_signs_temperature_celsius'] < 35.5, "points": self.weights.get('hypothermia_points', 20)},
            {"condition": lambda df: df['chronic_condition_flag'] > 0, "points": self.weights.get('chronic_condition_points', 15)},
            {"condition": lambda df: df['age'] > 65, "points": self.weights.get('age_over_65_points', 10)},
            {"condition": lambda df: df['age'] < 5, "points": self.weights.get('age_under_5_points', 10)},
            {"condition": lambda df: df['tb_contact_traced'] > 0, "points": self.weights.get('tb_contact_points', 20)},
            {"condition": lambda df: df['fall_detected_today'] > 0, "points": self.weights.get('fall_detected_points', 25)},
            {"condition": lambda df: df['ppe_compliant_flag'] == 0, "points": self.weights.get('no_ppe_points', 5)},
        ]

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all required columns exist and are of the correct numeric type
        by using the centralized data_cleaner and configurations from settings.
        """
        prepared_df = df.copy()
        
        # Use the declarative defaults from settings to ensure all required
        # columns for the model are present and correctly typed.
        numeric_defaults = getattr(settings, 'RISK_MODEL_NUMERIC_DEFAULTS', {})
        string_defaults = getattr(settings, 'RISK_MODEL_STRING_DEFAULTS', {})
        
        return data_cleaner.standardize_missing_values(
            prepared_df,
            string_cols_defaults=string_defaults,
            numeric_cols_defaults=numeric_defaults
        )

    def predict_risk_scores(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates risk scores for the DataFrame using a vectorized rules engine.
        """
        if not isinstance(health_df, pd.DataFrame) or health_df.empty:
            if 'ai_risk_score' not in health_df.columns:
                health_df['ai_risk_score'] = np.nan
            return health_df

        # --- DEFINITIVE FIX FOR TypeError ---
        # The prepare_data method must be called here to clean the data
        # before any rules are evaluated against it.
        df = self._prepare_data(health_df)
        
        risk_scores = pd.Series(0.0, index=df.index)
        
        for rule in self._get_rules_config():
            try:
                triggered_mask = rule["condition"](df)
                risk_scores.loc[triggered_mask] += rule["points"]
            except Exception as e:
                condition_str = str(rule.get('condition'))
                logger.error(f"Error evaluating risk rule ({condition_str}): {e}")

        df['ai_risk_score'] = risk_scores.clip(0, 100).round(1)
        logger.info(f"Generated {len(df)} AI risk scores.")
        return df

def calculate_risk_score(health_df: pd.DataFrame) -> pd.DataFrame:
    """
    Public factory function to calculate AI risk scores.
    """
    model = RiskPredictionModel()
    return model.predict_risk_scores(health_df)
