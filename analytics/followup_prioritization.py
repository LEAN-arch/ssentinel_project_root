# ssentinel_project_root/analytics/followup_prioritization.py
"""
Contains the Sentinel Follow-up Prioritization model.
Calculates a score indicating the urgency of a follow-up action using efficient, vectorized operations.
"""
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in followup_prioritization.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class FollowUpPrioritizer:
    """
    A rule-based model to calculate a follow-up priority score.
    Higher scores indicate a more urgent need for follow-up.
    This implementation is vectorized for high performance and configurable via settings.
    """
    def __init__(self):
        """Initializes the model, loading weights and thresholds from settings."""
        self.module_log_prefix = self.__class__.__name__
        
        # Load weights from settings, with robust defaults.
        default_weights = {
            'w_risk_score': 0.6, 'critical_vital_alert_points': 45.0,
            'pending_urgent_task_points': 35.0, 'poor_adherence_points': 15.0,
            'observed_fatigue_points': 12.0, 'task_overdue_factor_per_day': 1.5,
        }
        self.weights = self._get_setting('PRIORITY_MODEL_WEIGHTS', default_weights)
        logger.info(f"({self.module_log_prefix}) Initialized with configured weights.")

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        """Safely retrieves a value from settings, falling back to a default."""
        return getattr(settings, attr_name, default_value)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct type."""
        df_copy = df.copy()
        
        required_cols = {
            'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0,
            'fall_detected_today': 0, 'referral_status': "Not Referred",
            'condition': "Wellness Visit", 'medication_adherence_self_report': "Good",
            'signs_of_fatigue_observed_flag': 0, 'days_task_overdue': 0,
            'ai_risk_score': 0.0,
        }
        for col, default in required_cols.items():
            if col not in df_copy.columns:
                df_copy[col] = default

            if col not in ['referral_status', 'condition', 'medication_adherence_self_report']:
                df_copy[col] = convert_to_numeric(df_copy[col], default_value=default)
        
        return df_copy

    def _get_critical_vitals_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean Series indicating rows with critical vital signs."""
        spo2_mask = df['min_spo2_pct'] < self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90)
        temp_mask = df['vital_signs_temperature_celsius'] >= self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
        fall_mask = df['fall_detected_today'] > 0
        return spo2_mask | temp_mask | fall_mask

    def _get_pending_referral_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean Series for rows with pending critical referrals."""
        is_pending = df['referral_status'].astype(str).str.lower() == 'pending'
        
        key_conditions = self._get_setting('KEY_CONDITIONS_FOR_ACTION', [])
        if not key_conditions:
            return pd.Series(False, index=df.index)
            
        key_condition_pattern = '|'.join([re.escape(c) for c in key_conditions])
        has_key_condition = df['condition'].str.contains(key_condition_pattern, case=False, na=False, regex=True)
        
        return is_pending & has_key_condition

    def generate_priority_scores(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates follow-up priority scores for the DataFrame using vectorized operations.
        
        Args:
            health_df: The input DataFrame, ideally already processed by the RiskPredictionModel.
        
        Returns:
            The DataFrame with an added/updated 'ai_followup_priority_score' column.
        """
        if not isinstance(health_df, pd.DataFrame):
            return pd.DataFrame()
        if health_df.empty:
            health_df['ai_followup_priority_score'] = pd.Series(dtype=float)
            return health_df

        df = self._prepare_data(health_df)
        
        # --- Vectorized Score Calculation ---
        priority_scores = df['ai_risk_score'] * self.weights.get('w_risk_score', 0.6)
        priority_scores += self._get_critical_vitals_mask(df) * self.weights.get('critical_vital_alert_points', 45.0)
        priority_scores += self._get_pending_referral_mask(df) * self.weights.get('pending_urgent_task_points', 35.0)
        priority_scores += (df['medication_adherence_self_report'].astype(str).str.lower() == 'poor') * self.weights.get('poor_adherence_points', 15.0)
        priority_scores += (df['signs_of_fatigue_observed_flag'] > 0) * self.weights.get('observed_fatigue_points', 12.0)
        
        overdue_days_capped = df['days_task_overdue'].clip(upper=30)
        priority_scores += overdue_days_capped * self.weights.get('task_overdue_factor_per_day', 1.5)
        
        final_scores = priority_scores.clip(0, 100).round(1)
        
        logger.info(f"({self.module_log_prefix}) Generated {len(final_scores)} follow-up priority scores.")
        
        health_df['ai_followup_priority_score'] = final_scores
        
        return health_df
