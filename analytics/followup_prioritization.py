# sentinel_project_root/analytics/followup_prioritization.py
"""
Contains the Sentinel Follow-up Prioritization model.
Calculates a score indicating the urgency of a follow-up action using efficient, vectorized operations.
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
    logger_init.error(f"Critical import error in followup_prioritization.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class FollowUpPrioritizer:
    """
    A rule-based model to calculate a follow-up priority score.
    Higher scores indicate a more urgent need for follow-up.
    This implementation is vectorized for high performance.
    """
    def __init__(self):
        self.module_log_prefix = self.__class__.__name__
        self.weights = self._get_setting('PRIORITY_MODEL_WEIGHTS', {})
        logger.info(f"({self.module_log_prefix}) Initialized with configured weights.")

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct type."""
        clean_df = df.copy()
        # Define all columns the model might use and their default neutral values
        required_cols = {
            'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0,
            'fall_detected_today': 0, 'referral_status': "Unknown",
            'condition': "UnknownCondition", 'ai_risk_score': 0.0,
            'days_task_overdue': 0, 'medication_adherence_self_report': "Good",
            'signs_of_fatigue_observed_flag': 0
        }
        for col, default in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default
            # Ensure columns are numeric for calculation
            clean_df[col] = convert_to_numeric(clean_df[col], default_value=default)
        return clean_df

    def _get_critical_vitals_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean Series indicating rows with critical vital signs."""
        spo2_mask = df['min_spo2_pct'] < self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88)
        temp_mask = df['vital_signs_temperature_celsius'] >= self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0)
        fall_mask = df['fall_detected_today'] > 0
        return spo2_mask | temp_mask | fall_mask

    def _get_pending_referral_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean Series for rows with pending critical referrals."""
        is_pending = df['referral_status'].astype(str).str.lower() == 'pending'
        key_conditions = self._get_setting('KEY_CONDITIONS_FOR_ACTION', [])
        condition_str_lower = df['condition'].astype(str).str.lower()
        has_key_condition = condition_str_lower.str.contains('|'.join(key_conditions), case=False, na=False)
        return is_pending & has_key_condition

    def generate_priority_scores(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates follow-up priority scores for the DataFrame using vectorized operations.
        """
        if not isinstance(health_df, pd.DataFrame) or health_df.empty:
            if 'ai_followup_priority_score' not in health_df.columns:
                health_df['ai_followup_priority_score'] = np.nan
            return health_df

        df = self._prepare_data(health_df)
        
        # --- Vectorized Score Calculation ---
        # 1. Start with the base score from AI risk
        priority_scores = df['ai_risk_score'] * self.weights.get('w_risk_score', 0.5)

        # 2. Add points based on triggered rules (masks)
        priority_scores += self._get_critical_vitals_mask(df) * self.weights.get('critical_vital_alert_points', 40.0)
        priority_scores += self._get_pending_referral_mask(df) * self.weights.get('pending_urgent_task_points', 30.0)

        # 3. Add points for other contextual factors
        priority_scores += (df['medication_adherence_self_report'].astype(str).str.lower() == 'poor') * self.weights.get('poor_adherence_points', 15.0)
        priority_scores += (df['signs_of_fatigue_observed_flag'] > 0) * self.weights.get('observed_fatigue_points', 12.0)
        
        # 4. Add points for task being overdue (with a cap)
        overdue_days_capped = df['days_task_overdue'].clip(upper=30)
        priority_scores += overdue_days_capped * self.weights.get('task_overdue_factor_per_day', 1.5)
        
        # 5. Finalize scores by clipping to the 0-100 range
        final_scores = priority_scores.clip(0, 100).round(1)
        
        logger.info(f"({self.module_log_prefix}) Generated {len(final_scores)} follow-up priority scores.")
        df['ai_followup_priority_score'] = final_scores
        
        return df
