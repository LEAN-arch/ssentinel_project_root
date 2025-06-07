# sentinel_project_root/analytics/followup_prioritization.py
"""
Contains the Sentinel Follow-up Prioritization model.
Calculates a score indicating the urgency of a follow-up action using a declarative,
data-driven rules engine with efficient, vectorized operations.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in followup_prioritization.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class FollowUpPrioritizer:
    """
    A rule-based model to calculate a follow-up priority score.
    This implementation is fully vectorized for high performance and uses a declarative
    rules configuration for maintainability and easy extension.
    """
    def __init__(self):
        """Initializes the prioritizer with weights and rules from settings."""
        self.weights = self._get_setting('PRIORITY_MODEL_WEIGHTS', {})
        self.rules = self._get_rules_config()

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        """Safely gets a configuration value from the global settings."""
        return getattr(settings, attr_name, default_value)

    def _get_rules_config(self) -> List[Dict[str, Any]]:
        """
        Defines the declarative rule set for calculating the priority score.
        Each rule has a condition (a lambda that returns a boolean Series) and points to add.
        """
        key_cond_pattern = '|'.join(self._get_setting('KEY_CONDITIONS_FOR_ACTION', []))
        return [
            {"condition": lambda df: df['min_spo2_pct'] < self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "points": self.weights.get('critical_vital_alert_points', 40.0), "reason": "Critical SpO2"},
            {"condition": lambda df: df['vital_signs_temperature_celsius'] >= self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "points": self.weights.get('critical_vital_alert_points', 40.0), "reason": "High Fever"},
            {"condition": lambda df: df['fall_detected_today'] > 0, "points": self.weights.get('critical_vital_alert_points', 40.0), "reason": "Fall Detected"},
            {"condition": lambda df: (df['referral_status'].str.lower() == 'pending') & (df['condition'].str.contains(key_cond_pattern, case=False, na=False)), "points": self.weights.get('pending_urgent_task_points', 30.0), "reason": "Pending Urgent Referral"},
            {"condition": lambda df: df['medication_adherence_self_report'].str.lower() == 'poor', "points": self.weights.get('poor_adherence_points', 15.0), "reason": "Poor Adherence"},
            {"condition": lambda df: df['signs_of_fatigue_observed_flag'] > 0, "points": self.weights.get('observed_fatigue_points', 12.0), "reason": "Fatigue Observed"},
        ]

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct type."""
        clean_df = df.copy()
        # Define all columns the model might use and their default neutral values.
        required_cols = {
            'ai_risk_score': 0.0, 'days_task_overdue': 0,
            'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0, 'fall_detected_today': 0,
            'signs_of_fatigue_observed_flag': 0, 'referral_status': "Complete", 'condition': "No Condition",
            'medication_adherence_self_report': "Good"
        }
        for col, default in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default
            # Ensure columns used in calculations are numeric where appropriate.
            if pd.api.types.is_numeric_dtype(type(default)):
                 clean_df[col] = convert_to_numeric(clean_df[col], default_value=default)
            else: # Handle string columns
                 clean_df[col] = clean_df[col].astype(str).fillna(str(default))
        return clean_df

    def generate_priority_scores(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates follow-up priority scores and reasons using a vectorized rules engine.
        """
        if not isinstance(health_df, pd.DataFrame):
            return pd.DataFrame()
        if health_df.empty:
            health_df['ai_followup_priority_score'] = np.nan
            health_df['priority_reasons'] = ""
            return health_df

        df = self._prepare_data(health_df)
        
        # Initialize scores and reasons Series
        priority_scores = pd.Series(0.0, index=df.index)
        
        # --- Vectorized Score Calculation ---
        # 1. Base score from AI risk model
        priority_scores += df['ai_risk_score'] * self.weights.get('w_risk_score', 0.5)
        
        # 2. Dynamic points for overdue tasks
        overdue_days = df['days_task_overdue'].clip(upper=30)
        priority_scores += overdue_days * self.weights.get('task_overdue_factor_per_day', 1.5)

        # 3. Apply the declarative rule set and collect reasons
        triggered_reasons = []
        for rule in self.rules:
            try:
                triggered_mask = rule["condition"](df)
                if triggered_mask.any():
                    priority_scores.loc[triggered_mask] += rule["points"]
                    # For all triggered rows, add the reason string, otherwise add an empty string.
                    triggered_reasons.append(np.where(triggered_mask, rule["reason"], ""))
            except Exception as e:
                logger.error(f"Error evaluating priority rule '{rule.get('reason', 'Unnamed')}': {e}", exc_info=True)

        # 4. Concatenate all reasons into a single, clean string
        if triggered_reasons:
            reasons_df = pd.DataFrame(triggered_reasons).T
            df['priority_reasons'] = reasons_df.apply(lambda row: ', '.join(r for r in row if r), axis=1)
        else:
            df['priority_reasons'] = ""
        
        # 5. Finalize scores
        df['ai_followup_priority_score'] = priority_scores.clip(0, 100).round(1)
        
        logger.info(f"Generated {len(df)} follow-up priority scores.")
        return df


def calculate_followup_priority(health_df: pd.DataFrame) -> pd.DataFrame:
    """
    Public factory function to calculate follow-up priority scores for a given DataFrame.

    This function instantiates the `FollowUpPrioritizer` and applies its logic.
    It returns the input DataFrame with two new columns:
    
    - `ai_followup_priority_score` (float): A numeric score from 0-100 indicating urgency.
    - `priority_reasons` (str): A comma-separated string of reasons explaining the score.

    Args:
        health_df (pd.DataFrame): The input DataFrame containing patient health data.

    Returns:
        pd.DataFrame: The enriched DataFrame with priority scores and reasons.
    """
    prioritizer = FollowUpPrioritizer()
    return prioritizer.generate_priority_scores(health_df)
