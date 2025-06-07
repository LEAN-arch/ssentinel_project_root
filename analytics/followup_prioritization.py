# sentinel_project_root/analytics/followup_prioritization.py
# Contains the FollowUpPrioritizer class for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Optional

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric, standardize_missing_values
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in followup_prioritization.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

# --- Constants for Priority Factors ---
# Centralizing weights makes the model easier to understand, tune, and maintain.
PRIORITY_FACTORS: Dict[str, float] = {
    'BASE_AI_RISK_CONTRIBUTION_PCT': 0.40,
    'CRITICAL_VITAL_ALERT_BONUS': 40.0,
    'PENDING_URGENT_TASK_BONUS': 30.0,
    'ACUTE_CONDITION_SEVERITY_BONUS': 25.0,
    'POOR_ADHERENCE_BONUS': 15.0,
    'OBSERVED_FATIGUE_BONUS': 12.0,
    'TASK_OVERDUE_PENALTY_PER_DAY': 1.5,
    'MAX_OVERDUE_DAYS_FOR_PENALTY': 30 # Prevents runaway scores for very old tasks
}


class FollowUpPrioritizer:
    """
    Calculates a patient follow-up priority score (0-100) based on AI risk,
    vital sign alerts, task urgency, and contextual factors.
    This version is optimized for vectorized operations for improved performance.
    """
    def __init__(self):
        self.priority_factors = PRIORITY_FACTORS
        logger.info("FollowUpPrioritizer initialized with configured factors.")

    def _get_feature_value(self, features: pd.Series, key: str, default: Any = 0.0) -> Any:
        """Safely gets and converts a feature's value."""
        val = features.get(key, default)
        # Use a more targeted conversion based on default type
        if isinstance(default, (float, int)):
            return pd.to_numeric(val, errors='coerce').fillna(default)
        return val

    def calculate_priority_score(self, features: pd.Series) -> float:
        """
        Calculates a follow-up priority score for a single patient record.
        This method is kept for single-record prediction and testing.
        """
        if not isinstance(features, pd.Series):
            logger.error("calculate_priority_score expects a pandas Series.")
            return 0.0

        # --- Base Score from AI Risk ---
        ai_risk = self._get_feature_value(features, 'ai_risk_score')
        score = ai_risk * self.priority_factors['BASE_AI_RISK_CONTRIBUTION_PCT']
        
        # --- Additive Bonuses for High-Impact Events ---
        if self._get_feature_value(features, 'has_critical_vitals_alert', False):
            score += self.priority_factors['CRITICAL_VITAL_ALERT_BONUS']
        if self._get_feature_value(features, 'has_pending_urgent_task', False):
            score += self.priority_factors['PENDING_URGENT_TASK_BONUS']
        if self._get_feature_value(features, 'has_acute_condition_severity', False):
            score += self.priority_factors['ACUTE_CONDITION_SEVERITY_BONUS']

        # --- Contextual & Behavioral Factors ---
        adherence = str(self._get_feature_value(features, 'medication_adherence_self_report', 'Unknown')).lower()
        if adherence == 'poor':
            score += self.priority_factors['POOR_ADHERENCE_BONUS']
            
        if self._get_feature_value(features, 'signs_of_fatigue_observed_flag', False):
            score += self.priority_factors['OBSERVED_FATIGUE_BONUS']

        # --- Overdue Task Penalty ---
        overdue_days = self._get_feature_value(features, 'days_task_overdue', 0)
        overdue_penalty = min(overdue_days, self.priority_factors['MAX_OVERDUE_DAYS_FOR_PENALTY']) * self.priority_factors['TASK_OVERDUE_PENALTY_PER_DAY']
        score += overdue_penalty
        
        final_score = float(np.clip(score, 0, 100))
        logger.debug(f"Prio score for PID {features.get('patient_id', 'N/A')}: {final_score:.1f} (Base AI Risk: {ai_risk:.1f})")
        return final_score

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Generates follow-up priority scores for each row in the input DataFrame
        using efficient, vectorized operations.
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("Input DataFrame to generate_followup_priorities is empty.")
            return pd.Series(dtype='float64')

        # --- Data Preparation ---
        df = data_df.copy()
        numeric_defaults = {
            'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0,
            'fall_detected_today': 0, 'ai_risk_score': 0.0, 'days_task_overdue': 0,
            'signs_of_fatigue_observed_flag': 0
        }
        string_defaults = {
            'referral_status': "Unknown", 'condition': "UnknownCondition",
            'medication_adherence_self_report': "Unknown",
        }
        df = standardize_missing_values(df, string_defaults, numeric_defaults)

        # --- Vectorized Score Calculation ---
        
        # 1. Base score from AI risk
        scores = df['ai_risk_score'] * self.priority_factors['BASE_AI_RISK_CONTRIBUTION_PCT']

        # 2. Additive Bonuses for High-Impact Events
        # Critical vitals
        crit_vitals_mask = (
            (df['min_spo2_pct'] < settings.Thresholds.SPO2_CRITICAL_LOW) |
            (df['vital_signs_temperature_celsius'] >= settings.Thresholds.BODY_TEMP_HIGH_FEVER) |
            (df['fall_detected_today'] > 0)
        )
        scores.loc[crit_vitals_mask] += self.priority_factors['CRITICAL_VITAL_ALERT_BONUS']

        # Pending urgent referrals
        key_conds_pattern = '|'.join(settings.Semantics.KEY_CONDITIONS_FOR_ACTION)
        pending_referral_mask = (df['referral_status'].str.lower() == 'pending') & \
                                (df['condition'].str.contains(key_conds_pattern, case=False, na=False))
        scores.loc[pending_referral_mask] += self.priority_factors['PENDING_URGENT_TASK_BONUS']

        # Acute condition with severity
        pneumonia_mask = df['condition'].str.contains('pneumonia', case=False, na=False) & \
                         (df['min_spo2_pct'] < settings.Thresholds.SPO2_WARNING_LOW)
        other_acute_mask = df['condition'].str.contains('sepsis|severe dehydration|heat stroke', case=False, na=False)
        scores.loc[pneumonia_mask | other_acute_mask] += self.priority_factors['ACUTE_CONDITION_SEVERITY_BONUS']

        # 3. Contextual & Behavioral Factors
        scores.loc[df['medication_adherence_self_report'].str.lower() == 'poor'] += self.priority_factors['POOR_ADHERENCE_BONUS']
        scores.loc[df['signs_of_fatigue_observed_flag'] > 0] += self.priority_factors['OBSERVED_FATIGUE_BONUS']

        # 4. Overdue Task Penalty
        overdue_penalty = df['days_task_overdue'].clip(0, self.priority_factors['MAX_OVERDUE_DAYS_FOR_PENALTY']) * self.priority_factors['TASK_OVERDUE_PENALTY_PER_DAY']
        scores += overdue_penalty
        
        # 5. Final Clipping
        final_scores = scores.clip(0, 100).rename("ai_followup_priority_score")
        
        logger.info(f"Generated {len(final_scores)} follow-up priority scores using vectorized operations.")
        return final_scores
