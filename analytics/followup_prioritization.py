import pandas as pd
import numpy as np
import logging
from typing import Any, Dict

# --- Module Imports ---
try:
    from config import settings
except ImportError as e:
    # This setup is for standalone execution or if imports fail.
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in followup_prioritization.py: {e}. Ensure paths are correct.", exc_info=True)
    raise

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


class FollowUpPrioritizer:
    """
    Simulates logic to prioritize patients for follow-up based on AI risk,
    vital sign alerts, task urgency, and contextual factors.
    Score is intended to be between 0 and 100.
    """
    # FIXED: Renamed `init` to the correct Python constructor `__init__`.
    def __init__(self):
        """Initializes the prioritizer with configured weights."""
        self.priority_weights: Dict[str, float] = {
            'base_ai_risk_score_contribution_pct': 0.40,
            'critical_vital_alert_points': 40.0,
            'pending_urgent_task_points': 30.0,
            'acute_condition_severity_points': 25.0,
            'contextual_hazard_points': 15.0,
            'task_overdue_factor_per_day': 1.5,  # Capped effect
            'poor_adherence_points': 15.0,
            'observed_fatigue_points': 12.0
        }
        logger.info("FollowUpPrioritizer initialized with configured weights.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        """Checks for active critical vital sign alerts based on input features."""
        if not isinstance(features, pd.Series): return False

        min_spo2 = pd.to_numeric(features.get('min_spo2_pct'), errors='coerce')
        if pd.notna(min_spo2) and min_spo2 < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            return True

        temp_c_vital = pd.to_numeric(features.get('vital_signs_temperature_celsius'), errors='coerce')
        temp_c_skin = pd.to_numeric(features.get('max_skin_temp_celsius'), errors='coerce')
        effective_temp_c = temp_c_vital if pd.notna(temp_c_vital) else temp_c_skin

        if pd.notna(effective_temp_c) and effective_temp_c >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            return True
        
        fall_detected_str = str(features.get('fall_detected_today', '0')).strip().lower()
        if fall_detected_str in ['1', 'true', 'yes']:
            return True
            
        return False

    def _is_pending_urgent_task_or_referral(self, features: pd.Series) -> bool:
        """Checks if there's a pending urgent task or referral."""
        if not isinstance(features, pd.Series): return False

        referral_status = str(features.get('referral_status', '')).strip().lower()
        if referral_status == 'pending':
            condition_str = str(features.get('condition', '')).strip().lower()
            referral_reason_str = str(features.get('referral_reason', '')).strip().lower()
            
            is_key_condition = any(kc.lower() in condition_str for kc in settings.KEY_CONDITIONS_FOR_ACTION)
            is_urgent_reason = any(urgent_word in referral_reason_str for urgent_word in ['urgent', 'critical', 'emergency'])
            
            if is_key_condition or is_urgent_reason:
                return True
            
        return False

    def _has_acute_condition_with_severity_indicators(self, features: pd.Series) -> bool:
        """Checks for acute conditions combined with severity indicators."""
        if not isinstance(features, pd.Series): return False
        
        condition_str = str(features.get('condition', '')).strip().lower()
        
        min_spo2 = pd.to_numeric(features.get('min_spo2_pct'), errors='coerce')
        if "pneumonia" in condition_str and pd.notna(min_spo2) and min_spo2 < settings.ALERT_SPO2_WARNING_LOW_PCT:
            return True
            
        if any(crit_cond.lower() in condition_str for crit_cond in ["sepsis", "severe dehydration", "heat stroke"]):
            return True
            
        return False

    def _has_significant_contextual_hazard(self, features: pd.Series) -> bool:
        """Checks for significant contextual hazards affecting the patient."""
        if not isinstance(features, pd.Series): return False

        ambient_heat = pd.to_numeric(features.get('ambient_heat_index_c'), errors='coerce')
        if pd.notna(ambient_heat) and ambient_heat >= settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C:
            return True
        
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        """
        Calculates a follow-up priority score for a single patient/task.
        Score is between 0 and 100.
        """
        if not isinstance(features, pd.Series):
            logger.error("FollowUpPrioritizer.calculate_priority_score expects a pandas Series for features.")
            return 0.0
        
        priority_score = 0.0

        # Base score from AI risk
        ai_risk_raw = features.get('ai_risk_score', 0.0)
        ai_risk = pd.to_numeric(ai_risk_raw, errors='coerce')
        # Use 0.0 if the value is null/NaN after conversion
        ai_risk = 0.0 if pd.isna(ai_risk) else ai_risk
        priority_score += ai_risk * self.priority_weights['base_ai_risk_score_contribution_pct']

        # Add points based on triggered rules
        if self._has_active_critical_vitals_alert(features):
            priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task_or_referral(features):
            priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_with_severity_indicators(features):
            priority_score += self.priority_weights['acute_condition_severity_points']
        if self._has_significant_contextual_hazard(features):
            priority_score += self.priority_weights['contextual_hazard_points']

        # Add points for other contextual factors
        if str(features.get('medication_adherence_self_report', 'Unknown')).strip().lower() == 'poor':
            priority_score += self.priority_weights['poor_adherence_points']
        
        if str(features.get('signs_of_fatigue_observed_flag', '0')).strip().lower() in ['1', 'true', 'yes']:
             priority_score += self.priority_weights['observed_fatigue_points']

        # Add points for task being overdue
        overdue_days_cleaned = max(0, int(days_task_overdue)) if pd.notna(days_task_overdue) else 0
        overdue_days_capped = min(overdue_days_cleaned, 30) # Cap effect to avoid runaway scores
        priority_score += overdue_days_capped * self.priority_weights['task_overdue_factor_per_day']
        
        # Clip score to be within the 0-100 range and return as a float
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Generates follow-up priority scores for each row in the input DataFrame.
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("Input DataFrame to generate_followup_priorities is empty or invalid.")
            return pd.Series(dtype='float64')

        df_processed = data_df.copy()

        features_to_ensure: Dict[str, Any] = {
            'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0, 
            'max_skin_temp_celsius': 37.0, 'fall_detected_today': '0',
            'referral_status': 'Unknown', 'condition': 'UnknownCondition', 
            'referral_reason': 'N/A', 'ambient_heat_index_c': 25.0,
            'medication_adherence_self_report': 'Unknown',
            'signs_of_fatigue_observed_flag': '0', 'ai_risk_score': 0.0,
            'days_task_overdue': 0
        }

        for feature_col, default_val in features_to_ensure.items():
            if feature_col not in df_processed.columns:
                df_processed[feature_col] = default_val
            else:
                if pd.api.types.is_numeric_dtype(type(default_val)):
                    df_processed[feature_col] = pd.to_numeric(df_processed[feature_col], errors='coerce').fillna(default_val)
                else:
                    df_processed[feature_col] = df_processed[feature_col].fillna(str(default_val)).astype(str)
        
        df_processed['days_task_overdue'] = df_processed['days_task_overdue'].astype(int)

        try:
            priority_scores = df_processed.apply(
                lambda row: self.calculate_priority_score(row, row['days_task_overdue']),
                axis=1
            )
        except Exception as e_apply_prio:
            logger.error(f"Error during bulk application of calculate_priority_score: {e_apply_prio}", exc_info=True)
            return pd.Series([0.0] * len(df_processed), index=df_processed.index, dtype='float64')
        
        logger.info(f"Generated {len(priority_scores)} follow-up priority scores.")
        return priority_scores
