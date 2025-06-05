# sentinel_project_root/analytics/followup_prioritization.py
# Contains the FollowUpPrioritizer class for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict

from config import settings
from .protocol_executor import execute_escalation_protocol

logger = logging.getLogger(__name__)

class FollowUpPrioritizer:
    """
    Simulates logic to prioritize patients for follow-up based on AI risk,
    vital sign alerts, task urgency, and contextual factors.
    """
    def __init__(self):
        self.priority_weights: Dict[str, float] = {
            'base_ai_risk_score_contribution_pct': 0.40,
            'critical_vital_alert_points': 40.0,
            'pending_urgent_task_points': 30.0,
            'acute_condition_severity_points': 25.0,
            'contextual_hazard_points': 15.0,
            'task_overdue_factor_per_day': 1.5,
            'poor_adherence_points': 15.0,
            'observed_fatigue_points': 12.0
        }
        logger.info("FollowUpPrioritizer initialized.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        if not isinstance(features, pd.Series): return False
        min_spo2 = pd.to_numeric(features.get('min_spo2_pct'), errors='coerce')
        if pd.notna(min_spo2) and min_spo2 < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            return True
        temp_c = pd.to_numeric(features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius')), errors='coerce')
        if pd.notna(temp_c) and temp_c >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            return True
        if str(features.get('fall_detected_today', '0')).lower() in ['1', 'true', 'yes']:
            return True
        return False

    def _is_pending_urgent_task_or_referral(self, features: pd.Series) -> bool:
        if not isinstance(features, pd.Series): return False
        referral_status = str(features.get('referral_status', '')).lower()
        if referral_status == 'pending':
            condition_str = str(features.get('condition', '')).lower()
            referral_reason_str = str(features.get('referral_reason', '')).lower()
            if any(kc.lower() in condition_str for kc in settings.KEY_CONDITIONS_FOR_ACTION) or \
               any(urgent_word in referral_reason_str for urgent_word in ['urgent', 'critical', 'emergency']):
                return True
        return False

    def _has_acute_condition_with_severity_indicators(self, features: pd.Series) -> bool:
        if not isinstance(features, pd.Series): return False
        condition_str = str(features.get('condition', '')).lower()
        min_spo2 = pd.to_numeric(features.get('min_spo2_pct'), errors='coerce')
        if "pneumonia" in condition_str and pd.notna(min_spo2) and \
           min_spo2 < settings.ALERT_SPO2_WARNING_LOW_PCT:
            return True
        if any(crit_cond.lower() in condition_str for crit_cond in ["sepsis", "severe dehydration", "heat stroke"]):
            return True
        return False

    def _has_significant_contextual_hazard(self, features: pd.Series) -> bool:
        if not isinstance(features, pd.Series): return False
        ambient_heat = pd.to_numeric(features.get('ambient_heat_index_c'), errors='coerce')
        if pd.notna(ambient_heat) and ambient_heat >= settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C:
            return True
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        if not isinstance(features, pd.Series):
            logger.error("calculate_priority_score expects a pandas Series.")
            return 0.0
        
        priority_score = 0.0
        ai_risk = pd.to_numeric(features.get('ai_risk_score', 0.0), errors='coerce').fillna(0.0)
        priority_score += ai_risk * self.priority_weights['base_ai_risk_score_contribution_pct']

        if self._has_active_critical_vitals_alert(features):
            priority_score += self.priority_weights['critical_vital_alert_points']
        if self._is_pending_urgent_task_or_referral(features):
            priority_score += self.priority_weights['pending_urgent_task_points']
        if self._has_acute_condition_with_severity_indicators(features):
            priority_score += self.priority_weights['acute_condition_severity_points']
        if self._has_significant_contextual_hazard(features):
            priority_score += self.priority_weights['contextual_hazard_points']
        if str(features.get('medication_adherence_self_report', 'Unknown')).lower() == 'poor':
            priority_score += self.priority_weights['poor_adherence_points']
        if str(features.get('signs_of_fatigue_observed_flag', '0')).lower() in ['1', 'true', 'yes']:
             priority_score += self.priority_weights['observed_fatigue_points']
        
        overdue_days_capped = min(max(0, int(days_task_overdue)), 30)
        priority_score += overdue_days_capped * self.priority_weights['task_overdue_factor_per_day']
        
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("Input DataFrame to generate_followup_priorities is empty or invalid.")
            return pd.Series(dtype='float64')

        df_processed = data_df.copy()

        features_to_ensure_with_defaults = {
            'min_spo2_pct': 100.0, 'vital_signs_temperature_celsius': 37.0,
            'max_skin_temp_celsius': 37.0, 'fall_detected_today': '0',
            'referral_status': "Unknown", 'condition': "UnknownCondition",
            'referral_reason': "N/A", 'ambient_heat_index_c': 25.0,
            'medication_adherence_self_report': "Unknown",
            'signs_of_fatigue_observed_flag': '0', 'ai_risk_score': 0.0,
            'days_task_overdue': 0
        }
        for feature_col, default_val in features_to_ensure_with_defaults.items():
            if feature_col not in df_processed.columns:
                df_processed[feature_col] = default_val
            else:
                if isinstance(default_val, (float, int)):
                    df_processed[feature_col] = pd.to_numeric(df_processed[feature_col], errors='coerce').fillna(default_val)
                else:
                    df_processed[feature_col] = df_processed[feature_col].fillna(str(default_val)).astype(str)
        
        # Ensure days_task_overdue is int after filling
        df_processed['days_task_overdue'] = df_processed['days_task_overdue'].astype(int)

        priority_scores = df_processed.apply(
            lambda row: self.calculate_priority_score(row, row['days_task_overdue']),
            axis=1
        )
        logger.info(f"Generated {len(priority_scores)} follow-up priority scores.")
        return priority_scores
