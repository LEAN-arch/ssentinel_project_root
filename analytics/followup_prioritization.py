# sentinel_project_root/analytics/followup_prioritization.py
# Contains the FollowUpPrioritizer class for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Any

from config import settings # Use new settings module
from .protocol_executor import execute_escalation_protocol # For triggering protocols

logger = logging.getLogger(__name__)

class FollowUpPrioritizer:
    """
    Simulates logic to prioritize patients for follow-up or tasks based on a combination
    of AI risk scores, vital sign alerts, task urgency, and contextual factors.
    """
    def __init__(self):
        self.priority_weights: dict[str, float] = {
            'base_ai_risk_score_contribution_pct': 0.40, # Percentage of AI risk score to add
            'critical_vital_alert_points': 40.0,         # Points for active critical vital alert
            'pending_urgent_task_points': 30.0,          # Points for pending urgent referral/task
            'acute_condition_severity_points': 25.0,     # Points for specific severe conditions + vitals
            'contextual_hazard_points': 15.0,            # Points for environmental hazards affecting patient
            'task_overdue_factor_per_day': 1.5,          # Points added per day a task is overdue (capped)
            'poor_adherence_points': 15.0,               # Points for reported poor medication adherence
            'observed_fatigue_points': 12.0              # Points if CHW observed signs of fatigue
        }
        logger.info("FollowUpPrioritizer initialized with configured weights.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        """Checks for active critical vital sign alerts based on input features."""
        if not isinstance(features, pd.Series): return False

        # Critical Low SpO2
        min_spo2 = pd.to_numeric(features.get('min_spo2_pct'), errors='coerce')
        if pd.notna(min_spo2) and min_spo2 < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
            # Trigger escalation for critical SpO2 directly if this logic is primary
            # execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", features.to_dict(), {"SPO2_VALUE": min_spo2})
            return True

        # High Fever
        temp_c = pd.to_numeric(features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius')), errors='coerce')
        if pd.notna(temp_c) and temp_c >= settings.ALERT_BODY_TEMP_HIGH_FEVER_C:
            return True
        
        # Fall Detected
        fall_detected = str(features.get('fall_detected_today', '0')).lower()
        if fall_detected in ['1', 'true', 'yes']:
            # execute_escalation_protocol("PATIENT_FALL_DETECTED", features.to_dict())
            return True
            
        return False

    def _is_pending_urgent_task_or_referral(self, features: pd.Series) -> bool:
        """Checks if there's a pending urgent task or referral."""
        if not isinstance(features, pd.Series): return False

        referral_status = str(features.get('referral_status', '')).lower()
        if referral_status == 'pending':
            condition_str = str(features.get('condition', '')).lower()
            referral_reason_str = str(features.get('referral_reason', '')).lower()
            # Check if condition is a key actionable condition OR referral reason indicates urgency
            if any(kc.lower() in condition_str for kc in settings.KEY_CONDITIONS_FOR_ACTION) or \
               any(urgent_word in referral_reason_str for urgent_word in ['urgent', 'critical', 'emergency']):
                return True
        
        # Placeholder for a direct 'task_priority' field if it exists in data
        # if str(features.get('chw_task_priority', 'normal')).lower() == 'urgent':
        #     return True
            
        return False

    def _has_acute_condition_with_severity_indicators(self, features: pd.Series) -> bool:
        """Checks for acute conditions combined with severity indicators."""
        if not isinstance(features, pd.Series): return False
        
        condition_str = str(features.get('condition', '')).lower()
        
        # Example: Pneumonia with warning-level SpO2
        min_spo2 = pd.to_numeric(features.get('min_spo2_pct'), errors='coerce')
        if "pneumonia" in condition_str and pd.notna(min_spo2) and \
           min_spo2 < settings.ALERT_SPO2_WARNING_LOW_PCT: # Warning, not critical (handled above)
            return True
            
        # Example: Sepsis, Severe Dehydration, Heat Stroke are inherently severe
        if any(crit_cond.lower() in condition_str for crit_cond in ["sepsis", "severe dehydration", "heat stroke"]):
            return True
            
        return False

    def _has_significant_contextual_hazard(self, features: pd.Series) -> bool:
        """Checks for significant contextual hazards affecting the patient."""
        if not isinstance(features, pd.Series): return False

        ambient_heat = pd.to_numeric(features.get('ambient_heat_index_c'), errors='coerce')
        if pd.notna(ambient_heat) and ambient_heat >= settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C:
            # Potential protocol for CHW/patient in danger heat zone
            # execute_escalation_protocol("ENVIRONMENT_DANGER_HEAT_INDEX", features.to_dict(), {"HEAT_INDEX": ambient_heat})
            return True
        
        # Add other contextual hazards checks here (e.g., air quality, flooding alerts if available)
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

        # 1. Base from AI Risk Score
        ai_risk = pd.to_numeric(features.get('ai_risk_score', 0.0), errors='coerce').fillna(0.0)
        priority_score += ai_risk * self.priority_weights['base_ai_risk_score_contribution_pct']

        # 2. Critical Vital Sign Alert
        if self._has_active_critical_vitals_alert(features):
            priority_score += self.priority_weights['critical_vital_alert_points']

        # 3. Pending Urgent Task/Referral
        if self._is_pending_urgent_task_or_referral(features):
            priority_score += self.priority_weights['pending_urgent_task_points']

        # 4. Acute Condition with Severity Indicators
        if self._has_acute_condition_with_severity_indicators(features):
            priority_score += self.priority_weights['acute_condition_severity_points']

        # 5. Significant Contextual Hazard
        if self._has_significant_contextual_hazard(features):
            priority_score += self.priority_weights['contextual_hazard_points']

        # 6. Medication Adherence
        adherence_report = str(features.get('medication_adherence_self_report', 'Unknown')).lower()
        if adherence_report == 'poor':
            priority_score += self.priority_weights['poor_adherence_points']
        
        # 7. Observed Fatigue (by CHW)
        fatigue_observed = str(features.get('signs_of_fatigue_observed_flag', '0')).lower()
        if fatigue_observed in ['1', 'true', 'yes']:
             priority_score += self.priority_weights['observed_fatigue_points']

        # 8. Task Overdue Penalty (capped at 30 days for max effect)
        overdue_days_capped = min(int(days_task_overdue), 30) # Cap overdue impact
        priority_score += overdue_days_capped * self.priority_weights['task_overdue_factor_per_day']
        
        # Clip score to be between 0 and 100
        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Generates follow-up priority scores for each row in the input DataFrame.
        Assumes 'ai_risk_score' is already present.
        Adds 'days_task_overdue' if not present, defaulting to 0.
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("Input DataFrame to generate_followup_priorities is empty or invalid.")
            return pd.Series(dtype='float64')

        df_processed = data_df.copy()

        # Ensure 'days_task_overdue' column exists and is numeric
        if 'days_task_overdue' not in df_processed.columns:
            df_processed['days_task_overdue'] = 0
        else:
            df_processed['days_task_overdue'] = pd.to_numeric(df_processed['days_task_overdue'], errors='coerce').fillna(0).astype(int)
        
        # Ensure other potentially used feature columns exist with appropriate defaults if missing
        # This helps prevent KeyErrors in the apply function.
        # Defaults should be benign (not triggering points unless data explicitly says so).
        features_to_ensure_with_defaults = {
            'min_spo2_pct': 100.0, # Healthy default
            'vital_signs_temperature_celsius': 37.0, # Healthy default
            'max_skin_temp_celsius': 37.0, # Healthy default
            'fall_detected_today': '0', # String '0' for flag checks
            'referral_status': 'Unknown',
            'condition': 'Unknown',
            'referral_reason': 'N/A',
            # 'chw_task_priority': 'Normal', # If used
            'ambient_heat_index_c': 25.0, # Safe default
            'medication_adherence_self_report': 'Unknown',
            'signs_of_fatigue_observed_flag': '0', # String '0'
            'ai_risk_score': 0.0 # Default AI risk if missing
        }
        for feature_col, default_val in features_to_ensure_with_defaults.items():
            if feature_col not in df_processed.columns:
                df_processed[feature_col] = default_val
            else: # If column exists, fill NaNs with its benign default
                if isinstance(default_val, (float, int)):
                    df_processed[feature_col] = pd.to_numeric(df_processed[feature_col], errors='coerce').fillna(default_val)
                else: # String type
                    df_processed[feature_col] = df_processed[feature_col].fillna(str(default_val))


        # Apply the scoring function row-wise
        priority_scores = df_processed.apply(
            lambda row: self.calculate_priority_score(row, row['days_task_overdue']),
            axis=1
        )
        
        logger.info(f"Generated {len(priority_scores)} follow-up priority scores.")
        return priority_scores
