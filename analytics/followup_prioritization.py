# sentinel_project_root/analytics/followup_prioritization.py
# SME PLATINUM STANDARD - VECTORIZED FOLLOW-UP PRIORITIZATION

import logging

import numpy as np
import pandas as pd

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)

class FollowUpPrioritizer:
    """
    A rule-based model to calculate a follow-up priority score using fully
    vectorized operations for high performance.
    """
    def __init__(self):
        self.weights = settings.MODEL_WEIGHTS

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct type."""
        clean_df = df.copy()
        required_cols = {
            'ai_risk_score': 0.0,
            'days_task_overdue': 0,
            'min_spo2_pct': 100.0,
            'temperature': 37.0,
            'fall_detected_today': 0,
            'referral_status': "N/A",
            'diagnosis': "N/A",
            'medication_adherence_self_report': "Good"
        }
        for col, default in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default
            
            if isinstance(default, (int, float)):
                clean_df[col] = convert_to_numeric(clean_df[col], float, default)
            else:
                clean_df[col] = clean_df[col].astype(str).fillna(str(default))
        return clean_df

    def generate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates follow-up priority scores and reasons using a vectorized approach.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df.assign(ai_followup_priority_score=np.nan, priority_reasons="")

        prepared_df = self._prepare_data(df)
        scores = pd.Series(0.0, index=prepared_df.index)
        reasons = pd.Series("", index=prepared_df.index)

        # --- Base Score Components ---
        scores += prepared_df['ai_risk_score'] * self.weights.risk_score_multiplier
        scores += prepared_df['days_task_overdue'].clip(upper=30) * self.weights.days_overdue_multiplier

        # --- Event-Based Point Additions (Vectorized) ---
        # Rule: Critical vital signs
        crit_spo2 = prepared_df['min_spo2_pct'] < settings.ANALYTICS.spo2_critical_threshold_pct
        crit_temp = prepared_df['temperature'] >= settings.ANALYTICS.temp_high_fever_threshold_c
        fall = prepared_df['fall_detected_today'] > 0
        crit_vitals_mask = crit_spo2 | crit_temp | fall
        scores.loc[crit_vitals_mask] += self.weights.critical_vital_alert
        reasons.loc[crit_spo2, "reason_crit_spo2"] = "Critical SpO2"
        reasons.loc[crit_temp, "reason_crit_temp"] = "High Fever"
        reasons.loc[fall, "reason_fall"] = "Fall Detected"

        # Rule: Pending urgent referral
        key_dx_pattern = '|'.join(settings.KEY_DIAGNOSES)
        urgent_referral_mask = (
            (prepared_df['referral_status'].str.lower() == 'pending') &
            (prepared_df['diagnosis'].str.contains(key_dx_pattern, case=False, na=False))
        )
        scores.loc[urgent_referral_mask] += self.weights.pending_urgent_referral
        reasons.loc[urgent_referral_mask, "reason_referral"] = "Urgent Referral Pending"

        # Rule: Poor medication adherence
        poor_adherence_mask = prepared_df['medication_adherence_self_report'].str.lower() == 'poor'
        scores.loc[poor_adherence_mask] += self.weights.poor_med_adherence
        reasons.loc[poor_adherence_mask, "reason_adherence"] = "Poor Adherence"

        # --- Finalize and Assemble Output ---
        output_df = df.copy()
        output_df['ai_followup_priority_score'] = scores.clip(0, 100).round(1)
        
        # Combine all reason columns into a single comma-separated string
        reason_cols = [c for c in reasons.columns if c.startswith('reason_')]
        output_df['priority_reasons'] = reasons[reason_cols].apply(
            lambda row: ', '.join(row.dropna().astype(str)),
            axis=1
        )
        
        logger.info(f"Generated {len(output_df)} follow-up priority scores.")
        return output_df

def calculate_followup_priority(health_df: pd.DataFrame) -> pd.DataFrame:
    """
    Public factory function to calculate follow-up priority scores.
    Returns the input DataFrame with `ai_followup_priority_score` and `priority_reasons`.
    """
    return FollowUpPrioritizer().generate_scores(health_df)
