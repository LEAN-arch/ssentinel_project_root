# sentinel_project_root/analytics/risk_prediction.py
# SME PLATINUM STANDARD - VECTORIZED RISK PREDICTION MODEL

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    A rule-based model to simulate a patient risk score using fully vectorized
    operations for high performance. Higher scores indicate higher risk.
    """
    def __init__(self):
        self.weights = settings.MODEL_WEIGHTS

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist and are of the correct type."""
        clean_df = df.copy()
        # Define columns required by the model and their neutral default values
        required_cols = {
            'age': 50.0,
            'min_spo2_pct': 100.0,
            'temperature': 37.0,
            'chronic_condition_flag': 0,
            'tb_contact_traced': 0,
            'fall_detected_today': 0
        }
        # Add symptom cluster flags as required columns
        for cluster in settings.SYMPTOM_CLUSTERS.keys():
            required_cols[f'has_symptom_cluster_{cluster}'] = 0
            
        for col, default in required_cols.items():
            if col not in clean_df.columns:
                clean_df[col] = default
            # Ensure columns are numeric for calculations
            clean_df[col] = convert_to_numeric(clean_df[col], float, default)

        return clean_df

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates risk scores for the DataFrame using a vectorized rules engine.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df.assign(ai_risk_score=np.nan)

        prepared_df = self._prepare_data(df)
        scores = pd.Series(0.0, index=prepared_df.index)

        # --- Vectorized Rule Application ---
        
        # Age-based risk
        scores.loc[prepared_df['age'] > 65] += self.weights.base_age_gt_65
        scores.loc[prepared_df['age'] < 5] += self.weights.base_age_lt_5
        
        # Vital signs-based risk
        scores.loc[prepared_df['min_spo2_pct'] < settings.ANALYTICS.spo2_critical_threshold_pct] += self.weights.vital_spo2_critical
        scores.loc[prepared_df['temperature'] >= settings.ANALYTICS.temp_high_fever_threshold_c] += self.weights.vital_temp_critical
        
        # Symptom-based risk (using pre-enriched flags)
        if 'has_symptom_cluster_respiratory_distress' in prepared_df.columns:
            scores.loc[prepared_df['has_symptom_cluster_respiratory_distress'] > 0] += self.weights.symptom_cluster_severity_high
        if 'has_symptom_cluster_severe_febrile' in prepared_df.columns:
            scores.loc[prepared_df['has_symptom_cluster_severe_febrile'] > 0] += self.weights.symptom_cluster_severity_high
        if 'has_symptom_cluster_dehydration_shock' in prepared_df.columns:
            scores.loc[prepared_df['has_symptom_cluster_dehydration_shock'] > 0] += self.weights.symptom_cluster_severity_med

        # Contextual and comorbidity risk
        scores.loc[prepared_df['chronic_condition_flag'] > 0] += self.weights.comorbidity
        scores.loc[prepared_df['tb_contact_traced'] > 0] += self.weights.symptom_cluster_severity_high # High weight
        scores.loc[prepared_df['fall_detected_today'] > 0] += self.weights.vital_spo2_critical # High weight
        
        # Finalize scores
        output_df = df.copy()
        output_df['ai_risk_score'] = scores.clip(0, 100).round(1)
        
        logger.info(f"Generated {len(output_df)} AI risk scores.")
        return output_df

def calculate_risk_score(health_df: pd.DataFrame) -> pd.DataFrame:
    """Public factory function to calculate AI risk scores."""
    model = RiskPredictionModel()
    return model.predict_scores(health_df)
