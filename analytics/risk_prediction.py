import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Dict, Any

# --- Module Imports ---
try:
    from config import settings
    from .protocol_executor import execute_escalation_protocol
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in risk_prediction.py: {e}. Ensure paths are correct.", exc_info=True)
    raise

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


class RiskPredictionModel:
    """
    Simulates a pre-trained patient/worker risk prediction model.
    Uses rule-based logic with weights for core features.
    The risk score is a general indicator for further assessment.
    """
    # FIXED: Renamed `init` to the correct Python constructor `__init__`.
    def __init__(self):
        """Initializes the model with risk factors, weights, and base scores."""
        self.base_risk_factors: Dict[str, Dict[str, Any]] = {
            'age': {
                'weight': 0.6, 'threshold_low': settings.AGE_THRESHOLD_MODERATE, 'factor_low': -5,
                'threshold_high': settings.AGE_THRESHOLD_HIGH, 'factor_high': 10,
                'threshold_very_high': settings.AGE_THRESHOLD_VERY_HIGH, 'factor_very_high': 20
            },
            'min_spo2_pct': {
                'weight': 2.8, 'threshold_low': settings.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 35,
                'mid_threshold_low': settings.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 20
            },
            'vital_signs_temperature_celsius': {
                'weight': 2.2, 'threshold_high': settings.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15,
                'super_high_threshold': settings.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 28
            },
            'max_skin_temp_celsius': {
                'weight': 1.8, 'threshold_high': settings.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10,
                'super_high_threshold': settings.ALERT_BODY_TEMP_FEVER_C - 0.2, 'factor_super_high': 20
            },
            'stress_level_score': {
                'weight': 0.9, 'threshold_high': settings.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 8,
                'super_high_threshold': settings.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 15
            },
            'hrv_rmssd_ms': {
                'weight': 1.3, 'threshold_low': settings.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 18
            },
            'tb_contact_traced': {
                'weight': 1.5, 'is_flag': True, 'flag_value': '1', 'factor_true': 15
            },
            'fall_detected_today': {
                'weight': 2.5, 'is_flag': True, 'flag_value': '1', 'factor_true': 30
            },
            'ambient_heat_index_c': {
                'weight': 0.8, 'threshold_high': settings.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 10,
                'super_high_threshold': settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 18
            },
            'ppe_compliant_flag': {
                'weight': 1.2, 'is_flag': True, 'flag_value': '0', 'factor_true': 12
            }
        }
        self.condition_base_scores: Dict[str, float] = {
            cond.lower(): 25.0 for cond in settings.KEY_CONDITIONS_FOR_ACTION
        }
        self.condition_base_scores.update({
            "sepsis": 50.0, "severe dehydration": 45.0, "heat stroke": 48.0,
            "tb": 35.0, "hiv-positive": 28.0, "pneumonia": 40.0, "malaria": 22.0,
            "wellness visit": -20.0, "routine follow-up": -10.0,
            "minor cold": -8.0, "injury": 15.0
        })
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS: float = 20.0
        logger.info("RiskPredictionModel initialized with configured base scores and factors.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        """
        Determines a base risk score from a condition string, returning the highest score found.
        """
        if pd.isna(condition_str) or not isinstance(condition_str, str) or \
           condition_str.strip().lower() in ["", "unknown", "none", "n/a", "unknowncondition"]:
            return 0.0

        max_score = 0.0
        condition_lower = condition_str.lower()
        
        potential_conditions = re.split(r'[;,]\s*', condition_lower)
        
        for condition_part in potential_conditions:
            if not condition_part: continue

            # Check for exact match first
            part_score = self.condition_base_scores.get(condition_part, 0.0)
            
            # If no exact match, check for partial matches
            if part_score == 0.0:
                for known_cond, score in self.condition_base_scores.items():
                    if known_cond in condition_part:
                        part_score = max(part_score, score)
            max_score = max(max_score, part_score)
            
        return max_score

    def predict_risk_score(self, features: pd.Series) -> float:
        """
        Predicts a risk score for a single individual based on a Series of features.
        """
        if not isinstance(features, pd.Series):
            logger.error("RiskPredictionModel.predict_risk_score expects a pandas Series.")
            return 0.0

        calculated_risk = self._get_condition_base_score(str(features.get('condition', '')))

        if str(features.get('chronic_condition_flag', '0')).strip().lower() in ['1', 'true', 'yes']:
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

        for feature, params in self.base_risk_factors.items():
            value = features.get(feature)
            weight = params.get('weight', 1.0)
            
            if params.get('is_flag'):
                if str(value).strip().lower() == str(params.get('flag_value', '1')).lower():
                    calculated_risk += params.get('factor_true', 0) * weight
            else:
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.notna(numeric_value):
                    if 'super_high_threshold' in params and numeric_value >= params['super_high_threshold']:
                        calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_very_high' in params and numeric_value >= params['threshold_very_high']:
                        calculated_risk += params.get('factor_very_high', 0) * weight
                    elif 'threshold_high' in params and numeric_value >= params['threshold_high']:
                        calculated_risk += params.get('factor_high', 0) * weight
                    
                    if 'threshold_low' in params and numeric_value < params['threshold_low']:
                        calculated_risk += params.get('factor_low', 0) * weight
                    elif 'mid_threshold_low' in params and numeric_value < params['mid_threshold_low']:
                        calculated_risk += params.get('factor_mid_low', 0) * weight

        # Additional adjustments
        adherence = str(features.get('medication_adherence_self_report', 'Unknown')).lower()
        if adherence == 'poor': calculated_risk += 12.0
        elif adherence == 'fair': calculated_risk += 6.0

        distress_score = pd.to_numeric(features.get('rapid_psychometric_distress_score'), errors='coerce')
        if pd.notna(distress_score):
            calculated_risk += distress_score * 1.5

        if str(features.get('signs_of_fatigue_observed_flag', '0')).strip().lower() in ['1', 'true', 'yes']:
            calculated_risk += 10.0
            
        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Predicts risk scores for all rows in a DataFrame.
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("predict_bulk_risk_scores received an empty or invalid DataFrame.")
            return pd.Series(dtype='float64')
        
        # Ensure 'condition' column exists with a default fill for robustness.
        if 'condition' not in data_df.columns:
            data_df['condition'] = "UnknownCondition"
        else:
            data_df['condition'] = data_df['condition'].fillna("UnknownCondition")

        try:
            risk_scores_series = data_df.apply(self.predict_risk_score, axis=1)
        except Exception as e_apply:
            logger.error(f"Error during bulk application of predict_risk_score: {e_apply}", exc_info=True)
            return pd.Series([np.nan] * len(data_df), index=data_df.index, dtype='float64')

        logger.info(f"Bulk risk scores calculated for {len(risk_scores_series)} records.")
        return risk_scores_series
