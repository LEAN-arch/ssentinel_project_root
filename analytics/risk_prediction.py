# sentinel_project_root/analytics/risk_prediction.py
# Contains the RiskPredictionModel class for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Union

from config import settings
from .protocol_executor import execute_escalation_protocol # Relative import fine here

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    Simulates a patient/worker risk prediction model using rule-based logic.
    """
    def __init__(self):
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
                'weight': 1.5, 'is_flag': True, 'flag_value': '1', 'factor_true': 15 # Match string '1'
            },
            'fall_detected_today': {
                'weight': 2.5, 'is_flag': True, 'flag_value': '1', 'factor_true': 30 # Match string '1'
            },
            'ambient_heat_index_c': {
                'weight': 0.8, 'threshold_high': settings.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 10,
                'super_high_threshold': settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 18
            },
            'ppe_compliant_flag': {
                'weight': 1.2, 'is_flag': True, 'flag_value': '0', 'factor_true': 12 # Match string '0'
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
        logger.info("RiskPredictionModel initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or not isinstance(condition_str, str) or \
           condition_str.strip().lower() in ["unknown", "none", "n/a", "", "unknowncondition"]:
            return 0.0

        max_base_score = 0.0
        condition_input_lower = condition_str.lower()
        
        delimiters = [';', ',']
        potential_conditions = [condition_input_lower]
        for delim in delimiters:
            if delim in condition_input_lower:
                potential_conditions = [c.strip() for c in condition_input_lower.split(delim)]
                break # Use first delimiter found

        for part_cond_clean in potential_conditions:
            if not part_cond_clean:
                continue
            
            current_best_score_for_part = self.condition_base_scores.get(part_cond_clean, 0.0)
            if current_best_score_for_part == 0.0: # If no exact match, try partial
                for known_cond_key, score_val in self.condition_base_scores.items():
                    if known_cond_key in part_cond_clean:
                        current_best_score_for_part = max(current_best_score_for_part, score_val)
            max_base_score = max(max_base_score, current_best_score_for_part)
            
        return max_base_score

    def predict_risk_score(self, features: pd.Series) -> float:
        if not isinstance(features, pd.Series):
            logger.error("RiskPredictionModel.predict_risk_score expects a pandas Series.")
            return 0.0

        current_features = features.copy()
        calculated_risk = 0.0

        # Ensure essential features exist with benign defaults before calculation
        for key_feature, params_feature in self.base_risk_factors.items():
            if key_feature not in current_features:
                current_features[key_feature] = '0' if params_feature.get('is_flag') else np.nan

        if 'condition' not in current_features: current_features['condition'] = "UnknownCondition"
        if 'chronic_condition_flag' not in current_features: current_features['chronic_condition_flag'] = '0'
        if 'medication_adherence_self_report' not in current_features: current_features['medication_adherence_self_report'] = 'Unknown'
        if 'rapid_psychometric_distress_score' not in current_features: current_features['rapid_psychometric_distress_score'] = 0.0
        if 'signs_of_fatigue_observed_flag' not in current_features: current_features['signs_of_fatigue_observed_flag'] = '0'


        calculated_risk = self._get_condition_base_score(str(current_features.get('condition', '')))

        if str(current_features.get('chronic_condition_flag', '0')).lower() in ['1', 'true', 'yes']:
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

        for feature_key, params in self.base_risk_factors.items():
            value_feat = current_features.get(feature_key)
            weight_factor = params.get('weight', 1.0)
            
            if params.get('is_flag'):
                flag_trigger_value = str(params.get('flag_value', '1')).lower()
                if str(value_feat).lower() == flag_trigger_value:
                    calculated_risk += params.get('factor_true', 0) * weight_factor
                    if feature_key == 'fall_detected_today' and str(value_feat).lower() == '1': # Explicit check
                         execute_escalation_protocol("PATIENT_FALL_DETECTED", current_features.to_dict())
            elif pd.notna(value_feat):
                try:
                    numeric_value_feat = float(value_feat)
                    if 'super_high_threshold' in params and numeric_value_feat >= params['super_high_threshold']:
                        calculated_risk += params['factor_super_high'] * weight_factor
                    elif 'threshold_very_high' in params and numeric_value_feat >= params['threshold_very_high']:
                        calculated_risk += params['factor_very_high'] * weight_factor
                    elif 'threshold_high' in params and numeric_value_feat >= params['threshold_high']:
                        calculated_risk += params['factor_high'] * weight_factor
                    
                    if 'threshold_low' in params and numeric_value_feat < params['threshold_low']:
                        calculated_risk += params['factor_low'] * weight_factor
                        if feature_key == 'min_spo2_pct' and numeric_value_feat < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
                            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", current_features.to_dict(), {"SPO2_VALUE": numeric_value_feat})
                    elif 'mid_threshold_low' in params and numeric_value_feat < params['mid_threshold_low']:
                        calculated_risk += params['factor_mid_low'] * weight_factor
                except (ValueError, TypeError):
                    logger.debug(f"Could not convert feature '{feature_key}' value '{value_feat}' to float.")

        med_adherence = str(current_features.get('medication_adherence_self_report', 'Unknown')).lower()
        if med_adherence == 'poor': calculated_risk += 12.0
        elif med_adherence == 'fair': calculated_risk += 6.0

        psych_distress_val = pd.to_numeric(current_features.get('rapid_psychometric_distress_score'), errors='coerce')
        if pd.notna(psych_distress_val): calculated_risk += psych_distress_val * 1.5

        if str(current_features.get('signs_of_fatigue_observed_flag', '0')).lower() in ['1', 'true', 'yes']:
            calculated_risk += 10.0
            
        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("predict_bulk_risk_scores received an empty or invalid DataFrame.")
            return pd.Series(dtype='float64')
        
        # Ensure required columns exist with benign defaults, or apply might fail if predict_risk_score expects them.
        # predict_risk_score handles internal defaults, so less critical here, but 'condition' is good to ensure.
        df_to_process = data_df.copy()
        if 'condition' not in df_to_process.columns:
            df_to_process['condition'] = "UnknownCondition"
        else:
            df_to_process['condition'] = df_to_process['condition'].fillna("UnknownCondition")
        
        risk_scores_series = df_to_process.apply(self.predict_risk_score, axis=1)
        logger.info(f"Bulk risk scores calculated for {len(risk_scores_series)} records.")
        return risk_scores_series

        risk_scores_series = df_to_process.apply(self.predict_risk_score, axis=1)
        
        logger.info(f"Bulk risk scores calculated for {len(risk_scores_series)} records.")
        return risk_scores_series
