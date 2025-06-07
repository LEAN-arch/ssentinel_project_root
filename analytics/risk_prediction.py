# sentinel_project_root/analytics/risk_prediction.py
Contains the RiskPredictionModel class for Sentinel.
import pandas as pd
import numpy as np
import logging
import re  # Added this import
from typing import Optional, Dict, Any
from config import settings
from .protocol_executor import execute_escalation_protocol
logger = logging.getLogger(name)
class RiskPredictionModel:
"""
Simulates a pre-trained patient/worker risk prediction model.
Uses rule-based logic with weights for core features.
The risk score is intended to be a general indicator, potentially triggering
further assessment or specific protocols.
"""
def init(self):
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
'max_skin_temp_celsius': { # Alternative temperature, slightly less weight
'weight': 1.8, 'threshold_high': settings.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10,
'super_high_threshold': settings.ALERT_BODY_TEMP_FEVER_C - 0.2, 'factor_super_high': 20
},
'stress_level_score': { # Assumes a 0-100 score from another model/input
'weight': 0.9, 'threshold_high': settings.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 8,
'super_high_threshold': settings.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 15
},
'hrv_rmssd_ms': { # Lower HRV indicates higher stress/risk
'weight': 1.3, 'threshold_low': settings.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 18
},
'tb_contact_traced': { # Being a TB contact increases risk
'weight': 1.5, 'is_flag': True, 'flag_value': '1', 'factor_true': 15 # Match string '1' after conversion
},
'fall_detected_today': { # A fall is a significant risk event
'weight': 2.5, 'is_flag': True, 'flag_value': '1', 'factor_true': 30 # Match string '1'
},
'ambient_heat_index_c': { # High ambient heat poses risk
'weight': 0.8, 'threshold_high': settings.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 10,
'super_high_threshold': settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 18
},
'ppe_compliant_flag': { # Non-compliance increases risk
'weight': 1.2, 'is_flag': True, 'flag_value': '0', 'factor_true': 12 # Match string '0'
}
}
self.condition_base_scores: Dict[str, float] = {
cond.lower(): 25.0 for cond in settings.KEY_CONDITIONS_FOR_ACTION
}
self.condition_base_scores.update({ # More specific scores
"sepsis": 50.0, "severe dehydration": 45.0, "heat stroke": 48.0,
"tb": 35.0, "hiv-positive": 28.0, "pneumonia": 40.0, "malaria": 22.0,
"wellness visit": -20.0, "routine follow-up": -10.0,
"minor cold": -8.0, "injury": 15.0
})
self.CHRONIC_CONDITION_FLAG_RISK_POINTS: float = 20.0
logger.info("RiskPredictionModel initialized with configured base scores and factors.")
def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
    """
    Determines a base risk score from a condition string.
    Handles single conditions, delimited lists (';' or ','), and partial matches.
    Returns the score of the highest-risk condition found.
    """
    if pd.isna(condition_str) or not isinstance(condition_str, str) or \
       condition_str.strip().lower() in ["unknown", "none", "n/a", "", "unknowncondition"]: # Added unknowncondition
        return 0.0

    max_base_score = 0.0
    condition_input_lower = condition_str.lower()
    
    # Split if delimiters are present, otherwise treat as single condition
    potential_conditions = [condition_input_lower] # Default if no delimiters
    # Robust splitting for various delimiters
    if any(d in condition_input_lower for d in [';', ',']):
        # Use regex to split by multiple delimiters and handle spaces around them
        potential_conditions = [c.strip() for c in re.split(r'[;,]\s*', condition_input_lower) if c.strip()]


    for part_cond_clean in potential_conditions:
        if not part_cond_clean:
            continue

        # Check for exact match (case-insensitive) first for performance
        current_best_score_for_part = self.condition_base_scores.get(part_cond_clean, 0.0)
        
        if current_best_score_for_part == 0.0: # If no exact match, check for partial matches
            for known_cond_key, score_val in self.condition_base_scores.items():
                if known_cond_key in part_cond_clean: # known_cond_key is already lower
                    current_best_score_for_part = max(current_best_score_for_part, score_val)
        max_base_score = max(max_base_score, current_best_score_for_part)
        
    return max_base_score

def predict_risk_score(self, features: pd.Series) -> float:
    """
    Predicts a risk score for a single individual based on a Series of features.
    """
    if not isinstance(features, pd.Series):
        logger.error("RiskPredictionModel.predict_risk_score expects a pandas Series for features.")
        return 0.0

    current_features = features.copy() # Work on a copy

    # Ensure all expected base_risk_factors columns exist with a benign default
    # String '0' for flags (as they are compared as strings), np.nan for numeric
    for key_feature, params_feature in self.base_risk_factors.items():
        if key_feature not in current_features:
            current_features[key_feature] = '0' if params_feature.get('is_flag') else np.nan
    # Ensure other specific columns used are present
    if 'condition' not in current_features: current_features['condition'] = "UnknownCondition"
    if 'chronic_condition_flag' not in current_features: current_features['chronic_condition_flag'] = '0' # String '0' for flag
    if 'medication_adherence_self_report' not in current_features: current_features['medication_adherence_self_report'] = 'Unknown'
    if 'rapid_psychometric_distress_score' not in current_features: current_features['rapid_psychometric_distress_score'] = 0.0
    if 'signs_of_fatigue_observed_flag' not in current_features: current_features['signs_of_fatigue_observed_flag'] = '0' # String '0' for flag


    calculated_risk = self._get_condition_base_score(str(current_features.get('condition', '')))

    # Chronic condition flag check (value can be 0, 1, '0', '1', 'true', 'yes', etc.)
    chronic_flag_val = str(current_features.get('chronic_condition_flag', '0')).strip().lower()
    if chronic_flag_val in ['1', 'true', 'yes']:
        calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

    # Iterate through other base risk factors
    for feature_key_loop, params_loop in self.base_risk_factors.items():
        value_feat_raw = current_features.get(feature_key_loop)
        weight_factor = params_loop.get('weight', 1.0)
        
        if params_loop.get('is_flag'): # Handle boolean/flag features
            flag_trigger_value = str(params_loop.get('flag_value', '1')).lower() # Default trigger is '1'
            # Ensure value_feat_raw is string for comparison
            if str(value_feat_raw).strip().lower() == flag_trigger_value:
                calculated_risk += params_loop.get('factor_true', 0) * weight_factor
                # Specific protocol triggers for certain flags
                if feature_key_loop == 'fall_detected_today' and str(value_feat_raw).strip().lower() == '1':
                     execute_escalation_protocol("PATIENT_FALL_DETECTED", current_features.to_dict())
        
        else: # Handle numeric features with thresholds
            if pd.notna(value_feat_raw): # Only process if value is not NaN initially
                try:
                    # Convert to float for comparisons, handling potential conversion errors
                    numeric_value_feat = float(value_feat_raw)
                    # Order of checks matters for overlapping thresholds (e.g., high vs super_high)
                    if 'super_high_threshold' in params_loop and numeric_value_feat >= params_loop['super_high_threshold']:
                        calculated_risk += params_loop.get('factor_super_high', 0) * weight_factor
                    elif 'threshold_very_high' in params_loop and numeric_value_feat >= params_loop['threshold_very_high']:
                        calculated_risk += params_loop.get('factor_very_high', 0) * weight_factor
                    elif 'threshold_high' in params_loop and numeric_value_feat >= params_loop['threshold_high']:
                        calculated_risk += params_loop.get('factor_high', 0) * weight_factor
                    
                    # For low thresholds (e.g., SpO2, HRV)
                    if 'threshold_low' in params_loop and numeric_value_feat < params_loop['threshold_low']:
                        calculated_risk += params_loop.get('factor_low', 0) * weight_factor
                        if feature_key_loop == 'min_spo2_pct' and numeric_value_feat < settings.ALERT_SPO2_CRITICAL_LOW_PCT:
                            execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", current_features.to_dict(), {"SPO2_VALUE": numeric_value_feat})
                    elif 'mid_threshold_low' in params_loop and numeric_value_feat < params_loop['mid_threshold_low']: # e.g., SpO2 warning
                        calculated_risk += params_loop.get('factor_mid_low', 0) * weight_factor
                except (ValueError, TypeError):
                    logger.debug(f"Could not convert feature '{feature_key_loop}' value '{value_feat_raw}' to float for risk calc.")
                    # Skip this factor if value is not convertible to float

    # Additional adjustments based on other features
    med_adherence_val = str(current_features.get('medication_adherence_self_report', 'Unknown')).lower()
    if med_adherence_val == 'poor': calculated_risk += 12.0
    elif med_adherence_val == 'fair': calculated_risk += 6.0

    # Use pd.to_numeric for robust conversion before multiplication
    psych_distress_score_val = pd.to_numeric(current_features.get('rapid_psychometric_distress_score'), errors='coerce')
    if pd.notna(psych_distress_score_val):
        calculated_risk += psych_distress_score_val * 1.5 # Simple multiplier

    fatigue_flag_input_val = str(current_features.get('signs_of_fatigue_observed_flag', '0')).strip().lower()
    if fatigue_flag_input_val in ['1', 'true', 'yes']:
        calculated_risk += 10.0
        
    return float(np.clip(calculated_risk, 0, 100))

def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
    """
    Predicts risk scores for all rows in a DataFrame.
    Uses .apply() for row-wise prediction with the single-prediction logic.
    """
    if not isinstance(data_df, pd.DataFrame) or data_df.empty:
        logger.warning("predict_bulk_risk_scores received an empty or invalid DataFrame.")
        return pd.Series(dtype='float64')
    
    df_to_process = data_df.copy() # Work on a copy
    
    # `predict_risk_score` handles internal defaults, so extensive pre-filling isn't strictly needed here,
    # but ensuring 'condition' is present with a default fill helps.
    if 'condition' not in df_to_process.columns:
        df_to_process['condition'] = "UnknownCondition" # Default if column entirely missing
    else:
        # Fill NaNs in existing 'condition' column if any
        df_to_process['condition'] = df_to_process['condition'].fillna("UnknownCondition")

    try:
        risk_scores_series = df_to_process.apply(self.predict_risk_score, axis=1)
    except Exception as e_apply:
        logger.error(f"Error during bulk application of predict_risk_score: {e_apply}", exc_info=True)
        # Return a series of NaNs or default risk if bulk apply fails
        return pd.Series([np.nan] * len(df_to_process), index=df_to_process.index, dtype='float64')

    logger.info(f"Bulk risk scores calculated for {len(risk_scores_series)} records.")
    return risk_scores_series
