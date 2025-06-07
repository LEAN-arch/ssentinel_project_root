# ssentinel_project_root/analytics/alerting.py
"""
Provides a robust, data-driven, rule-based engine for generating alerts.
This centralized module replaces scattered alert logic.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Callable

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    # Assuming this module exists for executing escalations
    from .protocol_executor import execute_escalation_protocol
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in alerting.py: {e}", exc_info=True)
    # Define a dummy function if the import fails to allow the app to run
    def execute_escalation_protocol(*args, **kwargs):
        logger_init.warning("DUMMY CALL: execute_escalation_protocol called but module failed to import.")
    # raise # Commented out to allow functionality even if protocol_executor is missing

logger = logging.getLogger(__name__)

# --- Base Class for Alert Generation (Rules Engine) ---

class BaseAlertGenerator:
    """A base class for a data-driven alert generation rules engine."""
    
    def __init__(self, data_df: pd.DataFrame, source_context: str):
        if not isinstance(data_df, pd.DataFrame):
            data_df = pd.DataFrame()
        self.source_context = source_context
        self.df = data_df.copy() # Work on a copy
        self.ALERT_RULES: List[Dict[str, Any]] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _prepare_dataframe(self, column_config: Dict[str, Dict]) -> None:
        """A robust, reusable method to clean and prepare the source DataFrame."""
        for col, config in column_config.items():
            if col not in self.df.columns:
                self.df[col] = config.get('default')
            
            # Use robust helper for numeric conversion
            if config.get('type') == 'numeric':
                self.df[col] = convert_to_numeric(self.df[col], default_value=config.get('default'))

    def _evaluate_rules(self, data_row: pd.Series) -> List[Dict[str, Any]]:
        """Evaluates all defined rules against a single row of data."""
        triggered_alerts = []
        for rule in self.ALERT_RULES:
            metric_val = data_row.get(rule['metric'])
            if pd.isna(metric_val):
                continue

            # Safely get threshold, which can be a value or a function call
            threshold = rule['threshold']() if callable(rule['threshold']) else rule['threshold']
            
            condition_met = False
            op = rule['condition']
            if op == 'less_than' and metric_val < threshold: condition_met = True
            elif op == 'greater_than_or_equal' and metric_val >= threshold: condition_met = True
            elif op == 'equals' and metric_val == threshold: condition_met = True

            if condition_met:
                alert = rule['alert_details'].copy()
                alert['raw_priority_score'] = rule['priority_calculator'](metric_val, threshold)
                alert['patient_id'] = str(data_row.get('patient_id', 'UnknownPID'))
                alert['triggering_value'] = f"{rule.get('metric_name', rule['metric'])}: {metric_val:.1f}"
                alert['protocol_context'] = {
                    "patient_id": alert['patient_id'],
                    "triggering_metric": rule['metric'],
                    "triggering_value": metric_val,
                    "threshold": threshold,
                    "full_context": data_row.to_dict()
                }
                if rule.get('protocol_to_trigger'):
                    execute_escalation_protocol(rule['protocol_to_trigger'], alert['protocol_context'])
                
                triggered_alerts.append(alert)
        return triggered_alerts

# --- CHW-Specific Alert Generator ---

class CHWAlertGenerator(BaseAlertGenerator):
    """Generates prioritized alerts for Community Health Workers."""

    def __init__(self, patient_encounter_df: pd.DataFrame, for_date: Union[str, pd.Timestamp], zone_context: str):
        super().__init__(data_df=patient_encounter_df, source_context="CHWPatientAlertGen")
        self.processing_date = pd.to_datetime(for_date).date()
        
        column_config = {
            'min_spo2_pct': {"default": np.nan, "type": "numeric"},
            'vital_signs_temperature_celsius': {"default": np.nan, "type": "numeric"},
            'fall_detected_today': {"default": 0, "type": "numeric"},
            'ai_followup_priority_score': {"default": 0, "type": "numeric"},
        }
        self._prepare_dataframe(column_config)

        self.ALERT_RULES = [
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": lambda: self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90), "metric_name": "SpO2",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2"}, "priority_calculator": lambda v, t: 98 + max(0, t - v), "protocol_to_trigger": "PATIENT_CRITICAL_SPO2_LOW"},
            {"metric": "vital_signs_temperature_celsius", "condition": "greater_than_or_equal", "threshold": lambda: self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5), "metric_name": "Temp",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever"}, "priority_ calculator": lambda v, t: 95 + max(0, (v - t) * 2), "protocol_to_trigger": "PATIENT_HIGH_FEVER"},
            {"metric": "fall_detected_today", "condition": "greater_than_or_equal", "threshold": 1, "metric_name": "Fall(s)",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Fall Detected"}, "priority_calculator": lambda v, t: 92.0, "protocol_to_trigger": "PATIENT_FALL_DETECTED"},
            {"metric": "ai_followup_priority_score", "condition": "greater_than_or_equal", "threshold": lambda: self._get_setting('FATIGUE_INDEX_HIGH_THRESHOLD', 80), "metric_name": "AI Prio",
             "alert_details": {"alert_level": "WARNING", "primary_reason": "High AI Follow-up Prio."}, "priority_calculator": lambda v, t: min(90, v)},
        ]

    def generate(self, max_alerts_to_return: int = 15) -> List[Dict[str, Any]]:
        if self.df.empty: return []
        
        all_alerts = self.df.apply(self._evaluate_rules, axis=1).sum()
        if not all_alerts: return []

        # Deduplicate alerts, keeping only the highest-priority one for each patient
        deduped_alerts = {alert['patient_id']: alert for alert in sorted(all_alerts, key=lambda x: x['raw_priority_score'])}
        final_alerts = list(deduped_alerts.values())
        
        # Sort final list by level (Critical > Warning > Info) and then by priority score
        final_alerts.sort(key=lambda x: ({"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x.get("alert_level", "INFO"), 3), -x.get('raw_priority_score', 0.0)))
        
        logger.info(f"({self.source_context}) Generated {len(final_alerts)} unique CHW patient alerts for {self.processing_date}.")
        return final_alerts[:max_alerts_to_return]

# --- Backward Compatibility Functions ---

def generate_chw_patient_alerts(patient_encounter_data_df: Optional[pd.DataFrame], for_date: Union[str, pd.Timestamp], chw_zone_context_str: str, max_alerts_to_return: int = 15) -> List[Dict[str, Any]]:
    """Factory function to generate CHW alerts. Preserved for backward compatibility."""
    generator = CHWAlertGenerator(patient_encounter_data_df or pd.DataFrame(), for_date, chw_zone_context_str)
    return generator.generate(max_alerts_to_return=max_alerts_to_return)

def get_patient_alerts_for_clinic(health_df_period: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """
    Factory function to generate a DataFrame of patient alerts for clinic review.
    This is a compatibility stub. A full ClinicPatientAlerts class would be the proper implementation.
    """
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return pd.DataFrame()

    df = health_df_period.copy()
    high_risk_threshold = getattr(settings, 'RISK_SCORE_HIGH_THRESHOLD', 75)
    
    if 'ai_risk_score' not in df.columns:
        return pd.DataFrame()
        
    alerts_df = df[df['ai_risk_score'] >= high_risk_threshold].copy()
    if alerts_df.empty:
        return pd.DataFrame()
        
    alerts_df.sort_values('ai_risk_score', ascending=False, inplace=True)
    alerts_df['alert_reason'] = "High AI Risk Score (" + alerts_df['ai_risk_score'].astype(str) + ")"
    
    # Define columns to return, using .get() for safety
    cols_to_return = [
        'patient_id', 'age', 'gender', 'condition', 'alert_reason', 
        'ai_risk_score', 'encounter_date'
    ]
    
    return alerts_df[[c for c in cols_to_return if c in alerts_df.columns]].reset_index(drop=True)
