# sentinel_project_root/analytics/alerting.py
# Provides robust, data-driven alert generation for CHW and Clinic dashboards.

import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import date, datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from .protocol_executor import execute_escalation_protocol
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in alerting.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Base Class for Alert Generation (Rules Engine) ---

class BaseAlertGenerator:
    """A base class for a data-driven alert generation rules engine."""
    
    def __init__(self, data_df: pd.DataFrame, source_context: str):
        self.source_context = source_context
        self.df = self._prepare_dataframe(data_df)
        self.ALERT_RULES: List[Dict[str, Any]] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _prepare_dataframe(self, df: pd.DataFrame, column_config: Dict[str, Dict]) -> pd.DataFrame:
        """A robust, reusable method to clean and prepare the source DataFrame."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        
        prepared_df = df.copy()
        for col, config in column_config.items():
            if col not in prepared_df.columns:
                prepared_df[col] = config.get('default')
            else:
                if pd.api.types.is_object_dtype(prepared_df[col].dtype):
                    na_regex = r'^\s*(?:nan|none|n/a|nat|<na>|null)\s*$'
                    prepared_df[col].replace(na_regex, np.nan, regex=True, inplace=True)
                
                prepared_df[col] = convert_to_numeric(prepared_df[col], default_value=config.get('default'))
        return prepared_df

    def _evaluate_rules(self, data_row: pd.Series) -> List[Dict[str, Any]]:
        triggered_alerts = []
        for rule in self.ALERT_RULES:
            metric_val = data_row.get(rule['metric'])
            if pd.isna(metric_val): continue

            threshold = rule['threshold']() if callable(rule['threshold']) else rule['threshold']
            condition_met = False
            if rule['condition'] == 'less_than' and metric_val < threshold: condition_met = True
            elif rule['condition'] == 'greater_than_or_equal' and metric_val >= threshold: condition_met = True
            
            if condition_met:
                alert = rule['alert_details'].copy()
                alert['raw_priority_score'] = rule['priority_calculator'](metric_val, threshold)
                alert['patient_id'] = str(data_row.get('patient_id', 'UnknownPID'))
                alert['triggering_value'] = f"{rule.get('metric_name', rule['metric'])}: {metric_val:.1f}"
                alert['protocol_to_trigger'] = rule.get('protocol_to_trigger')
                alert['context_data'] = data_row.to_dict()
                triggered_alerts.append(alert)
        return triggered_alerts

    def generate(self, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the 'generate' method.")

# --- CHW-Specific Alert Generator ---

class CHWAlertGenerator(BaseAlertGenerator):
    """Generates prioritized alerts for Community Health Workers."""

    def __init__(self, patient_encounter_df: pd.DataFrame, for_date: Union[str, pd.Timestamp], zone_context: str):
        self.processing_date = pd.to_datetime(for_date).date()
        column_config = {
            'patient_id': {"default": f"UnknownPID_{self.processing_date.isoformat()}"},
            'min_spo2_pct': {"default": np.nan},
            'vital_signs_temperature_celsius': {"default": np.nan},
            'fall_detected_today': {"default": 0},
        }
        super().__init__(data_df=patient_encounter_df, source_context="CHWPatientAlertGen")
        self.df = self._prepare_dataframe(patient_encounter_df, column_config)
        self.ALERT_RULES = [
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "metric_name": "SpO2",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2"}, "priority_calculator": lambda v, t: 98.0 + max(0, t - v), "protocol_to_trigger": "PATIENT_CRITICAL_SPO2_LOW"},
            {"metric": "fall_detected_today", "condition": "greater_than_or_equal", "threshold": 1, "metric_name": "Fall(s)",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Fall Detected"}, "priority_calculator": lambda v, t: 92.0, "protocol_to_trigger": "PATIENT_FALL_DETECTED"},
        ]

    def _handle_triggered_protocols(self, alerts: List[Dict]):
        for alert in alerts:
            if alert.get('protocol_to_trigger'):
                execute_escalation_protocol(alert['protocol_to_trigger'], alert['context_data'])

    def generate(self, max_alerts_to_return: int = 15) -> List[Dict[str, Any]]:
        if self.df.empty: return []
        all_alerts = [alert for _, row in self.df.iterrows() for alert in self._evaluate_rules(row)]
        deduped_alerts = {alert['patient_id']: alert for alert in sorted(all_alerts, key=lambda x: x['raw_priority_score'])}
        final_alerts = list(deduped_alerts.values())
        self._handle_triggered_protocols(final_alerts)
        final_alerts.sort(key=lambda x: ({"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x["alert_level"], 3), -x['raw_priority_score']))
        logger.info(f"({self.source_context}) Generated {len(final_alerts)} unique CHW patient alerts for {self.processing_date}.")
        return final_alerts[:max_alerts_to_return]

# --- Clinic-Specific Alert Generator ---

class ClinicPatientAlerts(BaseAlertGenerator):
    """Generates a DataFrame of patient alerts suitable for a clinic dashboard review."""

    def __init__(self, health_data_df: pd.DataFrame, **_kwargs): # Accept kwargs for compatibility
        super().__init__(data_df=health_data_df, source_context="ClinicPatientAlerts")
        column_config = {
            'patient_id': {"default": "UnknownPID"}, 'encounter_date': {"default": pd.NaT}, 'ai_risk_score': {"default": 0.0},
            'min_spo2_pct': {"default": np.nan}, 'vital_signs_temperature_celsius': {"default": np.nan},
        }
        self.df = self._prepare_dataframe(health_data_df, column_config)
        self.ALERT_RULES = [
             {"metric": "ai_risk_score", "condition": "greater_than_or_equal", "threshold": self._get_setting('RISK_SCORE_HIGH_THRESHOLD', 80), "metric_name": "AI Risk",
             "alert_details": {"Alert Reason": "High AI Risk"}, "priority_calculator": lambda v, t: v},
             {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "metric_name": "SpO2",
             "alert_details": {"Alert Reason": "Critical SpO2"}, "priority_calculator": lambda v, t: 95.0},
        ]
    
    def _get_latest_encounter_per_patient(self) -> pd.DataFrame:
        if self.df.empty or 'patient_id' not in self.df.columns: return pd.DataFrame()
        return self.df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['patient_id'], keep='last')

    def generate_patient_review_list(self) -> pd.DataFrame:
        latest_encounters_df = self._get_latest_encounter_per_patient()
        if latest_encounters_df.empty: return pd.DataFrame()

        all_alerts = [alert for _, row in latest_encounters_df.iterrows() for alert in self._evaluate_rules(row)]
        if not all_alerts: return pd.DataFrame()
        
        alerts_df = pd.DataFrame(all_alerts)
        final_df = pd.merge(alerts_df, latest_encounters_df, on='patient_id', how='left')
        final_df['Priority Score'] = final_df['raw_priority_score'].round(1)
        final_df['alert_reason'] = final_df.apply(lambda row: f"{row['Alert Reason']} ({row['triggering_value']})", axis=1)
        final_df.sort_values('Priority Score', ascending=False, inplace=True)
        final_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
        
        output_cols_map = {'patient_id': 'patient_id', 'age': 'age', 'gender': 'gender', 'ai_risk_score': 'ai_risk_score',
                           'alert_reason': 'alert_reason', 'encounter_date_y': 'last_encounter_date', 'condition': 'diagnosis'}
        final_df.rename(columns=output_cols_map, inplace=True)
        
        final_cols = [v for k, v in output_cols_map.items() if v in final_df.columns]
        logger.info(f"({self.source_context}) Generated {len(final_df)} patient entries for clinic review list.")
        return final_df[final_cols].reset_index(drop=True)

# --- Public Factory Functions (for backward compatibility) ---

def generate_chw_patient_alerts(patient_encounter_data_df: Optional[pd.DataFrame], for_date: Union[str, pd.Timestamp], chw_zone_context_str: str, max_alerts_to_return: int = 15) -> List[Dict[str, Any]]:
    """Factory function to generate CHW alerts."""
    generator = CHWAlertGenerator(patient_encounter_data_df or pd.DataFrame(), for_date, chw_zone_context_str)
    return generator.generate(max_alerts_to_return=max_alerts_to_return)

# FIX: Add the missing factory function to maintain backward compatibility for any module that still imports the old name.
def get_patient_alerts_for_clinic(health_df_period: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """
    Factory function to generate a DataFrame of patient alerts for clinic review.
    This acts as a wrapper around the new ClinicPatientAlerts class.
    """
    generator = ClinicPatientAlerts(health_data_df=health_df_period or pd.DataFrame())
    return generator.generate_patient_review_list()
