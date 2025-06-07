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
        """Safely gets a configuration value from the global settings."""
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
                # Use vectorized replacement for common NA strings
                if pd.api.types.is_object_dtype(prepared_df[col].dtype):
                    na_regex = r'^\s*(?:nan|none|n/a|nat|<na>|null)\s*$'
                    prepared_df[col].replace(na_regex, np.nan, regex=True, inplace=True)
                
                prepared_df[col] = convert_to_numeric(prepared_df[col], default_value=config.get('default'))
        return prepared_df

    def _evaluate_rules(self, data_row: pd.Series) -> List[Dict[str, Any]]:
        """Evaluates all configured rules against a single row of data."""
        triggered_alerts = []
        for rule in self.ALERT_RULES:
            metric_val = data_row.get(rule['metric'])
            if pd.isna(metric_val):
                continue

            # Evaluate condition
            threshold = rule['threshold']() if callable(rule['threshold']) else rule['threshold']
            condition_met = False
            if rule['condition'] == 'less_than' and metric_val < threshold:
                condition_met = True
            elif rule['condition'] == 'greater_than_or_equal' and metric_val >= threshold:
                condition_met = True
            
            if condition_met:
                alert = rule['alert_details'].copy()
                alert['raw_priority_score'] = rule['priority_calculator'](metric_val, threshold)
                alert['patient_id'] = str(data_row.get('patient_id', 'UnknownPID'))
                alert['triggering_value'] = f"{rule.get('metric_name', rule['metric'])}: {metric_val:.1f}"
                alert['protocol_to_trigger'] = rule.get('protocol_to_trigger') # Tag for later execution
                alert['context_data'] = data_row.to_dict() # Pass full context for protocol
                triggered_alerts.append(alert)
        return triggered_alerts

    def generate(self, **kwargs) -> Any:
        """Main orchestration method to generate alerts."""
        raise NotImplementedError("Subclasses must implement the 'generate' method.")


# --- CHW-Specific Alert Generator ---

class CHWAlertGenerator(BaseAlertGenerator):
    """Generates prioritized alerts for Community Health Workers."""

    def __init__(self, patient_encounter_df: pd.DataFrame, for_date: Union[str, pd.Timestamp], zone_context: str):
        self.processing_date = pd.to_datetime(for_date).date()
        self.zone_context = zone_context
        
        column_config = {
            'patient_id': {"default": f"UnknownPID_{self.processing_date.isoformat()}"},
            'min_spo2_pct': {"default": np.nan},
            'vital_signs_temperature_celsius': {"default": np.nan},
            'fall_detected_today': {"default": 0},
            'ai_followup_priority_score': {"default": 0.0},
        }
        
        super().__init__(data_df=patient_encounter_df, source_context="CHWPatientAlertGen")
        self.df = self._prepare_dataframe(patient_encounter_df, column_config)
        
        self.ALERT_RULES = [
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "metric_name": "SpO2",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2", "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT"},
             "priority_calculator": lambda v, t: 98.0 + max(0, t - v), "protocol_to_trigger": "PATIENT_CRITICAL_SPO2_LOW"},
            
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_WARNING_LOW_PCT', 92), "metric_name": "SpO2",
             "alert_details": {"alert_level": "WARNING", "primary_reason": "Low SpO2", "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR"},
             "priority_calculator": lambda v, t: 75.0 + max(0, t - v)},
            
            {"metric": "vital_signs_temperature_celsius", "condition": "greater_than_or_equal", "threshold": self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "metric_name": "Temp",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever", "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT"},
             "priority_calculator": lambda v, t: 95.0 + max(0, (v - t) * 2.0)},
             
            {"metric": "fall_detected_today", "condition": "greater_than_or_equal", "threshold": 1, "metric_name": "Fall(s)",
             "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Fall Detected", "suggested_action_code": "ACTION_FALL_ASSESS_URGENT"},
             "priority_calculator": lambda v, t: 92.0, "protocol_to_trigger": "PATIENT_FALL_DETECTED"},
        ]

    def _handle_triggered_protocols(self, alerts: List[Dict]):
        """Executes escalation protocols for alerts that require them."""
        for alert in alerts:
            if alert.get('protocol_to_trigger'):
                execute_escalation_protocol(alert['protocol_to_trigger'], alert['context_data'])

    def generate(self, max_alerts_to_return: int = 15) -> List[Dict[str, Any]]:
        if self.df.empty:
            logger.warning(f"({self.source_context}) No valid data; no CHW alerts generated for {self.processing_date}.")
            return []

        all_alerts = []
        for _, row in self.df.iterrows():
            all_alerts.extend(self._evaluate_rules(row))

        # Deduplicate, keeping the highest priority alert per patient
        deduped_alerts = {}
        for alert in all_alerts:
            pid = alert['patient_id']
            if pid not in deduped_alerts or alert['raw_priority_score'] > deduped_alerts[pid]['raw_priority_score']:
                deduped_alerts[pid] = alert
        
        final_alerts = list(deduped_alerts.values())
        self._handle_triggered_protocols(final_alerts)

        # Sort by level (Critical > Warning), then by priority score descending
        final_alerts.sort(key=lambda x: ({"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(x["alert_level"], 3), -x['raw_priority_score']))
        
        logger.info(f"({self.source_context}) Generated {len(final_alerts)} unique CHW patient alerts for {self.processing_date}.")
        return final_alerts[:max_alerts_to_return]


# --- Clinic-Specific Alert Generator ---

class ClinicPatientAlerts(BaseAlertGenerator):
    """Generates a DataFrame of patient alerts suitable for a clinic dashboard review."""

    def __init__(self, health_data_df: pd.DataFrame, reporting_period_df: pd.DataFrame):
        # In a real scenario, reporting_period_df might be used for context, but here we focus on the main data.
        super().__init__(data_df=health_data_df, source_context="ClinicPatientAlerts")
        
        column_config = {
            'patient_id': {"default": "UnknownPID"}, 'encounter_date': {"default": pd.NaT}, 'ai_risk_score': {"default": 0.0},
            'min_spo2_pct': {"default": np.nan}, 'vital_signs_temperature_celsius': {"default": np.nan},
            'referral_status': {"default": "Unknown"}, 'condition': {"default": "N/A"}
        }
        self.df = self._prepare_dataframe(health_data_df, column_config)

        self.ALERT_RULES = [
             {"metric": "ai_risk_score", "condition": "greater_than_or_equal", "threshold": self._get_setting('RISK_SCORE_HIGH_THRESHOLD', 80), "metric_name": "AI Risk",
             "alert_details": {"Alert Reason": "High AI Risk"}, "priority_calculator": lambda v, t: v},
             
             {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "metric_name": "SpO2",
             "alert_details": {"Alert Reason": "Critical SpO2"}, "priority_calculator": lambda v, t: 95.0},
             
             {"metric": "vital_signs_temperature_celsius", "condition": "greater_than_or_equal", "threshold": self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "metric_name": "Temp",
             "alert_details": {"Alert Reason": "High Fever"}, "priority_calculator": lambda v, t: 90.0},
        ]
    
    def _get_latest_encounter_per_patient(self) -> pd.DataFrame:
        """Gets the most recent record for each patient."""
        if self.df.empty or 'patient_id' not in self.df.columns:
            return pd.DataFrame()
        return self.df.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['patient_id'], keep='last')

    def generate_patient_review_list(self) -> pd.DataFrame:
        """Main method to generate the formatted DataFrame for the clinic dashboard."""
        latest_encounters_df = self._get_latest_encounter_per_patient()
        if latest_encounters_df.empty:
            logger.warning(f"({self.source_context}) No latest encounter data to process.")
            return pd.DataFrame()

        all_alerts = []
        for _, row in latest_encounters_df.iterrows():
            all_alerts.extend(self._evaluate_rules(row))

        if not all_alerts:
            logger.info(f"({self.source_context}) No patients met criteria for clinic review list.")
            return pd.DataFrame()
        
        alerts_df = pd.DataFrame(all_alerts)
        
        # Format for final output: merge alert data with original record context
        final_df = pd.merge(alerts_df, latest_encounters_df, on='patient_id', how='left')
        final_df['Priority Score'] = final_df['raw_priority_score'].round(1)
        
        # Add the triggering value to the reason
        final_df['alert_reason'] = final_df.apply(lambda row: f"{row['Alert Reason']} ({row['triggering_value']})", axis=1)

        # Deduplicate, keeping only the highest priority reason per patient
        final_df.sort_values('Priority Score', ascending=False, inplace=True)
        final_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
        
        # Define and format final output columns
        output_cols_map = {
            'patient_id': 'patient_id', 'age': 'age', 'gender': 'gender', 'ai_risk_score': 'ai_risk_score',
            'alert_reason': 'alert_reason', 'encounter_date': 'last_encounter_date',
            'condition': 'diagnosis', # Assuming condition is the main diagnosis
        }
        final_df.rename(columns=output_cols_map, inplace=True)
        
        # Ensure all expected columns exist
        for col in output_cols_map.values():
            if col not in final_df.columns:
                final_df[col] = pd.NA

        logger.info(f"({self.source_context}) Generated {len(final_df)} patient entries for clinic review list.")
        return final_df[list(output_cols_map.values())].reset_index(drop=True)

# --- Public Factory Functions (for backward compatibility) ---

def generate_chw_patient_alerts(patient_encounter_data_df: Optional[pd.DataFrame], for_date: Union[str, pd.Timestamp], chw_zone_context_str: str, max_alerts_to_return: int = 15) -> List[Dict[str, Any]]:
    """Factory function to generate CHW alerts."""
    generator = CHWAlertGenerator(patient_encounter_data_df or pd.DataFrame(), for_date, chw_zone_context_str)
    return generator.generate(max_alerts_to_return=max_alerts_to_return)
