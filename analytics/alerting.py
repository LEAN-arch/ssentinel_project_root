# sentinel_project_root/analytics/alerting.py
# Provides robust, data-driven alert generation for CHW and Clinic dashboards.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import warnings

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from .protocol_executor import execute_escalation_protocol
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in alerting.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Base Class for Alert Generation (Rules Engine) ---

class BaseAlertGenerator:
    """A base class for a data-driven alert generation rules engine."""
    
    def __init__(self, data_df: pd.DataFrame):
        self.df = data_df.copy() if isinstance(data_df, pd.DataFrame) else pd.DataFrame()
        self.ALERT_RULES: List[Dict[str, Any]] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        """Safely gets a configuration value from the global settings."""
        return getattr(settings, attr_name, default_value)

    def _prepare_dataframe(self, column_config: Dict[str, Any]):
        """A robust, reusable method to clean and prepare the source DataFrame."""
        if self.df.empty: return
        
        for col, default in column_config.items():
            if col not in self.df.columns:
                self.df[col] = default
            else:
                self.df[col] = convert_to_numeric(self.df[col], default_value=default)
    
    def _evaluate_rules(self) -> pd.DataFrame:
        """
        Evaluates all configured rules against the entire DataFrame using vectorized operations.
        Returns a DataFrame of all triggered alerts.
        """
        all_alerts = []
        if self.df.empty: return pd.DataFrame()

        for rule in self.ALERT_RULES:
            metric_col = rule['metric']
            if metric_col not in self.df.columns: continue

            series = self.df[metric_col].dropna()
            if series.empty: continue
            
            threshold = rule['threshold']() if callable(rule['threshold']) else rule['threshold']
            
            if rule['condition'] == 'less_than':
                triggered_mask = series < threshold
            elif rule['condition'] == 'greater_than_or_equal':
                triggered_mask = series >= threshold
            else:
                continue
                
            if triggered_mask.any():
                triggered_df = self.df.loc[triggered_mask].copy()
                triggered_df['raw_priority_score'] = rule['priority_calculator'](triggered_df[metric_col], threshold)
                triggered_df['Alert Reason'] = rule['alert_details']['primary_reason']
                triggered_df['Triggering Value'] = triggered_df[metric_col].round(1).astype(str)
                triggered_df['Alert Level'] = rule['alert_details']['alert_level']
                triggered_df['protocol_to_trigger'] = rule.get('protocol_to_trigger')
                all_alerts.append(triggered_df)
        
        if not all_alerts: return pd.DataFrame()
        return pd.concat(all_alerts, ignore_index=True)

    def _deduplicate_alerts(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Keeps only the highest priority alert for each patient."""
        if alerts_df.empty: return pd.DataFrame()
        return alerts_df.sort_values('raw_priority_score', ascending=False).drop_duplicates('patient_id', keep='first')

    def generate(self) -> Any:
        """Main orchestration method to generate alerts."""
        raise NotImplementedError("Subclasses must implement the 'generate' method.")


# --- CHW-Specific Alert Generator ---

class CHWAlertGenerator(BaseAlertGenerator):
    """Generates prioritized alerts for Community Health Workers."""

    def __init__(self, patient_encounter_df: pd.DataFrame):
        super().__init__(patient_encounter_df)
        self._prepare_dataframe({
            'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
            'fall_detected_today': 0, 'ai_followup_priority_score': 0.0,
        })
        self.ALERT_RULES = [
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2"}, "priority_calculator": lambda v, t: 98 + (t - v), "protocol_to_trigger": "PATIENT_CRITICAL_SPO2_LOW"},
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_WARNING_LOW_PCT', 92), "alert_details": {"alert_level": "WARNING", "primary_reason": "Low SpO2"}, "priority_calculator": lambda v, t: 75 + (t - v)},
            {"metric": "vital_signs_temperature_celsius", "condition": "greater_than_or_equal", "threshold": self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever"}, "priority_calculator": lambda v, t: 95 + (v - t) * 2},
            {"metric": "fall_detected_today", "condition": "greater_than_or_equal", "threshold": 1, "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Fall Detected"}, "priority_calculator": lambda v, t: 92.0, "protocol_to_trigger": "PATIENT_FALL_DETECTED"},
        ]

    def _handle_triggered_protocols(self, alerts_df: pd.DataFrame):
        """Executes escalation protocols for alerts that require them."""
        protocol_alerts = alerts_df.dropna(subset=['protocol_to_trigger'])
        if not protocol_alerts.empty:
            with warnings.catch_warnings(): # Suppress SettingWithCopyWarning in protocol execution context
                warnings.simplefilter("ignore")
                for _, alert_row in protocol_alerts.iterrows():
                    execute_escalation_protocol(alert_row['protocol_to_trigger'], alert_row.to_dict())

    def generate(self, max_alerts: int = 15) -> List[Dict[str, Any]]:
        """Main method to generate a sorted list of unique CHW patient alerts."""
        all_alerts_df = self._evaluate_rules()
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df)
        
        if unique_alerts_df.empty: return []

        self._handle_triggered_protocols(unique_alerts_df)
        
        # Sort by level (Critical > Warning), then by priority score descending
        unique_alerts_df['level_sort'] = unique_alerts_df['Alert Level'].map({"CRITICAL": 0, "WARNING": 1, "INFO": 2}).fillna(3)
        final_alerts = unique_alerts_df.sort_values(['level_sort', 'raw_priority_score'], ascending=[True, False])
        
        return final_alerts.head(max_alerts).to_dict('records')


# --- Clinic Dashboard Alert Generator ---

class ClinicPatientAlertGenerator(BaseAlertGenerator):
    """Generates a DataFrame of patient alerts suitable for a clinic dashboard review."""

    def __init__(self, health_df: pd.DataFrame):
        super().__init__(health_df)
        self.df = self.df.sort_values('encounter_date', na_position='first').drop_duplicates('patient_id', keep='last')
        self._prepare_dataframe({
            'ai_risk_score': 0.0, 'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
        })
        self.ALERT_RULES = [
             {"metric": "ai_risk_score", "condition": "greater_than_or_equal", "threshold": self._get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60), "alert_details": {"alert_level": "INFO", "primary_reason": "High AI Risk"}, "priority_calculator": lambda v, t: v},
             {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 88), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical SpO2"}, "priority_calculator": lambda v, t: 95.0 + (t-v)},
             {"metric": "vital_signs_temperature_celsius", "condition": "greater_than_or_equal", "threshold": self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever"}, "priority_calculator": lambda v, t: 90.0 + (v-t)},
        ]
    
    def generate(self) -> pd.DataFrame:
        """Main method to generate the formatted DataFrame for the clinic dashboard."""
        all_alerts_df = self._evaluate_rules()
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df)

        if unique_alerts_df.empty:
            return pd.DataFrame()
        
        # Format the final output DataFrame
        unique_alerts_df['Alert Reason'] = unique_alerts_df.apply(lambda row: f"{row['Alert Reason']} ({row['Triggering Value']})", axis=1)
        unique_alerts_df['Priority Score'] = unique_alerts_df['raw_priority_score'].round(1)
        
        output_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
        # Reindex to ensure all columns are present with a predictable order
        return unique_alerts_df.reindex(columns=output_cols)


# --- Public Factory Functions ---

def generate_chw_patient_alerts(patient_encounter_df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """Factory function to generate prioritized alerts for a CHW."""
    generator = CHWAlertGenerator(patient_encounter_df)
    return generator.generate()

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Factory function to generate a list of flagged patients for clinic review."""
    generator = ClinicPatientAlertGenerator(health_df_period)
    return generator.generate()
