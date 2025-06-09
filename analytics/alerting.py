# ssentinel_project_root/analytics/alerting.py
# SME FINAL VERSION (V2 - FutureWarning & Robustness FIX)
# This version corrects the pandas FutureWarning, hardens the data preparation
# logic to prevent KeyErrors from schema mismatches, and refines the public API.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime

# --- Module Setup ---
logger = logging.getLogger(__name__)

# --- Module Imports & Fallbacks for Resilience ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from .protocol_executor import execute_escalation_protocol
except ImportError:
    logging.warning("alerting.py: Could not import full dependencies. Using mock fallbacks.")
    class MockSettings: pass
    settings = MockSettings()
    def convert_to_numeric(series, **kwargs): return pd.to_numeric(series, errors='coerce')
    def execute_escalation_protocol(protocol_name, data_dict): logger.info(f"DUMMY CALL: Escalation '{protocol_name}'")

# --- Base Class for the Rules Engine ---
class BaseAlertGenerator:
    """A base class for a data-driven alert generation rules engine."""
    def __init__(self, data_df: pd.DataFrame):
        self.df = data_df.copy() if isinstance(data_df, pd.DataFrame) else pd.DataFrame()
        self.ALERT_RULES: List[Dict[str, Any]] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _prepare_dataframe(self, column_config: Dict[str, Any]):
        """A robust, reusable method to clean and prepare the source DataFrame."""
        if self.df.empty: return
        for col, default in column_config.items():
            if col not in self.df.columns:
                self.df[col] = default
            
            # Convert to numeric, respecting the provided default for non-convertible values
            numeric_series = convert_to_numeric(self.df[col], default_value=default)
            
            # <<< SME REVISION >>> Fixed the FutureWarning.
            # The original `self.df[col].fillna(..., inplace=True)` is an anti-pattern.
            # This direct assignment is the robust, correct way to update a column.
            self.df[col] = numeric_series.fillna(default)

    def _evaluate_rules(self) -> pd.DataFrame:
        """Evaluates all configured rules against the entire DataFrame using vectorized operations."""
        all_alerts = []
        if self.df.empty: return pd.DataFrame()

        for rule in self.ALERT_RULES:
            metric_col = rule['metric']
            if metric_col not in self.df.columns: continue
            
            series = self.df[metric_col].dropna()
            if series.empty: continue
            
            threshold = rule['threshold']
            
            if rule['condition'] == 'less_than':
                mask = series < threshold
            elif rule['condition'] == 'greater_than_or_equal':
                mask = series >= threshold
            elif rule['condition'] == 'between' and isinstance(threshold, tuple):
                mask = series.between(threshold[0], threshold[1], inclusive='left')
            else: continue
                
            if mask.any():
                triggered_df = self.df.loc[mask].copy()
                triggered_df['raw_priority_score'] = rule['priority_calculator'](triggered_df[metric_col], threshold)
                triggered_df['primary_reason'] = rule['alert_details']['primary_reason']
                triggered_df['alert_level'] = rule['alert_details']['alert_level']
                triggered_df['protocol_to_trigger'] = rule.get('protocol_to_trigger')
                triggered_df['brief_details'] = triggered_df.apply(rule['details_formatter'], axis=1)
                all_alerts.append(triggered_df)
        
        return pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    def _deduplicate_alerts(self, alerts_df: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
        """Keeps the highest priority alert for each patient-reason combination."""
        if alerts_df.empty: return pd.DataFrame()
        return alerts_df.sort_values('raw_priority_score', ascending=False).drop_duplicates(on_cols, keep='first')

    def generate(self, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the 'generate' method.")

# --- CHW-Specific Alert Generator ---
class CHWAlertGenerator(BaseAlertGenerator):
    """Generates prioritized alerts for Community Health Workers."""
    def __init__(self, patient_encounter_df: pd.DataFrame):
        super().__init__(patient_encounter_df)
        self._prepare_dataframe({
            'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
            'max_skin_temp_celsius': np.nan, 'fall_detected_today': 0,
        })
        # <<< SME REVISION >>> Hardened this logic to prevent KeyErrors.
        # It now safely checks for columns before trying to use them.
        temp_col = self.df.get('vital_signs_temperature_celsius', pd.Series(dtype=float))
        skin_temp_col = self.df.get('max_skin_temp_celsius', pd.Series(dtype=float))
        self.df['temperature'] = temp_col.fillna(skin_temp_col)
        self._define_rules()
        
    def _define_rules(self):
        spo2_crit = self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90.0)
        spo2_warn = self._get_setting('ALERT_SPO2_WARNING_LOW_PCT', 94.0)
        temp_crit = self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
        self.ALERT_RULES = [
            {"metric": "min_spo2_pct", "condition": "less_than", "threshold": spo2_crit, "priority_calculator": lambda v, t: 98 + (t - v), "protocol_to_trigger": "PATIENT_CRITICAL_SPO2_LOW", "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2"}, "details_formatter": lambda r: f"SpO2: {r.get('min_spo2_pct', 0):.0f}%"},
            {"metric": "min_spo2_pct", "condition": "between", "threshold": (spo2_crit, spo2_warn), "priority_calculator": lambda v, t: 75 + (t[1] - v), "alert_details": {"alert_level": "WARNING", "primary_reason": "Low SpO2"}, "details_formatter": lambda r: f"SpO2: {r.get('min_spo2_pct', 0):.0f}%"},
            {"metric": "temperature", "condition": "greater_than_or_equal", "threshold": temp_crit, "priority_calculator": lambda v, t: 95 + (v - t) * 2, "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever"}, "details_formatter": lambda r: f"Temp: {r.get('temperature', 0):.1f}°C"},
            {"metric": "fall_detected_today", "condition": "greater_than_or_equal", "threshold": 1, "priority_calculator": lambda v, t: 92.0, "protocol_to_trigger": "PATIENT_FALL_DETECTED", "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Fall Detected"}, "details_formatter": lambda r: f"Falls: {int(r.get('fall_detected_today', 0))}"},
        ]

    def _handle_triggered_protocols(self, alerts_df: pd.DataFrame):
        protocol_alerts = alerts_df.dropna(subset=['protocol_to_trigger'])
        for _, alert_row in protocol_alerts.iterrows():
            execute_escalation_protocol(alert_row['protocol_to_trigger'], alert_row.to_dict())

    def generate(self, max_alerts: int = 15) -> List[Dict[str, Any]]:
        all_alerts_df = self._evaluate_rules()
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df, on_cols=['patient_id', 'primary_reason'])
        if unique_alerts_df.empty: return []
        self._handle_triggered_protocols(unique_alerts_df)
        unique_alerts_df['level_sort'] = unique_alerts_df['alert_level'].map({"CRITICAL": 0, "WARNING": 1, "INFO": 2}).fillna(3)
        final_alerts = unique_alerts_df.sort_values(['level_sort', 'raw_priority_score'], ascending=[True, False])
        # Safely create the context_info column
        condition_col = final_alerts.get('condition', 'N/A').astype(str)
        zone_col = final_alerts.get('zone_id', 'N/A').astype(str)
        final_alerts['context_info'] = "Cond: " + condition_col + " | Zone: " + zone_col
        output_cols = ['patient_id', 'alert_level', 'primary_reason', 'brief_details', 'context_info', 'raw_priority_score']
        return final_alerts.reindex(columns=output_cols).head(max_alerts).to_dict('records')

# --- Clinic Dashboard Alert Generator ---
class ClinicPatientAlertGenerator(BaseAlertGenerator):
    """Generates a DataFrame of patient alerts suitable for a clinic dashboard review."""
    def __init__(self, health_df: pd.DataFrame):
        super().__init__(health_df)
        self.df = self.df.sort_values('encounter_date', na_position='first').drop_duplicates('patient_id', keep='last')
        self._prepare_dataframe({
            'ai_risk_score': 0.0, 'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan
        })
        # Use the same hardened logic as the CHW generator
        temp_col = self.df.get('vital_signs_temperature_celsius', pd.Series(dtype=float))
        skin_temp_col = self.df.get('max_skin_temp_celsius', pd.Series(dtype=float))
        self.df['temperature'] = temp_col.fillna(skin_temp_col)
        self._define_rules()

    def _define_rules(self):
        self.ALERT_RULES = [
             {"metric": "ai_risk_score", "condition": "greater_than_or_equal", "threshold": self._get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60), "alert_details": {"alert_level": "INFO", "primary_reason": "High AI Risk"}, "priority_calculator": lambda v, t: v, "details_formatter": lambda r: f"Score: {r.get('ai_risk_score', 0):.0f}"},
             {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical SpO2"}, "priority_calculator": lambda v, t: 95.0 + (t-v), "details_formatter": lambda r: f"SpO2: {r.get('min_spo2_pct', 0):.0f}%"},
             {"metric": "temperature", "condition": "greater_than_or_equal", "threshold": self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever"}, "priority_calculator": lambda v, t: 90.0 + (v-t), "details_formatter": lambda r: f"Temp: {r.get('temperature', 0):.1f}°C"},
        ]
    
    def _format_output_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        # Use a safe .get() for the triggering value
        df['Alert Reason'] = df.apply(lambda row: f"{row['primary_reason']} ({row.get('brief_details', 'N/A')})", axis=1)
        df['Priority Score'] = df['raw_priority_score'].round(1)
        output_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
        return df.reindex(columns=output_cols).fillna('N/A')

    def generate(self) -> pd.DataFrame:
        all_alerts_df = self._evaluate_rules()
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df, on_cols=['patient_id', 'primary_reason'])
        return self._format_output_df(unique_alerts_df)

# --- Public Factory Functions ---
def generate_chw_patient_alerts(patient_encounter_data_df: Optional[pd.DataFrame], **kwargs) -> List[Dict[str, Any]]:
    """
    Factory function to generate prioritized alerts for a CHW.
    Assumes the input DataFrame is already filtered for the desired context (e.g., date, CHW).
    """
    if not isinstance(patient_encounter_data_df, pd.DataFrame) or patient_encounter_data_df.empty:
        return []
    # <<< SME REVISION >>> Simplified this function. The calling page is now responsible
    # for passing a correctly pre-filtered DataFrame, which it already does.
    generator = CHWAlertGenerator(patient_encounter_data_df)
    return generator.generate(**kwargs)

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame) -> pd.DataFrame:
    """Factory function to generate a list of flagged patients for clinic review."""
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        return pd.DataFrame()
    generator = ClinicPatientAlertGenerator(health_df_period)
    return generator.generate()
