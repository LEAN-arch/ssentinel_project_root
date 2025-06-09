# ssentinel_project_root/analytics/alerting.py
"""
Provides robust, data-driven alert generation for CHW and Clinic dashboards.
This is the centralized, high-performance rules engine for patient safety alerts.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime

logger = logging.getLogger(__name__)

# --- Module Imports & Fallbacks for Resilience ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from .protocol_executor import execute_escalation_protocol
except ImportError:
    logging.warning("Could not import full dependencies for alerting.py. Using fallbacks.")
    class MockSettings: pass
    settings = MockSettings()
    def execute_escalation_protocol(protocol_name, data_dict, **kwargs):
        logger.info(f"DUMMY CALL: Escalation '{protocol_name}' triggered for patient {data_dict.get('patient_id')}")
    def convert_to_numeric(series, **kwargs):
        return pd.to_numeric(series, errors='coerce')

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
            # `convert_to_numeric` should handle filling NaNs if the whole series is convertible
            self.df[col] = convert_to_numeric(self.df[col])
            self.df[col].fillna(default, inplace=True)
    
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
                triggered_mask = series < threshold
            elif rule['condition'] == 'greater_than_or_equal':
                triggered_mask = series >= threshold
            elif rule['condition'] == 'between' and isinstance(threshold, tuple):
                triggered_mask = series.between(threshold[0], threshold[1], inclusive='left')
            else:
                continue
                
            if triggered_mask.any():
                triggered_df = self.df.loc[triggered_mask].copy()
                triggered_df['raw_priority_score'] = rule['priority_calculator'](triggered_df[metric_col], threshold)
                triggered_df['primary_reason'] = rule['alert_details']['primary_reason']
                triggered_df['alert_level'] = rule['alert_details']['alert_level']
                triggered_df['protocol_to_trigger'] = rule.get('protocol_to_trigger')
                triggered_df['brief_details'] = triggered_df.apply(rule['details_formatter'], axis=1)
                all_alerts.append(triggered_df)
        
        return pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    def _deduplicate_alerts(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Keeps the highest priority alert for each patient-reason combination."""
        if alerts_df.empty: return pd.DataFrame()
        return alerts_df.sort_values('raw_priority_score', ascending=False).drop_duplicates(['patient_id', 'primary_reason'], keep='first')

    def generate(self, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the 'generate' method.")


class CHWAlertGenerator(BaseAlertGenerator):
    """Generates prioritized alerts for Community Health Workers."""
    def __init__(self, patient_encounter_df: pd.DataFrame):
        super().__init__(patient_encounter_df)
        self._prepare_dataframe({
            'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan,
            'max_skin_temp_celsius': np.nan, 'fall_detected_today': 0,
        })
        self.df['temperature'] = self.df['vital_signs_temperature_celsius'].fillna(self.df.get('max_skin_temp_celsius'))
        
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
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df)
        
        if unique_alerts_df.empty: return []

        self._handle_triggered_protocols(unique_alerts_df)
        
        unique_alerts_df['level_sort'] = unique_alerts_df['alert_level'].map({"CRITICAL": 0, "WARNING": 1, "INFO": 2}).fillna(3)
        final_alerts = unique_alerts_df.sort_values(['level_sort', 'raw_priority_score'], ascending=[True, False])
        
        final_alerts['context_info'] = "Cond: " + final_alerts.get('condition', 'N/A').astype(str) + " | Zone: " + final_alerts.get('zone_id', 'N/A').astype(str)
        output_cols = ['patient_id', 'alert_level', 'primary_reason', 'brief_details', 'context_info', 'raw_priority_score']
        return final_alerts.reindex(columns=output_cols).head(max_alerts).to_dict('records')

class ClinicPatientAlertGenerator(BaseAlertGenerator):
    """Generates a DataFrame of patient alerts suitable for a clinic dashboard review."""
    def __init__(self, health_df: pd.DataFrame):
        super().__init__(health_df)
        self.df = self.df.sort_values('encounter_date', na_position='first').drop_duplicates('patient_id', keep='last')
        self._prepare_dataframe({
            'ai_risk_score': 0.0, 'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan
        })
        # SME NOTE: FINAL FIX. Added standardization of the 'temperature' column here.
        self.df['temperature'] = self.df['vital_signs_temperature_celsius'].fillna(self.df.get('max_skin_temp_celsius'))
        
        self.ALERT_RULES = [
             {"metric": "ai_risk_score", "condition": "greater_than_or_equal", "threshold": self._get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60), "alert_details": {"alert_level": "INFO", "primary_reason": "High AI Risk"}, "priority_calculator": lambda v, t: v},
             {"metric": "min_spo2_pct", "condition": "less_than", "threshold": self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "Critical SpO2"}, "priority_calculator": lambda v, t: 95.0 + (t-v)},
             {"metric": "temperature", "condition": "greater_than_or_equal", "threshold": self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), "alert_details": {"alert_level": "CRITICAL", "primary_reason": "High Fever"}, "priority_calculator": lambda v, t: 90.0 + (v-t)},
        ]
    
    def _format_output_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        df['Alert Reason'] = df.apply(lambda row: f"{row['primary_reason']} ({row.get('triggering_value', 'N/A')})", axis=1)
        df['Priority Score'] = df['raw_priority_score'].round(1)
        output_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
        return df.reindex(columns=output_cols)

    def generate(self) -> pd.DataFrame:
        all_alerts_df = self._evaluate_rules()
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df)
        return self._format_output_df(unique_alerts_df)


# --- Public Factory Functions ---

def generate_chw_patient_alerts(patient_encounter_df: pd.DataFrame, for_date: Union[str, date_type], max_alerts: int = 15) -> List[Dict[str, Any]]:
    """Factory function to generate prioritized alerts for a CHW on a specific date."""
    if not isinstance(patient_encounter_df, pd.DataFrame) or patient_encounter_df.empty: return []
    try: processing_date = pd.to_datetime(for_date).date()
    except (AttributeError, ValueError): processing_date = datetime.now().date()
    df_today = patient_encounter_data_df[pd.to_datetime(patient_encounter_data_df['encounter_date']).dt.date == processing_date]
    if df_today.empty: return []
    generator = CHWAlertGenerator(df_today)
    return generator.generate(max_alerts=max_alerts)

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame) -> pd.DataFrame:
    """Factory function to generate a list of flagged patients for clinic review."""
    if not isinstance(health_df_period, pd.DataFrame): return pd.DataFrame()
    generator = ClinicPatientAlertGenerator(health_df_period)
    return generator.generate()
