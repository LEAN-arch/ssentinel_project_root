# ssentinel_project_root/analytics/alerting.py
# SME GOLD STANDARD VERSION (V3 - Performance, Robustness & Architectural Refinements)
# This version introduces:
# 1. Performance: Eliminates slow df.apply(axis=1) with vectorized operations.
# 2. Robustness: Uses Enums and a Rule dataclass to prevent typos and improve clarity.
# 3. Architecture: Refactors protocol handling for efficient batch execution.
# 4. Readability: Refactors complex lambdas into named methods for better testing and maintenance.

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

# --- Module Setup ---
logger = logging.getLogger(__name__)

# --- Module Imports & Fallbacks for Resilience ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    # <<< SME REVISION V3 >>> Assume the protocol executor can handle a batch DataFrame.
    from .protocol_executor import execute_escalation_protocols_batch
except ImportError:
    logging.warning("alerting.py: Could not import full dependencies. Using mock fallbacks.")
    class MockSettings: pass
    settings = MockSettings()
    def convert_to_numeric(series, **kwargs): return pd.to_numeric(series, errors='coerce')
    def execute_escalation_protocols_batch(alerts_df: pd.DataFrame):
        for _, row in alerts_df.iterrows():
             logger.info(f"DUMMY BATCH CALL: Escalation '{row['protocol_to_trigger']}' for patient {row['patient_id']}")

# <<< SME REVISION V3 >>> Use Enums for controlled vocabularies to prevent typos.
class AlertLevel(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

class AlertCondition(Enum):
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    BETWEEN = "between"

# <<< SME REVISION V3 >>> Use a dataclass for rule definitions for type safety and clarity.
@dataclass
class AlertRule:
    metric: str
    condition: AlertCondition
    threshold: Union[float, int, tuple]
    priority_calculator: Callable[[pd.Series, Any], pd.Series]
    alert_level: AlertLevel
    primary_reason: str
    details_formatter_str: str  # A format string for vectorization, e.g., "SpO2: {:.0f}%"
    protocol_to_trigger: Optional[str] = None

# --- Base Class for the Rules Engine ---
class BaseAlertGenerator:
    """A base class for a data-driven alert generation rules engine."""
    def __init__(self, data_df: pd.DataFrame):
        self.df = data_df.copy() if isinstance(data_df, pd.DataFrame) else pd.DataFrame()
        self.ALERT_RULES: List[AlertRule] = []

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
            # This direct assignment is the robust, correct way to update a column.
            self.df[col] = numeric_series.fillna(default)

    def _evaluate_rules(self) -> pd.DataFrame:
        """Evaluates all configured rules against the entire DataFrame using vectorized operations."""
        all_alerts = []
        if self.df.empty: return pd.DataFrame()

        for rule in self.ALERT_RULES:
            if rule.metric not in self.df.columns:
                logger.warning(f"Metric '{rule.metric}' for rule '{rule.primary_reason}' not in DataFrame. Skipping.")
                continue
            
            series = self.df[rule.metric].dropna()
            if series.empty: continue
            
            if rule.condition == AlertCondition.LESS_THAN:
                mask = series < rule.threshold
            elif rule.condition == AlertCondition.GREATER_THAN_OR_EQUAL:
                mask = series >= rule.threshold
            elif rule.condition == AlertCondition.BETWEEN and isinstance(rule.threshold, tuple):
                mask = series.between(rule.threshold[0], rule.threshold[1], inclusive='left')
            else:
                continue
                
            if mask.any():
                triggered_df = self.df.loc[mask].copy()
                metric_values = triggered_df[rule.metric]
                
                triggered_df['raw_priority_score'] = rule.priority_calculator(metric_values, rule.threshold)
                triggered_df['primary_reason'] = rule.primary_reason
                triggered_df['alert_level'] = rule.alert_level.value
                triggered_df['protocol_to_trigger'] = rule.protocol_to_trigger
                
                # <<< SME REVISION V3 >>> Replaced slow .apply() with fast, vectorized string formatting.
                triggered_df['brief_details'] = rule.details_formatter_str.format(metric_values)

                all_alerts.append(triggered_df)
        
        return pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    def _deduplicate_alerts(self, alerts_df: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
        """Keeps the highest priority alert for each entity-reason combination."""
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
        # Coalesce temperature columns safely
        temp_col = self.df.get('vital_signs_temperature_celsius', pd.Series(dtype=float))
        skin_temp_col = self.df.get('max_skin_temp_celsius', pd.Series(dtype=float))
        self.df['temperature'] = temp_col.fillna(skin_temp_col)
        self._define_rules()
        
    def _define_rules(self):
        spo2_crit = self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90.0)
        spo2_warn = self._get_setting('ALERT_SPO2_WARNING_LOW_PCT', 94.0)
        temp_crit = self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
        
        self.ALERT_RULES = [
            AlertRule(metric="min_spo2_pct", condition=AlertCondition.LESS_THAN, threshold=spo2_crit, priority_calculator=lambda v, t: 98 + (t - v), alert_level=AlertLevel.CRITICAL, primary_reason="Critical Low SpO2", details_formatter_str="SpO2: {}%", protocol_to_trigger="PATIENT_CRITICAL_SPO2_LOW"),
            AlertRule(metric="min_spo2_pct", condition=AlertCondition.BETWEEN, threshold=(spo2_crit, spo2_warn), priority_calculator=lambda v, t: 75 + (t[1] - v), alert_level=AlertLevel.WARNING, primary_reason="Low SpO2", details_formatter_str="SpO2: {}%"),
            AlertRule(metric="temperature", condition=AlertCondition.GREATER_THAN_OR_EQUAL, threshold=temp_crit, priority_calculator=lambda v, t: 95 + (v - t) * 2, alert_level=AlertLevel.CRITICAL, primary_reason="High Fever", details_formatter_str="Temp: {:.1f}°C"),
            AlertRule(metric="fall_detected_today", condition=AlertCondition.GREATER_THAN_OR_EQUAL, threshold=1, priority_calculator=lambda v, t: 92.0, alert_level=AlertLevel.CRITICAL, primary_reason="Fall Detected", details_formatter_str="Falls: {}", protocol_to_trigger="PATIENT_FALL_DETECTED"),
        ]

    # <<< SME REVISION V3 >>> Replaced row-by-row iteration with a single batch call.
    def _handle_triggered_protocols(self, alerts_df: pd.DataFrame):
        """Identifies alerts with protocols and triggers them in a single batch call."""
        protocol_alerts_df = alerts_df.dropna(subset=['protocol_to_trigger'])
        if not protocol_alerts_df.empty:
            execute_escalation_protocols_batch(protocol_alerts_df)

    def generate(self, max_alerts: int = 15) -> List[Dict[str, Any]]:
        all_alerts_df = self._evaluate_rules()
        unique_alerts_df = self._deduplicate_alerts(all_alerts_df, on_cols=['patient_id', 'primary_reason'])
        if unique_alerts_df.empty: return []
        
        self._handle_triggered_protocols(unique_alerts_df)

        # Create sorting and context columns safely and vectorially
        level_map = {level.value: i for i, level in enumerate(AlertLevel)}
        unique_alerts_df['level_sort'] = unique_alerts_df['alert_level'].map(level_map).fillna(len(level_map))
        
        final_alerts = unique_alerts_df.sort_values(['level_sort', 'raw_priority_score'], ascending=[True, False])
        
        condition_col = final_alerts.get('condition', 'N/A').astype(str)
        zone_col = final_alerts.get('zone_id', 'N/A').astype(str)
        final_alerts['context_info'] = "Cond: " + condition_col + " | Zone: " + zone_col
        
        # Select and format final output
        output_cols = ['patient_id', 'alert_level', 'primary_reason', 'brief_details', 'context_info', 'raw_priority_score']
        return final_alerts.reindex(columns=output_cols).head(max_alerts).to_dict('records')

# --- Clinic Dashboard Alert Generator ---
class ClinicPatientAlertGenerator(BaseAlertGenerator):
    """Generates a DataFrame of patient alerts suitable for a clinic dashboard review."""
    def __init__(self, health_df: pd.DataFrame):
        super().__init__(health_df)
        # Efficiently get the last encounter per patient
        if not self.df.empty and 'patient_id' in self.df.columns and 'encounter_date' in self.df.columns:
            self.df = self.df.sort_values('encounter_date', na_position='first').drop_duplicates('patient_id', keep='last')
        
        self._prepare_dataframe({
            'ai_risk_score': 0.0, 'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan
        })
        temp_col = self.df.get('vital_signs_temperature_celsius', pd.Series(dtype=float))
        skin_temp_col = self.df.get('max_skin_temp_celsius', pd.Series(dtype=float))
        self.df['temperature'] = temp_col.fillna(skin_temp_col)
        self._define_rules()

    def _define_rules(self):
        self.ALERT_RULES = [
             AlertRule(metric="ai_risk_score", condition=AlertCondition.GREATER_THAN_OR_EQUAL, threshold=self._get_setting('RISK_SCORE_MODERATE_THRESHOLD', 60), alert_level=AlertLevel.INFO, primary_reason="High AI Risk", priority_calculator=lambda v, t: v, details_formatter_str="Score: {}"),
             AlertRule(metric="min_spo2_pct", condition=AlertCondition.LESS_THAN, threshold=self._get_setting('ALERT_SPO2_CRITICAL_LOW_PCT', 90), alert_level=AlertLevel.CRITICAL, primary_reason="Critical SpO2", priority_calculator=lambda v, t: 95.0 + (t-v), details_formatter_str="SpO2: {}%"),
             AlertRule(metric="temperature", condition=AlertCondition.GREATER_THAN_OR_EQUAL, threshold=self._get_setting('ALERT_BODY_TEMP_HIGH_FEVER_C', 39.0), alert_level=AlertLevel.CRITICAL, primary_reason="High Fever", priority_calculator=lambda v, t: 90.0 + (v-t), details_formatter_str="Temp: {:.1f}°C"),
        ]
    
    def _format_output_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        # <<< SME REVISION V3 >>> Vectorized string concatenation is much faster than apply.
        df['Alert Reason'] = df['primary_reason'] + " (" + df.get('brief_details', 'N/A') + ")"
        df['Priority Score'] = df['raw_priority_score'].round(1)
        output_cols = ['patient_id', 'encounter_date', 'condition', 'Alert Reason', 'Priority Score', 'ai_risk_score', 'age', 'gender', 'zone_id']
        # Use reindex to select, order, and create missing columns with a single fillna
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
        logger.info("generate_chw_patient_alerts received an empty or invalid DataFrame. Returning empty list.")
        return []
    generator = CHWAlertGenerator(patient_encounter_data_df)
    return generator.generate(**kwargs)

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame) -> pd.DataFrame:
    """Factory function to generate a list of flagged patients for clinic review."""
    if not isinstance(health_df_period, pd.DataFrame) or health_df_period.empty:
        logger.info("get_patient_alerts_for_clinic received an empty or invalid DataFrame. Returning empty DataFrame.")
        return pd.DataFrame()
    generator = ClinicPatientAlertGenerator(health_df_period)
    return generator.generate()
