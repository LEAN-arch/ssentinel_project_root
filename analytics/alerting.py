# sentinel_project_root/analytics/alerting.py
# SME PLATINUM STANDARD - VECTORIZED ALERTING ENGINE

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import settings
from .protocol_executor import execute_protocols_for_alerts

logger = logging.getLogger(__name__)

# --- Enums and Dataclasses for Robustness and Clarity ---
class AlertLevel(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

class AlertCondition(Enum):
    LESS_THAN = auto()
    GREATER_THAN_EQUAL = auto()
    BETWEEN = auto()
    EQUALS = auto()

@dataclass(frozen=True)
class AlertRule:
    metric: str
    condition: AlertCondition
    threshold: Union[float, int, tuple]
    level: AlertLevel
    reason: str
    details_template: str
    priority_fn: Callable[[pd.Series, Any], pd.Series]
    protocol_id: Optional[str] = None

# --- Base Alert Generator Class ---
class AlertGenerator:
    def __init__(self, df: pd.DataFrame, rules: List[AlertRule]):
        self.df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        self.rules = rules

    def _evaluate_rules(self) -> pd.DataFrame:
        if self.df.empty: return pd.DataFrame()
        
        all_alerts = []
        for rule in self.rules:
            if rule.metric not in self.df.columns: continue
            series = self.df[rule.metric].dropna()
            if series.empty: continue

            if rule.condition == AlertCondition.LESS_THAN: mask = series < rule.threshold
            elif rule.condition == AlertCondition.GREATER_THAN_EQUAL: mask = series >= rule.threshold
            elif rule.condition == AlertCondition.BETWEEN: mask = series.between(rule.threshold[0], rule.threshold[1], inclusive='both')
            elif rule.condition == AlertCondition.EQUALS: mask = series == rule.threshold
            else: continue
            
            if mask.any():
                triggered_df = self.df.loc[mask].copy()
                metric_values = triggered_df[rule.metric]
                triggered_df['alert_level'] = rule.level.value
                triggered_df['reason'] = rule.reason
                triggered_df['protocol_id'] = rule.protocol_id
                triggered_df['priority'] = rule.priority_fn(metric_values, rule.threshold)
                triggered_df['details'] = triggered_df.apply(lambda row: rule.details_template.format(row[rule.metric]), axis=1)
                all_alerts.append(triggered_df)

        return pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    @staticmethod
    def _deduplicate_alerts(alerts_df: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
        if alerts_df.empty: return pd.DataFrame()
        return alerts_df.sort_values('priority', ascending=False).drop_duplicates(subset=on_cols, keep='first')

# --- Specific Alert Implementations ---
def generate_chw_alerts(patient_df: pd.DataFrame, max_alerts: int = 10) -> List[Dict[str, Any]]:
    if not isinstance(patient_df, pd.DataFrame) or patient_df.empty:
        return []

    spo2_crit = settings.ANALYTICS.spo2_critical_threshold_pct
    spo2_warn = settings.ANALYTICS.spo2_warning_threshold_pct
    temp_crit = settings.ANALYTICS.temp_high_fever_threshold_c
    
    CHW_RULES = [
        AlertRule(metric="min_spo2_pct", condition=AlertCondition.LESS_THAN, threshold=spo2_crit, level=AlertLevel.CRITICAL, reason="Critical Low SpO2", details_template="SpO2 at {:.0f}%", priority_fn=lambda v, t: 100 + (t - v), protocol_id="PATIENT_CRITICAL_SPO2_LOW"),
        AlertRule(metric="min_spo2_pct", condition=AlertCondition.BETWEEN, threshold=(spo2_crit, spo2_warn), level=AlertLevel.WARNING, reason="Low SpO2", details_template="SpO2 at {:.0f}%", priority_fn=lambda v, t: 75 + (t[1] - v)),
        AlertRule(metric="temperature", condition=AlertCondition.GREATER_THAN_EQUAL, threshold=temp_crit, level=AlertLevel.CRITICAL, reason="High Fever", details_template="Temp at {:.1f}°C", priority_fn=lambda v, t: 95 + (v - t) * 2),
        AlertRule(metric="fall_detected_today", condition=AlertCondition.EQUALS, threshold=1, level=AlertLevel.CRITICAL, reason="Fall Detected", details_template="Fall detected today", priority_fn=lambda v, t: 92.0, protocol_id="PATIENT_FALL_DETECTED"),
    ]
    
    df = patient_df.copy()
    temp_col = df.get('vital_signs_temperature_celsius', pd.Series(dtype=float))
    skin_temp_col = df.get('max_skin_temp_celsius', pd.Series(dtype=float))
    df['temperature'] = temp_col.fillna(skin_temp_col)
    
    generator = AlertGenerator(df, CHW_RULES)
    all_alerts = generator._evaluate_rules()
    unique_alerts = generator._deduplicate_alerts(all_alerts, on_cols=['patient_id', 'reason'])
    
    if unique_alerts.empty: return []
    execute_protocols_for_alerts(unique_alerts)

    final_alerts = unique_alerts.sort_values('priority', ascending=False)
    final_alerts['context'] = "Dx: " + final_alerts.get('diagnosis', 'N/A').astype(str) + " | Zone: " + final_alerts.get('zone_id', 'N/A').astype(str)
    output_cols = ['patient_id', 'alert_level', 'reason', 'details', 'context', 'priority']
    return final_alerts.reindex(columns=output_cols).head(max_alerts).to_dict('records')

def generate_clinic_patient_alerts(health_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(health_df, pd.DataFrame) or health_df.empty:
        return pd.DataFrame()

    risk_mod = settings.ANALYTICS.risk_score_moderate_threshold
    spo2_crit = settings.ANALYTICS.spo2_critical_threshold_pct
    temp_crit = settings.ANALYTICS.temp_high_fever_threshold_c
    
    CLINIC_RULES = [
        AlertRule(metric="ai_risk_score", condition=AlertCondition.GREATER_THAN_EQUAL, threshold=risk_mod, level=AlertLevel.INFO, reason="High AI Risk", details_template="Score: {:.0f}", priority_fn=lambda v, t: v),
        AlertRule(metric="min_spo2_pct", condition=AlertCondition.LESS_THAN, threshold=spo2_crit, level=AlertLevel.CRITICAL, reason="Critical SpO2", details_template="SpO2: {:.0f}%", priority_fn=lambda v, t: 100 + (t - v)),
        AlertRule(metric="temperature", condition=AlertCondition.GREATER_THAN_EQUAL, threshold=temp_crit, level=AlertLevel.CRITICAL, reason="High Fever", details_template="Temp: {:.1f}°C", priority_fn=lambda v, t: 95 + (v - t) * 2),
    ]

    df = health_df.copy()
    if 'patient_id' in df.columns and 'encounter_date' in df.columns:
        df = df.sort_values('encounter_date').drop_duplicates('patient_id', keep='last')
        
    temp_col = df.get('vital_signs_temperature_celsius', pd.Series(dtype=float))
    skin_temp_col = df.get('max_skin_temp_celsius', pd.Series(dtype=float))
    df['temperature'] = temp_col.fillna(skin_temp_col)
    
    generator = AlertGenerator(df, CLINIC_RULES)
    all_alerts = generator._evaluate_rules()
    unique_alerts = generator._deduplicate_alerts(all_alerts, on_cols=['patient_id', 'reason'])
    
    if unique_alerts.empty: return pd.DataFrame()

    unique_alerts['Alert Summary'] = unique_alerts['reason'] + " (" + unique_alerts.get('details', 'N/A') + ")"
    unique_alerts['Priority'] = unique_alerts['priority'].round(0).astype(int)
    
    output_cols = ['patient_id', 'encounter_date', 'diagnosis', 'Alert Summary', 'Priority', 'ai_risk_score', 'age', 'gender', 'zone_id']
    return unique_alerts.reindex(columns=output_cols).fillna('N/A').sort_values('Priority', ascending=False)
