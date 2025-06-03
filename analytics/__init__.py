# sentinel_project_root/analytics/__init__.py
# This file makes the 'analytics' directory a Python package.

from .alerting import (
    generate_chw_patient_alerts, # Renamed from generate_chw_patient_alerts_from_data
    get_patient_alerts_for_clinic # Moved from core_data_processing
)
from .followup_prioritization import FollowUpPrioritizer
from .orchestrator import apply_ai_models
from .protocol_executor import (
    execute_escalation_protocol,
    get_protocol_for_event,
    format_escalation_message
)
from .risk_prediction import RiskPredictionModel
from .supply_forecasting import (
    SupplyForecastingModel, # AI Simulated Model
    generate_simple_supply_forecast # Simple linear model (moved from core_data_processing as get_supply_forecast_data)
)


__all__ = [
    "generate_chw_patient_alerts",
    "get_patient_alerts_for_clinic",
    "FollowUpPrioritizer",
    "apply_ai_models",
    "execute_escalation_protocol",
    "get_protocol_for_event",
    "format_escalation_message",
    "RiskPredictionModel",
    "SupplyForecastingModel",
    "generate_simple_supply_forecast"
]
