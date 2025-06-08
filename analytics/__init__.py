# sentinel_project_root/analytics/__init__.py
"""
Initializes the analytics package, making key functions and classes
available at the top level for easier importing.
"""
# From orchestrator.py
from .orchestrator import apply_ai_models
# From risk_prediction.py
from .risk_prediction import calculate_risk_score
# From followup_prioritization.py
from .followup_prioritization import calculate_followup_priority
# From alerting.py
from .alerting import generate_chw_patient_alerts, get_patient_alerts_for_clinic
# From supply_forecasting.py
from .supply_forecasting import generate_simple_supply_forecast, forecast_supply_levels_advanced
# From protocol_executor.py
from .protocol_executor import execute_escalation_protocol

# Define the public API for the analytics package
__all__ = [
    "apply_ai_models",
    "calculate_risk_score",
    "calculate_followup_priority",
    "generate_chw_patient_alerts",
    "get_patient_alerts_for_clinic",
    "generate_simple_supply_forecast",
    "forecast_supply_levels_advanced",
    "execute_escalation_protocol"
]
