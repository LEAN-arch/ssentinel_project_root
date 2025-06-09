# sentinel_project_root/analytics/__init__.py
# SME PLATINUM STANDARD (V2 - ARCHITECTURAL ALIGNMENT)
# This version updates the public API to reflect the batch-processing
# architecture introduced in the submodules, promoting a more performant
# and consistent usage pattern.

"""
Initializes the analytics package, making key functions and classes
available at the top level for easier importing.

This __init__.py defines the public API for the package.
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

# <<< SME REVISION >>> Expose the batch function to align with modern architecture.
# From protocol_executor.py
from .protocol_executor import execute_escalation_protocols_batch
# NOTE: The single-item `execute_escalation_protocol` is no longer part of the
# primary public API to encourage more performant, batch-oriented workflows.

# --- Define the public API for the analytics package ---
# This list controls what is imported when a user does `from analytics import *`
# and is considered the canonical list of public-facing components.
__all__ = [
    # Core analytics functions
    "apply_ai_models",
    "calculate_risk_score",
    "calculate_followup_priority",

    # Alerting functions
    "generate_chw_patient_alerts",
    "get_patient_alerts_for_clinic",

    # Forecasting functions
    "generate_simple_supply_forecast",
    "forecast_supply_levels_advanced",

    # <<< SME REVISION >>> Advertise the performant batch function.
    # Protocol execution
    "execute_escalation_protocols_batch",
]
