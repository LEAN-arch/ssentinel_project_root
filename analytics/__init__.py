# sentinel_project_root/analytics/__init__.py
# SME PLATINUM STANDARD (V3 - FINAL INTEGRATED VERSION)
# This definitive version correctly exposes all high-level analytics functions,
# including the newly created `generate_kpi_analysis_table` for the refactored
# clinic dashboard, ensuring seamless system integration.

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

# From protocol_executor.py
from .protocol_executor import execute_escalation_protocols_batch

# <<< SME INTEGRATION >>> Import the new high-level KPI function from its dedicated module.
from .clinic_kpis import generate_kpi_analysis_table


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

    # Protocol execution
    "execute_escalation_protocols_batch",

    # <<< SME INTEGRATION >>> Add the new function to the public API.
    # High-level Dashboard Analytics
    "generate_kpi_analysis_table",
]
