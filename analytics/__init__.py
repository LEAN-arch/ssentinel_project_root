# sentinel_project_root/analytics/__init__.py
"""
Initializes the analytics package, defining its public API.

This package contains all business logic for AI/ML models, alerting,
forecasting, and complex KPI calculations.
"""

# --- Core AI/ML Model Orchestration ---
from .orchestrator import apply_ai_models

# --- Specific Model Logic (exposed for potential granular use) ---
from .risk_prediction import calculate_risk_score
from .followup_prioritization import calculate_followup_priority

# --- Alerting Engine ---
from .alerting import (
    generate_chw_alerts,
    generate_clinic_patient_alerts
)

# --- Supply Chain Forecasting ---
from .supply_forecasting import (
    generate_linear_forecast,
    generate_prophet_forecast
)

# --- Protocol Execution ---
from .protocol_executor import execute_protocol_for_event

# --- High-Level Dashboard Analytics ---
from .kpi_analyzer import generate_kpi_analysis_table


# --- Define the canonical public API for the package ---
__all__ = [
    # Orchestration
    "apply_ai_models",

    # Models
    "calculate_risk_score",
    "calculate_followup_priority",

    # Alerting
    "generate_chw_alerts",
    "generate_clinic_patient_alerts",

    # Forecasting
    "generate_linear_forecast",
    "generate_prophet_forecast",

    # Protocols
    "execute_protocol_for_event",

    # KPI Analysis
    "generate_kpi_analysis_table",
]
