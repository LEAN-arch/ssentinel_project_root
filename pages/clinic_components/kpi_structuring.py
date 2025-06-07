# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# Structures key clinic performance and disease-specific KPIs for Sentinel dashboards.

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from config import settings
except ImportError:
    # Fallback for standalone execution
    class FallbackSettings:
        TARGET_TEST_TURNAROUND_DAYS = 2.0
    settings = FallbackSettings()

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets an attribute from the settings object."""
    return getattr(settings, attr_name, default_value)

def _format_kpi_value(value: Any, precision: int = 1, is_count: bool = False) -> str:
    """Helper to format KPI values robustly, returning 'N/A' for invalid data."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    try:
        numeric_value = float(value)
        if is_count:
            return f"{int(numeric_value):,}"
        return f"{numeric_value:.{precision}f}"
    except (ValueError, TypeError):
        # Return the original string if it cannot be converted to a number
        return str(value)

def structure_main_clinic_kpis(kpis_summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Structures the main, high-level clinic KPIs for display using a data-driven approach.
    
    Args:
        kpis_summary: A dictionary of calculated KPI values from an aggregation step.
        
    Returns:
        A list of dictionaries, where each dictionary represents a KPI card for the UI.
    """
    if not isinstance(kpis_summary, dict):
        return []

    # A data-driven way to define KPIs. Easier to add, remove, or modify.
    kpi_definitions = [
        {"key": "total_encounters", "title": "Total Encounters", "icon": "👥", "is_count": True, "help": "Total patient encounters recorded in the selected period."},
        {"key": "unique_patients", "title": "Unique Patients", "icon": "🧍", "is_count": True, "help": "Number of distinct patients seen in the selected period."},
        {"key": "avg_test_turnaround_days", "title": "Avg. Lab TAT", "icon": "⏱️", "units": " days", "help": f"Average Test Turnaround Time. Target: < {_get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)} days."},
        {"key": "sample_rejection_rate_pct", "title": "Rejection Rate", "icon": "🗑️", "units": "%", "help": "Percentage of lab samples rejected during the period."}
    ]

    structured_kpis = []
    for kpi in kpi_definitions:
        value = kpis_summary.get(kpi["key"])
        structured_kpis.append({
            "title": kpi["title"],
            "value_str": _format_kpi_value(value, is_count=kpi.get("is_count", False)),
            "icon": kpi["icon"],
            "units": kpi.get("units", ""),
            "help_text": kpi.get("help", "")
        })
        
    return structured_kpis

def structure_disease_specific_clinic_kpis(kpis_summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Structures disease-specific and other key indicator KPIs.
    
    Args:
        kpis_summary: A dictionary of calculated KPI values from an aggregation step.
        
    Returns:
        A list of dictionaries, where each dictionary represents a KPI card for the UI.
    """
    if not isinstance(kpis_summary, dict):
        return []

    kpi_definitions = [
        {"key": "malaria_positivity_rate_pct", "title": "Malaria Positivity", "icon": "🦟", "units": "%", "help": "Percentage of malaria RDTs that were positive."},
        {"key": "tb_positivity_rate_pct", "title": "TB Positivity", "icon": "🫁", "units": "%", "help": "Percentage of TB tests (e.g., GeneXpert) that were positive."},
        {"key": "supply_items_critical_count", "title": "Critical Supplies", "icon": "📦", "is_count": True, "units": " items", "help": "Number of key supply items with critically low stock levels."},
        {"key": "patients_flagged_for_review_count", "title": "Flagged Patients", "icon": "🚩", "is_count": True, "help": "Number of patients flagged for clinical review due to high risk or other alerts."}
    ]

    structured_kpis = []
    for kpi in kpi_definitions:
        value = kpis_summary.get(kpi["key"])
        structured_kpis.append({
            "title": kpi["title"],
            "value_str": _format_kpi_value(value, is_count=kpi.get("is_count", False)),
            "icon": kpi["icon"],
            "units": kpi.get("units", ""),
            "help_text": kpi.get("help", "")
        })
        
    return structured_kpis
