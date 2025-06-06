# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# Structures key clinic performance and disease-specific KPIs for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in kpi_structuring.py: {e}. Ensure config.py is accessible.")
    class FallbackSettings:
        TARGET_TEST_TURNAROUND_DAYS = 2.0
        TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
        TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0
        KEY_TEST_TYPES_FOR_ANALYSIS: Dict[str, Dict[str, Any]] = {}
        CRITICAL_SUPPLY_DAYS_REMAINING = 7
    settings = FallbackSettings()
    logger_init.warning("kpi_structuring.py: Using fallback settings due to import error.")

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)

def _format_kpi_value(
    value: Any, 
    default_str: str = "N/A", 
    precision: Optional[int] = 1, 
    is_count: bool = False,
    is_percentage: bool = False
) -> str:
    """Helper to format KPI values robustly."""
    if pd.isna(value) or value is None:
        return default_str
    try:
        numeric_value = pd.to_numeric(value)
        if is_count:
            return f"{int(numeric_value):,}"
        if is_percentage:
            return f"{numeric_value:.{precision if precision is not None else 1}f}"
        if precision is not None:
            return f"{numeric_value:.{precision}f}"
        return str(numeric_value)
    except (ValueError, TypeError):
        return str(value) if str(value).strip() else default_str

# --- Refactored KPI Structuring with a Data-Driven Approach ---

def _create_kpi_dict(
    data_dict: Dict[str, Any],
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Creates a single formatted KPI dictionary from a configuration object.
    
    Args:
        data_dict: The source dictionary containing all raw KPI values.
        config: A dictionary defining the KPI to be created.
    
    Returns:
        A dictionary formatted for `render_kpi_card`, or None if value is missing.
    """
    value = data_dict.get(config["data_key"])
    
    # Don't render a card if the source data is completely missing
    if value is None:
        return None

    status_config = config.get("status_logic", {})
    target = float(_get_setting(status_config.get("target_setting", ""), status_config.get("default_target", 0)))
    
    status_level = "NO_DATA"
    numeric_value = pd.to_numeric(value, errors='coerce')
    if pd.notna(numeric_value):
        mode = status_config.get("mode", "lower_is_better")
        thresholds = status_config.get("thresholds", {})
        
        if mode == 'lower_is_better':
            if numeric_value <= target: status_level = "GOOD_PERFORMANCE"
            elif numeric_value <= target + thresholds.get("moderate_concern", 1.0): status_level = "MODERATE_CONCERN"
            else: status_level = "HIGH_CONCERN"
        elif mode == 'higher_is_better':
            if numeric_value >= target: status_level = "GOOD_PERFORMANCE"
            elif numeric_value >= target * thresholds.get("moderate_concern_factor", 0.8): status_level = "MODERATE_CONCERN"
            else: status_level = "HIGH_CONCERN"
        elif mode == 'zero_is_good':
            if numeric_value == 0: status_level = "GOOD_PERFORMANCE"
            elif numeric_value <= thresholds.get("moderate_concern", 2): status_level = "MODERATE_CONCERN"
            else: status_level = "HIGH_CONCERN"
    
    return {
        "title": config["title"],
        "value_str": _format_kpi_value(value, **config.get("formatting", {})),
        "units": config["units"],
        "icon": config["icon"],
        "status_level": status_level,
        "help_text": config.get("help_text_template", "").format(target=target)
    }

# Declarative configuration for main KPIs
MAIN_KPI_CONFIG = [
    {
        "data_key": "overall_avg_test_turnaround_conclusive_days",
        "title": "Overall Avg. TAT", "units": "days", "icon": "⏱️",
        "formatting": {"precision": 1},
        "status_logic": {
            "mode": "lower_is_better", "target_setting": "TARGET_TEST_TURNAROUND_DAYS", "default_target": 2.0,
            "thresholds": {"moderate_concern": 1.0}
        },
        "help_text_template": "Average Turnaround Time for conclusive tests. Target: ≤{target} days."
    },
    {
        "data_key": "perc_critical_tests_tat_met",
        "title": "% Critical TAT Met", "units": "%", "icon": "🎯",
        "formatting": {"is_percentage": True},
        "status_logic": {
            "mode": "higher_is_better", "target_setting": "TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY", "default_target": 85.0,
            "thresholds": {"moderate_concern_factor": 0.8}
        },
        "help_text_template": "Percentage of critical tests meeting TAT targets. Target: ≥{target:.0f}%."
    },
    {
        "data_key": "total_pending_critical_tests_patients",
        "title": "Pending Critical Tests", "units": "patients", "icon": "⏳",
        "formatting": {"is_count": True},
        "status_logic": {
            "mode": "zero_is_good", "default_target": 0,
            "thresholds": {"moderate_concern": 3}
        },
        "help_text_template": "Patients with critical tests pending. Target: {target}."
    },
    {
        "data_key": "sample_rejection_rate_perc",
        "title": "Sample Rejection Rate", "units": "%", "icon": "🧪",
        "formatting": {"is_percentage": True},
        "status_logic": {
            "mode": "lower_is_better", "target_setting": "TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY", "default_target": 5.0,
            "thresholds": {"moderate_concern": 2.0}
        },
        "help_text_template": "Overall rate of lab samples rejected. Target: <{target:.1f}%."
    }
]

def structure_main_clinic_kpis(
    kpi_data: Optional[Dict[str, Any]],
    period_str: str 
) -> List[Dict[str, Any]]:
    """Structures main clinic KPIs using a data-driven configuration."""
    if not isinstance(kpi_data, dict): return []
    
    structured_kpis = [_create_kpi_dict(kpi_data, config) for config in MAIN_KPI_CONFIG]
    return [kpi for kpi in structured_kpis if kpi is not None]

def structure_disease_specific_clinic_kpis(
    kpi_data: Optional[Dict[str, Any]],
    period_str: str
) -> List[Dict[str, Any]]:
    """Structures disease-specific KPIs and supply counts."""
    if not isinstance(kpi_data, dict): return []
    
    structured_kpis = []
    test_details = kpi_data.get("test_summary_details", {})
    key_tests_config = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
    
    if isinstance(test_details, dict):
        for internal_name, config in key_tests_config.items():
            if not (isinstance(config, dict) and config.get("critical")): continue
            
            stats = test_details.get(internal_name, {})
            kpi_config = {
                "data_key": "positive_rate_perc",
                "title": f"{config.get('disease_label_short', 'Test')} Positivity", "units": "%", "icon": config.get("icon", "🔬"),
                "formatting": {"is_percentage": True},
                "status_logic": {
                    "mode": "lower_is_better", "default_target": config.get("target_max_positivity_pct", 15.0),
                    "thresholds": {"moderate_concern": 5.0}
                },
                "help_text_template": f"Positivity for {config.get('display_name', internal_name)}. Target: <{{target:.1f}}%."
            }
            kpi_dict = _create_kpi_dict(stats, kpi_config)
            if kpi_dict:
                structured_kpis.append(kpi_dict)

    stockout_config = {
        "data_key": "key_drug_stockouts_count",
        "title": "Key Drug Stockouts", "units": "items", "icon": "💊",
        "formatting": {"is_count": True, "default_str": "0"},
        "status_logic": {
            "mode": "zero_is_good", "default_target": 0, "thresholds": {"moderate_concern": 2}
        },
        "help_text_template": f"Key drugs with < {_get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)} days of stock. Target: {{target}}."
    }
    stockout_kpi = _create_kpi_dict(kpi_data, stockout_config)
    if stockout_kpi:
        structured_kpis.append(stockout_kpi)
        
    return structured_kpis
