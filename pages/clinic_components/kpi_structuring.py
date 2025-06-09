# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# SME-EVALUATED AND REVISED VERSION
# This version fixes critical bugs in data retrieval and value formatting,
# and refactors status logic for clarity and efficiency.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Literal

# --- Module Imports & Setup ---
try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in kpi_structuring.py: {e}. Using fallback settings.", exc_info=True)
    
    class FallbackSettings:
        TARGET_TEST_TURNAROUND_DAYS = 2.0
        TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
        TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0
        KEY_TEST_TYPES_FOR_ANALYSIS = {}
        TARGET_MALARIA_POSITIVITY_RATE = 10.0
        CRITICAL_SUPPLY_DAYS_REMAINING = 7
        TARGET_PENDING_CRITICAL_TESTS = 0
        TARGET_DRUG_STOCKOUTS = 0
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

# Define literal types for status logic and levels for better static analysis.
StatusLogic = Literal["lower_is_better", "higher_is_better", "lower_is_better_count"]
StatusLevel = Literal["GOOD_PERFORMANCE", "ACCEPTABLE", "MODERATE_CONCERN", "HIGH_CONCERN", "NO_DATA"]


def _format_kpi_value(value: Any, default_str: str = "N/A", precision: int = 1, is_count: bool = False) -> str:
    """Helper to format KPI values robustly for display."""
    if pd.isna(value) or value is None:
        return default_str
    try:
        numeric_value = float(value)
        return f"{int(numeric_value):,}" if is_count else f"{numeric_value:.{precision}f}"
    except (ValueError, TypeError):
        return str(value) if str(value).strip() else default_str


class _KPIStructurer:
    """A data-driven class to structure raw KPI data into a display-ready format."""
    _MAIN_KPI_DEFINITIONS = [
        {"title": "Overall Avg. TAT", "source_key": "overall_avg_test_turnaround_conclusive_days", "target_setting": "TARGET_TEST_TURNAROUND_DAYS", "default_target": 2.0, "units": "days", "icon": "⏱️", "help_template": "Avg. Turnaround Time for conclusive tests. Target: ~{target:.1f} days.", "status_logic": "lower_is_better", "precision": 1},
        {"title": "% Critical Tests TAT Met", "source_key": "perc_critical_tests_tat_met", "target_setting": "TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY", "default_target": 85.0, "units": "%", "icon": "🎯", "help_template": "Critical tests meeting TAT targets. Target: ≥{target:.1f}%.", "status_logic": "higher_is_better", "precision": 1},
        {"title": "Pending Critical Tests", "source_key": "total_pending_critical_tests_patients", "target_setting": "TARGET_PENDING_CRITICAL_TESTS", "default_target": 0, "units": "patients", "icon": "⏳", "help_template": "Patients with pending critical tests. Target: {target}.", "status_logic": "lower_is_better_count", "is_count": True},
        {"title": "Sample Rejection Rate", "source_key": "sample_rejection_rate_perc", "target_setting": "TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY", "default_target": 5.0, "units": "%", "icon": "🧪", "help_template": "Overall rate of rejected lab samples. Target: <{target:.1f}%.", "status_logic": "lower_is_better", "precision": 1},
    ]

    def __init__(self, kpis_summary_data: Optional[Dict[str, Any]]):
        self.summary_data = kpis_summary_data if isinstance(kpis_summary_data, dict) else {}
        # EFFICIENCY: Build dynamic definitions once during initialization.
        self._disease_and_supply_kpi_definitions = self._build_dynamic_kpi_definitions()

    def _build_dynamic_kpi_definitions(self) -> List[Dict[str, Any]]:
        """Generates KPI definitions from settings configuration. Called only once."""
        kpi_defs = []
        key_tests = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
        for test_name, config in key_tests.items():
            if isinstance(config, dict):
                kpi_defs.append({
                    "title": f"{config.get('display_name', test_name)} Positivity", "source_key": f"test_summary_details.{test_name}.positive_rate_perc",
                    "target_value": float(config.get("target_max_positivity_pct", 10.0)), "units": "%", "icon": config.get("icon", "🔬"),
                    "help_template": "Positivity rate. Target: <{target:.1f}%.", "status_logic": "lower_is_better", "precision": 1
                })
        
        kpi_defs.append({
            "title": "Key Drug Stockouts", "source_key": "key_drug_stockouts_count",
            "target_setting": "TARGET_DRUG_STOCKOUTS", "default_target": 0, "units": "items", "icon": "💊",
            "help_template": "Key drugs with <{days_remaining} days of stock. Target: {target}.",
            "status_logic": "lower_is_better_count", "is_count": True, "precision": 0
        })
        return kpi_defs
    
    # --- REFACTORED STATUS LOGIC ---
    def _get_status_level(self, value: Optional[float], target: float, logic: StatusLogic) -> StatusLevel:
        """Determines the KPI status level based on its value, target, and performance logic."""
        if pd.isna(value):
            return "NO_DATA"

        if logic == "lower_is_better":
            if value > target * 1.5: return "HIGH_CONCERN"
            if value > target: return "MODERATE_CONCERN"
            return "GOOD_PERFORMANCE"
        
        if logic == "higher_is_better":
            if value >= target: return "GOOD_PERFORMANCE"
            if value >= target * 0.8: return "ACCEPTABLE"
            return "HIGH_CONCERN"
            
        if logic == "lower_is_better_count":
            value_int = int(value)
            if value_int == target: return "GOOD_PERFORMANCE"
            if value_int <= target + 2: return "ACCEPTABLE"
            return "HIGH_CONCERN"
        
        return "NO_DATA" # Fallback for unknown logic

    # --- BUG FIX: ROBUST NESTED VALUE RETRIEVAL ---
    def _get_nested_value(self, key_path: str) -> Any:
        """Safely retrieves a value from a nested dictionary using a dot-separated path."""
        keys = key_path.split('.')
        value = self.summary_data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                # This handles cases where a mid-path key exists but is None.
                return None
        return value

    def _build_kpi(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic method to build a single KPI dictionary from a configuration."""
        value = self._get_nested_value(config["source_key"])
        target = config.get("target_value", getattr(settings, config.get("target_setting", ""), config.get("default_target")))
        value_num = pd.to_numeric(value, errors='coerce')
        
        status = self._get_status_level(value_num, target, config["status_logic"])
        
        # --- BUG FIX: SIMPLIFIED AND CORRECTED VALUE/UNIT FORMATTING ---
        formatted_value = _format_kpi_value(value, precision=config.get("precision", 1), is_count=config.get("is_count", False))
        units = config.get("units", "")
        
        # Append unit symbol only if it's a percentage and value is valid.
        display_value = f"{formatted_value}{units}" if units == "%" and pd.notna(value) else formatted_value
        # The 'units' field should not contain the symbol itself, just the name.
        display_units = units if units != "%" else ""
        
        help_text = config["help_template"].format(
            target=target,
            days_remaining=getattr(settings, 'CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        )
        
        return {
            "title": config["title"],
            "value_str": display_value,
            "units": display_units,
            "icon": config["icon"],
            "status_level": status,
            "help_text": help_text
        }

    def structure_main_kpis(self) -> List[Dict[str, Any]]:
        """Structures all main clinic KPIs based on the predefined configuration."""
        return [self._build_kpi(conf) for conf in self._MAIN_KPI_DEFINITIONS]

    def structure_disease_and_supply_kpis(self) -> List[Dict[str, Any]]:
        """Structures all disease and supply KPIs based on the dynamically built configuration."""
        return [self._build_kpi(conf) for conf in self._disease_and_supply_kpi_definitions]


def structure_main_clinic_kpis(kpis_summary: Optional[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Public function to structure main clinic operational KPIs.

    Args:
        kpis_summary: A dictionary containing the aggregated KPI data.
        **kwargs: Catches any unused keyword arguments.

    Returns:
        A list of dictionaries, each representing a structured KPI ready for display.
    """
    structurer = _KPIStructurer(kpis_summary)
    return structurer.structure_main_kpis()

def structure_disease_specific_clinic_kpis(kpis_summary: Optional[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Public function to structure disease-specific and supply-chain KPIs.

    Args:
        kpis_summary: A dictionary containing the aggregated KPI data.
        **kwargs: Catches any unused keyword arguments.

    Returns:
        A list of dictionaries, each representing a structured KPI ready for display.
    """
    structurer = _KPIStructurer(kpis_summary)
    return structurer.structure_disease_and_supply_kpis()
