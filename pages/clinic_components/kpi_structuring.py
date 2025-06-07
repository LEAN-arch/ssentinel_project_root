# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# Structures key clinic performance and disease-specific KPIs for Sentinel dashboards.

import pandas as pd
import logging
from typing import Dict, Any, List, Optional

# --- Module Imports & Setup ---
try:
    from config import settings
except ImportError:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical("Fatal: Could not import settings. KPI structuring will fail.", exc_info=True)
    # Re-raise to prevent app startup with a critical missing dependency.
    raise

logger = logging.getLogger(__name__)

# --- Helper Function ---

def _format_kpi_value(value: Any, default_str: str = "N/A", precision: int = 1, is_count: bool = False) -> str:
    """Helper to format KPI values robustly for display."""
    if pd.isna(value) or value is None:
        return default_str
    try:
        numeric_value = pd.to_numeric(value)
        return f"{int(numeric_value):,}" if is_count else f"{numeric_value:.{precision}f}"
    except (ValueError, TypeError):
        return str(value) if str(value).strip() else default_str

# --- Main KPI Structuring Class ---

class KPIStructurer:
    """
    A data-driven class to structure raw KPI data into a display-ready format.
    
    This class uses configuration lists to define KPIs, making it easy to add,
    remove, or modify them without changing the core processing logic. It supports
    both simple and nested data sources for KPI values.
    """
    _MAIN_KPI_DEFINITIONS = [
        {"title": "Overall Avg. TAT", "source_key": "overall_avg_test_turnaround_conclusive_days", "target_setting": "TARGET_TEST_TURNAROUND_DAYS", "default_target": 2.0, "units": "days", "icon": "⏱️", "help_template": "Avg. Turnaround Time for conclusive tests. Target: ~{target:.1f} days.", "status_logic": "lower_is_better", "precision": 1},
        {"title": "% Critical Tests TAT Met", "source_key": "perc_critical_tests_tat_met", "target_setting": "TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY", "default_target": 85.0, "units": "%", "icon": "🎯", "help_template": "Critical tests meeting TAT targets. Target: ≥{target:.1f}%.", "status_logic": "higher_is_better", "precision": 1},
        {"title": "Pending Critical Tests", "source_key": "total_pending_critical_tests_patients", "target_setting": "TARGET_PENDING_CRITICAL_TESTS", "default_target": 0, "units": "patients", "icon": "⏳", "help_template": "Patients with pending critical tests. Target: {target}.", "status_logic": "lower_is_better_count", "is_count": True},
        {"title": "Sample Rejection Rate", "source_key": "sample_rejection_rate_perc", "target_setting": "TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY", "default_target": 5.0, "units": "%", "icon": "🧪", "help_template": "Overall rate of rejected lab samples. Target: <{target:.1f}%.", "status_logic": "lower_is_better", "precision": 1},
    ]
    
    _SUPPLY_KPI_DEFINITIONS = [
        {"title": "Key Drug Stockouts", "source_key": "key_drug_stockouts_count", "target_setting": "TARGET_DRUG_STOCKOUTS", "default_target": 0, "units": "items", "icon": "💊", "help_template": "Key drugs with <{days_remaining} days of stock. Target: {target}.", "status_logic": "lower_is_better_count", "is_count": True, "precision": 0},
    ]

    def __init__(self, kpis_summary_data: Optional[Dict[str, Any]]):
        self.summary_data = kpis_summary_data if isinstance(kpis_summary_data, dict) else {}

    def _get_nested_value(self, key_path: str) -> Any:
        """Safely retrieves a value from a nested dictionary using a dot-notated path."""
        keys = key_path.split('.')
        value = self.summary_data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _get_status_lower_is_better(self, value: float, target: float) -> str:
        """Returns status for metrics where a lower value is better (e.g., TAT, rejection rate)."""
        if value > target * 1.5: return "HIGH_CONCERN"
        if value > target: return "MODERATE_CONCERN"
        return "GOOD_PERFORMANCE"

    def _get_status_higher_is_better(self, value: float, target: float) -> str:
        """Returns status for metrics where a higher value is better (e.g., % TAT met)."""
        if value >= target: return "GOOD_PERFORMANCE"
        if value >= target * 0.8: return "ACCEPTABLE"
        return "HIGH_CONCERN"
        
    def _get_status_lower_is_better_count(self, value: int, target: int) -> str:
        """Returns status for count-based metrics where lower is better (e.g., stockouts)."""
        if value == target: return "GOOD_PERFORMANCE"
        if value <= target + 2: return "ACCEPTABLE"
        return "HIGH_CONCERN"

    def _build_kpi(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Universal method to build a single KPI dictionary from a configuration."""
        value = self._get_nested_value(config["source_key"])
        
        # Prioritize KPI-specific target value over global settings for flexibility.
        target = config.get("target_value")
        if target is None:
            target = getattr(settings, config.get("target_setting", ""), config.get("default_target"))

        status = "NO_DATA"
        if pd.notna(value):
            value_num = pd.to_numeric(value, errors='coerce')
            if pd.notna(value_num):
                logic_map = {"lower_is_better": self._get_status_lower_is_better, "higher_is_better": self._get_status_higher_is_better, "lower_is_better_count": self._get_status_lower_is_better_count}
                status_func = logic_map.get(config["status_logic"])
                if status_func: status = status_func(value_num, target)

        units = config.get("units", "")
        formatted_value = _format_kpi_value(value, precision=config.get("precision", 1), is_count=config.get("is_count", False))
        
        # Dynamically inject values into help text template.
        help_text = config["help_template"].format(
            target=target, 
            days_remaining=getattr(settings, 'CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        )
        
        return {
            "title": config["title"],
            "value_str": f"{formatted_value}{units}" if units == "%" and pd.notna(value) else formatted_value,
            "units": units if units != "%" else "",
            "icon": config["icon"],
            "status_level": status,
            "help_text": help_text
        }

    def structure_main_kpis(self) -> List[Dict[str, Any]]:
        """Structures all main clinic performance KPIs using the defined configuration."""
        return [self._build_kpi(conf) for conf in self._MAIN_KPI_DEFINITIONS]

    def structure_disease_and_supply_kpis(self) -> List[Dict[str, Any]]:
        """Dynamically generates and structures disease-specific and supply KPIs."""
        kpi_definitions = []
        
        # 1. Dynamically generate KPI definitions for each key test.
        key_tests = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
        for test_name, config in key_tests.items():
            if not isinstance(config, dict): continue
            
            kpi_definitions.append({
                "title": f"{config.get('display_name', test_name)} Positivity",
                "source_key": f"test_summary_details.{test_name}.positive_rate_perc",
                "target_value": float(config.get("target_max_positivity_pct", 10.0)),
                "units": "%", "icon": config.get("icon", "🔬"),
                "help_template": "Positivity rate. Target: <{target:.1f}%.",
                "status_logic": "lower_is_better", "precision": 1
            })
            
        # 2. Add static definitions for supply chain KPIs.
        kpi_definitions.extend(self._SUPPLY_KPI_DEFINITIONS)
        
        # 3. Build all KPIs using the universal builder.
        return [self._build_kpi(conf) for conf in kpi_definitions]


# --- Public Factory Functions ---

def structure_main_clinic_kpis(clinic_service_kpis_summary_data: Optional[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Public function to structure main clinic KPIs."""
    structurer = KPIStructurer(clinic_service_kpis_summary_data)
    return structurer.structure_main_kpis()

def structure_disease_specific_clinic_kpis(clinic_service_kpis_summary_data: Optional[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Public function to structure disease and supply KPIs."""
    structurer = KPIStructurer(clinic_service_kpis_summary_data)
    return structurer.structure_disease_and_supply_kpis()
