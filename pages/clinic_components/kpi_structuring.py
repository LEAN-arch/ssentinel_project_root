# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# SME PLATINUM STANDARD (V2 - MODEL-DRIVEN REFACTORING)
# This version refactors the configuration into Pydantic models and uses Enums
# for controlled vocabularies, making the entire system more robust, type-safe,
# and self-documenting.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field # <<< SME REVISION V2
from enum import Enum # <<< SME REVISION V2

# --- Module Imports & Setup ---
try:
    from config import settings
except ImportError:
    # ... (FallbackSettings remains the same)
    class FallbackSettings:
        TARGET_TEST_TURNAROUND_DAYS = 2.0
        TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
        TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0
        KEY_TEST_TYPES_FOR_ANALYSIS = {}
        TARGET_PENDING_CRITICAL_TESTS = 0
        TARGET_DRUG_STOCKOUTS = 0
        CRITICAL_SUPPLY_DAYS_REMAINING = 7
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

# <<< SME REVISION V2 >>> Use Enums for controlled, self-documenting vocabularies.
class StatusLevel(str, Enum):
    GOOD_PERFORMANCE = "GOOD_PERFORMANCE"
    ACCEPTABLE = "ACCEPTABLE"
    MODERATE_CONCERN = "MODERATE_CONCERN"
    HIGH_CONCERN = "HIGH_CONCERN"
    NO_DATA = "NO_DATA"

class KpiLogic(Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER_COUNT = "lower_is_better_count"
    IS_ZERO_TARGET = "is_zero_target"

# <<< SME REVISION V2 >>> Use a Pydantic model for type-safe, validated configuration.
class KpiConfig(BaseModel):
    """Defines the schema for a single Key Performance Indicator."""
    title: str
    source_key: str
    status_logic: KpiLogic
    icon: str
    help_template: str
    units: str = ""
    precision: int = 1
    is_count: bool = False
    target_setting: Optional[str] = None
    target_value: Optional[float] = None
    default_target: float

    def get_target(self) -> float:
        """Resolves the target value from settings or a default."""
        if self.target_value is not None:
            return self.target_value
        return getattr(settings, self.target_setting, self.default_target)

# --- Helper Functions ---
def _format_kpi_value(value: Any, precision: int, is_count: bool) -> str:
    # ... (function is good, just simplified the signature)
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"{int(value):,}" if is_count else f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)

# --- Core Structuring Class ---
class _KPIStructurer:
    """A model-driven engine for structuring raw KPI data into a display-ready format."""
    
    # <<< SME REVISION V2 >>> Configuration is now a list of Pydantic models.
    _MAIN_KPI_DEFINITIONS: List[KpiConfig] = [
        KpiConfig(title="Overall Avg. TAT", source_key="overall_avg_test_turnaround_conclusive_days", target_setting="TARGET_TEST_TURNAROUND_DAYS", default_target=2.0, units="days", icon="⏱️", help_template="Avg. Turnaround Time. Target: ~{target:.1f} days.", status_logic=KpiLogic.LOWER_IS_BETTER),
        KpiConfig(title="% Critical Tests TAT Met", source_key="perc_critical_tests_tat_met", target_setting="TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY", default_target=85.0, units="%", icon="🎯", help_template="Critical tests meeting TAT. Target: ≥{target:.1f}%.", status_logic=KpiLogic.HIGHER_IS_BETTER),
        KpiConfig(title="Pending Critical Tests", source_key="total_pending_critical_tests_patients", target_setting="TARGET_PENDING_CRITICAL_TESTS", default_target=0, units="patients", icon="⏳", is_count=True, help_template="Patients with pending critical tests. Target: {target}.", status_logic=KpiLogic.LOWER_IS_BETTER_COUNT),
        KpiConfig(title="Sample Rejection Rate", source_key="sample_rejection_rate_perc", target_setting="TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY", default_target=5.0, units="%", icon="🧪", help_template="Overall sample rejection rate. Target: <{target:.1f}%.", status_logic=KpiLogic.LOWER_IS_BETTER),
    ]

    def __init__(self, kpis_summary_data: Optional[Dict[str, Any]]):
        self.summary_data = kpis_summary_data or {}
        self._disease_and_supply_kpi_definitions = self._build_dynamic_kpi_definitions()

    def _build_dynamic_kpi_definitions(self) -> List[KpiConfig]:
        """Generates KPI configurations from settings, validating them as Pydantic models."""
        kpi_defs = []
        key_tests = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
        for test_name, config in key_tests.items():
            if isinstance(config, dict):
                kpi_defs.append(KpiConfig(
                    title=f"{config.get('display_name', test_name)} Positivity",
                    source_key=f"test_summary_details.{test_name}.positive_rate_perc",
                    target_value=float(config.get("target_max_positivity_pct", 10.0)),
                    default_target=10.0,
                    units="%", icon=config.get("icon", "🔬"),
                    help_template="Positivity rate. Target: <{target:.1f}%.",
                    status_logic=KpiLogic.LOWER_IS_BETTER,
                ))
        
        kpi_defs.append(KpiConfig(
            title="Key Drug Stockouts", source_key="key_drug_stockouts_count",
            target_setting="TARGET_DRUG_STOCKOUTS", default_target=0,
            units="items", icon="💊", is_count=True, precision=0,
            help_template="Key drugs with <{days_remaining} days of stock. Target: {target}.",
            status_logic=KpiLogic.IS_ZERO_TARGET,
        ))
        return kpi_defs
    
    # <<< SME REVISION V2 >>> Encapsulated status calculation logic.
    def _get_status_level(self, value: Optional[float], target: float, logic: KpiLogic) -> StatusLevel:
        if pd.isna(value): return StatusLevel.NO_DATA
        
        logic_map: Dict[KpiLogic, Callable[[float, float], StatusLevel]] = {
            KpiLogic.LOWER_IS_BETTER: lambda v, t: StatusLevel.HIGH_CONCERN if v > t * 1.5 else StatusLevel.MODERATE_CONCERN if v > t else StatusLevel.GOOD_PERFORMANCE,
            KpiLogic.HIGHER_IS_BETTER: lambda v, t: StatusLevel.GOOD_PERFORMANCE if v >= t else StatusLevel.ACCEPTABLE if v >= t * 0.8 else StatusLevel.HIGH_CONCERN,
            KpiLogic.LOWER_IS_BETTER_COUNT: lambda v, t: StatusLevel.GOOD_PERFORMANCE if int(v) == t else StatusLevel.ACCEPTABLE if int(v) <= t + 2 else StatusLevel.HIGH_CONCERN,
            KpiLogic.IS_ZERO_TARGET: lambda v, t: StatusLevel.GOOD_PERFORMANCE if int(v) == 0 else StatusLevel.HIGH_CONCERN,
        }
        return logic_map.get(logic, lambda v, t: StatusLevel.NO_DATA)(value, target)

    def _get_nested_value(self, key_path: str) -> Any:
        # ... (function remains good) ...
        keys, value = key_path.split('.'), self.summary_data
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value

    def _build_kpi(self, config: KpiConfig) -> Dict[str, Any]:
        """Generic method to build a single KPI dictionary from its validated configuration model."""
        value = self._get_nested_value(config.source_key)
        target = config.get_target()
        
        status = self._get_status_level(pd.to_numeric(value, errors='coerce'), target, config.status_logic)
        
        formatted_val = _format_kpi_value(value, config.precision, config.is_count)
        display_val = f"{formatted_val}{config.units}" if config.units == "%" and pd.notna(value) else formatted_val
        display_units = "" if config.units == "%" else config.units
        
        help_text = config.help_template.format(target=target, days_remaining=getattr(settings, 'CRITICAL_SUPPLY_DAYS_REMAINING', 7))
        
        return {"title": config.title, "value_str": display_val, "units": display_units, "icon": config.icon, "status_level": status.value, "help_text": help_text}

    def structure_kpis(self, definitions: List[KpiConfig]) -> List[Dict[str, Any]]:
        return [self._build_kpi(conf) for conf in definitions]

# --- Public API Functions ---
def structure_main_clinic_kpis(kpis_summary: Optional[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    structurer = _KPIStructurer(kpis_summary)
    return structurer.structure_kpis(structurer._MAIN_KPI_DEFINITIONS)

def structure_disease_specific_clinic_kpis(kpis_summary: Optional[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    structurer = _KPIStructurer(kpis_summary)
    return structurer.structure_kpis(structurer._disease_and_supply_kpi_definitions)
