# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# Structures key clinic performance and disease-specific KPIs for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# --- Module Imports & Setup ---
try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in kpi_structuring.py: {e}. Using fallback settings.")
    
    class FallbackSettings:
        TARGET_TEST_TURNAROUND_DAYS = 2.0; TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
        TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0; KEY_TEST_TYPES_FOR_ANALYSIS = {}
        TARGET_MALARIA_POSITIVITY_RATE = 10.0; CRITICAL_SUPPLY_DAYS_REMAINING = 7
    settings = FallbackSettings()

logger = logging.getLogger(__name__)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)


def _format_kpi_value(
    value: Any, default_str: str = "N/A", precision: int = 1, is_count: bool = False
) -> str:
    """Helper to format KPI values robustly."""
    if pd.isna(value) or value is None:
        return default_str
    try:
        numeric_value = pd.to_numeric(value)
        if is_count:
            return f"{int(numeric_value):,}"
        return f"{numeric_value:.{precision}f}"
    except (ValueError, TypeError):
        return str(value) if str(value).strip() else default_str


def structure_main_clinic_kpis(
    kpis_summary: Optional[Dict[str, Any]],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs into a list formatted for display.
    """
    structured_kpis = []
    if not isinstance(kpis_summary, dict): return structured_kpis

    # KPI 1: Overall Avg. TAT
    avg_tat = kpis_summary.get('overall_avg_test_turnaround_conclusive_days')
    tat_target = _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2.0)
    tat_status = "NO_DATA"
    if pd.notna(avg_tat):
        tat_status = "HIGH_CONCERN" if avg_tat > tat_target + 1.5 else "MODERATE_CONCERN" if avg_tat > tat_target else "ACCEPTABLE"
    structured_kpis.append({
        "title": "Overall Avg. TAT", "value_str": _format_kpi_value(avg_tat),
        "units": "days", "icon": "⏱️", "status_level": tat_status,
        "help_text": f"Average Turnaround Time for conclusive tests. Target: ~{tat_target} days."
    })

    # KPI 2: % Critical Tests TAT Met
    perc_met = kpis_summary.get('perc_critical_tests_tat_met')
    perc_target = _get_setting('TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY', 85.0)
    perc_status = "NO_DATA"
    if pd.notna(perc_met):
        perc_status = "GOOD_PERFORMANCE" if perc_met >= perc_target else "ACCEPTABLE" if perc_met >= perc_target * 0.8 else "HIGH_CONCERN"
    structured_kpis.append({
        "title": "% Critical Tests TAT Met", "value_str": _format_kpi_value(perc_met) + ("%" if pd.notna(perc_met) else ""),
        "icon": "🎯", "status_level": perc_status,
        "help_text": f"Percentage of critical tests meeting TAT targets. Target: ≥{perc_target:.1f}%."
    })

    # KPI 3: Pending Critical Tests
    pending_count = kpis_summary.get('total_pending_critical_tests_patients')
    pending_status = "NO_DATA"
    if pd.notna(pending_count):
        pending_status = "GOOD_PERFORMANCE" if pending_count == 0 else "ACCEPTABLE" if pending_count <= 3 else "HIGH_CONCERN"
    structured_kpis.append({
        "title": "Pending Critical Tests", "value_str": _format_kpi_value(pending_count, is_count=True),
        "units": "patients", "icon": "⏳", "status_level": pending_status,
        "help_text": "Number of patients with pending critical test results. Target: 0."
    })

    # KPI 4: Sample Rejection Rate
    rejection_rate = kpis_summary.get('sample_rejection_rate_perc')
    rejection_target = _get_setting('TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY', 5.0)
    rejection_status = "NO_DATA"
    if pd.notna(rejection_rate):
        rejection_status = "HIGH_CONCERN" if rejection_rate > rejection_target * 1.5 else "MODERATE_CONCERN" if rejection_rate > rejection_target else "GOOD_PERFORMANCE"
    structured_kpis.append({
        "title": "Sample Rejection Rate", "value_str": _format_kpi_value(rejection_rate) + ("%" if pd.notna(rejection_rate) else ""),
        "icon": "🧪", "status_level": rejection_status,
        "help_text": f"Overall rate of rejected lab samples. Target: < {rejection_target:.1f}%."
    })
    
    return structured_kpis


def structure_disease_specific_clinic_kpis(
    kpis_summary: Optional[Dict[str, Any]],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific and supply-chain KPIs.
    """
    structured_kpis = []
    if not isinstance(kpis_summary, dict): return structured_kpis

    # FIXED: Use .get() with a default empty dict to prevent crash if test_summary_details is missing.
    test_details = kpis_summary.get("test_summary_details", {})
    key_tests = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
    
    for test_name, config in key_tests.items():
        if not isinstance(config, dict): continue

        display_name = config.get("display_name", test_name)
        target_positivity = float(config.get("target_max_positivity_pct", 10.0))
        # FIXED: Use .get() with a default empty dict to prevent crash if a specific test has no data.
        stats = test_details.get(test_name, {})
        pos_rate = stats.get("positive_rate_perc")
        
        status = "NO_DATA"
        if pd.notna(pos_rate):
            pos_rate_num = pd.to_numeric(pos_rate, errors='coerce')
            if pd.notna(pos_rate_num):
                status = "HIGH_CONCERN" if pos_rate_num > target_positivity * 1.5 else "MODERATE_CONCERN" if pos_rate_num > target_positivity else "ACCEPTABLE"
        
        structured_kpis.append({
            "title": f"{display_name} Positivity",
            "value_str": _format_kpi_value(pos_rate) + ("%" if pd.notna(pos_rate) else ""),
            "icon": config.get("icon", "🔬"), "status_level": status,
            "help_text": f"Positivity rate for {display_name}. Context target: < {target_positivity:.1f}%."
        })

    # KPI: Key Drug Stockouts
    stockouts = kpis_summary.get('key_drug_stockouts_count')
    stockout_status = "NO_DATA"
    if pd.notna(stockouts):
        stockout_status = "GOOD_PERFORMANCE" if stockouts == 0 else "MODERATE_CONCERN" if stockouts <= 2 else "HIGH_CONCERN"
            
    structured_kpis.append({
        "title": "Key Drug Stockouts",
        "value_str": _format_kpi_value(stockouts, is_count=True, default_str="0"),
        "units": "items", "icon": "💊", "status_level": stockout_status,
        "help_text": f"Key drugs with < {_get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)} days of stock. Target: 0."
    })
    
    return structured_kpis
