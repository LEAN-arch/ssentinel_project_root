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
        KEY_TEST_TYPES_FOR_ANALYSIS: Dict[str, Dict[str, Any]] = {
            "Sputum-GeneXpert": {"display_name": "TB GeneXpert"},
            "RDT-Malaria": {"display_name": "Malaria RDT"},
            "HIV-Rapid": {"display_name": "HIV Rapid Test"}
        }
        TARGET_MALARIA_POSITIVITY_RATE = 10.0
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
    if pd.isna(value):
        return default_str
    try:
        numeric_value = pd.to_numeric(value, errors='raise')
        if is_count:
            return f"{int(numeric_value):,}"
        if is_percentage:
            return f"{numeric_value:.{precision if precision is not None and precision >=0 else 1}f}"
        if precision is not None:
            return f"{numeric_value:.{precision}f}"
        return str(int(numeric_value)) if isinstance(numeric_value, float) and numeric_value.is_integer() else str(numeric_value)
    except (ValueError, TypeError):
        return str(value) if str(value).strip() else default_str


def structure_main_clinic_kpis(
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]],
    reporting_period_context_str: str 
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs from a summary dictionary into a list
    of dictionaries, each formatted for display.
    """
    module_log_prefix = "ClinicMainKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring main clinic KPIs for: {reporting_period_context_str}")
    
    structured_kpis_list: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_kpis_list

    avg_overall_tat_val = clinic_service_kpis_summary_data.get('overall_avg_test_turnaround_conclusive_days')
    tat_target_val = float(_get_setting('TARGET_TEST_TURNAROUND_DAYS', 2.0))
    overall_tat_status = "NO_DATA"
    if pd.notna(avg_overall_tat_val):
        numeric_tat = pd.to_numeric(avg_overall_tat_val, errors='coerce')
        if pd.notna(numeric_tat):
            if numeric_tat > (tat_target_val + 1.5): overall_tat_status = "HIGH_CONCERN"
            elif numeric_tat > tat_target_val: overall_tat_status = "MODERATE_CONCERN"
            else: overall_tat_status = "ACCEPTABLE"
    structured_kpis_list.append({
        "title": "Overall Avg. TAT (Conclusive)", "value_str": _format_kpi_value(avg_overall_tat_val),
        "units": "days", "icon": "⏱️", "status_level": overall_tat_status,
        "help_text": f"Average Turnaround Time for all conclusive tests. Target: ~{tat_target_val} days."
    })

    perc_critical_tat_met_val = clinic_service_kpis_summary_data.get('perc_critical_tests_tat_met')
    target_perc_met_val = float(_get_setting('TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY', 85.0))
    critical_tat_status = "NO_DATA"
    if pd.notna(perc_critical_tat_met_val):
        numeric_perc_met = pd.to_numeric(perc_critical_tat_met_val, errors='coerce')
        if pd.notna(numeric_perc_met):
            if numeric_perc_met >= target_perc_met_val: critical_tat_status = "GOOD_PERFORMANCE"
            elif numeric_perc_met >= target_perc_met_val * 0.8: critical_tat_status = "ACCEPTABLE"
            else: critical_tat_status = "HIGH_CONCERN"
    structured_kpis_list.append({
        "title": "% Critical Tests TAT Met", "value_str": _format_kpi_value(perc_critical_tat_met_val, is_percentage=True),
        "units": "%", "icon": "🎯", "status_level": critical_tat_status,
        "help_text": f"Percentage of critical tests meeting TAT targets. Target: ≥{target_perc_met_val:.1f}%."
    })

    num_pending_critical_val = clinic_service_kpis_summary_data.get('total_pending_critical_tests_patients')
    pending_critical_status = "NO_DATA"
    if pd.notna(num_pending_critical_val):
        count_val = pd.to_numeric(num_pending_critical_val, errors='coerce')
        if pd.notna(count_val):
            if count_val == 0: pending_critical_status = "GOOD_PERFORMANCE"
            elif count_val <= 3: pending_critical_status = "ACCEPTABLE"
            elif count_val <= 10: pending_critical_status = "MODERATE_CONCERN"
            else: pending_critical_status = "HIGH_CONCERN"
    structured_kpis_list.append({
        "title": "Pending Critical Tests", "value_str": _format_kpi_value(num_pending_critical_val, is_count=True),
        "units": "patients", "icon": "⏳", "status_level": pending_critical_status,
        "help_text": "Number of unique patients with critical test results still pending. Target: 0."
    })

    rejection_rate_val = clinic_service_kpis_summary_data.get('sample_rejection_rate_perc')
    target_rejection_pct = float(_get_setting('TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY', 5.0))
    rejection_rate_status = "NO_DATA"
    if pd.notna(rejection_rate_val):
        numeric_rejection_rate = pd.to_numeric(rejection_rate_val, errors='coerce')
        if pd.notna(numeric_rejection_rate):
            if numeric_rejection_rate > target_rejection_pct * 1.5: rejection_rate_status = "HIGH_CONCERN"
            elif numeric_rejection_rate > target_rejection_pct: rejection_rate_status = "MODERATE_CONCERN"
            else: rejection_rate_status = "GOOD_PERFORMANCE"
    structured_kpis_list.append({
        "title": "Sample Rejection Rate", "value_str": _format_kpi_value(rejection_rate_val, is_percentage=True),
        "units":"%", "icon": "🧪", "status_level": rejection_rate_status,
        "help_text": f"Overall rate of lab samples rejected. Target: < {target_rejection_pct:.1f}%."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis_list)} main clinic KPIs.")
    return structured_kpis_list


def structure_disease_specific_clinic_kpis(
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]],
    reporting_period_context_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs (like test positivity rates) and 
    key drug stockout counts from summary data.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring disease & supply KPIs for: {reporting_period_context_str}")
    
    structured_kpis_list: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict):
        return structured_kpis_list

    test_summary_details = clinic_service_kpis_summary_data.get("test_summary_details", {})
    if not isinstance(test_summary_details, dict):
        logger.warning(f"({module_log_prefix}) 'test_summary_details' is not a dict. Cannot structure test KPIs.")
        test_summary_details = {}

    key_tests_config = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
    
    for internal_test_name, test_config in key_tests_config.items():
        if not isinstance(test_config, dict):
            continue

        display_name = test_config.get("display_name", internal_test_name)
        disease_label = test_config.get("disease_label_short", "Test")
        icon = test_config.get("icon", "🔬")
        target_max_positivity = float(test_config.get("target_max_positivity_pct", 10.0))

        # CORRECTED: Use the internal_test_name for data lookup, not the display name.
        stats_for_this_test = test_summary_details.get(internal_test_name, {})
        positivity_rate = stats_for_this_test.get("positive_rate_perc")
        
        status_val = "NO_DATA"
        if pd.notna(positivity_rate):
            numeric_pos_rate = pd.to_numeric(positivity_rate, errors='coerce')
            if pd.notna(numeric_pos_rate):
                if numeric_pos_rate > target_max_positivity * 1.5: status_val = "HIGH_CONCERN"
                elif numeric_pos_rate > target_max_positivity: status_val = "MODERATE_CONCERN"
                else: status_val = "ACCEPTABLE"
        
        structured_kpis_list.append({
            "title": f"{disease_label} Positivity Rate", "value_str": _format_kpi_value(positivity_rate, is_percentage=True),
            "units":"%", "icon": icon, "status_level": status_val,
            "help_text": f"Positivity for {display_name}. Target: < {target_max_positivity:.1f}%."
        })

    num_stockouts = clinic_service_kpis_summary_data.get('key_drug_stockouts_count')
    stockout_status = "NO_DATA"
    if pd.notna(num_stockouts):
        count_val = pd.to_numeric(num_stockouts, errors='coerce')
        if pd.notna(count_val):
            if count_val == 0: stockout_status = "GOOD_PERFORMANCE"
            elif count_val <= 2: stockout_status = "MODERATE_CONCERN"
            else: stockout_status = "HIGH_CONCERN"
            
    structured_kpis_list.append({
        "title": "Key Drug Stockouts",
        "value_str": _format_kpi_value(num_stockouts, is_count=True, default_str="0"),
        "units":"items", "icon": "💊", "status_level": stockout_status,
        "help_text": f"Key drugs with < {_get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)} days of stock."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis_list)} disease-specific & supply KPIs.")
    return structured_kpis_list
