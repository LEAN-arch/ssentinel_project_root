# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# Structures key clinic performance and disease-specific KPIs for Sentinel dashboards.
# Renamed from main_kpi_structurer.py

import pandas as pd # Not directly used for values, but good for type hints
import numpy as np # For np.nan
import logging
from typing import Dict, Any, List, Optional

from config import settings # Use new settings module

logger = logging.getLogger(__name__)


def structure_main_clinic_kpis( # Renamed function
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]], # Data from aggregation.get_clinic_summary_kpis
    reporting_period_context_str: str # For context in help_text if needed
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs from a summary dictionary into a list
    of dictionaries, each formatted for display (e.g., via visualization.ui_elements.render_kpi_card).

    Args:
        clinic_service_kpis_summary_data: A dictionary containing pre-calculated clinic service KPIs.
                                         Expected keys match outputs of `get_clinic_summary_kpis`.
        reporting_period_context_str: String describing the reporting period (for contextual help text).

    Returns:
        List of KPI dictionaries for main clinic performance.
    """
    module_log_prefix = "ClinicMainKPIStructure" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Structuring main clinic KPIs for period: {reporting_period_context_str}")
    
    structured_main_kpis_list: List[Dict[str, Any]] = [] # Renamed for clarity

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_main_kpis_list

    # Helper to safely get and format values
    def _get_kpi_val(key: str, default: Any = np.nan, precision: Optional[int] = 1) -> str:
        val = clinic_service_kpis_summary_data.get(key, default)
        if pd.isna(val):
            return "N/A"
        try:
            if precision is not None:
                return f"{float(val):.{precision}f}"
            return str(int(val)) if isinstance(val, (int, float)) and float(val).is_integer() else str(val)
        except (ValueError, TypeError):
            return "Error"


    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    avg_overall_tat_val = clinic_service_kpis_summary_data.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    overall_tat_status_level = "NO_DATA"
    general_tat_target = settings.TARGET_TEST_TURNAROUND_DAYS
    if pd.notna(avg_overall_tat_val):
        if avg_overall_tat_val > (general_tat_target + 1.5): overall_tat_status_level = "HIGH_CONCERN"
        elif avg_overall_tat_val > general_tat_target: overall_tat_status_level = "MODERATE_CONCERN"
        else: overall_tat_status_level = "ACCEPTABLE"
    structured_main_kpis_list.append({
        "title": "Overall Avg. TAT (Conclusive)", "value_str": _get_kpi_val('overall_avg_test_turnaround_conclusive_days', precision=1),
        "units": "days", "icon": "⏱️", "status_level": overall_tat_status_level,
        "help_text": f"Average Turnaround Time for all conclusive diagnostic tests. Target ref: ~{general_tat_target} days."
    })

    # 2. Percentage of CRITICAL Tests Meeting TAT Target
    perc_critical_tat_met_val = clinic_service_kpis_summary_data.get('perc_critical_tests_tat_met', np.nan)
    critical_tat_status_level = "NO_DATA"
    target_perc_met = settings.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(perc_critical_tat_met_val):
        if perc_critical_tat_met_val >= target_perc_met: critical_tat_status_level = "GOOD_PERFORMANCE"
        elif perc_critical_tat_met_val >= target_perc_met * 0.80: critical_tat_status_level = "ACCEPTABLE" # Allow some leeway
        elif perc_critical_tat_met_val >= target_perc_met * 0.60: critical_tat_status_level = "MODERATE_CONCERN"
        else: critical_tat_status_level = "HIGH_CONCERN"
    structured_main_kpis_list.append({
        "title": "% Critical Tests TAT Met", "value_str": _get_kpi_val('perc_critical_tests_tat_met', precision=1),
        "units": "%", "icon": "🎯", "status_level": critical_tat_status_level,
        "help_text": f"Percentage of critical diagnostic tests meeting defined TAT targets. Target: ≥{target_perc_met}%."
    })

    # 3. Total Pending Critical Tests (by unique patients)
    num_pending_critical_val = clinic_service_kpis_summary_data.get('total_pending_critical_tests_patients', 0)
    pending_critical_status_level = "NO_DATA"
    if pd.notna(num_pending_critical_val): # Ensure it's not NaN before int conversion logic
        count_val = int(num_pending_critical_val)
        if count_val == 0: pending_critical_status_level = "GOOD_PERFORMANCE"
        elif count_val <= 3: pending_critical_status_level = "ACCEPTABLE"
        elif count_val <= 10: pending_critical_status_level = "MODERATE_CONCERN"
        else: pending_critical_status_level = "HIGH_CONCERN"
    structured_main_kpis_list.append({
        "title": "Pending Critical Tests (Patients)", "value_str": _get_kpi_val('total_pending_critical_tests_patients', precision=0),
        "units": "patients", "icon": "⏳", "status_level": pending_critical_status_level,
        "help_text": "Number of unique patients with critical test results still pending. Target: 0."
    })

    # 4. Sample Rejection Rate (%)
    rejection_rate_val = clinic_service_kpis_summary_data.get('sample_rejection_rate_perc', np.nan)
    rejection_rate_status_level = "NO_DATA"
    target_rejection_pct = settings.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(rejection_rate_val):
        if rejection_rate_val > target_rejection_pct * 1.75: rejection_rate_status_level = "HIGH_CONCERN"
        elif rejection_rate_val > target_rejection_pct: rejection_rate_status_level = "MODERATE_CONCERN"
        else: rejection_rate_status_level = "GOOD_PERFORMANCE"
    structured_main_kpis_list.append({
        "title": "Sample Rejection Rate", "value_str": _get_kpi_val('sample_rejection_rate_perc', precision=1),
        "units":"%", "icon": "🧪", "status_level": rejection_rate_status_level, # Changed icon
        "help_text": f"Overall rate of laboratory samples rejected for testing. Target: < {target_rejection_pct}%."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_main_kpis_list)} main clinic KPIs.")
    return structured_main_kpis_list


def structure_disease_specific_clinic_kpis( # Renamed function
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]], # From aggregation.get_clinic_summary_kpis
    reporting_period_context_str: str # For context in help_text if needed
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs (e.g., test positivity rates) and key drug stockout counts
    from a summary dictionary into a list suitable for KPI card display.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructure" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for period: {reporting_period_context_str}")
    
    structured_disease_kpis_list: List[Dict[str, Any]] = [] # Renamed for clarity

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_disease_kpis_list

    # Test Positivity Rates for Key Configured Tests
    # Assumes clinic_service_kpis_summary_data['test_summary_details'] is a dict where keys are test_display_name
    test_summary_details_from_kpis = clinic_service_kpis_summary_data.get("test_summary_details", {})
    if not isinstance(test_summary_details_from_kpis, dict):
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not a dict in KPI summary. Cannot structure test positivity KPIs.")
        test_summary_details_from_kpis = {} # Ensure it's a dict for safe access

    # Define which tests to show positivity for, and their properties
    # These keys MUST match the `display_name` used in `settings.KEY_TEST_TYPES_FOR_ANALYSIS`
    # and thus the keys in `test_summary_details_from_kpis`.
    key_tests_for_positivity_display_config = {
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_positivity_pct": 15.0, "disease_label_short": "TB"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_positivity_pct": settings.TARGET_MALARIA_POSITIVITY_RATE, "disease_label_short": "Malaria"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_positivity_pct": 5.0, "disease_label_short": "HIV"} # Example target
    }

    for test_display_name_key, props_config in key_tests_for_positivity_display_config.items():
        test_stats_data = test_summary_details_from_kpis.get(test_display_name_key, {}) # Get stats for this display name
        positivity_rate_value = test_stats_data.get("positive_rate_perc", np.nan) # Key from get_clinic_summary_kpis
        
        pos_rate_status = "NO_DATA"
        target_max_pos = props_config.get("target_max_positivity_pct", 10.0) # Default target if not in config
        if pd.notna(positivity_rate_value):
            if positivity_rate_value > target_max_pos * 1.5: pos_rate_status = "HIGH_CONCERN"
            elif positivity_rate_value > target_max_pos : pos_rate_status = "MODERATE_CONCERN"
            else: pos_rate_status = "ACCEPTABLE"
        
        structured_disease_kpis_list.append({
            "title": f"{props_config['disease_label_short']} Positivity ({test_display_name_key})",
            "value_str": f"{positivity_rate_value:.1f}" if pd.notna(positivity_rate_value) else "N/A",
            "units":"%", "icon": props_config["icon"], "status_level": pos_rate_status,
            "help_text": f"Positivity rate for {test_display_name_key}. Target context: < {target_max_pos}%."
        })

    # Key Drug Stockouts Count
    num_key_drug_stockouts_val = clinic_service_kpis_summary_data.get('key_drug_stockouts_count', 0) # Default to 0
    stockout_status = "NO_DATA"
    if pd.notna(num_key_drug_stockouts_val):
        count_val = int(num_key_drug_stockouts_val)
        if count_val == 0: stockout_status = "GOOD_PERFORMANCE"
        elif count_val <= 2: stockout_status = "MODERATE_CONCERN"
        else: stockout_status = "HIGH_CONCERN"
    
    structured_disease_kpis_list.append({
        "title": "Key Drug Stockouts",
        "value_str": str(int(num_key_drug_stockouts_val)) if pd.notna(num_key_drug_stockouts_val) else "N/A",
        "units":"items", "icon": "💊", "status_level": stockout_status,
        "help_text": f"Number of key drugs/supplies with < {settings.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_disease_kpis_list)} disease-specific & supply KPIs.")
    return structured_disease_kpis_list
