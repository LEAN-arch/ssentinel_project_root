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

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
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
    of dictionaries, each formatted for display (e.g., via render_kpi_card).
    """
    module_log_prefix = "ClinicMainKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring main clinic KPIs for: {reporting_period_context_str}")
    
    structured_kpis_list: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_kpis_list

    # KPI 1: Overall Avg. TAT
    avg_overall_tat_val = clinic_service_kpis_summary_data.get('overall_avg_test_turnaround_conclusive_days')
    overall_tat_status = "NO_DATA"
    tat_target_val = float(_get_setting('TARGET_TEST_TURNAROUND_DAYS', 2.0))
    if pd.notna(avg_overall_tat_val):
        try:
            numeric_tat = float(avg_overall_tat_val)
            if numeric_tat > (tat_target_val + 1.5): overall_tat_status = "HIGH_CONCERN"
            elif numeric_tat > tat_target_val: overall_tat_status = "MODERATE_CONCERN"
            else: overall_tat_status = "ACCEPTABLE"
        except (ValueError, TypeError):
             logger.warning(f"({module_log_prefix}) Could not convert TAT value '{avg_overall_tat_val}' to float.")

    structured_kpis_list.append({
        "title": "Overall Avg. TAT (Conclusive)", 
        "value_str": _format_kpi_value(avg_overall_tat_val, precision=1),
        "units": "days", "icon": "⏱️", "status_level": overall_tat_status,
        "help_text": f"Average Turnaround Time for all conclusive diagnostic tests. Target ref: ~{tat_target_val} days."
    })

    # KPI 2: % Critical Tests TAT Met
    perc_critical_tat_met_val = clinic_service_kpis_summary_data.get('perc_critical_tests_tat_met')
    critical_tat_status = "NO_DATA"
    target_perc_met_val = float(_get_setting('TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY', 85.0))
    if pd.notna(perc_critical_tat_met_val):
        try:
            numeric_perc_met = float(perc_critical_tat_met_val)
            if numeric_perc_met >= target_perc_met_val: critical_tat_status = "GOOD_PERFORMANCE"
            elif numeric_perc_met >= target_perc_met_val * 0.80: critical_tat_status = "ACCEPTABLE"
            elif numeric_perc_met >= target_perc_met_val * 0.60: critical_tat_status = "MODERATE_CONCERN"
            else: critical_tat_status = "HIGH_CONCERN"
        except (ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Could not convert % Critical TAT Met value '{perc_critical_tat_met_val}' to float.")

    structured_kpis_list.append({
        "title": "% Critical Tests TAT Met", 
        "value_str": _format_kpi_value(perc_critical_tat_met_val, precision=1, is_percentage=True),
        "units": "%", "icon": "🎯", "status_level": critical_tat_status,
        "help_text": f"Percentage of critical diagnostic tests meeting defined TAT targets. Target: ≥{target_perc_met_val:.1f}%."
    })

    # KPI 3: Pending Critical Tests (Patients)
    num_pending_critical_val = clinic_service_kpis_summary_data.get('total_pending_critical_tests_patients')
    pending_critical_status = "NO_DATA"
    if pd.notna(num_pending_critical_val): 
        try:
            count_val = int(num_pending_critical_val)
            if count_val == 0: pending_critical_status = "GOOD_PERFORMANCE"
            elif count_val <= 3: pending_critical_status = "ACCEPTABLE"
            elif count_val <= 10: pending_critical_status = "MODERATE_CONCERN"
            else: pending_critical_status = "HIGH_CONCERN"
        except (ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Could not convert Pending Critical Tests value '{num_pending_critical_val}' to int.")
            
    structured_kpis_list.append({
        "title": "Pending Critical Tests (Patients)", 
        "value_str": _format_kpi_value(num_pending_critical_val, is_count=True),
        "units": "patients", "icon": "⏳", "status_level": pending_critical_status,
        "help_text": "Number of unique patients with critical test results still pending. Target: 0."
    })

    # KPI 4: Sample Rejection Rate
    rejection_rate_val = clinic_service_kpis_summary_data.get('sample_rejection_rate_perc')
    rejection_rate_status = "NO_DATA"
    target_rejection_pct = float(_get_setting('TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY', 5.0))
    if pd.notna(rejection_rate_val):
        try:
            numeric_rejection_rate = float(rejection_rate_val)
            if numeric_rejection_rate > target_rejection_pct * 1.75: rejection_rate_status = "HIGH_CONCERN"
            elif numeric_rejection_rate > target_rejection_pct: rejection_rate_status = "MODERATE_CONCERN"
            else: rejection_rate_status = "GOOD_PERFORMANCE"
        except(ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Could not convert Sample Rejection Rate value '{rejection_rate_val}' to float.")
            
    structured_kpis_list.append({
        "title": "Sample Rejection Rate", 
        "value_str": _format_kpi_value(rejection_rate_val, precision=1, is_percentage=True),
        "units":"%", "icon": "🧪", "status_level": rejection_rate_status,
        "help_text": f"Overall rate of laboratory samples rejected for testing. Target: < {target_rejection_pct:.1f}%."
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
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for: {reporting_period_context_str}")
    
    structured_kpis_list: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_kpis_list

    test_summary_details_data = clinic_service_kpis_summary_data.get("test_summary_details", {})
    if not isinstance(test_summary_details_data, dict):
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not a dict. Disease-specific test KPIs cannot be structured.")
        test_summary_details_data = {}

    key_tests_config = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})
    
    for internal_test_name, test_config in key_tests_config.items():
        if not isinstance(test_config, dict):
            continue

        display_name = test_config.get("display_name", internal_test_name)
        icon = test_config.get("icon", "🔬")
        target_max_positivity = float(test_config.get("target_max_positivity_pct", 10.0))

        stats_for_this_test = test_summary_details_data.get(internal_test_name, {})
        positivity_rate = stats_for_this_test.get("positive_rate_perc")
        
        status_val = "NO_DATA"
        if pd.notna(positivity_rate):
            try:
                numeric_pos_rate = float(positivity_rate)
                if numeric_pos_rate > target_max_positivity * 1.5: status_val = "HIGH_CONCERN"
                elif numeric_pos_rate > target_max_positivity: status_val = "MODERATE_CONCERN"
                else: status_val = "ACCEPTABLE"
            except (ValueError, TypeError):
                 logger.warning(f"({module_log_prefix}) Could not convert positivity rate '{positivity_rate}' for {display_name} to float.")
        
        # CORRECTED: Use the specific display_name in the title to provide clarity.
        structured_kpis_list.append({
            "title": f"{display_name} Positivity",
            "value_str": _format_kpi_value(positivity_rate, precision=1, is_percentage=True),
            "units":"%", "icon": icon, "status_level": status_val,
            "help_text": f"Positivity rate for {display_name}. Target context: < {target_max_positivity:.1f}%."
        })

    # KPI: Key Drug Stockouts
    num_key_drug_stockouts_val = clinic_service_kpis_summary_data.get('key_drug_stockouts_count')
    stockout_status = "NO_DATA"
    critical_supply_days_setting = int(_get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7))
    if pd.notna(num_key_drug_stockouts_val):
        try:
            count_stockout_val = int(num_key_drug_stockouts_val)
            if count_stockout_val == 0: stockout_status = "GOOD_PERFORMANCE"
            elif count_stockout_val <= 2: stockout_status = "MODERATE_CONCERN"
            else: stockout_status = "HIGH_CONCERN"
        except (ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Could not convert Key Drug Stockouts value '{num_key_drug_stockouts_val}' to int.")
            
    structured_kpis_list.append({
        "title": "Key Drug Stockouts",
        "value_str": _format_kpi_value(num_key_drug_stockouts_val, is_count=True, default_str="0"),
        "units":"items", "icon": "💊", "status_level": stockout_status,
        "help_text": f"Number of key drugs/supplies with < {critical_supply_days_setting} days of stock. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis_list)} disease-specific & supply KPIs.")
    return structured_kpis_list
