# sentinel_project_root/pages/clinic_components/kpi_structuring.py
# Structures key clinic performance and disease-specific KPIs for Sentinel dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from config import settings

logger = logging.getLogger(__name__)


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

    def _get_kpi_val(key: str, default: Any = np.nan, precision: Optional[int] = 1, is_count: bool = False) -> str: # Added is_count
        val = clinic_service_kpis_summary_data.get(key, default)
        if pd.isna(val):
            return "N/A"
        try:
            if is_count: # Format counts with commas
                return f"{int(val):,}"
            if precision is not None:
                return f"{float(val):.{precision}f}"
            # Fallback for non-count, no-precision specified (should be rare for KPIs)
            return str(int(val)) if isinstance(val, (int, float)) and float(val).is_integer() else str(val)
        except (ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Error formatting KPI value for key '{key}'. Value: {val}")
            return "Error"


    # 1. Overall Average Test Turnaround Time (TAT) for Conclusive Tests
    avg_overall_tat_val_main = clinic_service_kpis_summary_data.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    overall_tat_status_level_main = "NO_DATA"
    general_tat_target_main = settings.TARGET_TEST_TURNAROUND_DAYS
    if pd.notna(avg_overall_tat_val_main):
        if avg_overall_tat_val_main > (general_tat_target_main + 1.5): overall_tat_status_level_main = "HIGH_CONCERN"
        elif avg_overall_tat_val_main > general_tat_target_main: overall_tat_status_level_main = "MODERATE_CONCERN"
        else: overall_tat_status_level_main = "ACCEPTABLE"
    structured_kpis_list.append({
        "title": "Overall Avg. TAT (Conclusive)", "value_str": _get_kpi_val('overall_avg_test_turnaround_conclusive_days', precision=1),
        "units": "days", "icon": "⏱️", "status_level": overall_tat_status_level_main,
        "help_text": f"Average Turnaround Time for all conclusive diagnostic tests. Target ref: ~{general_tat_target_main} days."
    })

    # 2. Percentage of CRITICAL Tests Meeting TAT Target
    perc_critical_tat_met_val_main = clinic_service_kpis_summary_data.get('perc_critical_tests_tat_met', np.nan)
    critical_tat_status_level_main = "NO_DATA"
    target_perc_met_main = settings.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(perc_critical_tat_met_val_main):
        if perc_critical_tat_met_val_main >= target_perc_met_main: critical_tat_status_level_main = "GOOD_PERFORMANCE"
        elif perc_critical_tat_met_val_main >= target_perc_met_main * 0.80: critical_tat_status_level_main = "ACCEPTABLE"
        elif perc_critical_tat_met_val_main >= target_perc_met_main * 0.60: critical_tat_status_level_main = "MODERATE_CONCERN"
        else: critical_tat_status_level_main = "HIGH_CONCERN"
    structured_kpis_list.append({
        "title": "% Critical Tests TAT Met", "value_str": _get_kpi_val('perc_critical_tests_tat_met', precision=1),
        "units": "%", "icon": "🎯", "status_level": critical_tat_status_level_main,
        "help_text": f"Percentage of critical diagnostic tests meeting defined TAT targets. Target: ≥{target_perc_met_main}%."
    })

    # 3. Total Pending Critical Tests (by unique patients)
    num_pending_critical_val_main = clinic_service_kpis_summary_data.get('total_pending_critical_tests_patients', 0)
    pending_critical_status_level_main = "NO_DATA"
    if pd.notna(num_pending_critical_val_main): 
        count_val_main = int(num_pending_critical_val_main)
        if count_val_main == 0: pending_critical_status_level_main = "GOOD_PERFORMANCE"
        elif count_val_main <= 3: pending_critical_status_level_main = "ACCEPTABLE"
        elif count_val_main <= 10: pending_critical_status_level_main = "MODERATE_CONCERN"
        else: pending_critical_status_level_main = "HIGH_CONCERN"
    structured_kpis_list.append({
        "title": "Pending Critical Tests (Patients)", "value_str": _get_kpi_val('total_pending_critical_tests_patients', precision=0, is_count=True), # Use is_count
        "units": "patients", "icon": "⏳", "status_level": pending_critical_status_level_main,
        "help_text": "Number of unique patients with critical test results still pending. Target: 0."
    })

    # 4. Sample Rejection Rate (%)
    rejection_rate_val_main = clinic_service_kpis_summary_data.get('sample_rejection_rate_perc', np.nan)
    rejection_rate_status_level_main = "NO_DATA"
    target_rejection_pct_main = settings.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(rejection_rate_val_main):
        if rejection_rate_val_main > target_rejection_pct_main * 1.75: rejection_rate_status_level_main = "HIGH_CONCERN"
        elif rejection_rate_val_main > target_rejection_pct_main: rejection_rate_status_level_main = "MODERATE_CONCERN"
        else: rejection_rate_status_level_main = "GOOD_PERFORMANCE"
    structured_kpis_list.append({
        "title": "Sample Rejection Rate", "value_str": _get_kpi_val('sample_rejection_rate_perc', precision=1),
        "units":"%", "icon": "🧪", "status_level": rejection_rate_status_level_main,
        "help_text": f"Overall rate of laboratory samples rejected for testing. Target: < {target_rejection_pct_main}%."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis_list)} main clinic KPIs.")
    return structured_kpis_list


def structure_disease_specific_clinic_kpis(
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]],
    reporting_period_context_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs and key drug stockout counts.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for: {reporting_period_context_str}")
    
    structured_disease_kpis_list_val: List[Dict[str, Any]] = [] # Renamed to avoid conflict

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_disease_kpis_list_val

    test_summary_details_kpis = clinic_service_kpis_summary_data.get("test_summary_details", {})
    if not isinstance(test_summary_details_kpis, dict): # Check if it's a dict
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not a dict in KPI summary. Test positivity KPIs cannot be structured.")
        test_summary_details_kpis = {} # Ensure it's a dict for safe access

    key_tests_for_positivity_display_cfg_val = { # Use display_name from settings as key
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_positivity_pct": 15.0, "disease_label_short": "TB"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_positivity_pct": settings.TARGET_MALARIA_POSITIVITY_RATE, "disease_label_short": "Malaria"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_positivity_pct": 5.0, "disease_label_short": "HIV"} # Example target
    }

    for test_display_name_key_val, props_config_val in key_tests_for_positivity_display_cfg_val.items():
        test_stats_data_val = test_summary_details_kpis.get(test_display_name_key_val, {})
        positivity_rate_value_val = test_stats_data_val.get("positive_rate_perc", np.nan)
        
        pos_rate_status_val = "NO_DATA"
        target_max_pos_val = props_config_val.get("target_max_positivity_pct", 10.0)
        if pd.notna(positivity_rate_value_val):
            if positivity_rate_value_val > target_max_pos_val * 1.5: pos_rate_status_val = "HIGH_CONCERN"
            elif positivity_rate_value_val > target_max_pos_val : pos_rate_status_val = "MODERATE_CONCERN"
            else: pos_rate_status_val = "ACCEPTABLE"
        
        structured_disease_kpis_list_val.append({
            "title": f"{props_config_val['disease_label_short']} Positivity ({test_display_name_key_val})",
            "value_str": f"{positivity_rate_value_val:.1f}" if pd.notna(positivity_rate_value_val) else "N/A",
            "units":"%", "icon": props_config_val["icon"], "status_level": pos_rate_status_val,
            "help_text": f"Positivity rate for {test_display_name_key_val}. Target context: < {target_max_pos_val}%."
        })

    num_key_drug_stockouts_val_kpi = clinic_service_kpis_summary_data.get('key_drug_stockouts_count', 0)
    stockout_status_val_kpi = "NO_DATA"
    if pd.notna(num_key_drug_stockouts_val_kpi): # Ensure it's not NaN before int conversion logic
        count_val_stockout = int(num_key_drug_stockouts_val_kpi)
        if count_val_stockout == 0: stockout_status_val_kpi = "GOOD_PERFORMANCE"
        elif count_val_stockout <= 2: stockout_status_val_kpi = "MODERATE_CONCERN"
        else: stockout_status_val_kpi = "HIGH_CONCERN"
    
    structured_disease_kpis_list_val.append({
        "title": "Key Drug Stockouts",
        "value_str": str(int(num_key_drug_stockouts_val_kpi)) if pd.notna(num_key_drug_stockouts_val_kpi) else "N/A",
        "units":"items", "icon": "💊", "status_level": stockout_status_val_kpi,
        "help_text": f"Number of key drugs/supplies with < {settings.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_disease_kpis_list_val)} disease-specific & supply KPIs.")
    return structured_disease_kpis_list_val
