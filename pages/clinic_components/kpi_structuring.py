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
    
    structured_kpis_list_main: List[Dict[str, Any]] = [] # Renamed to avoid conflict

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_kpis_list_main

    def _get_kpi_val_main(key: str, default: Any = np.nan, precision: Optional[int] = 1, is_count: bool = False) -> str:
        val = clinic_service_kpis_summary_data.get(key, default)
        if pd.isna(val):
            return "N/A"
        try:
            if is_count:
                return f"{int(val):,}"
            if precision is not None:
                return f"{float(val):.{precision}f}"
            return str(int(val)) if isinstance(val, (int, float)) and float(val).is_integer() else str(val)
        except (ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Error formatting KPI value for key '{key}'. Value: {val}")
            return "Error"

    avg_overall_tat_val = clinic_service_kpis_summary_data.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    overall_tat_status = "NO_DATA"; tat_target_val = settings.TARGET_TEST_TURNAROUND_DAYS
    if pd.notna(avg_overall_tat_val):
        if avg_overall_tat_val > (tat_target_val + 1.5): overall_tat_status = "HIGH_CONCERN"
        elif avg_overall_tat_val > tat_target_val: overall_tat_status = "MODERATE_CONCERN"
        else: overall_tat_status = "ACCEPTABLE"
    structured_kpis_list_main.append({
        "title": "Overall Avg. TAT (Conclusive)", "value_str": _get_kpi_val_main('overall_avg_test_turnaround_conclusive_days', precision=1),
        "units": "days", "icon": "⏱️", "status_level": overall_tat_status,
        "help_text": f"Average Turnaround Time for all conclusive diagnostic tests. Target ref: ~{tat_target_val} days."
    })

    perc_critical_tat_met_val = clinic_service_kpis_summary_data.get('perc_critical_tests_tat_met', np.nan)
    critical_tat_status = "NO_DATA"; target_perc_met_val = settings.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(perc_critical_tat_met_val):
        if perc_critical_tat_met_val >= target_perc_met_val: critical_tat_status = "GOOD_PERFORMANCE"
        elif perc_critical_tat_met_val >= target_perc_met_val * 0.80: critical_tat_status = "ACCEPTABLE"
        elif perc_critical_tat_met_val >= target_perc_met_val * 0.60: critical_tat_status = "MODERATE_CONCERN"
        else: critical_tat_status = "HIGH_CONCERN"
    structured_kpis_list_main.append({
        "title": "% Critical Tests TAT Met", "value_str": _get_kpi_val_main('perc_critical_tests_tat_met', precision=1),
        "units": "%", "icon": "🎯", "status_level": critical_tat_status,
        "help_text": f"Percentage of critical diagnostic tests meeting defined TAT targets. Target: ≥{target_perc_met_val}%."
    })

    num_pending_critical_val = clinic_service_kpis_summary_data.get('total_pending_critical_tests_patients', 0)
    pending_critical_status = "NO_DATA"
    if pd.notna(num_pending_critical_val): 
        count_val = int(num_pending_critical_val)
        if count_val == 0: pending_critical_status = "GOOD_PERFORMANCE"
        elif count_val <= 3: pending_critical_status = "ACCEPTABLE"
        elif count_val <= 10: pending_critical_status = "MODERATE_CONCERN"
        else: pending_critical_status = "HIGH_CONCERN"
    structured_kpis_list_main.append({
        "title": "Pending Critical Tests (Patients)", "value_str": _get_kpi_val_main('total_pending_critical_tests_patients', precision=0, is_count=True),
        "units": "patients", "icon": "⏳", "status_level": pending_critical_status,
        "help_text": "Number of unique patients with critical test results still pending. Target: 0."
    })

    rejection_rate_val = clinic_service_kpis_summary_data.get('sample_rejection_rate_perc', np.nan)
    rejection_rate_status = "NO_DATA"; target_rejection_pct = settings.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(rejection_rate_val):
        if rejection_rate_val > target_rejection_pct * 1.75: rejection_rate_status = "HIGH_CONCERN"
        elif rejection_rate_val > target_rejection_pct: rejection_rate_status = "MODERATE_CONCERN"
        else: rejection_rate_status = "GOOD_PERFORMANCE"
    structured_kpis_list_main.append({
        "title": "Sample Rejection Rate", "value_str": _get_kpi_val_main('sample_rejection_rate_perc', precision=1),
        "units":"%", "icon": "🧪", "status_level": rejection_rate_status,
        "help_text": f"Overall rate of laboratory samples rejected for testing. Target: < {target_rejection_pct}%."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis_list_main)} main clinic KPIs.")
    return structured_kpis_list_main


def structure_disease_specific_clinic_kpis(
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]],
    reporting_period_context_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs and key drug stockout counts.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for: {reporting_period_context_str}")
    
    structured_disease_kpis_list: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data provided. Returning empty list.")
        return structured_disease_kpis_list

    test_summary_details_data = clinic_service_kpis_summary_data.get("test_summary_details", {})
    if not isinstance(test_summary_details_data, dict):
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not a dict. Test KPIs cannot be structured.")
        test_summary_details_data = {}

    key_tests_config_disease = {
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_positivity_pct": 15.0, "disease_label_short": "TB"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_positivity_pct": settings.TARGET_MALARIA_POSITIVITY_RATE, "disease_label_short": "Malaria"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_positivity_pct": 5.0, "disease_label_short": "HIV"}
    }

    for test_disp_name_disease, props_disease in key_tests_config_disease.items():
        stats_disease = test_summary_details_data.get(test_disp_name_disease, {})
        pos_rate_disease = stats_disease.get("positive_rate_perc", np.nan)
        status_disease = "NO_DATA"; target_max_disease = props_disease.get("target_max_positivity_pct", 10.0)
        if pd.notna(pos_rate_disease):
            if pos_rate_disease > target_max_disease * 1.5: status_disease = "HIGH_CONCERN"
            elif pos_rate_disease > target_max_disease: status_disease = "MODERATE_CONCERN"
            else: status_disease = "ACCEPTABLE"
        
        structured_disease_kpis_list.append({
            "title": f"{props_disease['disease_label_short']} Positivity ({test_disp_name_disease})",
            "value_str": f"{pos_rate_disease:.1f}" if pd.notna(pos_rate_disease) else "N/A",
            "units":"%", "icon": props_disease["icon"], "status_level": status_disease,
            "help_text": f"Positivity rate for {test_disp_name_disease}. Target context: < {target_max_disease}%."
        })

    num_key_drug_stockouts = clinic_service_kpis_summary_data.get('key_drug_stockouts_count', 0)
    stockout_status_supply = "NO_DATA"
    if pd.notna(num_key_drug_stockouts):
        count_stockout = int(num_key_drug_stockouts)
        if count_stockout == 0: stockout_status_supply = "GOOD_PERFORMANCE"
        elif count_stockout <= 2: stockout_status_supply = "MODERATE_CONCERN"
        else: stockout_status_supply = "HIGH_CONCERN"
    
    structured_disease_kpis_list.append({
        "title": "Key Drug Stockouts",
        "value_str": str(int(num_key_drug_stockouts)) if pd.notna(num_key_drug_stockouts) else "N/A",
        "units":"items", "icon": "💊", "status_level": stockout_status_supply,
        "help_text": f"Number of key drugs/supplies with < {settings.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_disease_kpis_list)} disease-specific & supply KPIs.")
    return structured_disease_kpis_list
