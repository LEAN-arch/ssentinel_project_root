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
    
    structured_kpis: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data. Returning empty list.")
        return structured_kpis

    def _get_val(key: str, default: Any = np.nan, precision: Optional[int] = 1) -> str:
        val = clinic_service_kpis_summary_data.get(key, default)
        if pd.isna(val): return "N/A"
        try:
            if precision is not None: return f"{float(val):.{precision}f}"
            return str(int(val)) if isinstance(val, (int, float)) and float(val).is_integer() else str(val)
        except: return "Error"

    # Overall Avg. TAT
    avg_tat = clinic_service_kpis_summary_data.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    tat_status = "NO_DATA"; tat_target = settings.TARGET_TEST_TURNAROUND_DAYS
    if pd.notna(avg_tat):
        if avg_tat > (tat_target + 1.5): tat_status = "HIGH_CONCERN"
        elif avg_tat > tat_target: tat_status = "MODERATE_CONCERN"
        else: tat_status = "ACCEPTABLE"
    structured_kpis.append({"title": "Overall Avg. TAT (Conclusive)", "value_str": _get_val('overall_avg_test_turnaround_conclusive_days', precision=1),
                           "units": "days", "icon": "⏱️", "status_level": tat_status, "help_text": f"Avg. TAT for conclusive tests. Target ref: ~{tat_target} days."})

    # % Critical Tests TAT Met
    perc_crit_tat = clinic_service_kpis_summary_data.get('perc_critical_tests_tat_met', np.nan)
    crit_tat_status = "NO_DATA"; target_perc_met = settings.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY
    if pd.notna(perc_crit_tat):
        if perc_crit_tat >= target_perc_met: crit_tat_status = "GOOD_PERFORMANCE"
        elif perc_crit_tat >= target_perc_met * 0.80: crit_tat_status = "ACCEPTABLE"
        elif perc_crit_tat >= target_perc_met * 0.60: crit_tat_status = "MODERATE_CONCERN"
        else: crit_tat_status = "HIGH_CONCERN"
    structured_kpis.append({"title": "% Critical Tests TAT Met", "value_str": _get_val('perc_critical_tests_tat_met', precision=1),
                           "units": "%", "icon": "🎯", "status_level": crit_tat_status, "help_text": f"% critical tests meeting TAT. Target: ≥{target_perc_met}%."})

    # Pending Critical Tests (Patients)
    pending_crit = clinic_service_kpis_summary_data.get('total_pending_critical_tests_patients', 0)
    pending_status = "NO_DATA"
    if pd.notna(pending_crit):
        count = int(pending_crit)
        if count == 0: pending_status = "GOOD_PERFORMANCE"
        elif count <= 3: pending_status = "ACCEPTABLE"
        elif count <= 10: pending_status = "MODERATE_CONCERN"
        else: pending_status = "HIGH_CONCERN"
    structured_kpis.append({"title": "Pending Critical Tests (Patients)", "value_str": _get_val('total_pending_critical_tests_patients', precision=0),
                           "units": "patients", "icon": "⏳", "status_level": pending_status, "help_text": "Unique patients with critical test results pending. Target: 0."})

    # Sample Rejection Rate
    rejection_rate = clinic_service_kpis_summary_data.get('sample_rejection_rate_perc', np.nan)
    rejection_status = "NO_DATA"; target_reject_pct = settings.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY
    if pd.notna(rejection_rate):
        if rejection_rate > target_reject_pct * 1.75: rejection_status = "HIGH_CONCERN"
        elif rejection_rate > target_reject_pct: rejection_status = "MODERATE_CONCERN"
        else: rejection_status = "GOOD_PERFORMANCE"
    structured_kpis.append({"title": "Sample Rejection Rate", "value_str": _get_val('sample_rejection_rate_perc', precision=1),
                           "units":"%", "icon": "🧪", "status_level": rejection_status, "help_text": f"Overall lab sample rejection rate. Target: < {target_reject_pct}%."})
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis)} main clinic KPIs.")
    return structured_kpis


def structure_disease_specific_clinic_kpis(
    clinic_service_kpis_summary_data: Optional[Dict[str, Any]],
    reporting_period_context_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs and key drug stockout counts.
    """
    module_log_prefix = "ClinicDiseaseSupplyKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring disease-specific & supply KPIs for: {reporting_period_context_str}")
    
    structured_kpis: List[Dict[str, Any]] = []

    if not isinstance(clinic_service_kpis_summary_data, dict) or not clinic_service_kpis_summary_data:
        logger.warning(f"({module_log_prefix}) No clinic service KPI summary data. Returning empty list.")
        return structured_kpis

    test_summary_details = clinic_service_kpis_summary_data.get("test_summary_details", {})
    if not isinstance(test_summary_details, dict):
        logger.warning(f"({module_log_prefix}) 'test_summary_details' missing or not dict. Test KPIs cannot be structured.")
        test_summary_details = {}

    key_tests_positivity_config = { # Key = display_name from settings.KEY_TEST_TYPES_FOR_ANALYSIS
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert", {}).get("display_name", "TB GeneXpert"):
            {"icon": "🫁", "target_max_pos_pct": 15.0, "label_short": "TB"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria", {}).get("display_name", "Malaria RDT"):
            {"icon": "🦟", "target_max_pos_pct": settings.TARGET_MALARIA_POSITIVITY_RATE, "label_short": "Malaria"},
        settings.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid", {}).get("display_name", "HIV Rapid Test"):
            {"icon": "🩸", "target_max_pos_pct": 5.0, "label_short": "HIV"}
    }

    for test_disp_name, props in key_tests_positivity_config.items():
        stats = test_summary_details.get(test_disp_name, {})
        pos_rate = stats.get("positive_rate_perc", np.nan)
        status = "NO_DATA"; target_max = props.get("target_max_pos_pct", 10.0)
        if pd.notna(pos_rate):
            if pos_rate > target_max * 1.5: status = "HIGH_CONCERN"
            elif pos_rate > target_max: status = "MODERATE_CONCERN"
            else: status = "ACCEPTABLE"
        
        structured_kpis.append({
            "title": f"{props['label_short']} Positivity ({test_disp_name})",
            "value_str": f"{pos_rate:.1f}" if pd.notna(pos_rate) else "N/A",
            "units":"%", "icon": props["icon"], "status_level": status,
            "help_text": f"Positivity rate for {test_disp_name}. Target context: < {target_max}%."
        })

    # Key Drug Stockouts
    stockouts = clinic_service_kpis_summary_data.get('key_drug_stockouts_count', 0)
    stockout_status_val = "NO_DATA" # Renamed to avoid conflict
    if pd.notna(stockouts):
        count = int(stockouts)
        if count == 0: stockout_status_val = "GOOD_PERFORMANCE"
        elif count <= 2: stockout_status_val = "MODERATE_CONCERN"
        else: stockout_status_val = "HIGH_CONCERN"
    
    structured_kpis.append({
        "title": "Key Drug Stockouts",
        "value_str": str(int(stockouts)) if pd.notna(stockouts) else "N/A",
        "units":"items", "icon": "💊", "status_level": stockout_status_val,
        "help_text": f"Key drugs/supplies with < {settings.CRITICAL_SUPPLY_DAYS_REMAINING} days of stock. Target: 0."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis)} disease-specific & supply KPIs.")
    return structured_kpis
