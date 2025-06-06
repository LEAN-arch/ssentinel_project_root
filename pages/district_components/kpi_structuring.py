# sentinel_project_root/pages/district_components/kpi_structuring.py
# Structures district-level summary KPIs for Sentinel DHO dashboards.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from config import settings

logger = logging.getLogger(__name__)


def structure_district_summary_kpis(
    district_kpis_summary_input: Optional[Dict[str, Any]],
    district_enriched_zone_df_context: Optional[pd.DataFrame] = None, 
    reporting_period_context_str: str = "Latest Aggregated Data"
) -> List[Dict[str, Any]]:
    """
    Structures district summary KPIs into a list for display (e.g., via render_kpi_card).
    """
    module_log_prefix = "DistrictKPIStructure"
    logger.info(f"({module_log_prefix}) Structuring district summary KPIs for: {reporting_period_context_str}")
    
    structured_kpis: List[Dict[str, Any]] = []

    if not isinstance(district_kpis_summary_input, dict) or not district_kpis_summary_input:
        logger.warning(f"({module_log_prefix}) No district KPI summary data provided. Returning empty list.")
        return structured_kpis

    def _get_val(key: str, default: Any = np.nan, precision: Optional[int] = 1, is_count: bool = False) -> str:
        val = district_kpis_summary_input.get(key, default)
        if pd.isna(val): return "N/A"
        try:
            if is_count: return f"{int(val):,}"
            if precision is not None: return f"{float(val):.{precision}f}"
            return str(val)
        except (ValueError, TypeError): return "Error"

    total_zones_val = 0
    if isinstance(district_enriched_zone_df_context, pd.DataFrame) and 'zone_id' in district_enriched_zone_df_context.columns:
        total_zones_val = district_enriched_zone_df_context['zone_id'].nunique()
    elif 'total_zones_in_df' in district_kpis_summary_input:
        total_zones_val = district_kpis_summary_input['total_zones_in_df']
    structured_kpis.append({"title": "Total Operational Zones", "value_str": f"{int(total_zones_val):,}", "icon": "🗺️", "status_level": "NEUTRAL", "help_text": "Total number of defined operational zones in the district."})

    structured_kpis.append({"title": "Total District Population", "value_str": _get_val('total_population_district', precision=0, is_count=True), "icon": "👥", "status_level": "NEUTRAL", "help_text": "Estimated total population across all zones."})

    avg_risk_val = district_kpis_summary_input.get('population_weighted_avg_ai_risk_score', np.nan)
    risk_status = "NO_DATA"
    if pd.notna(avg_risk_val):
        if avg_risk_val >= settings.RISK_SCORE_HIGH_THRESHOLD: risk_status = "HIGH_CONCERN"
        elif avg_risk_val >= settings.RISK_SCORE_MODERATE_THRESHOLD: risk_status = "MODERATE_CONCERN"
        else: risk_status = "ACCEPTABLE"
    structured_kpis.append({"title": "Avg. AI Risk Score (Pop. Weighted)", "value_str": _get_val('population_weighted_avg_ai_risk_score', precision=1), "icon": "⚠️", "status_level": risk_status, "help_text": "Population-weighted average AI risk score across zones."})

    high_risk_zones_count = district_kpis_summary_input.get('zones_meeting_high_risk_criteria_count', 0)
    hr_zone_status = "NO_DATA"
    if pd.notna(high_risk_zones_count) and total_zones_val > 0:
        perc_hr_zones = (int(high_risk_zones_count) / total_zones_val) * 100
        if perc_hr_zones > 25: hr_zone_status = "HIGH_CONCERN"
        elif perc_hr_zones > 10: hr_zone_status = "MODERATE_CONCERN"
        else: hr_zone_status = "ACCEPTABLE"
    elif pd.notna(high_risk_zones_count) and int(high_risk_zones_count) == 0:
        hr_zone_status = "GOOD_PERFORMANCE"
    structured_kpis.append({"title": "High-Risk Zones Count", "value_str": _get_val('zones_meeting_high_risk_criteria_count', precision=0, is_count=True), "icon": "🚩", "status_level": hr_zone_status, "help_text": f"Number of zones with average AI risk score ≥ {settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."})
    
    key_dis_prev = district_kpis_summary_input.get('district_overall_key_disease_prevalence_per_1000', np.nan)
    prev_status = "NO_DATA"
    target_prev_1k = 50.0 
    if pd.notna(key_dis_prev):
        if key_dis_prev > target_prev_1k * 1.5: prev_status = "HIGH_CONCERN"
        elif key_dis_prev > target_prev_1k : prev_status = "MODERATE_CONCERN"
        else: prev_status = "ACCEPTABLE"
    structured_kpis.append({"title": "Key Disease Prevalence (/1k Pop.)", "value_str": _get_val('district_overall_key_disease_prevalence_per_1000', precision=1), "icon": "🔬", "status_level": prev_status, "help_text": "Combined prevalence of key monitored diseases per 1,000 population."})

    # CORRECTED: Instead of looping and creating many KPIs, create one KPI for the top condition.
    top_cond_name = district_kpis_summary_input.get("top_active_condition_name", "N/A")
    top_cond_count = district_kpis_summary_input.get("top_active_condition_count", 0)
    
    if top_cond_name != "N/A":
        top_cond_status = "ACCEPTABLE"
        if top_cond_count > 50: top_cond_status = "HIGH_CONCERN" # Example thresholds
        elif top_cond_count > 10: top_cond_status = "MODERATE_CONCERN"
        
        structured_kpis.append({
            "title": f"Top Active Condition ({top_cond_name})", 
            "value_str": f"{top_cond_count:,}",
            "icon": "🤒", 
            "status_level": top_cond_status,
            "help_text": f"The most prevalent key condition currently active across the district."
        })

    logger.info(f"({module_log_prefix}) Structured {len(structured_kpis)} district summary KPIs.")
    return structured_kpis
