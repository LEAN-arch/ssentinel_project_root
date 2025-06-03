# sentinel_project_root/pages/district_components/kpi_structuring.py
# Structures district-wide KPIs for Sentinel Health Co-Pilot DHO dashboards.
# Renamed from kpi_structurer_district.py

import pandas as pd # Not directly used for values, but for type hints
import numpy as np  # For np.nan
import logging
from typing import Dict, Any, List, Optional

from config import settings # Use new settings module

logger = logging.getLogger(__name__)


def structure_district_summary_kpis( # Renamed function
    district_kpis_summary_input: Optional[Dict[str, Any]], # Data from aggregation.get_district_summary_kpis
    district_enriched_zone_df_context: Optional[pd.DataFrame] = None, # Enriched zone DF for context (e.g., total zone count)
    reporting_period_context_str: str = "Latest Aggregated Data" # Renamed for clarity
) -> List[Dict[str, Any]]:
    """
    Structures district-wide KPIs from a summary dictionary into a list of KPI dictionaries,
    formatted for display (e.g., via visualization.ui_elements.render_kpi_card).

    Args:
        district_kpis_summary_input: Dict from `aggregation.get_district_summary_kpis`.
        district_enriched_zone_df_context: Optional. The enriched zone DataFrame used to calculate
                                           the summary, primarily for context like total zone count.
        reporting_period_context_str: String describing the reporting period (for contextual help text).

    Returns:
        List of structured KPI dictionaries.
    """
    module_log_prefix = "DistrictKPIStructure" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Structuring district KPIs for period: {reporting_period_context_str}")
    
    structured_district_kpis_output_list: List[Dict[str, Any]] = [] # Renamed for clarity

    if not isinstance(district_kpis_summary_input, dict) or not district_kpis_summary_input:
        logger.warning(f"({module_log_prefix}) No district overall KPI summary data provided. Returning empty list.")
        return structured_district_kpis_output_list

    # Helper to safely get and format values from the input summary dictionary
    def _get_formatted_kpi_value(key: str, default_val: Any = np.nan, precision: Optional[int] = 1) -> str:
        val_raw = district_kpis_summary_input.get(key, default_val)
        if pd.isna(val_raw):
            return "N/A"
        try:
            if precision is not None: # Format as float with specified precision
                return f"{float(val_raw):.{precision}f}"
            # If no precision, try to format as int if it's a whole number, else string
            return str(int(val_raw)) if isinstance(val_raw, (int, float)) and float(val_raw).is_integer() else str(val_raw)
        except (ValueError, TypeError):
            logger.warning(f"({module_log_prefix}) Could not format KPI value for key '{key}' (raw: {val_raw}). Returning 'Error'.")
            return "Error" # Fallback for unformattable values

    # Determine total number of zones for percentage calculations if DataFrame context is provided
    num_total_zones_in_district = 0
    if isinstance(district_enriched_zone_df_context, pd.DataFrame) and not district_enriched_zone_df_context.empty:
        if 'zone_id' in district_enriched_zone_df_context.columns: # Prefer unique zone_id count
            num_total_zones_in_district = district_enriched_zone_df_context['zone_id'].nunique()
        else: # Fallback to length of DataFrame if 'zone_id' is somehow missing
            num_total_zones_in_district = len(district_enriched_zone_df_context)
            logger.debug(f"({module_log_prefix}) 'zone_id' col not found in context DF, using DF length ({num_total_zones_in_district}) for total zone count.")
    
    # --- KPI Definition and Structuring Logic ---

    # 1. Avg. Population AI Risk Score (Weighted by zone population)
    avg_pop_ai_risk_val = district_kpis_summary_input.get('population_weighted_avg_ai_risk_score', np.nan)
    pop_ai_risk_status = "NO_DATA"
    if pd.notna(avg_pop_ai_risk_val):
        if avg_pop_ai_risk_val >= settings.RISK_SCORE_HIGH_THRESHOLD: pop_ai_risk_status = "HIGH_RISK"
        elif avg_pop_ai_risk_val >= settings.RISK_SCORE_MODERATE_THRESHOLD: pop_ai_risk_status = "MODERATE_RISK"
        else: pop_ai_risk_status = "ACCEPTABLE" # Low risk is acceptable
    structured_district_kpis_output_list.append({
        "title": "Avg. Population AI Risk", 
        "value_str": _get_formatted_kpi_value('population_weighted_avg_ai_risk_score', precision=1),
        "units": "score", "icon": "🎯", "status_level": pop_ai_risk_status, 
        "help_text": "Population-weighted average AI risk score across all zones in the district."
    })

    # 2. Facility Coverage Score (District Average, Population Weighted)
    avg_facility_coverage_val = district_kpis_summary_input.get('district_avg_facility_coverage_score', np.nan)
    facility_coverage_status = "NO_DATA"
    if pd.notna(avg_facility_coverage_val): # Thresholds for status (can be refined)
        if avg_facility_coverage_val >= 80: facility_coverage_status = "GOOD_PERFORMANCE" 
        elif avg_facility_coverage_val >= settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT: facility_coverage_status = "ACCEPTABLE" # Meeting minimum target
        elif avg_facility_coverage_val >= settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT * 0.75 : facility_coverage_status = "MODERATE_CONCERN"
        else: facility_coverage_status = "HIGH_CONCERN" # Significantly below target
    structured_district_kpis_output_list.append({
        "title": "Facility Coverage Score (Avg %)", 
        "value_str": _get_formatted_kpi_value('district_avg_facility_coverage_score', precision=1),
        "units": "%", "icon": "🏥", "status_level": facility_coverage_status, 
        "help_text": f"Population-weighted average facility coverage score. Target > {settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%."
    })

    # 3. High AI Risk Zones (Count and Percentage of total zones)
    count_high_risk_zones_val = district_kpis_summary_input.get('zones_meeting_high_risk_criteria_count', 0)
    percent_high_risk_zones_str = "N/A"
    high_risk_zones_status = "ACCEPTABLE" # Default if no high-risk zones
    
    if pd.notna(count_high_risk_zones_val):
        num_hr_zones_int = int(count_high_risk_zones_val)
        if num_total_zones_in_district > 0:
            percent_val = (num_hr_zones_int / num_total_zones_in_district) * 100
            percent_high_risk_zones_str = f"{percent_val:.0f}%"
            # Status based on percentage of zones being high risk
            if percent_val > 30: high_risk_zones_status = "HIGH_CONCERN" 
            elif num_hr_zones_int > 0: high_risk_zones_status = "MODERATE_CONCERN" # Any high-risk zone is a concern
        elif num_hr_zones_int > 0 : # Have count, but no total_zones to calculate percentage
             percent_high_risk_zones_str = "(% unavail.)" # Indicate percentage cannot be calculated
             high_risk_zones_status = "MODERATE_CONCERN" # Still a concern if any such zones exist
        # If num_hr_zones_int is 0, status remains ACCEPTABLE (default)

    value_display_hr_zones = f"{int(count_high_risk_zones_val) if pd.notna(count_high_risk_zones_val) else '0'} ({percent_high_risk_zones_str})"
    structured_district_kpis_output_list.append({
        "title": "High AI Risk Zones", 
        "value_str": value_display_hr_zones,
        "units": "zones", "icon": "⚠️", "status_level": high_risk_zones_status, 
        "help_text": f"Number (and percentage) of zones with average AI risk score ≥ {settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."
    })

    # 4. Overall Key Disease Prevalence per 1,000 Population (District-wide)
    district_prevalence_val = district_kpis_summary_input.get('district_overall_key_disease_prevalence_per_1000', np.nan)
    prevalence_status = "NO_DATA"
    if pd.notna(district_prevalence_val): # Example thresholds (highly context-dependent)
        if district_prevalence_val > 50: prevalence_status = "HIGH_CONCERN" 
        elif district_prevalence_val > 20: prevalence_status = "MODERATE_CONCERN"
        else: prevalence_status = "ACCEPTABLE"
    structured_district_kpis_output_list.append({
        "title": "Key Disease Prevalence", 
        "value_str": _get_formatted_kpi_value('district_overall_key_disease_prevalence_per_1000', precision=1),
        "units": "/1k pop", "icon": "📈", "status_level": prevalence_status, 
        "help_text": "Combined prevalence of specified key infectious diseases per 1,000 population across the district."
    })

    # 5. Dynamically add KPIs for total active cases of each key condition
    default_condition_kpi_icon = "🌡️" # Generic health icon
    condition_icon_map_simple = {"TB": "🫁", "Malaria": "🦟", "HIV": "🩸", "Pneumonia": "💨", "Sepsis": "☣️", "Dehydration": "💧", "Heat": "☀️"}

    for condition_key_cfg in settings.KEY_CONDITIONS_FOR_ACTION:
        # Construct the metric key name as used in aggregation.get_district_summary_kpis output
        metric_key_for_condition = f"district_total_active_{condition_key_cfg.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        total_active_cases_for_cond = district_kpis_summary_input.get(metric_key_for_condition, 0) # Default to 0
        
        condition_burden_status = "ACCEPTABLE" # Default
        if pd.notna(total_active_cases_for_cond):
            num_cases_cond_int = int(total_active_cases_for_cond)
            # Example thresholds (can be refined per condition in settings if needed)
            # Using TB burden as a general high threshold reference from config
            if num_cases_cond_int > settings.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS * 1.5: # Significantly high
                condition_burden_status = "HIGH_CONCERN" 
            elif num_cases_cond_int > settings.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS * 0.5: # Moderately concerning
                condition_burden_status = "MODERATE_CONCERN"
        
        icon_for_this_cond_kpi = default_condition_kpi_icon # Default icon
        for keyword, icon_char_map in condition_icon_map_simple.items():
            if keyword.lower() in condition_key_cfg.lower():
                icon_for_this_cond_kpi = icon_char_map
                break
        
        display_name_for_cond_kpi_title = condition_key_cfg.replace("(Severe)", "").strip() # Cleaner title for UI
        
        structured_district_kpis_output_list.append({
            "title": f"Total Active {display_name_for_cond_kpi_title} Cases", 
            "value_str": _get_formatted_kpi_value(metric_key_for_condition, precision=0), 
            "units": "cases", "icon": icon_for_this_cond_kpi, "status_level": condition_burden_status, 
            "help_text": f"Total active {display_name_for_cond_kpi_title} cases identified across the district."
        })

    # 6. District Avg Patient Daily Steps (Population Weighted Wellness Proxy)
    avg_district_steps_val = district_kpis_summary_input.get('district_population_weighted_avg_steps', np.nan)
    steps_status = "NO_DATA"
    if pd.notna(avg_district_steps_val):
        if avg_district_steps_val >= settings.TARGET_DAILY_STEPS * 0.85: steps_status = "GOOD_PERFORMANCE" # >=85% of target
        elif avg_district_steps_val >= settings.TARGET_DAILY_STEPS * 0.60: steps_status = "ACCEPTABLE" # 60-84%
        elif avg_district_steps_val >= settings.TARGET_DAILY_STEPS * 0.40: steps_status = "MODERATE_CONCERN" # 40-59%
        else: steps_status = "HIGH_CONCERN" # <40%
    structured_district_kpis_output_list.append({
        "title": "Avg. Patient Steps (Pop. Wt.)", 
        "value_str": _get_formatted_kpi_value('district_population_weighted_avg_steps', precision=0),
        "units": "steps/day", "icon": "👣", "status_level": steps_status, 
        "help_text": f"Population-weighted average daily steps from patient data. Target ref: {settings.TARGET_DAILY_STEPS:,.0f} steps."
    })
    
    # 7. District Avg Clinic CO2 Levels (Average of Zonal Means from IoT data)
    avg_district_clinic_co2_val = district_kpis_summary_input.get('district_avg_clinic_co2_ppm', np.nan)
    clinic_co2_status = "NO_DATA"
    if pd.notna(avg_district_clinic_co2_val):
        if avg_district_clinic_co2_val > settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM : clinic_co2_status = "HIGH_RISK" # Matches env alert levels
        elif avg_district_clinic_co2_val > settings.ALERT_AMBIENT_CO2_HIGH_PPM: clinic_co2_status = "MODERATE_RISK"
        else: clinic_co2_status = "ACCEPTABLE"
    structured_district_kpis_output_list.append({
        "title": "Avg. Clinic CO2 (District)", 
        "value_str": _get_formatted_kpi_value('district_avg_clinic_co2_ppm', precision=0),
        "units": "ppm", "icon": "💨", "status_level": clinic_co2_status, 
        "help_text": f"District average of zonal mean clinic CO2 levels. Aim for < {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm for good ventilation."
    })
    
    logger.info(f"({module_log_prefix}) Structured {len(structured_district_kpis_output_list)} district-level KPIs.")
    return structured_district_kpis_output_list
