# sentinel_project_root/pages/district_components/intervention_planning.py
# Prepares data for identifying priority zones for intervention in Sentinel DHO dashboards.

import pandas as pd
import numpy as np
import logging
import re # For dynamic column name creation
from typing import Dict, Any, Optional, List, Callable

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def get_district_intervention_criteria_options(
    district_zone_df_sample_check: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Defines available intervention criteria based on settings and DataFrame columns.
    """
    module_log_prefix = "DistrictInterventionCriteria"
    all_criteria: Dict[str, Dict[str, Any]] = {
        f"High Avg. AI Risk (Zone Score ≥ {settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE})": {
            "lambda_func": lambda r: pd.to_numeric(r.get('avg_risk_score'), errors='coerce') >= settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE,
            "required_cols": ['avg_risk_score'], "description": "Zones with average AI patient risk score exceeding threshold."
        },
        f"Low Facility Coverage (< {settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%)": {
            "lambda_func": lambda r: pd.to_numeric(r.get('facility_coverage_score'), errors='coerce') < settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT,
            "required_cols": ['facility_coverage_score'], "description": "Zones with facility coverage below target."
        },
        f"High Avg. Clinic CO2 (≥ {settings.ALERT_AMBIENT_CO2_HIGH_PPM} ppm)": {
            "lambda_func": lambda r: pd.to_numeric(r.get('zone_avg_co2'), errors='coerce') >= settings.ALERT_AMBIENT_CO2_HIGH_PPM,
            "required_cols": ['zone_avg_co2'], "description": "Zones with high average clinic CO2 levels."
        },
        f"High Critical Test TAT (> {settings.TARGET_TEST_TURNAROUND_DAYS + 1} days avg)": {
            "lambda_func": lambda r: pd.to_numeric(r.get('avg_test_turnaround_critical'), errors='coerce') > (settings.TARGET_TEST_TURNAROUND_DAYS + 1),
            "required_cols": ['avg_test_turnaround_critical'], "description": f"Zones with critical test TAT > {settings.TARGET_TEST_TURNAROUND_DAYS + 1} days."
        }
    }
    # Default threshold for other conditions if not specified elsewhere
    default_other_cond_burden_thresh = settings.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS 

    for cond_key in settings.KEY_CONDITIONS_FOR_ACTION:
        col_name_cond = f"active_{re.sub(r'[^a-z0-9_]+', '_', cond_key.lower().replace('(severe)','').strip())}_cases"
        disp_cond_name = cond_key.replace("(Severe)", "").strip()
        # Example: Use a specific threshold for TB, default for others. This could be enhanced with a config dict.
        burden_thresh = settings.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS if "TB" in disp_cond_name.upper() else default_other_cond_burden_thresh
        
        crit_name = f"High {disp_cond_name} Burden (≥ {burden_thresh} cases)"
        if crit_name not in all_criteria: # Avoid overwriting if manually defined with more nuance
            all_criteria[crit_name] = {
                "lambda_func": lambda r, c=col_name_cond, t=burden_thresh: pd.to_numeric(r.get(c), errors='coerce') >= t,
                "required_cols": [col_name_cond], "description": f"Zones with ≥ {burden_thresh} active {disp_cond_name} cases."
            }
    
    if not isinstance(district_zone_df_sample_check, pd.DataFrame) or district_zone_df_sample_check.empty:
        logger.debug(f"({module_log_prefix}) No zone DF sample. Returning all defined criteria.")
        return all_criteria

    available_criteria: Dict[str, Dict[str, Any]] = {}
    for crit_name, crit_cfg in all_criteria.items():
        req_cols = crit_cfg["required_cols"]
        if all(c in district_zone_df_sample_check.columns for c in req_cols) and \
           all(district_zone_df_sample_check[c].notna().any() for c in req_cols):
            available_criteria[crit_name] = crit_cfg
        else:
            logger.debug(f"({module_log_prefix}) Criterion '{crit_name}' excluded: required cols {req_cols} missing/all NaN in sample.")
            
    if not available_criteria: logger.warning(f"({module_log_prefix}) No intervention criteria available after checking DF sample.")
    return available_criteria


def identify_priority_zones_for_intervention_planning(
    enriched_district_zone_df: Optional[pd.DataFrame],
    selected_criteria_display_names_list: List[str],
    available_intervention_criteria_config: Dict[str, Dict[str, Any]],
    reporting_period_context_str: str = "Latest Aggregated Data"
) -> Dict[str, Any]:
    """
    Identifies priority zones based on selected intervention criteria.
    """
    module_log_prefix = "DistrictInterventionPlanPrep"
    logger.info(f"({module_log_prefix}) Identifying priority zones for: {reporting_period_context_str} using criteria: {selected_criteria_display_names_list}")
    
    output_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str, "applied_criteria_display_names": [],
        "priority_zones_for_intervention_df": pd.DataFrame(), "data_availability_notes": []
    }

    if not isinstance(enriched_district_zone_df, pd.DataFrame) or enriched_district_zone_df.empty:
        note = "Enriched District Zone DF missing/empty. Cannot plan intervention."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
        return output_data
    if not selected_criteria_display_names_list:
        note = "No intervention criteria selected. No zones will be flagged."
        logger.info(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
        return output_data
    
    overall_flag_mask = pd.Series([False] * len(enriched_district_zone_df), index=enriched_district_zone_df.index)
    applied_criteria_names: List[str] = []
    zone_flag_reasons: Dict[Any, List[str]] = {idx: [] for idx in enriched_district_zone_df.index}

    for crit_disp_name in selected_criteria_display_names_list:
        crit_cfg = available_intervention_criteria_config.get(crit_disp_name)
        if not crit_cfg or 'lambda_func' not in crit_cfg or 'required_cols' not in crit_cfg:
            logger.warning(f"({module_log_prefix}) Config invalid for criterion: '{crit_disp_name}'. Skipping.")
            output_data["data_availability_notes"].append(f"Invalid config for criterion: {crit_disp_name}.")
            continue
        
        req_cols = crit_cfg['required_cols']
        if not all(c in enriched_district_zone_df.columns for c in req_cols):
            note = f"Criterion '{crit_disp_name}' skipped: required cols {req_cols} not in zone DF."
            logger.warning(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
            continue
        
        try:
            current_mask = enriched_district_zone_df.apply(crit_cfg['lambda_func'], axis=1)
            if isinstance(current_mask, pd.Series) and current_mask.dtype == bool:
                overall_flag_mask |= current_mask.fillna(False)
                applied_criteria_names.append(crit_disp_name)
                for flagged_idx in enriched_district_zone_df.index[current_mask.fillna(False)]:
                    if flagged_idx in zone_flag_reasons: zone_flag_reasons[flagged_idx].append(crit_disp_name)
            else:
                logger.warning(f"({module_log_prefix}) Criterion '{crit_disp_name}' lambda did not return boolean Series. Type: {type(current_mask)}. Skipping.")
                output_data["data_availability_notes"].append(f"Invalid output from criterion logic: {crit_disp_name}.")
        except Exception as e_lambda:
            logger.error(f"({module_log_prefix}) Error applying lambda for '{crit_disp_name}': {e_lambda}", exc_info=True)
            output_data["data_availability_notes"].append(f"Error processing criterion: {crit_disp_name}.")

    output_data["applied_criteria_display_names"] = applied_criteria_names
    df_priority_zones = enriched_district_zone_df[overall_flag_mask].copy()

    if not df_priority_zones.empty:
        df_priority_zones['flagging_reasons_summary_text'] = df_priority_zones.index.map(
            lambda idx: "; ".join(zone_flag_reasons.get(idx, ["Unknown Reason"]))
        )
        display_cols_interv = ['name', 'population', 'avg_risk_score', 'flagging_reasons_summary_text']
        for crit_name_app in applied_criteria_names:
            crit_details_app = available_intervention_criteria_config.get(crit_name_app)
            if crit_details_app and 'required_cols' in crit_details_app:
                for req_col_app in crit_details_app['required_cols']:
                    if req_col_app not in display_cols_interv and req_col_app in df_priority_zones.columns:
                        display_cols_interv.append(req_col_app)
        
        final_display_cols = [c for c in display_cols_interv if c in df_priority_zones.columns]
        sort_col_interv = next((c for c in ['avg_risk_score', 'population'] if c in final_display_cols), None)
        
        if sort_col_interv:
            output_data["priority_zones_for_intervention_df"] = df_priority_zones[final_display_cols].sort_values(by=sort_col_interv, ascending=False)
        else:
            output_data["priority_zones_for_intervention_df"] = df_priority_zones[final_display_cols]
        logger.info(f"({module_log_prefix}) Identified {len(df_priority_zones)} priority zones using: {applied_criteria_names}")
    else:
        note = "No zones meet selected intervention criteria based on available data."
        logger.info(f"({module_log_prefix}) {note}"); output_data["data_availability_notes"].append(note)
        example_cols_empty_interv = ['name', 'population', 'avg_risk_score', 'flagging_reasons_summary_text']
        output_data["priority_zones_for_intervention_df"] = pd.DataFrame(columns=example_cols_empty_interv)

    return output_data
