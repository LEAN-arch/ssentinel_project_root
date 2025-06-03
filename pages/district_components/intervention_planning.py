# sentinel_project_root/pages/district_components/intervention_planning.py
# Prepares data for identifying priority zones for intervention in Sentinel DHO dashboards.
# Renamed from intervention_data_preparer_district.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable # Callable for lambda functions

from config import settings # Use new settings module
from data_processing.helpers import convert_to_numeric # For ensuring numeric types

logger = logging.getLogger(__name__)


def get_district_intervention_criteria_options( # Renamed function
    district_zone_df_sample_check: Optional[pd.DataFrame] = None # Sample of enriched zone DF
) -> Dict[str, Dict[str, Any]]:
    """
    Defines and returns available intervention criteria based on app settings and DataFrame columns.
    Each criterion includes a display name, a lambda function for evaluation against the DataFrame,
    a list of required columns from the DataFrame for that lambda, and a description.

    Args:
        district_zone_df_sample_check: A small sample (e.g., .head(2)) of the enriched zone DataFrame.
                                       Used to validate column existence and non-null data.

    Returns:
        Dict[str, Dict[str, Any]]: Configuration for available intervention criteria.
            Format: {Display Name: {"lambda_func": Callable, 
                                    "required_cols": List[str], 
                                    "description": str}}
    """
    module_log_prefix = "DistrictInterventionCriteria"
    
    # Define all potential intervention criteria.
    # Column names in 'required_cols' MUST match those produced by data_processing.enrichment.enrich_zone_geodata_with_health_aggregates.
    # Lambda functions should handle NaNs gracefully if possible (e.g., by using .get(col, pd.Series(dtype=float)) or pre-filtering NaNs).
    all_criteria_definitions: Dict[str, Dict[str, Any]] = {
        f"High Avg. AI Risk (Zone Score ≥ {settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE})": {
            "lambda_func": lambda df_row: pd.to_numeric(df_row.get('avg_risk_score'), errors='coerce') >= settings.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE,
            "required_cols": ['avg_risk_score'],
            "description": "Zones where the average AI-calculated patient risk score meets or exceeds the district high-risk threshold."
        },
        f"Low Facility Coverage (< {settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%)": {
            "lambda_func": lambda df_row: pd.to_numeric(df_row.get('facility_coverage_score'), errors='coerce') < settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT,
            "required_cols": ['facility_coverage_score'], # This column needs to be calculated during enrichment
            "description": "Zones with a facility coverage score below the district's minimum acceptable target."
        },
        f"High Avg. Clinic CO2 (Ventilation Risk ≥ {settings.ALERT_AMBIENT_CO2_HIGH_PPM} ppm)": {
            "lambda_func": lambda df_row: pd.to_numeric(df_row.get('zone_avg_co2'), errors='coerce') >= settings.ALERT_AMBIENT_CO2_HIGH_PPM,
            "required_cols": ['zone_avg_co2'], # From IoT aggregation during enrichment
            "description": "Zones where average clinic CO2 levels suggest potential ventilation issues or overcrowding."
        },
        f"High Critical Test TAT (> {settings.TARGET_TEST_TURNAROUND_DAYS + 1} days avg)": { # Example: target + 1 day buffer
            "lambda_func": lambda df_row: pd.to_numeric(df_row.get('avg_test_turnaround_critical'), errors='coerce') > (settings.TARGET_TEST_TURNAROUND_DAYS + 1),
            "required_cols": ['avg_test_turnaround_critical'], # From health data aggregation during enrichment
            "description": f"Zones where average turnaround time for critical tests exceeds target by more than 1 day."
        },
        # Placeholder for CHW Density - requires 'chw_density_per_10k' to be calculated in enrichment
        # "Low CHW Density (< 2 CHW per 10k Pop.)": { # Example target, make configurable
        #     "lambda_func": lambda df_row: pd.to_numeric(df_row.get('chw_density_per_10k'), errors='coerce') < 2.0,
        #     "required_cols": ['chw_density_per_10k'], 
        #     "description": "Zones with CHW coverage below 2 CHWs per 10,000 population (example target)."
        # }
    }

    # Dynamically add criteria for high burden of other key conditions from settings.KEY_CONDITIONS_FOR_ACTION
    for condition_key_interv_cfg in settings.KEY_CONDITIONS_FOR_ACTION:
        condition_col_name_interv_metric = f"active_{condition_key_interv_cfg.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
        display_condition_name_for_interv = condition_key_interv_cfg.replace("(Severe)", "").strip()
        
        # Use a specific burden threshold if defined for this condition, else use a general one (e.g., based on TB threshold)
        # This part can be made more sophisticated with condition-specific thresholds in settings.py
        burden_threshold_for_this_condition = settings.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS # Default to TB's
        # Example: if "TB" in display_condition_name_for_interv, use its specific threshold.
        # Otherwise, could use a fraction or a different configured value.
        # For simplicity, we'll use the TB threshold for all for now if not TB itself to show concept.
        # A better approach would be a dict in settings: CONDITION_BURDEN_THRESHOLDS = {"TB": 10, "Malaria": 50, ...}
        
        criterion_display_name = f"High {display_condition_name_for_interv} Burden (≥ {burden_threshold_for_this_condition} cases)"
        
        # Avoid re-adding if a very similar key (e.g. for TB) already exists from static definitions
        if criterion_display_name not in all_criteria_definitions:
            all_criteria_definitions[criterion_display_name] = {
                # Need to handle lambda scoping carefully if col_name and thres are not passed as defaults
                "lambda_func": lambda df_row, col=condition_col_name_interv_metric, thres=burden_threshold_for_this_condition: \
                               pd.to_numeric(df_row.get(col), errors='coerce') >= thres,
                "required_cols": [condition_col_name_interv_metric],
                "description": f"Zones with a high number of active {display_condition_name_for_interv} cases (≥ {burden_threshold_for_this_condition})."
            }
    
    if not isinstance(district_zone_df_sample_check, pd.DataFrame) or district_zone_df_sample_check.empty:
        logger.debug(f"({module_log_prefix}) No zone DataFrame sample provided. Returning all defined criteria without column validation.")
        return all_criteria_definitions

    # Filter criteria: only include if all required columns exist in the DataFrame sample
    # AND each required column has at least one non-NaN value in the sample.
    available_criteria_for_intervention: Dict[str, Dict[str, Any]] = {}
    for crit_display_name, crit_config_details in all_criteria_definitions.items():
        required_cols_for_criterion = crit_config_details["required_cols"]
        
        cols_present_in_sample = all(col_req in district_zone_df_sample_check.columns for col_req in required_cols_for_criterion)
        cols_have_data_in_sample = cols_present_in_sample and \
                                   all(district_zone_df_sample_check[col_req].notna().any() for col_req in required_cols_for_criterion)
                                   
        if cols_present_in_sample and cols_have_data_in_sample:
            available_criteria_for_intervention[crit_display_name] = crit_config_details
        else:
            logger.debug(
                f"({module_log_prefix}) Intervention criterion '{crit_display_name}' excluded: "
                f"One or more required columns ({required_cols_for_criterion}) are missing from DataFrame sample "
                f"or contain only NaN values."
            )
    
    if not available_criteria_for_intervention:
        logger.warning(f"({module_log_prefix}) No intervention criteria found to be available after checking DataFrame sample.")
        
    return available_criteria_for_intervention


def identify_priority_zones_for_intervention_planning( # Renamed function
    enriched_district_zone_df: Optional[pd.DataFrame], # Enriched DataFrame (not GeoDataFrame)
    selected_criteria_display_names_list: List[str], # List of display names chosen by user from UI
    available_intervention_criteria_config: Dict[str, Dict[str, Any]], # Output from get_district_intervention_criteria_options
    reporting_period_context_str: str = "Latest Aggregated Data" # Renamed
) -> Dict[str, Any]:
    """
    Identifies priority zones based on selected intervention criteria applied to the enriched zone DataFrame.
    Zones meeting ANY of the selected criteria will be flagged.
    """
    module_log_prefix = "DistrictInterventionPlanPrep" # Renamed for clarity
    logger.info(
        f"({module_log_prefix}) Identifying priority zones for: {reporting_period_context_str} "
        f"using criteria: {selected_criteria_display_names_list}"
    )
    
    # Initialize output structure
    output_intervention_planning_data: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "applied_criteria_display_names": [], # List of display names of criteria that were successfully applied
        "priority_zones_for_intervention_df": pd.DataFrame(), # Default to empty DF
        "data_availability_notes": []
    }

    if not isinstance(enriched_district_zone_df, pd.DataFrame) or enriched_district_zone_df.empty:
        note_msg = "Enriched District Zone DataFrame is missing or empty. Cannot proceed with intervention planning."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        output_intervention_planning_data["data_availability_notes"].append(note_msg)
        return output_intervention_planning_data

    if not selected_criteria_display_names_list:
        note_msg = "No intervention criteria were selected by the user. No zones will be flagged."
        logger.info(f"({module_log_prefix}) {note_msg}")
        output_intervention_planning_data["data_availability_notes"].append(note_msg)
        return output_intervention_planning_data
    
    # Initialize a boolean Series for combining criteria (zones meeting ANY selected criterion)
    # This mask will be True for any zone that meets at least one selected criterion.
    overall_flagging_mask_for_zones = pd.Series([False] * len(enriched_district_zone_df), index=enriched_district_zone_df.index)
    successfully_applied_criteria_names: List[str] = []
    
    # Store reasons for flagging each zone (which criteria it met)
    # Key: zone index (from enriched_district_zone_df.index), Value: List of criteria names
    zone_flagging_reasons_map: Dict[Any, List[str]] = {
        zone_idx_val: [] for zone_idx_val in enriched_district_zone_df.index
    }

    for criterion_display_name_user_selected in selected_criteria_display_names_list:
        criterion_configuration_details = available_intervention_criteria_config.get(criterion_display_name_user_selected)
        
        if not criterion_configuration_details or \
           'lambda_func' not in criterion_configuration_details or \
           'required_cols' not in criterion_configuration_details:
            logger.warning(
                f"({module_log_prefix}) Configuration details missing or invalid for selected criterion: "
                f"'{criterion_display_name_user_selected}'. Skipping this criterion."
            )
            output_intervention_planning_data["data_availability_notes"].append(
                f"Invalid configuration for criterion: {criterion_display_name_user_selected}."
            )
            continue

        # Before applying lambda, ensure all its required columns are actually in the main DataFrame
        required_cols_for_this_criterion = criterion_configuration_details['required_cols']
        if not all(col_r in enriched_district_zone_df.columns for col_r in required_cols_for_this_criterion):
            note_msg = (f"Criterion '{criterion_display_name_user_selected}' skipped: "
                        f"One or more required columns ({required_cols_for_this_criterion}) not found in the provided zone DataFrame.")
            logger.warning(f"({module_log_prefix}) {note_msg}")
            output_intervention_planning_data["data_availability_notes"].append(note_msg)
            continue
        
        try:
            # Apply the lambda function row-wise to get a boolean mask for the current criterion
            # The lambda function in config is designed to take a row (pd.Series)
            current_criterion_mask_series = enriched_district_zone_df.apply(
                criterion_configuration_details['lambda_func'], axis=1
            )
            
            if isinstance(current_criterion_mask_series, pd.Series) and current_criterion_mask_series.dtype == bool:
                # Combine with OR logic: a zone is flagged if it meets *any* selected criterion
                overall_flagging_mask_for_zones |= current_criterion_mask_series.fillna(False)
                successfully_applied_criteria_names.append(criterion_display_name_user_selected)
                
                # Record this criterion as a reason for flagging for all relevant zones
                for flagged_zone_idx in enriched_district_zone_df.index[current_criterion_mask_series.fillna(False)]:
                    if flagged_zone_idx in zone_flagging_reasons_map: # Should always be true
                        zone_flagging_reasons_map[flagged_zone_idx].append(criterion_display_name_user_selected)
            else:
                logger.warning(
                    f"({module_log_prefix}) Criterion '{criterion_display_name_user_selected}' lambda function "
                    f"did not return a valid boolean Series. Type was: {type(current_criterion_mask_series)}. Skipping."
                )
                output_intervention_planning_data["data_availability_notes"].append(
                    f"Invalid output from criterion logic: {criterion_display_name_user_selected}."
                )
        except Exception as e_apply_criterion_lambda:
            logger.error(
                f"({module_log_prefix}) Error applying lambda for criterion '{criterion_display_name_user_selected}': "
                f"{e_apply_criterion_lambda}", exc_info=True
            )
            output_intervention_planning_data["data_availability_notes"].append(
                f"Error during processing of criterion: {criterion_display_name_user_selected}."
            )

    output_intervention_planning_data["applied_criteria_display_names"] = successfully_applied_criteria_names
    
    # Filter the DataFrame to get only the zones that met one or more criteria
    df_priority_zones_identified = enriched_district_zone_df[overall_flagging_mask_for_zones].copy() # Use .copy()

    if not df_priority_zones_identified.empty:
        # Add a column detailing the reasons (which criteria) for flagging each zone
        df_priority_zones_identified['flagging_reasons_summary_text'] = df_priority_zones_identified.index.map(
            lambda zone_idx_map_val: "; ".join(zone_flagging_reasons_map.get(zone_idx_map_val, ["Unknown Reason"]))
        )

        # Select relevant columns for display in the intervention planning table
        # Start with basic identifiers and add the actual metric values that triggered flagging.
        display_columns_for_intervention_table = ['name', 'population', 'avg_risk_score', 'flagging_reasons_summary_text'] 
        
        # Add columns related to the actually applied criteria to show their values
        for crit_name_applied_val in successfully_applied_criteria_names:
            crit_details_applied_val = available_intervention_criteria_config.get(crit_name_applied_val)
            if crit_details_applied_val and 'required_cols' in crit_details_applied_val:
                for req_col_name_applied in crit_details_applied_val['required_cols']:
                    if req_col_name_applied not in display_columns_for_intervention_table and \
                       req_col_name_applied in df_priority_zones_identified.columns:
                        display_columns_for_intervention_table.append(req_col_name_applied)
        
        # Ensure all selected display_columns_for_intervention_table actually exist before trying to select them (safety net)
        final_display_cols_for_intervention = [
            col_disp_final for col_disp_final in display_columns_for_intervention_table 
            if col_disp_final in df_priority_zones_identified.columns
        ]
        
        # Sort flagged zones, e.g., by average risk score (descending) or population
        # This makes the most "at-risk" (by one metric) zones appear first.
        sort_col_for_intervention_table = 'avg_risk_score' if 'avg_risk_score' in final_display_cols_for_intervention else \
                                          ('population' if 'population' in final_display_cols_for_intervention else None)
        
        sort_ascending_for_intervention = False # Default: higher risk or higher population first
        if sort_col_for_intervention_table == 'population': # For population, could be either, but often higher pop is higher priority
            sort_ascending_for_intervention = False 

        if sort_col_for_intervention_table:
             output_intervention_planning_data["priority_zones_for_intervention_df"] = df_priority_zones_identified[final_display_cols_for_intervention].sort_values(
                by=sort_col_for_intervention_table, ascending=sort_ascending_for_intervention
            )
        else: # No obvious sort column, just use the selected columns in their current order
            output_intervention_planning_data["priority_zones_for_intervention_df"] = df_priority_zones_identified[final_display_cols_for_intervention]
        
        logger.info(
            f"({module_log_prefix}) Identified {len(df_priority_zones_identified)} priority zones based on criteria: "
            f"{successfully_applied_criteria_names}"
        )
    else: # No zones met the combined criteria based on user selection
        note_msg = "No zones currently meet the selected combination of intervention criteria based on available data."
        logger.info(f"({module_log_prefix}) {note_msg}")
        output_intervention_planning_data["data_availability_notes"].append(note_msg)
        # Ensure priority_zones_for_intervention_df is an empty DF with some schema if no zones flagged
        # This helps UI components expecting a DataFrame.
        example_cols_if_empty_intervention = ['name', 'population', 'avg_risk_score', 'flagging_reasons_summary_text']
        output_intervention_planning_data["priority_zones_for_intervention_df"] = pd.DataFrame(columns=example_cols_if_empty_intervention)

    return output_intervention_planning_data
