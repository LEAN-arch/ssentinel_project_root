# sentinel_project_root/pages/clinic_components/testing_insights.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime # Added datetime

try:
    from config import settings
    from data_processing.aggregation import get_trend_data # Ensure this is robust
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in testing_insights.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_INSIGHTS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_INSIGHTS_PATTERN = r'^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_INSIGHTS if s) + r')$' if COMMON_NA_STRINGS_INSIGHTS else None

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


def _prepare_testing_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str,
    default_patient_id_prefix: str
) -> pd.DataFrame:
    """Prepares the DataFrame for testing insights analysis."""
    df_prepared = df.copy()
    for col_name, config in cols_config.items():
        default_value = config.get("default")
        target_type_str = config.get("type")

        if col_name not in df_prepared.columns:
            if col_name == 'patient_id': default_value = default_patient_id_prefix
            
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            elif isinstance(default_value, (list, dict)): 
                 df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value
        
        current_col_dtype = df_prepared[col_name].dtype
        if target_type_str in [float, "datetime"] and pd.api.types.is_object_dtype(current_col_dtype):
            if NA_REGEX_INSIGHTS_PATTERN:
                try:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_INSIGHTS_PATTERN, np.nan, regex=True)
                except Exception as e_regex:
                     logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value)
            # No explicit int conversion here, float is generally fine for these metrics
            elif target_type_str == str:
                df_prepared[col_name] = df_prepared[col_name].fillna(str(default_value)).astype(str)
                if NA_REGEX_INSIGHTS_PATTERN:
                    df_prepared[col_name] = df_prepared[col_name].replace(NA_REGEX_INSIGHTS_PATTERN, str(default_value), regex=True)
                df_prepared[col_name] = df_prepared[col_name].str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', default_patient_id_prefix).fillna(default_patient_id_prefix)
    return df_prepared


def prepare_clinic_lab_testing_insights_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    clinic_overall_kpis_summary: Optional[Dict[str, Any]], # From aggregation.py
    reporting_period_context_str: str,
    focus_test_group_display_name: str = "All Critical Tests Summary" # Allows focusing on a specific group
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including summaries, trends,
    overdue tests, and rejection reasons.
    """
    module_log_prefix = "ClinicTestInsightsPrep"
    logger.info(f"({module_log_prefix}) Preparing testing insights. Focus: '{focus_test_group_display_name}', Period: {reporting_period_context_str}")

    # Define default column names for output DataFrames for schema consistency
    default_crit_summary_cols = ["Test Group (Critical)", "Positivity (%)", "Avg. TAT (Days)", "% Met TAT Target", "Pending (Patients)", "Rejected (Patients)", "Total Conclusive Tests"]
    default_overdue_cols = ['patient_id', 'test_type', 'Sample Collection/Registered Date', 'days_pending', 'overdue_threshold_days', 'condition']
    default_rejection_cols = ['Rejection Reason', 'Count', 'Percentage (%)']
    default_rejected_examples_cols = ['patient_id', 'test_type', 'sample_collection_date', 'encounter_date', 'rejection_reason_clean', 'condition']

    insights_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str, 
        "selected_focus_area": focus_test_group_display_name,
        "all_critical_tests_summary_table_df": pd.DataFrame(columns=default_crit_summary_cols),
        "focused_test_group_kpis_dict": None, # Will store dict of KPIs for the focused group
        "focused_test_group_tat_trend_series": pd.Series(dtype='float64'),
        "focused_test_group_volume_trend_df": pd.DataFrame(columns=['date', 'Conclusive Tests', 'Pending Tests']), # Date index
        "overdue_pending_tests_list_df": pd.DataFrame(columns=default_overdue_cols),
        "sample_rejection_reasons_summary_df": pd.DataFrame(columns=default_rejection_cols),
        "top_rejected_samples_examples_df": pd.DataFrame(columns=default_rejected_examples_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note = "No health data provided for testing insights preparation. All outputs will be default/empty."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        return insights_output
    
    # Ensure clinic_overall_kpis_summary and its nested structure are valid
    if not isinstance(clinic_overall_kpis_summary, dict) or \
       not isinstance(clinic_overall_kpis_summary.get("test_summary_details"), dict):
        note = "Clinic overall KPI summary or 'test_summary_details' is missing or has an invalid format. Aggregated test group metrics will be unavailable. Test trends will be calculated from raw data if possible."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        # Initialize to prevent errors, but data will be missing for summary table
        clinic_overall_kpis_summary = {"test_summary_details": {}} 
            
    # --- Data Preparation ---
    pid_prefix_insights = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15]
    insights_cols_cfg = {
        'test_type': {"default": "UnknownTest_Insights", "type": str}, 
        'test_result': {"default": "UnknownResult_Insights", "type": str},
        'sample_status': {"default": "UnknownStatus_Insights", "type": str}, 
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Primary date for trends
        'test_turnaround_days': {"default": np.nan, "type": float}, 
        'patient_id': {"default": f"UPID_Ins_{pid_prefix_insights}", "type": str},
        'sample_collection_date': {"default": pd.NaT, "type": "datetime"}, 
        'sample_registered_lab_date': {"default": pd.NaT, "type": "datetime"},
        'rejection_reason': {"default": "NoReasonGiven", "type": str},
        'condition': {"default": "UnknownCondition", "type": str} # Added for context
    }
    df_tests_src = _prepare_testing_dataframe(
        filtered_health_df_for_clinic_period, insights_cols_cfg, 
        module_log_prefix, f"UPID_Ins_{pid_prefix_insights}"
    )
    df_tests_src.dropna(subset=['encounter_date', 'test_type', 'patient_id'], inplace=True) # Essential for most analyses
    if df_tests_src.empty:
        note = "No valid records with encounter_date, test_type, and patient_id after cleaning. Testing insights skipped."
        logger.warning(f"({module_log_prefix}) {note}")
        insights_output["processing_notes"].append(note)
        return insights_output
        
    # --- Critical Tests Summary Table (from pre-aggregated KPIs) ---
    test_summary_details_from_kpis = clinic_overall_kpis_summary.get("test_summary_details", {})
    key_test_configs_setting = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})

    if focus_test_group_display_name == "All Critical Tests Summary":
        critical_tests_summary_list: List[Dict[str, Any]] = []
        if test_summary_details_from_kpis and isinstance(key_test_configs_setting, dict):
            for internal_test_name, test_props_config in key_test_configs_setting.items():
                if isinstance(test_props_config, dict) and test_props_config.get("critical", False): # Check if test is marked critical
                    display_name = test_props_config.get("display_name", internal_test_name)
                    stats = test_summary_details_from_kpis.get(display_name, {}) # Get stats by display_name
                    critical_tests_summary_list.append({
                        "Test Group (Critical)": display_name, 
                        "Positivity (%)": stats.get("positive_rate_perc", np.nan),
                        "Avg. TAT (Days)": stats.get("avg_tat_days", np.nan), 
                        "% Met TAT Target": stats.get("perc_met_tat_target", np.nan),
                        "Pending (Patients)": stats.get("pending_count_patients", 0), 
                        "Rejected (Patients)": stats.get("rejected_count_patients", 0),
                        "Total Conclusive Tests": stats.get("total_conclusive_tests", 0)
                    })
            if critical_tests_summary_list:
                insights_output["all_critical_tests_summary_table_df"] = pd.DataFrame(critical_tests_summary_list)
            else: insights_output["processing_notes"].append("No critical test data found in summary, or no tests are configured as critical in KEY_TEST_TYPES_FOR_ANALYSIS.")
        else: insights_output["processing_notes"].append("'test_summary_details' from overall KPIs missing or KEY_TEST_TYPES_FOR_ANALYSIS not configured for 'All Critical Tests' summary.")
    
    # --- Focused Test Group KPIs & Trends (if a specific group is selected) ---
    elif focus_test_group_display_name and isinstance(key_test_configs_setting, dict):
        focused_test_internal_name = None
        focused_test_config = {}
        for internal_name, config_props in key_test_configs_setting.items():
            if isinstance(config_props, dict) and config_props.get("display_name") == focus_test_group_display_name:
                focused_test_internal_name = internal_name
                focused_test_config = config_props
                break
        
        if focused_test_internal_name and focused_test_config:
            # Populate KPIs from pre-aggregated summary if available
            if focus_test_group_display_name in test_summary_details_from_kpis:
                focused_stats = test_summary_details_from_kpis[focus_test_group_display_name]
                insights_output["focused_test_group_kpis_dict"] = {
                    "Positivity Rate (%)": focused_stats.get("positive_rate_perc", np.nan), 
                    "Avg. TAT (Days)": focused_stats.get("avg_tat_days", np.nan),
                    "% Met TAT Target": focused_stats.get("perc_met_tat_target", np.nan), 
                    "Pending Tests (Patients)": focused_stats.get("pending_count_patients", 0),
                    "Rejected Samples (Patients)": focused_stats.get("rejected_count_patients", 0), 
                    "Total Conclusive Tests": focused_stats.get("total_conclusive_tests", 0)
                }
            else:
                insights_output["processing_notes"].append(f"No pre-aggregated stats for selected focused test group: '{focus_test_group_display_name}'. Trends calculated from raw data if possible.")

            # Calculate Trends for the focused test group from df_tests_src
            # 'types_in_group' allows a display name to map to multiple raw test_type values
            test_types_in_focused_group = focused_test_config.get("types_in_group", [focused_test_internal_name])
            if isinstance(test_types_in_focused_group, str): # Ensure it's a list
                test_types_in_focused_group = [test_types_in_focused_group]

            df_focused_group_data = df_tests_src[df_tests_src['test_type'].isin(test_types_in_focused_group)].copy()

            if not df_focused_group_data.empty:
                # TAT Trend
                if 'test_turnaround_days' in df_focused_group_data.columns:
                    df_focus_tat_calc = df_focused_group_data[
                        df_focused_group_data['test_turnaround_days'].notna() &
                        ~df_focused_group_data.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(["pending", "unknownresult_insights", "rejected", "indeterminate"]) # Conclusive for TAT
                    ].copy()
                    if not df_focus_tat_calc.empty:
                        tat_trend_series = get_trend_data(df=df_focus_tat_calc, value_col='test_turnaround_days', date_col='encounter_date', period='D', agg_func='mean', source_context=f"{module_log_prefix}/TATTrend/{focus_test_group_display_name}")
                        if isinstance(tat_trend_series, pd.Series) and not tat_trend_series.empty: 
                            insights_output["focused_test_group_tat_trend_series"] = tat_trend_series
                
                # Volume Trend (Conclusive vs. Pending)
                conclusive_mask_volume = ~df_focused_group_data.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(["pending", "unknownresult_insights", "rejected", "indeterminate"])
                pending_mask_volume = df_focused_group_data.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending'
                
                conclusive_volume_trend = get_trend_data(df=df_focused_group_data[conclusive_mask_volume], value_col='patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Conclusive Tests")
                pending_volume_trend = get_trend_data(df=df_focused_group_data[pending_mask_volume], value_col='patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending Tests")
                
                volume_trends_to_concat = [s for s in [conclusive_volume_trend, pending_volume_trend] if isinstance(s, pd.Series) and not s.empty]
                if volume_trends_to_concat:
                    df_volume_trends_final = pd.concat(volume_trends_to_concat, axis=1).fillna(0).reset_index()
                    # Ensure 'encounter_date' from index is named 'date' or whatever the plot expects
                    date_col_name_from_index = df_volume_trends_final.columns[0] 
                    insights_output["focused_test_group_volume_trend_df"] = df_volume_trends_final.rename(columns={date_col_name_from_index: 'date'})
            else:
                 insights_output["processing_notes"].append(f"No raw data found for test types in group '{focus_test_group_display_name}' for trend calculation.")
        else: insights_output["processing_notes"].append(f"Configuration key for focused test group '{focus_test_group_display_name}' not found in settings.KEY_TEST_TYPES_FOR_ANALYSIS.")
    else: # Focus group name not specified or not found in KPI summary (and not "All Critical")
        insights_output["processing_notes"].append(f"No specific data or configuration found for focused test group: '{focus_test_group_display_name}'.")


    # --- Overdue Pending Tests ---
    # Determine the most relevant date column for calculating "pending since"
    # Prioritize sample_collection_date, then sample_registered_lab_date, then encounter_date
    date_col_for_overdue_calc = 'encounter_date' # Fallback
    if 'sample_collection_date' in df_tests_src.columns and df_tests_src['sample_collection_date'].notna().any():
        date_col_for_overdue_calc = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_tests_src.columns and df_tests_src['sample_registered_lab_date'].notna().any():
        date_col_for_overdue_calc = 'sample_registered_lab_date'
    
    df_pending_tests_raw = df_tests_src[
        (df_tests_src.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending') & 
        (df_tests_src[date_col_for_overdue_calc].notna())
    ].copy()

    if not df_pending_tests_raw.empty:
        # Ensure the date column is datetime
        df_pending_tests_raw[date_col_for_overdue_calc] = pd.to_datetime(df_pending_tests_raw[date_col_for_overdue_calc], errors='coerce')
        df_pending_tests_raw.dropna(subset=[date_col_for_overdue_calc], inplace=True) # Remove if date became NaT

        if not df_pending_tests_raw.empty:
            current_processing_date = pd.Timestamp('now').normalize() # Today's date for "days pending" calculation
            df_pending_tests_raw['days_pending'] = (current_processing_date - df_pending_tests_raw[date_col_for_overdue_calc]).dt.days
            
            def get_overdue_threshold_days(test_type_str: str) -> int:
                test_config = key_test_configs_setting.get(test_type_str, {}) # Check internal name first
                if not test_config: # Try finding by display name if internal name not found
                    test_config = next((cfg for cfg in key_test_configs_setting.values() if isinstance(cfg,dict) and cfg.get("display_name") == test_type_str),{})

                target_tat = test_config.get('target_tat_days', _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2))
                buffer_days = _get_setting('OVERDUE_TEST_BUFFER_DAYS', 2) # e.g., 2 days buffer
                overdue_fallback = _get_setting('OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK', 7)
                
                threshold = overdue_fallback # Default if target_tat is not sensible
                if pd.notna(target_tat) and float(target_tat) > 0:
                    threshold = int(float(target_tat) + buffer_days)
                return max(1, threshold) # Ensure at least 1 day threshold
            
            df_pending_tests_raw['overdue_threshold_days'] = df_pending_tests_raw['test_type'].apply(get_overdue_threshold_days)
            
            df_actually_overdue = df_pending_tests_raw[df_pending_tests_raw['days_pending'] > df_pending_tests_raw['overdue_threshold_days']]
            
            if not df_actually_overdue.empty:
                df_overdue_display_prep = df_actually_overdue.rename(columns={date_col_for_overdue_calc: "Sample Collection/Registered Date"})
                # Ensure all default_overdue_cols are present, adding NaNs if not
                for col in default_overdue_cols:
                    if col not in df_overdue_display_prep.columns: df_overdue_display_prep[col] = np.nan
                insights_output["overdue_pending_tests_list_df"] = df_overdue_display_prep[default_overdue_cols].sort_values('days_pending', ascending=False).reset_index(drop=True)
            else: insights_output["processing_notes"].append("No tests currently pending longer than their target TAT + buffer days.")
        else: insights_output["processing_notes"].append("No valid pending tests with dates for overdue calculation after cleaning dates.")
    else: insights_output["processing_notes"].append("No tests with 'Pending' status found for overdue evaluation.")


    # --- Sample Rejection Analysis ---
    if 'sample_status' in df_tests_src.columns and 'rejection_reason' in df_tests_src.columns:
        df_rejected_samples = df_tests_src[df_tests_src.get('sample_status', pd.Series(dtype=str)).astype(str).str.lower() == 'rejected'].copy()
        if not df_rejected_samples.empty:
            # Clean rejection reasons: strip, handle NAs, group common unknowns
            df_rejected_samples['rejection_reason_clean'] = df_rejected_samples['rejection_reason'].astype(str).str.strip()
            df_rejected_samples.loc[df_rejected_samples['rejection_reason_clean'].isin(list(COMMON_NA_STRINGS_INSIGHTS) + ["UnknownReason_Insights", ""]), 'rejection_reason_clean'] = 'Unknown/Not Specified'
            
            rejection_summary = df_rejected_samples['rejection_reason_clean'].value_counts().reset_index()
            rejection_summary.columns = ['Rejection Reason', 'Count'] # Match default_rejection_cols
            total_rejections = rejection_summary['Count'].sum()
            if total_rejections > 0:
                rejection_summary['Percentage (%)'] = ((rejection_summary['Count'] / total_rejections) * 100).round(1)
            else:
                rejection_summary['Percentage (%)'] = 0.0
            insights_output["sample_rejection_reasons_summary_df"] = rejection_summary

            # Prepare example rejected samples
            cols_for_rejected_examples_display = []
            for col in default_rejected_examples_cols: # Ensure only existing columns are selected
                if col in df_rejected_samples.columns:
                    cols_for_rejected_examples_display.append(col)
                elif col == 'rejection_reason_clean' and 'rejection_reason' in df_rejected_samples.columns : # Use original if clean not directly there (should be)
                    cols_for_rejected_examples_display.append('rejection_reason')


            if cols_for_rejected_examples_display:
                insights_output["top_rejected_samples_examples_df"] = df_rejected_samples[cols_for_rejected_examples_display].head(15).reset_index(drop=True)
        else: insights_output["processing_notes"].append("No rejected samples recorded in this period for rejection analysis.")
    else: insights_output["processing_notes"].append("'sample_status' or 'rejection_reason' columns missing. Sample rejection analysis skipped.")
    
    logger.info(f"({module_log_prefix}) Clinic testing insights preparation finished. Number of notes: {len(insights_output['processing_notes'])}")
    return insights_output
