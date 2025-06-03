# sentinel_project_root/pages/clinic_components/testing_insights.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.
# Renamed from testing_insights_analyzer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
from data_processing.aggregation import get_trend_data # For TAT/Volume trends
from data_processing.helpers import convert_to_numeric # For data cleaning

logger = logging.getLogger(__name__)


def prepare_clinic_lab_testing_insights_data( # Renamed function
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame], # Health data already filtered for clinic & period
    clinic_overall_kpis_summary: Optional[Dict[str, Any]], # From aggregation.get_clinic_summary_kpis, contains 'test_summary_details'
    reporting_period_context_str: str, # Renamed for clarity
    focus_test_group_display_name: str = "All Critical Tests Summary" # Renamed, default view
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including performance metrics
    for critical tests, trends for a focused test group (if selected), overdue tests,
    and sample rejection analysis.
    """
    module_log_prefix = "ClinicTestInsightsPrep" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Preparing testing insights. Focus: '{focus_test_group_display_name}', Period: {reporting_period_context_str}")

    # Initialize output structure with defaults for DataFrames to ensure consistent schema
    # Columns for these DFs should match what the UI components expect.
    default_critical_summary_cols = ["Test Group (Critical)", "Positivity (%)", "Avg. TAT (Days)",
                                     "% Met TAT Target", "Pending (Patients)", "Rejected (Patients)",
                                     "Total Conclusive Tests"]
    default_overdue_cols = ['patient_id', 'test_type', 'Sample Collection/Registered Date', 'days_pending', 'overdue_threshold_days']
    default_rejection_cols = ['Rejection Reason', 'Count']
    default_rejected_examples_cols = ['patient_id', 'test_type', 'sample_collection_date', 'encounter_date', 'rejection_reason_clean']

    insights_output_dict: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "selected_focus_area": focus_test_group_display_name,
        "all_critical_tests_summary_table_df": pd.DataFrame(columns=default_critical_summary_cols),
        "focused_test_group_kpis_dict": None,      # Dict of KPIs for a specific selected group
        "focused_test_group_tat_trend_series": pd.Series(dtype='float64'),  # Empty Series default
        "focused_test_group_volume_trend_df": pd.DataFrame(columns=['date', 'Conclusive Tests', 'Pending Tests']), # Date as first col
        "overdue_pending_tests_list_df": pd.DataFrame(columns=default_overdue_cols),
        "sample_rejection_reasons_summary_df": pd.DataFrame(columns=default_rejection_cols),
        "top_rejected_samples_examples_df": pd.DataFrame(columns=default_rejected_examples_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note_msg = "No health data provided for testing insights analysis. All outputs will be empty/default."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        insights_output_dict["processing_notes"].append(note_msg)
        return insights_output_dict
    
    # Validate clinic_overall_kpis_summary and its critical sub-dictionary 'test_summary_details'
    # This summary is used for aggregated views like "All Critical Tests".
    if not isinstance(clinic_overall_kpis_summary, dict) or \
       "test_summary_details" not in clinic_overall_kpis_summary or \
       not isinstance(clinic_overall_kpis_summary.get("test_summary_details"), dict):
        note_msg = ("Clinic overall KPI summary or its 'test_summary_details' is missing/invalid. "
                    "Aggregated metrics for test groups (e.g., 'All Critical Tests') will be unavailable.")
        logger.warning(f"({module_log_prefix}) {note_msg}")
        insights_output_dict["processing_notes"].append(note_msg)
        # Ensure test_summary_details exists as an empty dict for safe access later, even if flawed.
        if not isinstance(clinic_overall_kpis_summary, dict): clinic_overall_kpis_summary = {}
        if not isinstance(clinic_overall_kpis_summary.get("test_summary_details"), dict):
            clinic_overall_kpis_summary["test_summary_details"] = {} # Safe empty dict
            
    df_tests_src_cleaned = filtered_health_df_for_clinic_period.copy() # Work on a copy
    
    # --- Data Preparation for raw data analysis (overdue, rejection) ---
    # Ensure essential columns for these calculations exist with proper types/defaults
    # These are columns from the raw health_df that this component directly uses.
    test_insights_cols_config = {
        'test_type': {"default": "UnknownTest_Insights", "type": str},
        'test_result': {"default": "UnknownResult_Insights", "type": str},
        'sample_status': {"default": "UnknownStatus_Insights", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, # Primary date for trends
        'test_turnaround_days': {"default": np.nan, "type": float},
        'patient_id': {"default": f"UnknownPID_Insights_{reporting_period_context_str[:10]}", "type": str},
        'sample_collection_date': {"default": pd.NaT, "type": "datetime"},
        'sample_registered_lab_date': {"default": pd.NaT, "type": "datetime"},
        'rejection_reason': {"default": "UnknownReason", "type": str}
    }
    common_na_insights_prep = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown']

    for col, config_item_insights in test_insights_cols_config.items():
        if col not in df_tests_src_cleaned.columns:
            df_tests_src_cleaned[col] = config_item_insights["default"]
        
        if config_item_insights["type"] == "datetime":
            df_tests_src_cleaned[col] = pd.to_datetime(df_tests_src_cleaned[col], errors='coerce')
        elif config_item_insights["type"] == float:
            df_tests_src_cleaned[col] = convert_to_numeric(df_tests_src_cleaned[col], default_value=config_item_insights["default"])
        elif config_item_insights["type"] == str:
            df_tests_src_cleaned[col] = df_tests_src_cleaned[col].astype(str).fillna(str(config_item_insights["default"]))
            df_tests_src_cleaned[col] = df_tests_src_cleaned[col].replace(common_na_insights_prep, str(config_item_insights["default"]), regex=False).str.strip()


    # --- A. Data for Selected Focus Area (All Critical or Specific Test Group) ---
    # This part primarily uses the pre-aggregated `clinic_overall_kpis_summary`.
    test_summary_details_map_kpis = clinic_overall_kpis_summary.get("test_summary_details", {})

    if focus_test_group_display_name == "All Critical Tests Summary":
        critical_tests_summary_for_table_list: List[Dict[str, Any]] = []
        if test_summary_details_map_kpis: # Only proceed if detailed stats map is available
            for test_disp_name_iter, test_stats_iter in test_summary_details_map_kpis.items():
                # Determine if this test_disp_name_iter corresponds to a critical test in settings
                original_test_key_from_config = next(
                    (k_orig for k_orig, v_cfg in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
                     if v_cfg.get("display_name") == test_disp_name_iter), None
                )
                if original_test_key_from_config and settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_test_key_from_config, {}).get("critical"):
                    critical_tests_summary_for_table_list.append({
                        "Test Group (Critical)": test_disp_name_iter,
                        "Positivity (%)": test_stats_iter.get("positive_rate_perc", np.nan),
                        "Avg. TAT (Days)": test_stats_iter.get("avg_tat_days", np.nan),
                        "% Met TAT Target": test_stats_iter.get("perc_met_tat_target", np.nan),
                        "Pending (Patients)": test_stats_iter.get("pending_count_patients", 0),
                        "Rejected (Patients)": test_stats_iter.get("rejected_count_patients", 0), # Assuming this key exists from aggregation
                        "Total Conclusive Tests": test_stats_iter.get("total_conclusive_tests", 0)
                    })
            if critical_tests_summary_for_table_list:
                insights_output_dict["all_critical_tests_summary_table_df"] = pd.DataFrame(critical_tests_summary_for_table_list)
            else:
                insights_output_dict["processing_notes"].append("No data for critical tests found in summary, or no tests are configured as critical.")
        else:
            insights_output_dict["processing_notes"].append("Detailed test statistics map ('test_summary_details') missing for 'All Critical Tests' summary table.")

    elif focus_test_group_display_name in test_summary_details_map_kpis: # A specific test group is selected
        stats_for_focused_group = test_summary_details_map_kpis[focus_test_group_display_name]
        insights_output_dict["focused_test_group_kpis_dict"] = { # Store as a dict for easy UI access
            "Positivity Rate (%)": stats_for_focused_group.get("positive_rate_perc", np.nan),
            "Avg. TAT (Days)": stats_for_focused_group.get("avg_tat_days", np.nan),
            "% Met TAT Target": stats_for_focused_group.get("perc_met_tat_target", np.nan),
            "Pending Tests (Patients)": stats_for_focused_group.get("pending_count_patients", 0),
            "Rejected Samples (Patients)": stats_for_focused_group.get("rejected_count_patients", 0),
            "Total Conclusive Tests": stats_for_focused_group.get("total_conclusive_tests", 0)
        }
        
        # For trends (TAT, Volume), we need to use the raw `df_tests_src_cleaned`
        # Find the original config key for the selected display name to filter raw data
        original_key_for_focused_trend = next(
            (k_orig_f for k_orig_f, v_cfg_f in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
             if v_cfg_f.get("display_name") == focus_test_group_display_name), None
        )
        if original_key_for_focused_trend:
            # Some "display names" might aggregate multiple raw test_types. Handle this if defined in config.
            raw_test_keys_for_group_trend = settings.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_focused_trend].get("types_in_group", [original_key_for_focused_trend])
            if isinstance(raw_test_keys_for_group_trend, str): raw_test_keys_for_group_trend = [raw_test_keys_for_group_trend]

            # TAT Trend for the focused group (Daily average TAT of conclusive tests)
            if 'test_turnaround_days' in df_tests_src_cleaned.columns and 'encounter_date' in df_tests_src_cleaned.columns:
                df_focused_group_for_tat_trend = df_tests_src_cleaned[
                    (df_tests_src_cleaned['test_type'].isin(raw_test_keys_for_group_trend)) &
                    (df_tests_src_cleaned['test_turnaround_days'].notna()) & # Must have a TAT value
                    (~df_tests_src_cleaned.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin( # Conclusive results
                        ['pending','unknownresult_insights','rejected','indeterminate']))
                ].copy()
                if not df_focused_group_for_tat_trend.empty:
                    tat_trend_series_focused = get_trend_data(
                        df=df_focused_group_for_tat_trend, value_col='test_turnaround_days', 
                        date_col='encounter_date', period='D', agg_func='mean', 
                        source_context=f"{module_log_prefix}/TATTrend/{focus_test_group_display_name}"
                    )
                    insights_output_dict["focused_test_group_tat_trend_series"] = tat_trend_series_focused if isinstance(tat_trend_series_focused, pd.Series) and not tat_trend_series_focused.empty else pd.Series(dtype='float64')
            
            # Volume Trend (Daily: Conclusive vs. Pending) for the focused group
            if 'patient_id' in df_tests_src_cleaned.columns and 'encounter_date' in df_tests_src_cleaned.columns:
                df_focused_group_for_vol_trend = df_tests_src_cleaned[df_tests_src_cleaned['test_type'].isin(raw_test_keys_for_group_trend)].copy()
                if not df_focused_group_for_vol_trend.empty:
                    conclusive_mask_for_vol_trend = ~df_focused_group_for_vol_trend.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(
                                                        ['pending','unknownresult_insights','rejected','indeterminate'])
                    
                    series_conclusive_vol_trend = get_trend_data(
                        df=df_focused_group_for_vol_trend[conclusive_mask_for_vol_trend], value_col='patient_id', # Count unique patients or encounters
                        date_col='encounter_date', period='D', agg_func='count').rename("Conclusive Tests") # Using count for volume
                    
                    series_pending_vol_trend = get_trend_data(
                        df=df_focused_group_for_vol_trend[df_focused_group_for_vol_trend.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending'], 
                        value_col='patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending Tests")
                    
                    volume_trends_list_to_concat = [s for s in [series_conclusive_vol_trend, series_pending_vol_trend] if isinstance(s, pd.Series) and not s.empty]
                    if volume_trends_list_to_concat:
                        # Concatenate into a DataFrame, ensure date index becomes a column for plotting
                        df_vol_trend_concat = pd.concat(volume_trends_list_to_concat, axis=1).fillna(0).reset_index()
                        df_vol_trend_concat.rename(columns={'index': 'date', 'encounter_date': 'date'}, inplace=True) # Standardize date column name
                        insights_output_dict["focused_test_group_volume_trend_df"] = df_vol_trend_concat
        else: # Original config key not found for the selected display name
            insights_output_dict["processing_notes"].append(f"Could not find original configuration key for test group '{focus_test_group_display_name}' to generate its trends.")
    else: # Selected display name not in the summary map (should not happen if UI uses keys from map)
        insights_output_dict["processing_notes"].append(f"No detailed aggregated stats found in summary for selected test group: '{focus_test_group_display_name}'. Trends might be affected.")


    # --- B. Overdue Pending Tests (Calculated from raw period data: df_tests_src_cleaned) ---
    # Prioritize date for pending calculation: sample_collection_date > sample_registered_lab_date > encounter_date
    date_col_for_overdue_pending_calc = 'encounter_date' # Default fallback
    if 'sample_collection_date' in df_tests_src_cleaned.columns and df_tests_src_cleaned['sample_collection_date'].notna().any():
        date_col_for_overdue_pending_calc = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_tests_src_cleaned.columns and df_tests_src_cleaned['sample_registered_lab_date'].notna().any():
        date_col_for_overdue_pending_calc = 'sample_registered_lab_date'
    
    df_pending_tests_for_overdue = df_tests_src_cleaned[
        (df_tests_src_cleaned.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending') & 
        (df_tests_src_cleaned[date_col_for_overdue_pending_calc].notna()) # Must have a valid date
    ].copy()

    if not df_pending_tests_for_overdue.empty:
        # Ensure the chosen date column is indeed datetime (should be from prep, but double check)
        df_pending_tests_for_overdue[date_col_for_overdue_pending_calc] = pd.to_datetime(df_pending_tests_for_overdue[date_col_for_overdue_pending_calc], errors='coerce')
        df_pending_tests_for_overdue.dropna(subset=[date_col_for_overdue_pending_calc], inplace=True)

        if not df_pending_tests_for_overdue.empty:
            current_processing_date_ts = pd.Timestamp('now').normalize() # Today at midnight for consistent "days_pending"
            df_pending_tests_for_overdue['days_pending'] = (current_processing_date_ts - df_pending_tests_for_overdue[date_col_for_overdue_pending_calc]).dt.days
            
            # Helper to get specific TAT target for a test type from settings
            def get_overdue_threshold_for_test(test_type_str: str) -> int:
                test_config_details = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_str)
                buffer_days_for_overdue = 2 # Allowable buffer beyond target TAT before flagging as "overdue"
                target_tat_days = settings.TARGET_TEST_TURNAROUND_DAYS # General default TAT
                if test_config_details and 'target_tat_days' in test_config_details and pd.notna(test_config_details['target_tat_days']):
                    target_tat_days = test_config_details['target_tat_days']
                
                # Use a general fallback if target_tat_days is not sensible
                overdue_thresh = int(target_tat_days + buffer_days_for_overdue) if pd.notna(target_tat_days) and target_tat_days > 0 else \
                                 int(settings.OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK + buffer_days_for_overdue)
                return max(1, overdue_thresh) # Ensure threshold is at least 1 day
            
            df_pending_tests_for_overdue['overdue_threshold_days'] = df_pending_tests_for_overdue['test_type'].apply(get_overdue_threshold_for_test)
            
            df_actually_overdue_tests = df_pending_tests_for_overdue[
                df_pending_tests_for_overdue['days_pending'] > df_pending_tests_for_overdue['overdue_threshold_days']
            ]
            
            if not df_actually_overdue_tests.empty:
                # Select and rename columns for display in the UI table
                cols_to_display_for_overdue = ['patient_id', 'test_type', date_col_for_overdue_pending_calc, 'days_pending', 'overdue_threshold_days']
                df_overdue_for_display = df_actually_overdue_tests.rename(columns={date_col_for_overdue_pending_calc:"Sample Collection/Registered Date"})
                
                final_overdue_display_cols = [col_disp for col_disp in default_overdue_cols if col_disp in df_overdue_for_display.columns]
                insights_output_dict["overdue_pending_tests_list_df"] = df_overdue_for_display[final_overdue_display_cols].sort_values('days_pending', ascending=False)
            else:
                insights_output_dict["processing_notes"].append("No tests currently pending longer than their specific target TAT + buffer within the period.")
        else:
            insights_output_dict["processing_notes"].append("No valid pending tests with dates found for overdue calculation after data cleaning.")
    else:
        insights_output_dict["processing_notes"].append("No tests with 'Pending' status found in the period for overdue status evaluation.")


    # --- C. Sample Rejection Analysis (From raw period data: df_tests_src_cleaned) ---
    if 'sample_status' in df_tests_src_cleaned.columns and 'rejection_reason' in df_tests_src_cleaned.columns:
        df_rejected_samples_raw = df_tests_src_cleaned[
            df_tests_src_cleaned.get('sample_status', pd.Series(dtype=str)).astype(str).str.lower() == 'rejected'
        ].copy()
        
        if not df_rejected_samples_raw.empty:
            # Clean rejection reasons: fillna, strip, and map common NAs to "Unknown Reason"
            df_rejected_samples_raw['rejection_reason_clean'] = df_rejected_samples_raw['rejection_reason'].astype(str).str.strip()
            df_rejected_samples_raw.loc[
                df_rejected_samples_raw['rejection_reason_clean'].isin(common_na_insights_prep + ["UnknownReason"]), 
                'rejection_reason_clean'
            ] = 'Unknown Reason' # Standardize unknown
            
            df_rejection_reason_counts = df_rejected_samples_raw['rejection_reason_clean'].value_counts().reset_index()
            df_rejection_reason_counts.columns = ['Rejection Reason', 'Count']
            insights_output_dict["sample_rejection_reasons_summary_df"] = df_rejection_reason_counts

            # Provide a list of example rejected samples for review in the UI
            # Add 'sample_collection_date' if available for more context to the user
            cols_for_rejected_examples = ['patient_id', 'test_type']
            if 'sample_collection_date' in df_rejected_samples_raw.columns: 
                cols_for_rejected_examples.append('sample_collection_date')
            cols_for_rejected_examples.append('encounter_date') # Date of encounter when rejection might have been logged
            cols_for_rejected_examples.append('rejection_reason_clean')
            
            # Ensure only existing columns are selected to prevent KeyErrors
            final_rejected_examples_cols_to_show = [col_ex for col_ex in cols_for_rejected_examples if col_ex in df_rejected_samples_raw.columns]
            
            insights_output_dict["top_rejected_samples_examples_df"] = df_rejected_samples_raw[
                final_rejected_examples_cols_to_show
            ].head(15) # Show top N examples for review
        else:
            insights_output_dict["processing_notes"].append("No rejected samples recorded in this period for rejection analysis.")
    else:
        insights_output_dict["processing_notes"].append("Sample status or rejection reason data columns missing; skipping rejection analysis.")
    
    logger.info(f"({module_log_prefix}) Clinic testing insights data preparation finished. Notes: {len(insights_output_dict['processing_notes'])}")
    return insights_output_dict
