# sentinel_project_root/pages/clinic_components/testing_insights.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.

import pandas as pd
import numpy as np
import logging
import re # For string operations
from typing import Dict, Any, Optional, List

from config import settings
from data_processing.aggregation import get_trend_data
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def prepare_clinic_lab_testing_insights_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    clinic_overall_kpis_summary: Optional[Dict[str, Any]],
    reporting_period_context_str: str,
    focus_test_group_display_name: str = "All Critical Tests Summary"
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights.
    """
    module_log_prefix = "ClinicTestInsightsPrep"
    logger.info(f"({module_log_prefix}) Preparing testing insights. Focus: '{focus_test_group_display_name}', Period: {reporting_period_context_str}")

    default_crit_summary_cols = ["Test Group (Critical)", "Positivity (%)", "Avg. TAT (Days)", "% Met TAT Target", "Pending (Patients)", "Rejected (Patients)", "Total Conclusive Tests"]
    default_overdue_cols = ['patient_id', 'test_type', 'Sample Collection/Registered Date', 'days_pending', 'overdue_threshold_days']
    default_rejection_cols = ['Rejection Reason', 'Count']
    default_rejected_examples_cols = ['patient_id', 'test_type', 'sample_collection_date', 'encounter_date', 'rejection_reason_clean']

    insights_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str, "selected_focus_area": focus_test_group_display_name,
        "all_critical_tests_summary_table_df": pd.DataFrame(columns=default_crit_summary_cols),
        "focused_test_group_kpis_dict": None, "focused_test_group_tat_trend_series": pd.Series(dtype='float64'),
        "focused_test_group_volume_trend_df": pd.DataFrame(columns=['date', 'Conclusive Tests', 'Pending Tests']),
        "overdue_pending_tests_list_df": pd.DataFrame(columns=default_overdue_cols),
        "sample_rejection_reasons_summary_df": pd.DataFrame(columns=default_rejection_cols),
        "top_rejected_samples_examples_df": pd.DataFrame(columns=default_rejected_examples_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        note = "No health data for testing insights. Outputs will be empty/default."
        logger.warning(f"({module_log_prefix}) {note}"); insights_output["processing_notes"].append(note)
        return insights_output
    
    if not isinstance(clinic_overall_kpis_summary, dict) or \
       not isinstance(clinic_overall_kpis_summary.get("test_summary_details"), dict):
        note = "Clinic overall KPI summary or 'test_summary_details' missing/invalid. Aggregated test group metrics unavailable."
        logger.warning(f"({module_log_prefix}) {note}"); insights_output["processing_notes"].append(note)
        clinic_overall_kpis_summary = {"test_summary_details": {}} # Ensure safe access
            
    df_tests_src = filtered_health_df_for_clinic_period.copy()
    
    insights_cols_cfg = {
        'test_type': {"default": "UTest_Insights", "type": str}, 'test_result': {"default": "URes_Insights", "type": str},
        'sample_status': {"default": "UStat_Insights", "type": str}, 'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'test_turnaround_days': {"default": np.nan, "type": float}, 'patient_id': {"default": f"UPID_Ins_{reporting_period_context_str[:10]}", "type": str},
        'sample_collection_date': {"default": pd.NaT, "type": "datetime"}, 'sample_registered_lab_date': {"default": pd.NaT, "type": "datetime"},
        'rejection_reason': {"default": "UReason", "type": str}
    }
    common_na_insights = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    na_regex_insights = r'^(?:' + '|'.join(re.escape(s) for s in common_na_insights if s) + r')$'

    for col, cfg in insights_cols_cfg.items():
        if col not in df_tests_src.columns: df_tests_src[col] = cfg["default"]
        if cfg["type"] == "datetime": df_tests_src[col] = pd.to_datetime(df_tests_src[col], errors='coerce')
        elif cfg["type"] == float: df_tests_src[col] = convert_to_numeric(df_tests_src[col], default_value=cfg["default"])
        elif cfg["type"] == str:
            df_tests_src[col] = df_tests_src[col].astype(str).fillna(str(cfg["default"]))
            if any(common_na_insights): df_tests_src[col] = df_tests_src[col].replace(na_regex_insights, str(cfg["default"]), regex=True)
            df_tests_src[col] = df_tests_src[col].str.strip()

    test_summary_details = clinic_overall_kpis_summary.get("test_summary_details", {}) # Safe access
    if focus_test_group_display_name == "All Critical Tests Summary":
        crit_summary_list: List[Dict[str, Any]] = []
        if test_summary_details:
            for test_disp_name, stats in test_summary_details.items():
                orig_key = next((k for k, v_cfg in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == test_disp_name), None)
                if orig_key and settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(orig_key, {}).get("critical"):
                    crit_summary_list.append({"Test Group (Critical)": test_disp_name, "Positivity (%)": stats.get("positive_rate_perc", np.nan),
                                              "Avg. TAT (Days)": stats.get("avg_tat_days", np.nan), "% Met TAT Target": stats.get("perc_met_tat_target", np.nan),
                                              "Pending (Patients)": stats.get("pending_count_patients", 0), "Rejected (Patients)": stats.get("rejected_count_patients", 0),
                                              "Total Conclusive Tests": stats.get("total_conclusive_tests", 0)})
            if crit_summary_list: insights_output["all_critical_tests_summary_table_df"] = pd.DataFrame(crit_summary_list)
            else: insights_output["processing_notes"].append("No critical test data in summary or no tests configured as critical.")
        else: insights_output["processing_notes"].append("'test_summary_details' missing for 'All Critical Tests' summary.")
    elif focus_test_group_display_name in test_summary_details:
        focused_stats = test_summary_details[focus_test_group_display_name]
        insights_output["focused_test_group_kpis_dict"] = {
            "Positivity Rate (%)": focused_stats.get("positive_rate_perc", np.nan), "Avg. TAT (Days)": focused_stats.get("avg_tat_days", np.nan),
            "% Met TAT Target": focused_stats.get("perc_met_tat_target", np.nan), "Pending Tests (Patients)": focused_stats.get("pending_count_patients", 0),
            "Rejected Samples (Patients)": focused_stats.get("rejected_count_patients", 0), "Total Conclusive Tests": focused_stats.get("total_conclusive_tests", 0)
        }
        orig_key_focus = next((k for k, v_cfg in settings.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == focus_test_group_display_name), None)
        if orig_key_focus:
            raw_keys_group = settings.KEY_TEST_TYPES_FOR_ANALYSIS[orig_key_focus].get("types_in_group", [orig_key_focus])
            if isinstance(raw_keys_group, str): raw_keys_group = [raw_keys_group]
            if 'test_turnaround_days' in df_tests_src.columns and 'encounter_date' in df_tests_src.columns:
                df_focus_tat = df_tests_src[(df_tests_src['test_type'].isin(raw_keys_group)) & (df_tests_src['test_turnaround_days'].notna()) & 
                                            (~df_tests_src.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(['pending','ures_insights','rejected','indeterminate']))].copy()
                if not df_focus_tat.empty:
                    tat_trend = get_trend_data(df=df_focus_tat, value_col='test_turnaround_days', date_col='encounter_date', period='D', agg_func='mean', source_context=f"{module_log_prefix}/TATTrend/{focus_test_group_display_name}")
                    if isinstance(tat_trend, pd.Series) and not tat_trend.empty: insights_output["focused_test_group_tat_trend_series"] = tat_trend
            if 'patient_id' in df_tests_src.columns and 'encounter_date' in df_tests_src.columns:
                df_focus_vol = df_tests_src[df_tests_src['test_type'].isin(raw_keys_group)].copy()
                if not df_focus_vol.empty:
                    concl_mask_vol = ~df_focus_vol.get('test_result', pd.Series(dtype=str)).astype(str).str.lower().isin(['pending','ures_insights','rejected','indeterminate'])
                    concl_vol_trend = get_trend_data(df=df_focus_vol[concl_mask_vol], value_col='patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Conclusive Tests")
                    pend_vol_trend = get_trend_data(df=df_focus_vol[df_focus_vol.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending'], value_col='patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending Tests")
                    vol_trends_concat = [s for s in [concl_vol_trend, pend_vol_trend] if isinstance(s, pd.Series) and not s.empty]
                    if vol_trends_concat:
                        df_vol_concat = pd.concat(vol_trends_concat, axis=1).fillna(0).reset_index().rename(columns={'index': 'date', 'encounter_date': 'date'})
                        insights_output["focused_test_group_volume_trend_df"] = df_vol_concat
        else: insights_output["processing_notes"].append(f"Could not find config key for '{focus_test_group_display_name}' for trends.")
    else: insights_output["processing_notes"].append(f"No aggregated stats for selected test group: '{focus_test_group_display_name}'.")

    # Overdue Pending Tests
    date_col_overdue = 'encounter_date' # Default
    if 'sample_collection_date' in df_tests_src.columns and df_tests_src['sample_collection_date'].notna().any(): date_col_overdue = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_tests_src.columns and df_tests_src['sample_registered_lab_date'].notna().any(): date_col_overdue = 'sample_registered_lab_date'
    
    df_pending_overdue = df_tests_src[(df_tests_src.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending') & (df_tests_src[date_col_overdue].notna())].copy()
    if not df_pending_overdue.empty:
        df_pending_overdue[date_col_overdue] = pd.to_datetime(df_pending_overdue[date_col_overdue], errors='coerce')
        df_pending_overdue.dropna(subset=[date_col_overdue], inplace=True)
        if not df_pending_overdue.empty:
            current_date_ts = pd.Timestamp('now').normalize()
            df_pending_overdue['days_pending'] = (current_date_ts - df_pending_overdue[date_col_overdue]).dt.days
            def get_overdue_thresh(test_type: str) -> int:
                cfg = settings.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type, {})
                buffer = 2; target_tat = settings.TARGET_TEST_TURNAROUND_DAYS
                if cfg and 'target_tat_days' in cfg and pd.notna(cfg['target_tat_days']): target_tat = cfg['target_tat_days']
                thresh = int(target_tat + buffer) if pd.notna(target_tat) and target_tat > 0 else int(settings.OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK + buffer)
                return max(1, thresh)
            df_pending_overdue['overdue_threshold_days'] = df_pending_overdue['test_type'].apply(get_overdue_thresh)
            df_overdue = df_pending_overdue[df_pending_overdue['days_pending'] > df_pending_overdue['overdue_threshold_days']]
            if not df_overdue.empty:
                cols_overdue_display = ['patient_id', 'test_type', date_col_overdue, 'days_pending', 'overdue_threshold_days']
                df_overdue_display = df_overdue.rename(columns={date_col_overdue:"Sample Collection/Registered Date"})
                insights_output["overdue_pending_tests_list_df"] = df_overdue_display[[c for c in default_overdue_cols if c in df_overdue_display.columns]].sort_values('days_pending', ascending=False)
            else: insights_output["processing_notes"].append("No tests pending longer than target TAT + buffer.")
        else: insights_output["processing_notes"].append("No valid pending tests with dates for overdue calculation.")
    else: insights_output["processing_notes"].append("No tests with 'Pending' status for overdue evaluation.")

    # Sample Rejection Analysis
    if 'sample_status' in df_tests_src.columns and 'rejection_reason' in df_tests_src.columns:
        df_rejected_raw = df_tests_src[df_tests_src.get('sample_status', pd.Series(dtype=str)).astype(str).str.lower() == 'rejected'].copy()
        if not df_rejected_raw.empty:
            df_rejected_raw['rejection_reason_clean'] = df_rejected_raw['rejection_reason'].astype(str).str.strip()
            df_rejected_raw.loc[df_rejected_raw['rejection_reason_clean'].isin(common_na_insights + ["UReason"]), 'rejection_reason_clean'] = 'Unknown Reason'
            insights_output["sample_rejection_reasons_summary_df"] = df_rejected_raw['rejection_reason_clean'].value_counts().reset_index().rename(columns={'index': 'Rejection Reason', 'rejection_reason_clean':'Count'})
            
            cols_rejected_ex = ['patient_id', 'test_type']
            if 'sample_collection_date' in df_rejected_raw.columns: cols_rejected_ex.append('sample_collection_date')
            cols_rejected_ex.extend(['encounter_date', 'rejection_reason_clean'])
            final_rejected_ex_cols = [c for c in cols_rejected_ex if c in df_rejected_raw.columns]
            insights_output["top_rejected_samples_examples_df"] = df_rejected_raw[final_rejected_ex_cols].head(15)
        else: insights_output["processing_notes"].append("No rejected samples recorded for rejection analysis.")
    else: insights_output["processing_notes"].append("Sample status or rejection reason data missing; skipping rejection analysis.")
    
    logger.info(f"({module_log_prefix}) Clinic testing insights prep finished. Notes: {len(insights_output['processing_notes'])}")
    return insights_output
