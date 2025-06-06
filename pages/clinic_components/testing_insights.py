# sentinel_project_root/pages/clinic_components/testing_insights.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.

import pandas as pd
import numpy as np
import logging
import re 
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

try:
    from config import settings
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in testing_insights.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# Common NA strings for robust replacement
COMMON_NA_STRINGS_INSIGHTS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_INSIGHTS_PATTERN = r'^\s*$' + (r'|^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_INSIGHTS if s) + r')$' if any(COMMON_NA_STRINGS_INSIGHTS) else '')

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
            elif target_type_str == str:
                df_prepared[col_name] = df_prepared[col_name].fillna(str(default_value)).astype(str).str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT: df_prepared[col_name] = pd.NaT
            else: df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        # CORRECTED: Use the correct variable `default_patient_id_prefix` that is passed as an argument.
        df_prepared['patient_id'] = df_prepared['patient_id'].replace('', default_patient_id_prefix).fillna(default_patient_id_prefix)
    return df_prepared


def prepare_clinic_lab_testing_insights_data(
    filtered_health_df_for_clinic_period: Optional[pd.DataFrame],
    clinic_overall_kpis_summary: Optional[Dict[str, Any]],
    reporting_period_context_str: str,
    focus_test_group_display_name: str = "All Critical Tests Summary"
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including summaries, trends,
    overdue tests, and rejection reasons.
    """
    module_log_prefix = "ClinicTestInsightsPrep"
    logger.info(f"({module_log_prefix}) Preparing testing insights. Focus: '{focus_test_group_display_name}', Period: {reporting_period_context_str}")

    default_crit_summary_cols = ["Test Group (Critical)", "Positivity (%)", "Avg. TAT (Days)", "% Met TAT Target", "Pending (Patients)", "Rejected (Patients)", "Total Conclusive Tests"]
    default_overdue_cols = ['patient_id', 'test_type', 'Sample Collection/Registered Date', 'days_pending', 'overdue_threshold_days', 'condition']
    
    insights_output: Dict[str, Any] = {
        "all_critical_tests_summary_table_df": pd.DataFrame(columns=default_crit_summary_cols),
        "overdue_pending_tests_list_df": pd.DataFrame(columns=default_overdue_cols),
        "processing_notes": []
    }

    if not isinstance(filtered_health_df_for_clinic_period, pd.DataFrame) or filtered_health_df_for_clinic_period.empty:
        insights_output["processing_notes"].append("No health data provided for testing insights.")
        return insights_output
    
    if not isinstance(clinic_overall_kpis_summary, dict) or not isinstance(clinic_overall_kpis_summary.get("test_summary_details"), dict):
        insights_output["processing_notes"].append("KPI summary data is missing or invalid.")
        clinic_overall_kpis_summary = {"test_summary_details": {}}
            
    pid_prefix_insights = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15]
    insights_cols_cfg = {
        'test_type': {"default": "UnknownTest_Insights", "type": str}, 
        'test_result': {"default": "UnknownResult_Insights", "type": str},
        'sample_status': {"default": "UnknownStatus_Insights", "type": str}, 
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'test_turnaround_days': {"default": np.nan, "type": float}, 
        'patient_id': {"default": f"UPID_Ins_{pid_prefix_insights}", "type": str},
        'sample_collection_date': {"default": pd.NaT, "type": "datetime"}, 
        'sample_registered_lab_date': {"default": pd.NaT, "type": "datetime"},
        'rejection_reason': {"default": "NoReasonGiven", "type": str},
        'condition': {"default": "UnknownCondition", "type": str}
    }
    df_tests_src = _prepare_testing_dataframe(
        filtered_health_df_for_clinic_period, insights_cols_cfg, 
        module_log_prefix, f"UPID_Ins_{pid_prefix_insights}"
    )
    df_tests_src.dropna(subset=['encounter_date', 'test_type', 'patient_id'], inplace=True)
    if df_tests_src.empty:
        insights_output["processing_notes"].append("No valid test records after cleaning.")
        return insights_output
        
    test_summary_details_from_kpis = clinic_overall_kpis_summary.get("test_summary_details", {})
    key_test_configs_setting = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})

    if focus_test_group_display_name == "All Critical Tests Summary":
        critical_tests_summary_list: List[Dict[str, Any]] = []
        if test_summary_details_from_kpis and isinstance(key_test_configs_setting, dict):
            for internal_test_name, test_props_config in key_test_configs_setting.items():
                if isinstance(test_props_config, dict) and test_props_config.get("critical", False):
                    display_name = test_props_config.get("display_name", internal_test_name)
                    stats = test_summary_details_from_kpis.get(internal_test_name, {})
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
    
    date_col_for_overdue_calc = 'sample_collection_date'
    if date_col_for_overdue_calc not in df_tests_src.columns or df_tests_src[date_col_for_overdue_calc].isnull().all():
        date_col_for_overdue_calc = 'encounter_date'
    
    df_pending_tests_raw = df_tests_src[
        (df_tests_src.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending') & 
        (df_tests_src[date_col_for_overdue_calc].notna())
    ].copy()

    if not df_pending_tests_raw.empty:
        df_pending_tests_raw[date_col_for_overdue_calc] = pd.to_datetime(df_pending_tests_raw[date_col_for_overdue_calc], errors='coerce')
        df_pending_tests_raw.dropna(subset=[date_col_for_overdue_calc], inplace=True)

        if not df_pending_tests_raw.empty:
            current_processing_date = pd.Timestamp('now').normalize()
            date_series = df_pending_tests_raw[date_col_for_overdue_calc]
            if date_series.dt.tz is not None:
                date_series = date_series.dt.tz_localize(None)
            df_pending_tests_raw['days_pending'] = (current_processing_date - date_series).dt.days
            
            def get_overdue_threshold_days(test_type_str: str) -> int:
                test_config = key_test_configs_setting.get(test_type_str, {})
                target_tat = test_config.get('target_tat_days', _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2))
                buffer_days = _get_setting('OVERDUE_TEST_BUFFER_DAYS', 2)
                return max(1, int(pd.to_numeric(target_tat, errors='coerce').fillna(7)) + buffer_days)
            
            df_pending_tests_raw['overdue_threshold_days'] = df_pending_tests_raw['test_type'].apply(get_overdue_threshold_days)
            
            df_actually_overdue = df_pending_tests_raw[df_pending_tests_raw['days_pending'] > df_pending_tests_raw['overdue_threshold_days']]
            
            if not df_actually_overdue.empty:
                df_overdue_display = df_actually_overdue.rename(columns={date_col_for_overdue_calc: "Sample Collection/Registered Date"})
                insights_output["overdue_pending_tests_list_df"] = df_overdue_display.reindex(columns=default_overdue_cols).sort_values('days_pending', ascending=False).reset_index(drop=True)
    
    return insights_output
