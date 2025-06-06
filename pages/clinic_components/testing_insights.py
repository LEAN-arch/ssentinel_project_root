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
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in testing_insights.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

COMMON_NA_STRINGS_INSIGHTS = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
NA_REGEX_INSIGHTS_PATTERN = r'^\s*$' + (r'|^(?:' + '|'.join(re.escape(s) for s in COMMON_NA_STRINGS_INSIGHTS if s) + r')$' if any(COMMON_NA_STRINGS_INSIGHTS) else '')

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
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
            df_prepared[col_name] = default_value
        
        series = df_prepared[col_name]
        if target_type_str in [float, "datetime"] and pd.api.types.is_object_dtype(series.dtype):
            if NA_REGEX_INSIGHTS_PATTERN:
                series = series.replace(NA_REGEX_INSIGHTS_PATTERN, np.nan, regex=True)
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(series, errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(series, default_value=default_value)
            elif target_type_str == str:
                df_prepared[col_name] = series.fillna(str(default_value)).astype(str).str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting '{col_name}': {e_conv}. Using defaults.", exc_info=True)
            df_prepared[col_name] = default_value
            
    if 'patient_id' in df_prepared.columns:
        # CORRECTED: Use the correct variable name passed into the function.
        df_prepared['patient_id'].replace('', default_patient_id_prefix, inplace=True)
        df_prepared['patient_id'].fillna(default_patient_id_prefix, inplace=True)
    return df_prepared


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
    logger.info(f"({module_log_prefix}) Preparing insights. Focus: '{focus_test_group_display_name}', Period: {reporting_period_context_str}")

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
            
    pid_prefix = reporting_period_context_str.replace(" ", "_").replace("-", "")[:15]
    insights_cols_cfg = {
        'test_type': {"default": "UnknownTest", "type": str}, 'test_result': {"default": "UnknownResult", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"}, 'test_turnaround_days': {"default": np.nan, "type": float},
        'patient_id': {"default": f"UPID_Ins_{pid_prefix}", "type": str}, 'sample_collection_date': {"default": pd.NaT, "type": "datetime"},
        'condition': {"default": "UnknownCondition", "type": str}
    }
    df_tests_src = _prepare_testing_dataframe(filtered_health_df_for_clinic_period, insights_cols_cfg, module_log_prefix, f"UPID_Ins_{pid_prefix}")
    df_tests_src.dropna(subset=['encounter_date', 'test_type', 'patient_id'], inplace=True)

    if df_tests_src.empty:
        insights_output["processing_notes"].append("No valid test records after cleaning.")
        return insights_output
        
    test_summary_details = clinic_overall_kpis_summary.get("test_summary_details", {})
    key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})

    critical_tests_summary = []
    if test_summary_details and isinstance(key_test_configs, dict):
        for internal_name, config in key_test_configs.items():
            if isinstance(config, dict) and config.get("critical", False):
                display_name = config.get("display_name", internal_name)
                # CORRECTED: Use the internal test name for data lookup.
                stats = test_summary_details.get(internal_name, {})
                critical_tests_summary.append({
                    "Test Group (Critical)": display_name, 
                    "Positivity (%)": stats.get("positive_rate_perc", np.nan),
                    "Avg. TAT (Days)": stats.get("avg_tat_days", np.nan), 
                    "% Met TAT Target": stats.get("perc_met_tat_target", np.nan),
                    "Pending (Patients)": stats.get("pending_count_patients", 0), 
                    "Rejected (Patients)": stats.get("rejected_count_patients", 0),
                    "Total Conclusive Tests": stats.get("total_conclusive_tests", 0)
                })
        if critical_tests_summary:
            insights_output["all_critical_tests_summary_table_df"] = pd.DataFrame(critical_tests_summary)

    date_col_overdue = 'sample_collection_date' if 'sample_collection_date' in df_tests_src.columns and df_tests_src['sample_collection_date'].notna().any() else 'encounter_date'
    
    df_pending = df_tests_src[(df_tests_src['test_result'].str.lower() == 'pending') & (df_tests_src[date_col_overdue].notna())].copy()

    if not df_pending.empty:
        df_pending['days_pending'] = (pd.Timestamp('now').normalize() - df_pending[date_col_overdue]).dt.days
        
        def get_overdue_threshold(test_type: str) -> int:
            config = key_test_configs.get(test_type, {})
            tat = config.get('target_tat_days', _get_setting('OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK', 7))
            return max(1, int(tat) + _get_setting('OVERDUE_TEST_BUFFER_DAYS', 2))
        
        df_pending['overdue_threshold_days'] = df_pending['test_type'].apply(get_overdue_threshold)
        df_overdue = df_pending[df_pending['days_pending'] > df_pending['overdue_threshold_days']]
        
        if not df_overdue.empty:
            df_overdue_display = df_overdue.rename(columns={date_col_overdue: "Sample Collection/Registered Date"})
            insights_output["overdue_pending_tests_list_df"] = df_overdue_display[default_overdue_cols].sort_values('days_pending', ascending=False).reset_index(drop=True)

    return insights_output
