# sentinel_project_root/pages/clinic_components/testing_insights.py
# Prepares detailed data for laboratory testing performance and trends for Sentinel.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List

# --- Module Imports ---
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


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)


def prepare_clinic_lab_testing_insights_data(
    kpis_summary: Optional[Dict[str, Any]],
    filtered_health_df: Optional[pd.DataFrame] = None,
    **kwargs 
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including summaries, trends,
    and lists of overdue tests.
    """
    module_log_prefix = "ClinicTestInsightsPrep"
    
    default_summary_cols = ["Test Group (Critical)", "Positivity (%)", "Avg. TAT (Days)", "% Met TAT Target", "Pending (Patients)"]
    default_overdue_cols = ['patient_id', 'test_type', 'Sample Collection/Registered Date', 'days_pending', 'overdue_threshold_days']
    
    insights: Dict[str, Any] = {
        "all_critical_tests_summary_table_df": pd.DataFrame(columns=default_summary_cols),
        "overdue_pending_tests_list_df": pd.DataFrame(columns=default_overdue_cols),
        "avg_tat_by_test_df": pd.DataFrame(),
        "rejection_reasons_df": pd.DataFrame(),
        "processing_notes": []
    }

    test_summary_details = kpis_summary.get("test_summary_details", {}) if isinstance(kpis_summary, dict) else {}
    key_test_configs = _get_setting('KEY_TEST_TYPES_FOR_ANALYSIS', {})

    # --- Critical Tests Summary Table ---
    if test_summary_details and isinstance(key_test_configs, dict):
        summary_list = []
        for test_name, config in key_test_configs.items():
            if isinstance(config, dict) and config.get("critical"):
                stats = test_summary_details.get(test_name, {})
                summary_list.append({
                    "Test Group (Critical)": config.get("display_name", test_name),
                    "Positivity (%)": stats.get("positive_rate_perc"),
                    "Avg. TAT (Days)": stats.get("avg_tat_days"),
                    "% Met TAT Target": stats.get("perc_met_tat_target"),
                    "Pending (Patients)": stats.get("pending_count_patients", 0),
                })
        if summary_list:
            insights["all_critical_tests_summary_table_df"] = pd.DataFrame(summary_list)

    if not isinstance(filtered_health_df, pd.DataFrame) or filtered_health_df.empty:
        insights["processing_notes"].append("Health data not provided for detailed insights (overdue tests, rejection reasons).")
        return insights
        
    df_tests = filtered_health_df.copy()

    # --- Overdue Pending Tests List ---
    date_col = 'sample_collection_date' if 'sample_collection_date' in df_tests and df_tests['sample_collection_date'].notna().any() else 'encounter_date'
    df_pending = df_tests[(df_tests.get('test_result', pd.Series(dtype=str)).astype(str).str.lower() == 'pending') & (df_tests[date_col].notna())].copy()

    if not df_pending.empty:
        df_pending[date_col] = pd.to_datetime(df_pending[date_col], errors='coerce').dt.tz_localize(None)
        df_pending.dropna(subset=[date_col], inplace=True)
        
        if not df_pending.empty:
            df_pending['days_pending'] = (pd.Timestamp('now').normalize() - df_pending[date_col]).dt.days
            
            def get_overdue_threshold(test_type: str) -> int:
                """Safely calculate the overdue threshold for a given test type."""
                test_config = key_test_configs.get(test_type, {})
                target_tat = test_config.get('target_tat_days', _get_setting('TARGET_TEST_TURNAROUND_DAYS', 2))
                buffer_days = _get_setting('OVERDUE_TEST_BUFFER_DAYS', 2)
                
                # FIXED: Wrap the input 'tat' in a pandas Series before using pandas methods.
                # This prevents the AttributeError on scalar values (like integers).
                # .iloc[0] is used to extract the scalar value back out.
                numeric_tat = pd.to_numeric(pd.Series([target_tat]), errors='coerce').fillna(2).iloc[0]
                
                return int(numeric_tat) + buffer_days
            
            df_pending['overdue_threshold_days'] = df_pending['test_type'].apply(get_overdue_threshold)
            
            df_overdue = df_pending[df_pending['days_pending'] > df_pending['overdue_threshold_days']]
            
            if not df_overdue.empty:
                df_display = df_overdue.rename(columns={date_col: "Sample Collection/Registered Date"})
                insights["overdue_pending_tests_list_df"] = df_display.reindex(columns=default_overdue_cols).sort_values('days_pending', ascending=False).reset_index(drop=True)

    # --- Average TAT by Test Type (for new plot) ---
    if 'test_turnaround_days' in df_tests.columns and 'test_type' in df_tests.columns:
        df_tat = df_tests[['test_type', 'test_turnaround_days']].dropna()
        if not df_tat.empty:
            avg_tat_by_test = df_tat.groupby('test_type')['test_turnaround_days'].mean().round(1).sort_values(ascending=False).reset_index()
            avg_tat_by_test.rename(columns={'test_type': 'Test Type', 'test_turnaround_days': 'Average TAT (Days)'}, inplace=True)
            insights['avg_tat_by_test_df'] = avg_tat_by_test

    # --- Sample Rejection Reasons (for new plot) ---
    if 'sample_status' in df_tests.columns and 'rejection_reason' in df_tests.columns:
        df_rejected = df_tests[df_tests['sample_status'].str.lower() == 'rejected by lab'].copy()
        if not df_rejected.empty:
            rejection_counts = df_rejected['rejection_reason'].value_counts().reset_index()
            rejection_counts.columns = ['Reason', 'Count']
            insights['rejection_reasons_df'] = rejection_counts

    logger.info(f"({module_log_prefix}) Testing insights preparation complete.")
    return insights
