# sentinel_project_root/pages/clinic_components/testing_insights.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR); logger = logging.getLogger(__name__)
    logger.critical(f"Critical import error in testing_insights.py: {e}", exc_info=True); raise

logger = logging.getLogger(__name__)

SUMMARY_COLS = ["Test Group (Critical)", "Positivity (%)", "Avg. TAT (Days)", "% Met TAT Target", "Pending (Patients)"]
OVERDUE_COLS = ['patient_id', 'test_type', 'Sample Collection/Registered Date', 'days_pending', 'overdue_threshold_days']
REJECTION_COLS = ['Reason', 'Count']

class TestingInsightsPreparer:
    """Encapsulates logic for preparing detailed laboratory testing insights."""
    def __init__(self, kpis_summary: Optional[Dict[str, Any]], health_df_period: Optional[pd.DataFrame]):
        self.kpis_summary = kpis_summary if isinstance(kpis_summary, dict) else {}
        self.df_health = health_df_period.copy() if isinstance(health_df_period, pd.DataFrame) else pd.DataFrame()
        self.key_test_configs = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
        self.notes: List[str] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    def _format_value(self, value: Any, precision: int = 1, suffix: str = "", default: str = "N/A", is_int: bool = False) -> str:
        if pd.isna(value): return default
        try:
            num_value = pd.to_numeric(value)
            return f"{int(num_value):,}" if is_int else f"{num_value:,.{precision}f}{suffix}"
        except (ValueError, TypeError): return default

    def _prepare_summary_table(self) -> pd.DataFrame:
        summary_list = []
        test_details = self.kpis_summary.get("test_summary_details", {})
        critical_tests = [k for k, v in self.key_test_configs.items() if isinstance(v, dict) and v.get("critical")]
        if not critical_tests: self.notes.append("Config note: No tests are marked as 'critical'."); return pd.DataFrame(columns=SUMMARY_COLS)
        
        for test_name in critical_tests:
            config = self.key_test_configs.get(test_name, {})
            # The test_details from aggregation now uses internal names, not display names, as keys
            stats = test_details.get(test_name, {})
            summary_list.append({
                "Test Group (Critical)": config.get("display_name", test_name),
                "Positivity (%)": self._format_value(stats.get("positive_rate_perc"), suffix="%"),
                "Avg. TAT (Days)": self._format_value(stats.get("avg_tat_days")),
                "% Met TAT Target": self._format_value(stats.get("perc_met_tat_target"), suffix="%"),
                "Pending (Patients)": self._format_value(stats.get("pending_count_patients", 0), is_int=True),
            })
        return pd.DataFrame(summary_list, columns=SUMMARY_COLS)

    def _prepare_overdue_list(self) -> pd.DataFrame:
        if self.df_health.empty: self.notes.append("Overdue analysis skipped: no health data."); return pd.DataFrame(columns=OVERDUE_COLS)
        date_col = 'sample_collection_date' if 'sample_collection_date' in self.df_health and self.df_health['sample_collection_date'].notna().any() else 'encounter_date'
        required_cols = ['test_result', 'test_type', 'patient_id', date_col]
        if not all(col in self.df_health.columns for col in required_cols): return pd.DataFrame(columns=OVERDUE_COLS)
        
        df_pending = self.df_health[(self.df_health['test_result'].astype(str).str.lower() == 'pending') & (self.df_health[date_col].notna())].copy()
        if df_pending.empty: return pd.DataFrame(columns=OVERDUE_COLS)

        # --- DEFINITIVE FIX FOR Timezone Error ---
        now_naive = pd.Timestamp.now().normalize()
        df_pending['days_pending'] = (now_naive - df_pending[date_col].dt.normalize()).dt.days

        df_pending = df_pending[df_pending['days_pending'] >= 0]
        if df_pending.empty: return pd.DataFrame(columns=OVERDUE_COLS)

        default_tat = self._get_setting('TARGET_TEST_TURNAROUND_DAYS', 2)
        buffer = self._get_setting('OVERDUE_TEST_BUFFER_DAYS', 2)
        threshold_map = {name: int(pd.to_numeric(conf.get('target_tat_days', default_tat), errors='coerce').fillna(default_tat)) + buffer for name, conf in self.key_test_configs.items()}
        df_pending['overdue_threshold_days'] = df_pending['test_type'].map(threshold_map)
        
        df_overdue = df_pending[df_pending['days_pending'] > df_pending['overdue_threshold_days']]
        if df_overdue.empty: return pd.DataFrame(columns=OVERDUE_COLS)
        
        return df_overdue.rename(columns={date_col: "Sample Collection/Registered Date"})[OVERDUE_COLS].sort_values('days_pending', ascending=False).reset_index(drop=True)

    def _prepare_rejection_reasons(self) -> pd.DataFrame:
        if self.df_health.empty or 'sample_status' not in self.df_health.columns or 'rejection_reason' not in self.df_health.columns: return pd.DataFrame(columns=REJECTION_COLS)
        df_rejected = self.df_health[self.df_health['sample_status'].astype(str).str.lower() == 'rejected by lab'].copy()
        df_rejected.dropna(subset=['rejection_reason'], inplace=True)
        df_rejected['rejection_reason'] = df_rejected['rejection_reason'].astype(str).str.strip()
        df_rejected = df_rejected[df_rejected['rejection_reason'] != '']
        if df_rejected.empty: return pd.DataFrame(columns=REJECTION_COLS)
        top_n = self._get_setting('TESTING_TOP_N_REJECTION_REASONS', 10)
        return df_rejected['rejection_reason'].value_counts().nlargest(top_n).reset_index().rename(columns={'index': 'Reason', 'rejection_reason': 'Count'})

    def prepare(self) -> Dict[str, Any]:
        """Orchestrates all testing insights calculations."""
        return {
            "all_critical_tests_summary_table_df": self._prepare_summary_table(),
            "overdue_pending_tests_list_df": self._prepare_overdue_list(),
            "rejection_reasons_df": self._prepare_rejection_reasons(),
            "processing_notes": self.notes,
        }

def prepare_clinic_lab_testing_insights_data(kpis_summary: Optional[Dict[str, Any]], health_df_period: Optional[pd.DataFrame], **kwargs) -> Dict[str, Any]:
    """Factory function to prepare structured data for detailed testing insights."""
    preparer = TestingInsightsPreparer(kpis_summary=kpis_summary, health_df_period=health_df_period)
    return preparer.prepare()
