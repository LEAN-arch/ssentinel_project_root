# sentinel_project_root/pages/clinic_components/testing_insights.py
# SME PLATINUM STANDARD (V2 - ARCHITECTURAL REFACTOR)
# This version refactors the logic into a more declarative and robust
# architecture using:
# 1. A fluent `.pipe()` based data pipeline for clarity in overdue test processing.
# 2. Pydantic models for a self-validating, explicit output contract.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field # <<< SME REVISION V2

# --- Module Imports & Setup ---
try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR); logger = logging.getLogger(__name__)
    logger.critical(f"Critical import error in testing_insights.py: {e}", exc_info=True); raise

logger = logging.getLogger(__name__)

# --- Pydantic Models for a Strong Output Contract ---
# <<< SME REVISION V2
class TestingInsightsResult(BaseModel):
    """Defines the structured output for the testing insights component."""
    summary_table: List[Dict[str, str]] = Field(alias="all_critical_tests_summary_table_df")
    overdue_tests: List[Dict[str, Any]] = Field(alias="overdue_pending_tests_list_df")
    rejection_reasons: List[Dict[str, Any]] = Field(alias="rejection_reasons_df")
    processing_notes: List[str]

    class Config:
        # Convert DataFrames to list of records during model creation
        from_attributes = True

# --- Main Preparer Class ---
class TestingInsightsPreparer:
    """Encapsulates all logic for preparing detailed laboratory testing insights."""

    def __init__(self, kpis_summary: Optional[Dict[str, Any]], health_df_period: Optional[pd.DataFrame]):
        self.kpis_summary = kpis_summary or {}
        self.df_health = health_df_period.copy() if isinstance(health_df_period, pd.DataFrame) else pd.DataFrame()
        self.key_test_configs = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
        self.notes: List[str] = []

    # ... _get_setting and _format_value helpers remain the same ...
    def _get_setting(self, attr_name: str, default_value: Any) -> Any: ...
    def _format_value(self, value: Any, precision: int = 1, suffix: str = "", default: str = "N/A", is_int: bool = False) -> str: ...

    def _prepare_summary_table(self) -> pd.DataFrame:
        # ... This method is already clean and declarative, no changes needed ...
        summary_list = []
        test_details = self.kpis_summary.get("test_summary_details", {})
        critical_tests = [k for k, v in self.key_test_configs.items() if isinstance(v, dict) and v.get("critical")]
        if not critical_tests: self.notes.append("Config note: No tests marked as 'critical'.")
        
        for test_name in critical_tests:
            config = self.key_test_configs.get(test_name, {})
            stats = test_details.get(test_name, {})
            summary_list.append({
                "Test Group (Critical)": config.get("display_name", test_name),
                "Positivity (%)": self._format_value(stats.get("positive_rate_perc"), suffix="%"),
                "Avg. TAT (Days)": self._format_value(stats.get("avg_tat_days")),
                "% Met TAT Target": self._format_value(stats.get("perc_met_tat_target"), suffix="%"),
                "Pending (Patients)": self._format_value(stats.get("pending_count_patients", 0), is_int=True),
            })
        return pd.DataFrame(summary_list)

    def _prepare_rejection_reasons(self) -> pd.DataFrame:
        # ... This method is also excellent, no changes needed ...
        if self.df_health.empty or 'sample_status' not in self.df_health.columns: return pd.DataFrame()
        df_rejected = self.df_health[self.df_health['sample_status'].astype(str).str.lower() == 'rejected by lab'].copy()
        if df_rejected.empty: return pd.DataFrame()
        
        top_n = self._get_setting('TESTING_TOP_N_REJECTION_REASONS', 10)
        rejection_counts = df_rejected['rejection_reason'].dropna().value_counts().nlargest(top_n).reset_index()
        return rejection_counts.rename(columns={'index': 'Reason', 'rejection_reason': 'Count'})

    # --- Overdue Tests Pipeline ---
    # <<< SME REVISION V2 >>> Break down the logic into composable pipeline stages.
    def _filter_to_pending_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 1: Filter to valid, pending tests."""
        date_col = 'sample_collection_date' if 'sample_collection_date' in df.columns and df['sample_collection_date'].notna().any() else 'encounter_date'
        
        if not all(c in df.columns for c in ['test_result', 'test_type', date_col]):
            self.notes.append("Overdue analysis skipped: missing required columns.")
            return pd.DataFrame()
            
        return df[
            (df['test_result'].astype(str).str.lower() == 'pending') & (df[date_col].notna())
        ].copy().rename(columns={date_col: "collection_date"})

    def _calculate_days_pending(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 2: Calculate how many days each test has been pending."""
        if df.empty: return df
        now_naive = pd.Timestamp.now().normalize()
        df['collection_date'] = df['collection_date'].dt.normalize()
        df['days_pending'] = (now_naive - df['collection_date']).dt.days
        return df[df['days_pending'] >= 0] # Filter out future-dated records

    def _identify_overdue_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 3: Compare pending days against test-specific thresholds."""
        if df.empty: return df
        
        def get_threshold(test_type: str) -> int:
            config = self.key_test_configs.get(test_type, {})
            tat = pd.to_numeric(config.get('target_tat_days', 2), errors='coerce')
            buffer = self._get_setting('OVERDUE_TEST_BUFFER_DAYS', 2)
            return int(np.nan_to_num(tat, nan=2)) + buffer

        threshold_map = {name: get_threshold(name) for name in df['test_type'].unique()}
        df['overdue_threshold_days'] = df['test_type'].map(threshold_map)
        
        return df[df['days_pending'] > df['overdue_threshold_days']]

    def _format_overdue_for_ui(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline Stage 4: Select, rename, and sort columns for UI display."""
        if df.empty: return df
        
        display_cols = {
            'patient_id': 'patient_id', 'test_type': 'test_type', 'collection_date': 'Sample Collection/Registered Date',
            'days_pending': 'days_pending', 'overdue_threshold_days': 'overdue_threshold_days'
        }
        return df.rename(columns=display_cols)[list(display_cols.values())].sort_values('days_pending', ascending=False).reset_index(drop=True)

    # --- Main Orchestration Method ---
    def prepare(self) -> TestingInsightsResult:
        """Orchestrates all testing insights calculations and returns a consolidated, typed result."""
        logger.info("Starting testing insights data preparation.")
        
        # <<< SME REVISION V2 >>> Use a fluent .pipe() based pipeline for overdue tests.
        overdue_df = self.df_health.pipe(self._filter_to_pending_tests) \
                                   .pipe(self._calculate_days_pending) \
                                   .pipe(self._identify_overdue_tests) \
                                   .pipe(self._format_overdue_for_ui)
        
        result_data = {
            "all_critical_tests_summary_table_df": self._prepare_summary_table().to_dict('records'),
            "overdue_pending_tests_list_df": overdue_df.to_dict('records'),
            "rejection_reasons_df": self._prepare_rejection_reasons().to_dict('records'),
            "processing_notes": list(set(self.notes)),
        }
        
        logger.info("Testing insights preparation complete.")
        return TestingInsightsResult.model_validate(result_data)

# --- Public API Function ---
def prepare_clinic_lab_testing_insights_data(
    kpis_summary: Optional[Dict[str, Any]],
    health_df_period: Optional[pd.DataFrame],
    **kwargs
) -> Dict[str, Any]:
    """Public factory function to prepare structured data for detailed testing insights."""
    preparer = TestingInsightsPreparer(kpis_summary=kpis_summary, health_df_period=health_df_period)
    # Return as a dictionary to maintain compatibility with the UI component.
    return preparer.prepare().model_dump(by_alias=True)
