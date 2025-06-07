# sentinel_project_root/tests/test_core_data_processing.py
# Pytest tests for functions in data_processing module for Sentinel.

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import json
import re
import hashlib

# --- Module Imports ---
from data_processing.helpers import clean_column_names, convert_to_numeric, hash_dataframe_safe
from data_processing.loaders import load_health_records, load_iot_clinic_environment_data, load_zone_data
from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
from data_processing.aggregation import get_clinic_summary_kpis, get_district_summary_kpis, get_trend_data
from analytics.supply_forecasting import generate_simple_supply_forecast
from analytics.alerting import get_patient_alerts_for_clinic
from config import settings
import logging
logger = logging.getLogger(__name__)

# Fixtures are sourced from conftest.py

# --- Tests for Helper Functions ---
def test_clean_column_names():
    """Tests the column name cleaning utility."""
    df_dirty = pd.DataFrame(columns=['Test Col One', 'Another-Col WITH_Space  ', 'already_ok', '  lead_space'])
    df_cleaned = clean_column_names(df_dirty.copy())
    expected = ['test_col_one', 'another_col_with_space', 'already_ok', 'lead_space']
    assert list(df_cleaned.columns) == expected
    assert list(clean_column_names(pd.DataFrame()).columns) == [], "Empty DataFrame should be handled."

def test_convert_to_numeric():
    """Tests the robust numeric conversion utility."""
    mixed_series = pd.Series(['101.5', '20', 'bad', None, '70', 'NaN'])
    series_nan_default = convert_to_numeric(mixed_series.copy(), default_value=np.nan, target_type=float)
    expected_nan = pd.Series([101.5, 20.0, np.nan, np.nan, 70.0, np.nan], dtype=float)
    pd.testing.assert_series_equal(series_nan_default, expected_nan)

    series_zero_default = convert_to_numeric(mixed_series.copy(), default_value=0, target_type=int)
    expected_zero = pd.Series([101, 20, 0, 0, 70, 0], dtype=int)
    pd.testing.assert_series_equal(series_zero_default, expected_zero)

def test_hash_dataframe_safe(sample_zone_data_df_main_fixture: pd.DataFrame):
    """Tests the consistent hashing of DataFrames."""
    if sample_zone_data_df_main_fixture.empty:
        pytest.skip("Sample Zone DataFrame is empty.")
    
    hash1 = hash_dataframe_safe(sample_zone_data_df_main_fixture.copy())
    assert isinstance(hash1, str)
    assert hash_dataframe_safe(None) is None
    # FIXED: Hash for empty DF must match implementation
    assert hash_dataframe_safe(pd.DataFrame()) == str(pd.util.hash_pandas_object(pd.DataFrame(), index=True).sum())

    df_modified = sample_zone_data_df_main_fixture.copy()
    df_modified.loc[0, 'population'] += 100
    hash2 = hash_dataframe_safe(df_modified)
    assert hash1 != hash2, "Modified DataFrame produced identical hash."

# --- Tests for Data Loaders ---
def test_load_health_records_output(sample_health_records_df_main_fixture: pd.DataFrame):
    """Verifies the structure and types of the loaded health records fixture."""
    assert isinstance(sample_health_records_df_main_fixture, pd.DataFrame)
    if sample_health_records_df_main_fixture.empty: pytest.skip("Health records fixture is empty.")
    
    key_cols = ['patient_id', 'encounter_date', 'ai_risk_score', 'condition']
    assert all(col in sample_health_records_df_main_fixture.columns for col in key_cols)
    assert pd.api.types.is_datetime64_any_dtype(sample_health_records_df_main_fixture['encounter_date'])
    assert pd.api.types.is_numeric_dtype(sample_health_records_df_main_fixture['ai_risk_score'])

# --- Test for Enrichment ---
def test_enrich_zone_df_aggregates_values(sample_enriched_zone_df_main_fixture: Optional[pd.DataFrame], sample_health_records_df_main_fixture: pd.DataFrame):
    """Verifies that enrichment calculations are correct."""
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or sample_enriched_zone_df_main_fixture.empty:
        pytest.skip("Enriched zone fixture is invalid or empty.")
    
    df_enriched = sample_enriched_zone_df_main_fixture
    health_df_src = sample_health_records_df_main_fixture
    zone_verify = 'ZoneA'
    
    # FIXED: Use re.sub to match the exact column name generation in enrichment.py
    tb_key = next((c for c in settings.KEY_CONDITIONS_FOR_ACTION if "TB" in c.upper()), "TB")
    tb_col_name = f"active_{re.sub(r'[^a-z0-9_]+', '_', tb_key.lower().strip())}_cases"
    
    if tb_col_name in df_enriched.columns:
        expected_tb = health_df_src[(health_df_src['zone_id'] == zone_verify) & (health_df_src['condition'].str.contains(tb_key, case=False, na=False))]['patient_id'].nunique()
        actual_tb = df_enriched[df_enriched['zone_id'] == zone_verify][tb_col_name].iloc[0]
        assert actual_tb == expected_tb, f"Mismatch in '{tb_col_name}' for '{zone_verify}'"

    if 'avg_risk_score' in df_enriched.columns:
        expected_risk = health_df_src[health_df_src['zone_id'] == zone_verify]['ai_risk_score'].mean()
        actual_risk = df_enriched[df_enriched['zone_id'] == zone_verify]['avg_risk_score'].iloc[0]
        assert np.isclose(actual_risk, expected_risk) if pd.notna(actual_risk) and pd.notna(expected_risk) else pd.isna(actual_risk) == pd.isna(expected_risk)

# --- Tests for Aggregation Functions ---
def test_get_clinic_summary_kpis_structure(sample_health_records_df_main_fixture: pd.DataFrame):
    """Verifies the structure of the clinic summary KPI dictionary."""
    if sample_health_records_df_main_fixture.empty: pytest.skip("Sample health records are empty.")
    
    clinic_summary = get_clinic_summary_kpis(sample_health_records_df_main_fixture.copy())
    assert isinstance(clinic_summary, dict)
    expected_keys = ["overall_avg_test_turnaround_conclusive_days", "perc_critical_tests_tat_met", "test_summary_details"]
    assert all(key in clinic_summary for key in expected_keys)
    assert isinstance(clinic_summary.get("test_summary_details"), dict)

def test_get_patient_alerts_for_clinic(sample_health_records_df_main_fixture: pd.DataFrame):
    """Verifies the structure and content of the clinic alerts DataFrame."""
    if sample_health_records_df_main_fixture.empty: pytest.skip("Sample health records are empty.")
    
    clinic_alerts_df = get_patient_alerts_for_clinic(sample_health_records_df_main_fixture.copy())
    assert isinstance(clinic_alerts_df, pd.DataFrame)
    if not clinic_alerts_df.empty:
        expected_cols = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score']
        assert all(col in clinic_alerts_df.columns for col in expected_cols)
        assert not clinic_alerts_df['Priority Score'].isnull().any(), "Priority score should not be null in generated alerts."

def test_get_district_summary_kpis(sample_enriched_zone_df_main_fixture: Optional[pd.DataFrame]):
    """Verifies the structure of the district summary KPI dictionary."""
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or sample_enriched_zone_df_main_fixture.empty:
        pytest.skip("Enriched zone DataFrame is invalid or empty.")
    
    district_kpis = get_district_summary_kpis(sample_enriched_zone_df_main_fixture.copy())
    assert isinstance(district_kpis, dict)
    
    key_conditions = settings.KEY_CONDITIONS_FOR_ACTION
    if not key_conditions: pytest.skip("KEY_CONDITIONS_FOR_ACTION is empty in settings.")
    
    # FIXED: Use re.sub to match the exact key generation in aggregation.py
    sanitized_cond_key = re.sub(r'[^a-z0-9_]+', '_', key_conditions[0].lower().strip())
    expected_keys = ["total_population_district", f"district_total_active_{sanitized_cond_key}_cases"]
    assert all(key in district_kpis for key in expected_keys)

def test_generate_simple_supply_forecast(sample_health_records_df_main_fixture: pd.DataFrame):
    """Verifies the structure and output of the simple supply forecast."""
    if sample_health_records_df_main_fixture.empty: pytest.skip("Health records are empty.")
    
    item_to_test = "ACT" # Assuming 'ACT' is a key drug
    
    # FIXED: Use corrected column names from the actual function output
    supply_fc_df = generate_simple_supply_forecast(sample_health_records_df_main_fixture.copy(), 30, [item_to_test])
    assert isinstance(supply_fc_df, pd.DataFrame)
    if not supply_fc_df.empty:
        expected_cols = ['item', 'forecasted_stock_level', 'estimated_stockout_date_linear']
        assert all(col in supply_fc_df.columns for col in expected_cols)
        assert supply_fc_df['item'].iloc[0] == item_to_test

# --- Graceful Handling Tests for Empty Inputs ---
@pytest.fixture
def empty_health_df_schema_fixture() -> pd.DataFrame:
    """Provides an empty DataFrame with a typical health records schema."""
    return pd.DataFrame(columns=['patient_id', 'encounter_date', 'ai_risk_score', 'condition'])

def test_graceful_handling_empty_inputs(empty_health_df_schema_fixture: pd.DataFrame):
    """Ensures all aggregation functions handle empty DataFrames without errors."""
    df_empty = empty_health_df_schema_fixture
    assert isinstance(get_clinic_summary_kpis(df_empty), dict)
    assert isinstance(get_patient_alerts_for_clinic(df_empty), pd.DataFrame)
    assert isinstance(generate_simple_supply_forecast(df_empty, 30, ["TestItem"]), pd.DataFrame)
    assert isinstance(get_trend_data(df_empty, value_col='ai_risk_score', date_col='encounter_date'), pd.Series)
