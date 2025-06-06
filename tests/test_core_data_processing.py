# sentinel_project_root/tests/test_core_data_processing.py
# Pytest tests for functions in data_processing module for Sentinel.

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import json
import re # For condition name processing
import hashlib # Added for hash test

from data_processing.helpers import clean_column_names, convert_to_numeric, hash_dataframe_safe
from data_processing.loaders import (load_health_records, load_iot_clinic_environment_data, load_zone_data)
from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
from data_processing.aggregation import (
    get_overall_kpis, get_chw_summary_kpis, get_clinic_summary_kpis,
    get_clinic_environmental_summary_kpis, get_district_summary_kpis, get_trend_data
)
from analytics.supply_forecasting import generate_simple_supply_forecast
from analytics.alerting import get_patient_alerts_for_clinic
from config import settings
import logging
logger = logging.getLogger(__name__)

# Fixtures are sourced from conftest.py

# --- Tests for Helper Functions ---
def test_clean_column_names():
    df_dirty = pd.DataFrame(columns=['Test Col One', 'Another-Col WITH_Space  ', 'already_ok', '  lead_space', 'trail_space ', 'col(paren)'])
    df_cleaned = clean_column_names(df_dirty.copy())
    expected = ['test_col_one', 'another_col_with_space', 'already_ok', 'lead_space', 'trail_space', 'col_paren']
    assert list(df_cleaned.columns) == expected
    assert list(clean_column_names(pd.DataFrame()).columns) == []
    assert list(clean_column_names(df_cleaned.copy()).columns) == expected # Idempotency

def test_convert_to_numeric():
    mixed_series = pd.Series(['101.5', '20', 'bad', None, '70', True, False, 'NaN', '88.00'])
    
    series_nan_default = convert_to_numeric(mixed_series.copy(), default_value=np.nan, target_type=float)
    expected_nan = pd.Series([101.5, 20.0, np.nan, np.nan, 70.0, 1.0, 0.0, np.nan, 88.0], dtype=float)
    pd.testing.assert_series_equal(series_nan_default, expected_nan, check_dtype=True)

    series_zero_default = convert_to_numeric(mixed_series.copy(), default_value=0, target_type=float)
    expected_zero = pd.Series([101.5, 20.0, 0.0, 0.0, 70.0, 1.0, 0.0, 0.0, 88.0], dtype=float)
    pd.testing.assert_series_equal(series_zero_default, expected_zero, check_dtype=True)

    series_int_default = convert_to_numeric(pd.Series(['10', '20', 'bad', '30.0']), default_value=0, target_type=int)
    expected_int = pd.Series([10, 20, 0, 30], dtype=int)
    pd.testing.assert_series_equal(series_int_default, expected_int, check_dtype=True)
    
    # Test with already numeric series
    num_series = pd.Series([1.0, 2.5, np.nan])
    pd.testing.assert_series_equal(convert_to_numeric(num_series.copy()), pd.Series([1.0, 2.5, np.nan], dtype=float))
    pd.testing.assert_series_equal(convert_to_numeric(pd.Series([], dtype=object)), pd.Series([], dtype=float))


def test_hash_dataframe_safe(sample_zone_data_df_main_fixture: pd.DataFrame):
    if not isinstance(sample_zone_data_df_main_fixture, pd.DataFrame) or sample_zone_data_df_main_fixture.empty:
        pytest.skip("Sample Zone DataFrame for hashing invalid/empty.")
    
    df_hash = sample_zone_data_df_main_fixture.copy()
    hash1 = hash_dataframe_safe(df_hash)
    assert isinstance(hash1, str) and hash1 is not None
    assert hash_dataframe_safe(None) is None
    # Corrected hash for empty DataFrame to match implementation
    assert hash_dataframe_safe(pd.DataFrame()) == hashlib.sha256("empty_dataframe_cols:".encode('utf-8')).hexdigest()

    df_modified = df_hash.copy()
    if 'population' in df_modified.columns and not df_modified.empty:
        orig_pop = df_modified.loc[0, 'population']
        df_modified.loc[0, 'population'] = (orig_pop + 100) if pd.notna(orig_pop) else 100
        hash2 = hash_dataframe_safe(df_modified)
        assert isinstance(hash2, str) and hash1 != hash2
    else: logger.warning("Skipping hash modification sub-test; 'population' missing or df empty.")

# --- Tests for Data Loaders ---
def test_load_health_records_output(sample_health_records_df_main_fixture: pd.DataFrame):
    df = sample_health_records_df_main_fixture # Fixture is already processed output
    assert isinstance(df, pd.DataFrame)
    if df.empty: pytest.skip("Health records fixture empty.")
    key_cols = ['patient_id', 'encounter_date', 'ai_risk_score', 'condition', 'zone_id']
    assert all(col in df.columns for col in key_cols)
    assert pd.api.types.is_datetime64_any_dtype(df['encounter_date'])
    assert pd.api.types.is_numeric_dtype(df.get('ai_risk_score', pd.Series(dtype=float))) # Check if exists and numeric

def test_load_iot_data_output(sample_iot_clinic_df_main_fixture: pd.DataFrame):
    df = sample_iot_clinic_df_main_fixture
    assert isinstance(df, pd.DataFrame)
    if df.empty: pytest.skip("IoT data fixture empty.")
    key_cols = ['timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm', 'zone_id']
    assert all(col in df.columns for col in key_cols)
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    assert pd.api.types.is_numeric_dtype(df.get('avg_co2_ppm', pd.Series(dtype=float))) or df.get('avg_co2_ppm', pd.Series(dtype=float)).isnull().all()

def test_load_zone_data_output(sample_zone_data_df_main_fixture: pd.DataFrame):
    df = sample_zone_data_df_main_fixture
    assert isinstance(df, pd.DataFrame)
    if df.empty: pytest.skip("Zone data fixture empty.")
    key_cols = ['zone_id', 'name', 'population', 'geometry_obj', 'crs'] # geometry_obj expected
    assert all(col in df.columns for col in key_cols)
    if 'geometry_obj' in df.columns and df['geometry_obj'].notna().any():
        first_geom = df['geometry_obj'].dropna().iloc[0]
        assert isinstance(first_geom, dict) and "type" in first_geom and "coordinates" in first_geom
    assert 'crs' in df.columns and df['crs'].notna().all()
    assert df['crs'].iloc[0].upper() == settings.DEFAULT_CRS_STANDARD.upper()

# --- Test for Enrichment ---
def test_enrich_zone_df_aggregates_values(sample_enriched_zone_df_main_fixture: Optional[pd.DataFrame], sample_health_records_df_main_fixture: pd.DataFrame):
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or sample_enriched_zone_df_main_fixture.empty or sample_health_records_df_main_fixture.empty:
        pytest.skip("Cannot test enrichment values with empty/invalid fixtures.")
    
    df_enriched = sample_enriched_zone_df_main_fixture
    health_df_src = sample_health_records_df_main_fixture
    zone_verify = 'ZoneA'
    if zone_verify not in df_enriched.get('zone_id', pd.Series(dtype=str)).tolist():
        pytest.skip(f"Test zone '{zone_verify}' not found in enriched zone fixture.")
    
    tb_key = next((c for c in settings.KEY_CONDITIONS_FOR_ACTION if "TB" in c.upper()), "TB")
    # CORRECTED: Use re.sub to match the exact column name generation in enrichment.py
    tb_col = f"active_{re.sub(r'[^a-z0-9_]+', '_', tb_key.lower().strip())}_cases"
    if tb_col in df_enriched.columns:
        expected_tb_zone_a = health_df_src[(health_df_src['zone_id'] == zone_verify) & (health_df_src.get('condition', pd.Series(dtype=str)).astype(str).str.contains(tb_key, case=False, na=False))]['patient_id'].nunique()
        actual_tb_zone_a = df_enriched[df_enriched['zone_id'] == zone_verify][tb_col].iloc[0]
        assert actual_tb_zone_a == expected_tb_zone_a, f"Mismatch in '{tb_col}' for '{zone_verify}'."

    if 'avg_risk_score' in df_enriched.columns:
        expected_risk_zone_a = health_df_src[health_df_src['zone_id'] == zone_verify]['ai_risk_score'].mean()
        actual_risk_zone_a = df_enriched[df_enriched['zone_id'] == zone_verify]['avg_risk_score'].iloc[0]
        if pd.notna(expected_risk_zone_a) and pd.notna(actual_risk_zone_a):
            assert np.isclose(actual_risk_zone_a, expected_risk_zone_a, rtol=1e-3)
        else: assert pd.isna(actual_risk_zone_a) == pd.isna(expected_risk_zone_a)

# --- Tests for Aggregation Functions ---
def test_get_clinic_summary_kpis_structure(sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty: pytest.skip("Sample health records empty for clinic KPI structure test.")
    df_clinic_period = sample_health_records_df_main_fixture[
        (sample_health_records_df_main_fixture['encounter_date'] >= pd.Timestamp('2023-10-01')) &
        (sample_health_records_df_main_fixture['encounter_date'] <= pd.Timestamp('2023-10-25'))
    ].copy()
    if df_clinic_period.empty: pytest.skip("No data for period for clinic summary KPI test.")
    
    clinic_summary = get_clinic_summary_kpis(df_clinic_period)
    assert isinstance(clinic_summary, dict)
    expected_keys = ["overall_avg_test_turnaround_conclusive_days", "perc_critical_tests_tat_met", "test_summary_details"]
    assert all(key in clinic_summary for key in expected_keys)
    assert isinstance(clinic_summary.get("test_summary_details"), dict)
    
    # This check is more robust, not tied to a specific test name
    if clinic_summary.get("test_summary_details"):
        first_test_detail_key = next(iter(clinic_summary["test_summary_details"]))
        malaria_detail = clinic_summary["test_summary_details"][first_test_detail_key]
        expected_detail_sub_keys = ["positive_rate_perc", "avg_tat_days", "total_conclusive_tests"]
        assert all(sub_key in malaria_detail for sub_key in expected_detail_sub_keys)

def test_get_clinic_env_summary_kpis(sample_iot_clinic_df_main_fixture: pd.DataFrame):
    if sample_iot_clinic_df_main_fixture.empty: pytest.skip("Sample IoT data empty for env summary KPI test.")
    iot_summary = get_clinic_environmental_summary_kpis(sample_iot_clinic_df_main_fixture)
    assert isinstance(iot_summary, dict)
    assert 'avg_co2_overall_ppm' in iot_summary and 'rooms_co2_very_high_alert_latest_count' in iot_summary

    latest_readings = sample_iot_clinic_df_main_fixture.sort_values('timestamp', na_position='first').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
    co2_very_high_thresh = settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM
    if 'avg_co2_ppm' in latest_readings.columns:
        latest_readings['avg_co2_ppm_num'] = pd.to_numeric(latest_readings['avg_co2_ppm'], errors='coerce')
        expected_co2_alerts = (latest_readings['avg_co2_ppm_num'] > co2_very_high_thresh).sum()
        assert iot_summary.get('rooms_co2_very_high_alert_latest_count', -1) == expected_co2_alerts
    else: assert iot_summary.get('rooms_co2_very_high_alert_latest_count', 0) == 0

def test_get_patient_alerts_for_clinic(sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty: pytest.skip("Sample health records empty for clinic patient alert test.")
    risk_mod_thresh = settings.RISK_SCORE_MODERATE_THRESHOLD
    df_alerts_in = sample_health_records_df_main_fixture[
        (pd.to_datetime(sample_health_records_df_main_fixture['encounter_date']).dt.date >= date(2023,10,1)) &
        (pd.to_numeric(sample_health_records_df_main_fixture.get('ai_risk_score'), errors='coerce').fillna(0) >= risk_mod_thresh)
    ].copy()
    if df_alerts_in.empty: pytest.skip("No relevant data for clinic patient alert test.")
    
    clinic_alerts_df = get_patient_alerts_for_clinic(df_alerts_in)
    assert isinstance(clinic_alerts_df, pd.DataFrame)
    if not clinic_alerts_df.empty:
        expected_cols = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'ai_risk_score']
        assert all(col in clinic_alerts_df.columns for col in expected_cols)
        assert not clinic_alerts_df['Priority Score'].isnull().any()

def test_get_district_summary_kpis(sample_enriched_zone_df_main_fixture: Optional[pd.DataFrame]):
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or sample_enriched_zone_df_main_fixture.empty:
        pytest.skip("Enriched zone DF fixture empty/invalid for district KPI test.")
    district_kpis = get_district_summary_kpis(sample_enriched_zone_df_main_fixture)
    assert isinstance(district_kpis, dict)
    key_cond_list_dist = settings.KEY_CONDITIONS_FOR_ACTION
    if not key_cond_list_dist: pytest.fail("KEY_CONDITIONS_FOR_ACTION empty in settings.")
    # CORRECTED: Use re.sub to match the exact key generation in aggregation.py
    sanitized_cond_key = re.sub(r'[^a-z0-9_]+', '_', key_cond_list_dist[0].lower().strip())
    expected_keys = ["total_population_district", "population_weighted_avg_ai_risk_score",
                     f"district_total_active_{sanitized_cond_key}_cases"]
    assert all(key in district_kpis for key in expected_keys)
    if district_kpis.get("total_population_district", 0) > 0:
        assert pd.notna(district_kpis.get("population_weighted_avg_ai_risk_score"))

def test_generate_simple_supply_forecast(sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty: pytest.skip("Health records empty for simple supply forecast test.")
    item_test = None; key_drugs = settings.KEY_DRUG_SUBSTRINGS_SUPPLY
    if 'item' in sample_health_records_df_main_fixture.columns and key_drugs:
        for drug_sub in key_drugs:
            if sample_health_records_df_main_fixture['item'].astype(str).str.contains(drug_sub, case=False, na=False).any():
                item_test = sample_health_records_df_main_fixture[sample_health_records_df_main_fixture['item'].astype(str).str.contains(drug_sub, case=False, na=False)]['item'].iloc[0]
                break
    if not item_test: pytest.skip("No key drugs found in sample data for supply forecast test.")

    # Using the corrected function signature for generate_simple_supply_forecast
    supply_fc_df = generate_simple_supply_forecast(sample_health_records_df_main_fixture, 30, [item_test])
    assert isinstance(supply_fc_df, pd.DataFrame)
    if not supply_fc_df.empty:
        expected_cols = ['item', 'current_stock_level', 'estimated_stockout_date']
        assert all(col in supply_fc_df.columns for col in expected_cols)
        assert supply_fc_df['item'].iloc[0] == item_test

# --- Graceful Handling Tests ---
def test_graceful_handling_empty_inputs(empty_health_df_schema_fixture: pd.DataFrame, empty_iot_df_schema_fixture: pd.DataFrame, empty_enriched_zone_df_schema_fixture: pd.DataFrame):
    assert isinstance(get_overall_kpis(empty_health_df_schema_fixture.copy()), dict)
    assert isinstance(get_chw_summary_kpis(empty_health_df_schema_fixture.copy(), for_date=date.today()), dict)
    clinic_sum_empty = get_clinic_summary_kpis(empty_health_df_schema_fixture.copy())
    assert isinstance(clinic_sum_empty, dict) and "test_summary_details" in clinic_sum_empty
    assert isinstance(get_clinic_environmental_summary_kpis(empty_iot_df_schema_fixture.copy()), dict)
    assert isinstance(get_patient_alerts_for_clinic(empty_health_df_schema_fixture.copy()), pd.DataFrame)
    assert isinstance(get_district_summary_kpis(empty_enriched_zone_df_schema_fixture.copy()), dict)
    assert isinstance(get_trend_data(empty_health_df_schema_fixture.copy(), value_col='ai_risk_score', date_col='encounter_date'), pd.Series)
    assert isinstance(generate_simple_supply_forecast(empty_health_df_schema_fixture.copy(), 30, ["TestItem"]), pd.DataFrame)

def test_handling_missing_cols_in_kpis(sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty: pytest.skip("Sample data empty for missing cols KPI test.")
    
    df_no_risk = sample_health_records_df_main_fixture.drop(columns=['ai_risk_score'], errors='ignore')
    kpis_no_risk = get_overall_kpis(df_no_risk.copy())
    assert pd.isna(kpis_no_risk.get('avg_patient_ai_risk_period', np.nan))

    df_no_cond = sample_health_records_df_main_fixture.drop(columns=['condition'], errors='ignore')
    kpis_no_cond = get_overall_kpis(df_no_cond.copy())
    key_conds_test = settings.KEY_CONDITIONS_FOR_ACTION
    if key_conds_test:
        # CORRECTED: Use re.sub to match the exact key generation in aggregation.py
        example_key = f"active_{re.sub(r'[^a-z0-9_]+', '_', key_conds_test[0].lower().strip())}_cases_period"
        assert kpis_no_cond.get(example_key, 0) == 0
