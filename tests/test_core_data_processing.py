# sentinel_project_root/tests/test_core_data_processing.py
# Pytest tests for functions in data_processing module for Sentinel.
# Path reflects old structure, tests adapted for new module structure.

import pytest
import pandas as pd
# GeoPandas is removed. Tests for GeoPandas-specific behavior are removed or adapted.
import numpy as np
from datetime import date, datetime, timedelta
import json # For checking 'geometry_obj' if it's dict

# Import functions and modules from the new 'data_processing' structure
from data_processing.helpers import clean_column_names, convert_to_numeric, hash_dataframe_safe
from data_processing.loaders import (
    load_health_records,
    load_iot_clinic_environment_data,
    load_zone_data # This now returns a DataFrame, not GeoDataFrame
)
from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
from data_processing.aggregation import (
    get_overall_kpis,
    get_chw_summary_kpis,
    get_clinic_summary_kpis,
    get_clinic_environmental_summary_kpis,
    get_district_summary_kpis,
    get_trend_data
)
# For supply forecast, import the simple model from analytics
from analytics.supply_forecasting import generate_simple_supply_forecast
# For patient alerts, import from analytics
from analytics.alerting import get_patient_alerts_for_clinic

from config import settings # Use new settings module
import logging
logger = logging.getLogger(__name__)


# --- Tests for Helper Functions (data_processing.helpers) ---
def test_clean_column_names_utility(): # Renamed test
    df_dirty_cols_test = pd.DataFrame(columns=['Test Column One', 'Another-Col WITH_Space  ', 'already_good_col', '  leading_space_col', 'trailing_space_col ', 'col(with)paren'])
    df_cleaned_cols_res = clean_column_names(df_dirty_cols_test.copy())
    expected_cleaned_cols = ['test_column_one', 'another_col_with_space', 'already_good_col', 'leading_space_col', 'trailing_space_col', 'col_with_paren']
    assert list(df_cleaned_cols_res.columns) == expected_cleaned_cols, "Column cleaning utility did not produce the expected column names."
    
    df_empty_for_clean = pd.DataFrame()
    assert list(clean_column_names(df_empty_for_clean.copy()).columns) == [], "Cleaning column names of an empty DataFrame failed."
    # Test idempotency of cleaning
    assert list(clean_column_names(df_cleaned_cols_res.copy()).columns) == expected_cleaned_cols, "Re-cleaning already cleaned column names changed them."

def test_convert_to_numeric_utility(): # Renamed test
    mixed_dirty_series_test = pd.Series(['105.5', '28.0', 'invalid_num', None, '60', True, False, 'NaN']) # Added 'NaN' string
    
    # Test conversion with default_value=np.nan
    series_to_nan_res = convert_to_numeric(mixed_dirty_series_test.copy(), default_value=np.nan)
    expected_series_nan = pd.Series([105.5, 28.0, np.nan, np.nan, 60.0, 1.0, 0.0, np.nan], dtype=float) # True/False become 1.0/0.0
    pd.testing.assert_series_equal(series_to_nan_res, expected_series_nan, check_dtype=True, check_exact=False, rtol=1e-5)

    # Test conversion with default_value=0
    series_to_zero_res = convert_to_numeric(mixed_dirty_series_test.copy(), default_value=0)
    expected_series_zero = pd.Series([105.5, 28.0, 0.0, 0.0, 60.0, 1.0, 0.0, 0.0], dtype=float)
    pd.testing.assert_series_equal(series_to_zero_res, expected_series_zero, check_dtype=True, check_exact=False, rtol=1e-5)

    # Test with already numeric series
    numeric_series_input = pd.Series([1.0, 2.5, 3.0, np.nan])
    pd.testing.assert_series_equal(convert_to_numeric(numeric_series_input.copy()), pd.Series([1.0, 2.5, 3.0, np.nan], dtype=float), check_dtype=True)
    # Test with empty series
    pd.testing.assert_series_equal(convert_to_numeric(pd.Series([], dtype=object)), pd.Series([], dtype=float), check_dtype=True)


def test_hash_dataframe_safe_utility(sample_zone_data_df_main_fixture: pd.DataFrame): # Renamed, uses DataFrame
    if not isinstance(sample_zone_data_df_main_fixture, pd.DataFrame) or \
       sample_zone_data_df_main_fixture.empty:
        pytest.skip("Sample Zone DataFrame for hashing utility is invalid or empty.")
    
    df_to_hash_test = sample_zone_data_df_main_fixture.copy()
    hash_val_1_test = hash_dataframe_safe(df_to_hash_test)
    assert isinstance(hash_val_1_test, str) and hash_val_1_test is not None, "Hashing valid DataFrame failed with safe hasher."
    
    assert hash_dataframe_safe(None) is None, "Hashing None DataFrame should return None with safe hasher."
    assert hash_dataframe_safe(pd.DataFrame()) == "empty_dataframe", "Hashing empty DataFrame returned unexpected value with safe hasher."

    # Test that modification changes the hash
    df_modified_for_hash_test = df_to_hash_test.copy()
    if 'population' in df_modified_for_hash_test.columns and not df_modified_for_hash_test.empty:
        original_pop_val_test = df_modified_for_hash_test.loc[0, 'population']
        df_modified_for_hash_test.loc[0, 'population'] = (original_pop_val_test + 1500) if pd.notna(original_pop_val_test) else 1500
        hash_val_2_test = hash_dataframe_safe(df_modified_for_hash_test)
        assert isinstance(hash_val_2_test, str) and hash_val_1_test != hash_val_2_test, "Modified DataFrame did not produce a different hash with safe hasher."
    else:
        logger.warning("Skipping DataFrame hash modification sub-test; 'population' col missing or DataFrame empty.")


# --- Tests for Data Loading (data_processing.loaders) ---
# These tests use fixtures that simulate the *output* of the load functions.
def test_load_health_records_simulated_output_check(sample_health_records_df_main_fixture: pd.DataFrame): # Renamed fixture
    df_health_test = sample_health_records_df_main_fixture # This fixture already applies AI models
    assert isinstance(df_health_test, pd.DataFrame), "Health records fixture is not a pandas DataFrame."
    if df_health_test.empty: pytest.skip("Health records fixture is empty for schema/type check.")
    
    key_cols_health_check = ['patient_id', 'encounter_date', 'ai_risk_score', 'min_spo2_pct', 'condition', 'zone_id']
    for col_hc in key_cols_health_check:
        assert col_hc in df_health_test.columns, f"Key column '{col_hc}' missing in health records fixture."
    assert pd.api.types.is_datetime64_any_dtype(df_health_test['encounter_date']), "'encounter_date' in health records is not datetime type."
    assert pd.api.types.is_numeric_dtype(df_health_test['ai_risk_score']), "'ai_risk_score' in health records is not numeric type."

def test_load_iot_data_simulated_output_check(sample_iot_clinic_df_main_fixture: pd.DataFrame): # Renamed fixture
    df_iot_test = sample_iot_clinic_df_main_fixture
    assert isinstance(df_iot_test, pd.DataFrame), "IoT data fixture is not a pandas DataFrame."
    if df_iot_test.empty: pytest.skip("IoT data fixture is empty for schema/type check.")
        
    key_cols_iot_check = ['timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm', 'zone_id']
    for col_ic in key_cols_iot_check:
        assert col_ic in df_iot_test.columns, f"Key column '{col_ic}' missing in IoT data fixture."
    assert pd.api.types.is_datetime64_any_dtype(df_iot_test['timestamp']), "'timestamp' in IoT data is not datetime type."
    assert pd.api.types.is_numeric_dtype(df_iot_test['avg_co2_ppm']) or df_iot_test['avg_co2_ppm'].isnull().all(), \
        "'avg_co2_ppm' in IoT data is not numeric or all NaN."

def test_load_zone_data_simulated_output_check(sample_zone_data_df_main_fixture: pd.DataFrame): # Renamed fixture, now DataFrame
    df_zone_test = sample_zone_data_df_main_fixture # This fixture simulates the output of the new load_zone_data
    assert isinstance(df_zone_test, pd.DataFrame), "Zone data fixture is not a pandas DataFrame."
    if df_zone_test.empty: pytest.skip("Zone data fixture is empty for schema/type check.")
        
    # Key columns expected from the new loader (which includes geometry as string/dict and CRS info)
    key_cols_zone_check = ['zone_id', 'name', 'population', 'geometry', 'geometry_obj', 'crs']
    for col_zc in key_cols_zone_check:
        assert col_zc in df_zone_test.columns, f"Key column '{col_zc}' missing in zone data DataFrame fixture."
    
    # Check 'geometry_obj' type if column exists and has data
    if 'geometry_obj' in df_zone_test.columns and df_zone_test['geometry_obj'].notna().any():
        first_geom_obj = df_zone_test['geometry_obj'].dropna().iloc[0]
        assert isinstance(first_geom_obj, dict), "'geometry_obj' should contain Python dicts parsed from GeoJSON."
        assert "type" in first_geom_obj and "coordinates" in first_geom_obj, "geometry_obj dict lacks GeoJSON structure."
        
    assert 'crs' in df_zone_test.columns and df_zone_test['crs'].notna().all(), "Zone data DataFrame missing 'crs' or has NaNs."
    assert df_zone_test['crs'].iloc[0].upper() == getattr(settings, 'DEFAULT_CRS_STANDARD', "EPSG:4326").upper(), \
        "Zone data DataFrame has incorrect CRS information."


# --- Test for Enrichment Logic (data_processing.enrichment) ---
def test_enrich_zone_df_with_aggregates_values_check( # Renamed test and fixture
    sample_enriched_zone_df_main_fixture: Optional[pd.DataFrame], # Now a DataFrame
    sample_health_records_df_main_fixture: pd.DataFrame
):
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or \
       sample_enriched_zone_df_main_fixture.empty or \
       sample_health_records_df_main_fixture.empty:
        pytest.skip("Cannot test enrichment values with empty/invalid input fixtures for Sentinel (DataFrame version).")
    
    df_enriched_test = sample_enriched_zone_df_main_fixture
    health_df_source_test = sample_health_records_df_main_fixture
    
    zone_to_verify_enrich = 'ZoneA' # Assuming 'ZoneA' exists in sample data
    if zone_to_verify_enrich not in df_enriched_test.get('zone_id', pd.Series(dtype=str)).tolist():
        pytest.skip(f"Test zone '{zone_to_verify_enrich}' not found in the enriched zone DataFrame fixture.")
    
    # Example 1: active_tb_cases for ZoneA
    # Construct dynamic column name as per enrichment logic (lowercase, underscore, no "(Severe)")
    tb_condition_key_from_settings = next((c for c in getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', []) if "TB" in c.upper()), "TB") # Find actual TB key
    tb_col_dynamic_enrich = f"active_{tb_condition_key_from_settings.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases"
    
    if tb_col_dynamic_enrich in df_enriched_test.columns:
        expected_tb_in_zone_a_test = health_df_source_test[
            (health_df_source_test['zone_id'] == zone_to_verify_enrich) & 
            (health_df_source_test.get('condition', pd.Series(dtype=str)).astype(str).str.contains(tb_condition_key_from_settings, case=False, na=False))
        ]['patient_id'].nunique()
        
        actual_tb_in_zone_a_test = df_enriched_test[df_enriched_test['zone_id'] == zone_to_verify_enrich][tb_col_dynamic_enrich].iloc[0]
        assert actual_tb_in_zone_a_test == expected_tb_in_zone_a_test, f"Mismatch in '{tb_col_dynamic_enrich}' for '{zone_to_verify_enrich}'."

    # Example 2: avg_risk_score for ZoneA
    if 'avg_risk_score' in df_enriched_test.columns:
        expected_avg_risk_zone_a_test = health_df_source_test[health_df_source_test['zone_id'] == zone_to_verify_enrich]['ai_risk_score'].mean()
        actual_avg_risk_zone_a_test = df_enriched_test[df_enriched_test['zone_id'] == zone_to_verify_enrich]['avg_risk_score'].iloc[0]
        
        if pd.notna(expected_avg_risk_zone_a_test) and pd.notna(actual_avg_risk_zone_a_test):
            assert np.isclose(actual_avg_risk_zone_a_test, expected_avg_risk_zone_a_test, rtol=1e-3, equal_nan=False), \
                f"Mismatch in 'avg_risk_score' for '{zone_to_verify_enrich}'."
        else: # Check if both are NaN, or one is NaN (depends on expected behavior if source data causes NaN mean)
            assert pd.isna(actual_avg_risk_zone_a_test) == pd.isna(expected_avg_risk_zone_a_test), \
                f"NaN state mismatch for 'avg_risk_score' in '{zone_to_verify_enrich}'. Expected NaN: {pd.isna(expected_avg_risk_zone_a_test)}, Actual NaN: {pd.isna(actual_avg_risk_zone_a_test)}."


# --- Tests for KPI and Summary Calculation Functions (data_processing.aggregation) ---

def test_get_clinic_summary_kpis_structure_check(sample_health_records_df_main_fixture: pd.DataFrame): # Renamed
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records empty for clinic summary KPI structure test.")
    
    # Use a relevant slice of data for the summary test
    df_clinic_period_test = sample_health_records_df_main_fixture[
        (sample_health_records_df_main_fixture['encounter_date'] >= pd.Timestamp('2023-10-01')) &
        (sample_health_records_df_main_fixture['encounter_date'] <= pd.Timestamp('2023-10-20')) & # Wider period
        (sample_health_records_df_main_fixture['clinic_id'] == 'CLINIC01') # Focus on one clinic
    ].copy()

    if df_clinic_period_test.empty: 
        pytest.skip("No data for CLINIC01 in the specified period for clinic summary KPI test.")
    
    clinic_summary_output_test = get_clinic_summary_kpis(df_clinic_period_test) # Use renamed function
    assert isinstance(clinic_summary_output_test, dict), "get_clinic_summary_kpis did not return a dict."
    
    expected_top_level_kpi_keys = [
        "overall_avg_test_turnaround_conclusive_days", "perc_critical_tests_tat_met",
        "total_pending_critical_tests_patients", "sample_rejection_rate_perc",
        "key_drug_stockouts_count", "test_summary_details" # This contains nested details
    ]
    for key_kpi in expected_top_level_kpi_keys:
        assert key_kpi in clinic_summary_output_test, f"Key '{key_kpi}' missing in get_clinic_summary_kpis output."
    
    assert isinstance(clinic_summary_output_test.get("test_summary_details"), dict), "'test_summary_details' in clinic summary is not a dict."
    
    # Check structure of one entry in test_summary_details using a configured test type
    malaria_rdt_config_key_test = "RDT-Malaria" # Original key from settings.KEY_TEST_TYPES_FOR_ANALYSIS
    key_test_types_config = getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {})
    if malaria_rdt_config_key_test in key_test_types_config:
        malaria_display_name_test = key_test_types_config[malaria_rdt_config_key_test].get("display_name", malaria_rdt_config_key_test)
        
        if malaria_display_name_test in clinic_summary_output_test.get("test_summary_details", {}):
            malaria_test_detail_check = clinic_summary_output_test["test_summary_details"][malaria_display_name_test]
            expected_detail_metric_sub_keys = [
                "positive_rate_perc", "avg_tat_days", "perc_met_tat_target",
                "pending_count_patients", "rejected_count_patients", "total_conclusive_tests"
            ]
            for detail_sub_key in expected_detail_metric_sub_keys:
                assert detail_sub_key in malaria_test_detail_check, \
                    f"Detail key '{detail_sub_key}' missing for '{malaria_display_name_test}' in test_summary_details."
        else:
            logger.info(f"Test display name '{malaria_display_name_test}' not found in test_summary_details. This might be okay if no such tests in sample period for CLINIC01.")


def test_get_clinic_environmental_summary_kpis_check(sample_iot_clinic_df_main_fixture: pd.DataFrame): # Renamed
    if sample_iot_clinic_df_main_fixture.empty:
        pytest.skip("Sample IoT data fixture empty for environmental summary KPI test.")
    
    iot_summary_output_test = get_clinic_environmental_summary_kpis(sample_iot_clinic_df_main_fixture) # Use renamed
    assert isinstance(iot_summary_output_test, dict), "get_clinic_environmental_summary_kpis did not return a dict."
    
    assert 'avg_co2_overall_ppm' in iot_summary_output_test, "Missing 'avg_co2_overall_ppm' in IoT environmental summary."
    assert 'rooms_co2_very_high_alert_latest_count' in iot_summary_output_test, "Missing 'rooms_co2_very_high_alert_latest_count' in IoT summary."
    
    # Validate alert count based on sample data and config thresholds from settings
    latest_readings_per_room_test = sample_iot_clinic_df_main_fixture.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
    co2_very_high_thresh_test = getattr(settings, 'ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500)
    
    # Ensure avg_co2_ppm is numeric for comparison
    if 'avg_co2_ppm' in latest_readings_per_room_test.columns:
        latest_readings_per_room_test['avg_co2_ppm_numeric'] = pd.to_numeric(latest_readings_per_room_test['avg_co2_ppm'], errors='coerce')
        expected_co2_alerts_test = (latest_readings_per_room_test['avg_co2_ppm_numeric'] > co2_very_high_thresh_test).sum()
        assert iot_summary_output_test.get('rooms_co2_very_high_alert_latest_count', -1) == expected_co2_alerts_test, \
            "Mismatch in calculated 'rooms_co2_very_high_alert_latest_count'."
    else:
        assert iot_summary_output_test.get('rooms_co2_very_high_alert_latest_count', 0) == 0, \
            "'avg_co2_ppm' missing, so very high alerts should be 0."


def test_get_patient_alerts_for_clinic_check(sample_health_records_df_main_fixture: pd.DataFrame): # Renamed from test_get_patient_alerts_for_clinic_sentinel
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records fixture empty for clinic patient alert testing.")
    
    # Filter for records that are more likely to trigger alerts (e.g., high risk score)
    # This helps ensure the alert generation logic has something to work with.
    risk_moderate_thresh_test = getattr(settings, 'RISK_SCORE_MODERATE_THRESHOLD', 60)
    df_alerts_input_test = sample_health_records_df_main_fixture[
        (sample_health_records_df_main_fixture['encounter_date'].dt.date >= date(2023,10,5)) & # Example period
        (sample_health_records_df_main_fixture['encounter_date'].dt.date <= date(2023,10,15)) & 
        (pd.to_numeric(sample_health_records_df_main_fixture.get('ai_risk_score'), errors='coerce').fillna(0) >= risk_moderate_thresh_test)
    ].copy()

    if df_alerts_input_test.empty:
        pytest.skip("No relevant data (e.g., high risk patients) in sample for clinic patient alert testing for the chosen period/filters.")
    
    clinic_alerts_df_output_test = get_patient_alerts_for_clinic(df_alerts_input_test) # Uses function from analytics.alerting
    assert isinstance(clinic_alerts_df_output_test, pd.DataFrame), "get_patient_alerts_for_clinic did not return a DataFrame."
    
    if not clinic_alerts_df_output_test.empty:
        expected_clinic_alert_cols_check = ['patient_id', 'encounter_date', 'Alert Reason', 'Priority Score', 'condition', 'ai_risk_score']
        for col_ac in expected_clinic_alert_cols_check:
            assert col_ac in clinic_alerts_df_output_test.columns, f"Clinic alerts DataFrame missing expected column: {col_ac}"
        assert not clinic_alerts_df_output_test['Priority Score'].isnull().any(), "'Priority Score' in clinic alerts DataFrame should not be NaN if records exist."


def test_get_district_summary_kpis_check(sample_enriched_zone_df_main_fixture: Optional[pd.DataFrame]): # Renamed, uses DataFrame
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or sample_enriched_zone_df_main_fixture.empty:
        pytest.skip("Enriched zone DataFrame fixture empty or invalid for district KPI test.")
        
    district_kpis_output_test = get_district_summary_kpis(sample_enriched_zone_df_main_fixture) # Uses aggregation function
    assert isinstance(district_kpis_output_test, dict), "get_district_summary_kpis did not return a dict."
    
    key_cond_list_for_dist_kpi = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
    if not key_cond_list_for_dist_kpi: pytest.fail("KEY_CONDITIONS_FOR_ACTION is empty in settings, cannot form dynamic KPI key.")

    expected_district_kpi_keys_check = [
        "total_population_district", "population_weighted_avg_ai_risk_score",
        "zones_meeting_high_risk_criteria_count", "district_avg_facility_coverage_score",
        f"district_total_active_{key_cond_list_for_dist_kpi[0].lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases", # Example dynamic key
        "district_overall_key_disease_prevalence_per_1000", "district_population_weighted_avg_steps"
    ]
    for key_dk in expected_district_kpi_keys_check:
        assert key_dk in district_kpis_output_test, f"District KPI key '{key_dk}' missing from output of get_district_summary_kpis."
    
    # Check logic for population-weighted averages if population is present and positive
    if district_kpis_output_test.get("total_population_district", 0) > 0:
        assert pd.notna(district_kpis_output_test.get("population_weighted_avg_ai_risk_score")), \
            "Population-weighted AI risk is NaN when total_population_district > 0."
        assert pd.notna(district_kpis_output_test.get("district_avg_facility_coverage_score")), \
            "Population-weighted facility coverage is NaN when total_population_district > 0."
    else: # If total population is 0, weighted averages might be NaN or simple mean based on impl.
        assert pd.isna(district_kpis_output_test.get("population_weighted_avg_ai_risk_score", np.nan)) or \
               isinstance(district_kpis_output_test.get("population_weighted_avg_ai_risk_score"), float), \
               "Population-weighted AI risk has unexpected value (should be NaN or float) with zero total population."


def test_generate_simple_supply_forecast_check(sample_health_records_df_main_fixture: pd.DataFrame): # Renamed, uses analytics function
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Health records fixture empty for simple supply forecast test.")
    
    item_for_forecast_test_simple = None
    key_drug_substrings_test = getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if 'item' in sample_health_records_df_main_fixture.columns and key_drug_substrings_test:
        for drug_substr_test in key_drug_substrings_test:
            if sample_health_records_df_main_fixture['item'].astype(str).str.contains(drug_substr_test, case=False, na=False).any():
                item_for_forecast_test_simple = sample_health_records_df_main_fixture[
                    sample_health_records_df_main_fixture['item'].astype(str).str.contains(drug_substr_test, case=False, na=False)
                ]['item'].iloc[0]
                break # Found a testable item
    if not item_for_forecast_test_simple:
        pytest.skip("No key drugs from settings.KEY_DRUG_SUBSTRINGS_SUPPLY found in sample data. Cannot test simple supply forecast.")

    supply_forecast_df_output_simple = generate_simple_supply_forecast(
        sample_health_records_df_main_fixture, item_filter_list=[item_for_forecast_test_simple]
    )
    assert isinstance(supply_forecast_df_output_simple, pd.DataFrame), "generate_simple_supply_forecast did not return a DataFrame."
    
    if not supply_forecast_df_output_simple.empty:
        expected_supply_fc_cols_simple = ['item', 'forecast_date', 'forecasted_stock_level', 
                                          'forecasted_days_of_supply', 'estimated_stockout_date_linear', 
                                          'initial_stock_at_forecast_start', 'base_consumption_rate_per_day']
        for col_sfc in expected_supply_fc_cols_simple:
            assert col_sfc in supply_forecast_df_output_simple.columns, f"Simple supply forecast DataFrame missing column: {col_sfc}"
        assert supply_forecast_df_output_simple['item'].iloc[0] == item_for_forecast_test_simple, "Forecasted item name mismatch in simple forecast."
        assert pd.api.types.is_datetime64_any_dtype(supply_forecast_df_output_simple['forecast_date']), \
            "'forecast_date' column in simple supply forecast is not datetime type."
        # estimated_stockout_date_linear can be NaT if stockout is beyond forecast period
        assert pd.api.types.is_datetime64_any_dtype(supply_forecast_df_output_simple['estimated_stockout_date_linear']) or \
               supply_forecast_df_output_simple['estimated_stockout_date_linear'].isnull().all(), \
               "'estimated_stockout_date_linear' has incorrect type or non-NaT nulls in simple forecast."


# --- Graceful Handling Tests for Empty or Flawed Inputs (using new structure) ---
def test_graceful_handling_empty_inputs_for_summaries( # Renamed
    empty_health_df_schema_fixture: pd.DataFrame, 
    empty_iot_df_schema_fixture: pd.DataFrame, 
    empty_enriched_zone_df_schema_fixture: pd.DataFrame # Now a DataFrame schema
):
    # Test aggregation functions with empty DataFrames matching expected schemas
    assert isinstance(get_overall_kpis(empty_health_df_schema_fixture.copy()), dict), "get_overall_kpis failed on empty schema input."
    assert isinstance(get_chw_summary_kpis(empty_health_df_schema_fixture.copy(), for_date=date.today()), dict), "get_chw_summary_kpis failed on empty schema."
    
    clinic_summary_empty_res_test = get_clinic_summary_kpis(empty_health_df_schema_fixture.copy())
    assert isinstance(clinic_summary_empty_res_test, dict) and "test_summary_details" in clinic_summary_empty_res_test, \
        "get_clinic_summary_kpis structure error on empty schema input."
    
    assert isinstance(get_clinic_environmental_summary_kpis(empty_iot_df_schema_fixture.copy()), dict), \
        "get_clinic_environmental_summary_kpis failed on empty schema."
    
    assert isinstance(get_patient_alerts_for_clinic(empty_health_df_schema_fixture.copy()), pd.DataFrame), \
        "get_patient_alerts_for_clinic (analytics) failed on empty schema." # from analytics
        
    assert isinstance(get_district_summary_kpis(empty_enriched_zone_df_schema_fixture.copy()), dict), \
        "get_district_summary_kpis failed on empty enriched zone schema."
        
    assert isinstance(get_trend_data(empty_health_df_schema_fixture.copy(), value_col='ai_risk_score', date_col='encounter_date'), pd.Series), \
        "get_trend_data failed on empty schema (should return empty Series)."
        
    assert isinstance(generate_simple_supply_forecast(empty_health_df_schema_fixture.copy()), pd.DataFrame), \
        "generate_simple_supply_forecast (analytics) failed on empty schema." # from analytics


def test_handling_missing_critical_columns_in_kpi_summaries(sample_health_records_df_main_fixture: pd.DataFrame): # Renamed
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample data fixture empty, cannot test missing columns impact on KPI summaries.")
    
    # Test get_overall_kpis
    df_no_risk_kpi = sample_health_records_df_main_fixture.drop(columns=['ai_risk_score'], errors='ignore')
    kpis_output_no_risk_test = get_overall_kpis(df_no_risk_kpi.copy())
    assert pd.isna(kpis_output_no_risk_test.get('avg_patient_ai_risk_period', np.nan)), \
        "avg_patient_ai_risk_period should be NaN when 'ai_risk_score' column is missing in get_overall_kpis."

    df_no_condition_kpi = sample_health_records_df_main_fixture.drop(columns=['condition'], errors='ignore')
    kpis_output_no_condition_test = get_overall_kpis(df_no_condition_kpi.copy())
    key_cond_list_test_kpi = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
    if key_cond_list_test_kpi:
        example_cond_key_kpi = key_cond_list_test_kpi[0]
        formatted_cond_key_for_kpi = f"active_{example_cond_key_kpi.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_period"
        assert kpis_output_no_condition_test.get(formatted_cond_key_for_kpi, 0) == 0, \
            f"KPI '{formatted_cond_key_for_kpi}' should default to 0 when 'condition' column is missing in get_overall_kpis."
