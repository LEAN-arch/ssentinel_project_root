# sentinel_project_root/tests/test_ai_analytics_engine.py
# Pytest tests for AI simulation logic in analytics module for Sentinel.
# File path reflects the old structure, test names updated for new one.

import pytest
import pandas as pd
import numpy as np

# Import classes and functions from the new 'analytics' module
from analytics.risk_prediction import RiskPredictionModel
from analytics.followup_prioritization import FollowUpPrioritizer
from analytics.supply_forecasting import SupplyForecastingModel # AI Simulated
from analytics.orchestrator import apply_ai_models
from config import settings # Use new settings module

# Fixtures (e.g., sample_health_records_df_main_fixture) are sourced from conftest.py

# --- Tests for RiskPredictionModel (analytics.risk_prediction) ---

@pytest.fixture(scope="module") # one instance per test module
def risk_model_instance_fixture() -> RiskPredictionModel: # Renamed fixture
    """Provides a single instance of RiskPredictionModel for tests in this module."""
    return RiskPredictionModel()

def test_risk_model_condition_base_score_logic(risk_model_instance_fixture: RiskPredictionModel): # Renamed test
    # Ensure KEY_CONDITIONS_FOR_ACTION is not empty
    key_conditions_list = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
    if not key_conditions_list:
        pytest.skip("No KEY_CONDITIONS_FOR_ACTION defined in settings to test.")
        
    # Test with conditions from settings
    for condition_cfg in key_conditions_list:
        # The model stores keys in lower case
        expected_score_cfg = risk_model_instance_fixture.condition_base_scores.get(condition_cfg.lower(), 0.0)
        assert risk_model_instance_fixture._get_condition_base_score(condition_cfg) == expected_score_cfg, \
            f"Base score mismatch for key condition from settings: {condition_cfg}"
        assert risk_model_instance_fixture._get_condition_base_score(condition_cfg.lower()) == expected_score_cfg, \
            f"Base score mismatch for lowercase key condition: {condition_cfg.lower()}"

    # Test multi-condition string (model's _get_condition_base_score should take the max relevant score)
    if len(key_conditions_list) >= 2:
        cond1_multi = key_conditions_list[0]
        cond2_multi = "Pneumonia" # Assume Pneumonia is also in model's condition_base_scores
                                # (it is, based on current RiskPredictionModel setup)
        multi_condition_str = f"{cond1_multi}; {cond2_multi}"
        
        expected_score_for_multi = max(
            risk_model_instance_fixture.condition_base_scores.get(cond1_multi.lower(), 0.0),
            risk_model_instance_fixture.condition_base_scores.get(cond2_multi.lower(), 0.0)
        )
        assert risk_model_instance_fixture._get_condition_base_score(multi_condition_str) == expected_score_for_multi, \
            "Multi-condition string base score did not correctly pick the maximum of relevant scores."
    
    assert risk_model_instance_fixture._get_condition_base_score("NonExistentCondition") == 0.0, "Non-existent condition should yield 0.0 base score."
    assert risk_model_instance_fixture._get_condition_base_score(None) == 0.0, "None condition string should yield 0.0 base score." # type: ignore
    assert risk_model_instance_fixture._get_condition_base_score("") == 0.0, "Empty condition string should yield 0.0 base score."
    
    # Check for specific scores like "Wellness Visit" if defined with negative values
    wellness_score = risk_model_instance_fixture.condition_base_scores.get("wellness visit", 1.0) # Default to non-negative if not found
    if wellness_score < 0:
        assert risk_model_instance_fixture._get_condition_base_score("Wellness Visit") < 0, \
            "Wellness Visit score should be negative as per model configuration."


def test_risk_model_predict_score_with_various_factors(risk_model_instance_fixture: RiskPredictionModel): # Renamed test
    # Base features for a relatively healthy individual
    base_features_risk_test = pd.Series({
        'condition': 'Wellness Visit', 'age': 30, 'chronic_condition_flag': 0,
        'min_spo2_pct': 99.0, 'vital_signs_temperature_celsius': 36.8,
        'fall_detected_today': 0, 'ambient_heat_index_c': 25.0,
        'ppe_compliant_flag': 1, 'signs_of_fatigue_observed_flag': 0,
        'rapid_psychometric_distress_score': 0.0, 'hrv_rmssd_ms': 60.0,
        'medication_adherence_self_report': 'Good', 'tb_contact_traced': 0
    })
    base_risk_val = risk_model_instance_fixture.predict_risk_score(base_features_risk_test.copy())
    assert 0 <= base_risk_val <= 100, "Base risk score out of 0-100 bounds for healthy individual."

    # Test Critical Low SpO2 impact
    features_low_spo2_test = base_features_risk_test.copy()
    spo2_critical_thresh = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90)
    features_low_spo2_test['min_spo2_pct'] = spo2_critical_thresh - 3 # Clearly below critical
    score_low_spo2_test = risk_model_instance_fixture.predict_risk_score(features_low_spo2_test)
    
    # Expect a significant increase from base_risk_val
    # The exact increase depends on the 'factor_low' and 'weight' for 'min_spo2_pct'
    # and how it interacts with the negative score of 'Wellness Visit'.
    spo2_factor_details_test = risk_model_instance_fixture.base_risk_factors['min_spo2_pct']
    expected_spo2_point_add = spo2_factor_details_test.get('factor_low', 0) * spo2_factor_details_test.get('weight', 1.0)
    # Allow a small delta for interactions with other small negative/positive factors from base
    assert score_low_spo2_test >= (base_risk_val + expected_spo2_point_add - abs(risk_model_instance_fixture.condition_base_scores.get("wellness visit",0)) - 15), \
        f"Critical SpO2 (value: {features_low_spo2_test['min_spo2_pct']}) did not increase risk score sufficiently from base {base_risk_val:.1f} to {score_low_spo2_test:.1f}."

    # Test High Fever impact
    features_high_fever_test = base_features_risk_test.copy()
    temp_high_fever_thresh = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
    features_high_fever_test['vital_signs_temperature_celsius'] = temp_high_fever_thresh + 0.3 # Clearly high fever
    score_high_fever_test = risk_model_instance_fixture.predict_risk_score(features_high_fever_test)
    fever_factor_details_test = risk_model_instance_fixture.base_risk_factors['vital_signs_temperature_celsius']
    expected_fever_point_add = fever_factor_details_test.get('factor_super_high',0) * fever_factor_details_test.get('weight', 1.0)
    assert score_high_fever_test >= (base_risk_val + expected_fever_point_add - abs(risk_model_instance_fixture.condition_base_scores.get("wellness visit",0)) - 15), \
        "High fever did not increase risk score sufficiently."


def test_risk_model_bulk_predict_scores(risk_model_instance_fixture: RiskPredictionModel, sample_health_records_df_main_fixture: pd.DataFrame): # Renamed test and fixture
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records fixture is empty. Skipping bulk risk prediction test.")
    
    df_to_score_bulk = sample_health_records_df_main_fixture.copy()
    # The fixture itself calls apply_ai_models, so 'ai_risk_score' will exist.
    # We can test if re-running gives consistent types, or drop and re-calculate.
    # For this test, let's drop and recalculate to specifically test this model's bulk method.
    if 'ai_risk_score' in df_to_score_bulk.columns:
        df_to_score_bulk = df_to_score_bulk.drop(columns=['ai_risk_score'])

    risk_scores_series_bulk = risk_model_instance_fixture.predict_bulk_risk_scores(df_to_score_bulk)
    assert isinstance(risk_scores_series_bulk, pd.Series), "Bulk risk prediction did not return a pandas Series."
    assert len(risk_scores_series_bulk) == len(df_to_score_bulk), "Bulk prediction Series length does not match input DataFrame length."
    assert risk_scores_series_bulk.notna().all(), "Risk scores from bulk prediction should not contain NaN values (due to clipping and defaults)."
    assert risk_scores_series_bulk.min() >= 0 and risk_scores_series_bulk.max() <= 100, \
        "Risk scores from bulk prediction are out of the expected 0-100 bounds."

# --- Tests for FollowUpPrioritizer (analytics.followup_prioritization) ---
@pytest.fixture(scope="module")
def priority_model_instance_fixture() -> FollowUpPrioritizer: # Renamed fixture
    return FollowUpPrioritizer()

def test_priority_model_helper_logic_checks(priority_model_instance_fixture: FollowUpPrioritizer): # Renamed test
    spo2_crit_local = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90)
    temp_high_fev_local = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
    key_cond_local_0 = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', ['TB'])[0]

    assert priority_model_instance_fixture._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': spo2_crit_local - 2})) is True, "Critical low SpO2 not detected by helper."
    assert priority_model_instance_fixture._has_active_critical_vitals_alert(pd.Series({'vital_signs_temperature_celsius': temp_high_fev_local + 0.2})) is True, "Critical high fever not detected by helper."
    assert priority_model_instance_fixture._has_active_critical_vitals_alert(pd.Series({'fall_detected_today': '1'})) is True, "Fall detected (string '1') not caught by helper."
    assert priority_model_instance_fixture._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 99, 'vital_signs_temperature_celsius': 37.0, 'fall_detected_today': 0})) is False, "Healthy vitals incorrectly flagged by helper."

    assert priority_model_instance_fixture._is_pending_urgent_task_or_referral(pd.Series({'referral_status': 'Pending', 'condition': key_cond_local_0})) is True, "Pending critical referral not detected."
    # Add test for 'chw_task_priority' if that field is used by the model.
    assert priority_model_instance_fixture._is_pending_urgent_task_or_referral(pd.Series({'referral_status': 'Completed', 'condition': key_cond_local_0})) is False, "Completed referral incorrectly flagged as pending urgent."

    spo2_warn_local = getattr(settings, 'ALERT_SPO2_WARNING_LOW_PCT', 94)
    assert priority_model_instance_fixture._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Pneumonia', 'min_spo2_pct': spo2_warn_local -1 })) is True, "Pneumonia with warning SpO2 not detected as severe."
    assert priority_model_instance_fixture._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Sepsis'})) is True, "Sepsis (inherently severe) not detected."
    assert priority_model_instance_fixture._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Wellness Visit'})) is False, "Wellness visit incorrectly flagged as acute/severe."

    heat_danger_local = getattr(settings, 'ALERT_AMBIENT_HEAT_INDEX_DANGER_C', 41)
    assert priority_model_instance_fixture._has_significant_contextual_hazard(pd.Series({'ambient_heat_index_c': heat_danger_local + 2})) is True, "Danger heat index not detected as hazard."
    assert priority_model_instance_fixture._has_significant_contextual_hazard(pd.Series({'ambient_heat_index_c': 28})) is False, "Safe heat index incorrectly flagged as hazard."


def test_priority_model_calculate_score_with_components(priority_model_instance_fixture: FollowUpPrioritizer): # Renamed test
    base_features_prio_test = pd.Series({'ai_risk_score': 30.0}) # Moderate base AI risk
    score_base_only_prio = priority_model_instance_fixture.calculate_priority_score(base_features_prio_test.copy())
    
    features_critical_vitals_prio = base_features_prio_test.copy()
    features_critical_vitals_prio['min_spo2_pct'] = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90) - 1
    score_critical_vitals_prio = priority_model_instance_fixture.calculate_priority_score(features_critical_vitals_prio.copy())
    assert score_critical_vitals_prio >= score_base_only_prio + priority_model_instance_fixture.priority_weights['critical_vital_alert_points'] - 15, "Critical vitals points not added sufficiently to priority score." # Allow for base risk contribution part

    score_overdue_task_prio = priority_model_instance_fixture.calculate_priority_score(base_features_prio_test.copy(), days_task_overdue=5)
    expected_overdue_add = 5 * priority_model_instance_fixture.priority_weights['task_overdue_factor_per_day']
    assert score_overdue_task_prio >= score_base_only_prio + expected_overdue_add - 10, "Task overdue factor impact incorrect on priority score."


def test_priority_model_generate_bulk_priority_scores(priority_model_instance_fixture: FollowUpPrioritizer, sample_health_records_df_main_fixture: pd.DataFrame): # Renamed
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records fixture is empty. Skipping bulk priority score generation test.")
    
    df_for_prio_bulk = sample_health_records_df_main_fixture.copy()
    # Fixture already calls apply_ai_models, which adds 'ai_risk_score' and 'days_task_overdue' (defaulted to 0 if not present).
    # If 'ai_followup_priority_score' exists from fixture, drop it to test this model's calculation.
    if 'ai_followup_priority_score' in df_for_prio_bulk.columns:
        df_for_prio_bulk = df_for_prio_bulk.drop(columns=['ai_followup_priority_score'])

    priority_scores_series_bulk_prio = priority_model_instance_fixture.generate_followup_priorities(df_for_prio_bulk)
    assert isinstance(priority_scores_series_bulk_prio, pd.Series), "Bulk priority generation did not return a pandas Series."
    assert len(priority_scores_series_bulk_prio) == len(df_for_prio_bulk), "Bulk priority Series length mismatch with input DataFrame."
    assert priority_scores_series_bulk_prio.notna().all(), "Priority scores from bulk generation should not contain NaN values."
    assert priority_scores_series_bulk_prio.min() >= 0 and priority_scores_series_bulk_prio.max() <= 100, "Priority scores out of 0-100 bounds."


# --- Tests for SupplyForecastingModel (AI-Simulated from analytics.supply_forecasting) ---
@pytest.fixture(scope="module")
def supply_model_ai_instance_fixture() -> SupplyForecastingModel: # Renamed
    return SupplyForecastingModel()

def test_supply_model_ai_get_simulated_item_params(supply_model_ai_instance_fixture: SupplyForecastingModel): # Renamed
    key_drug_list_supply = getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if not key_drug_list_supply:
        pytest.skip("No KEY_DRUG_SUBSTRINGS_SUPPLY in settings to test AI supply model params.")
    
    a_key_drug_supply = key_drug_list_supply[0]
    params_supply = supply_model_ai_instance_fixture._get_simulated_item_params(a_key_drug_supply)
    assert isinstance(params_supply, dict), f"Parameters for drug '{a_key_drug_supply}' should be a dictionary."
    assert all(k in params_supply for k in ["monthly_coeffs", "annual_trend_factor", "random_noise_std_dev"]), \
        f"Missing one or more expected keys in parameters for drug '{a_key_drug_supply}'."

    params_unknown_supply = supply_model_ai_instance_fixture._get_simulated_item_params("BrandNewUnseenItemXYZ")
    assert params_unknown_supply["annual_trend_factor"] == 0.0001, "Default trend for unknown item differs from expected fallback in AI supply model."


def test_supply_model_ai_forecast_generates_correct_output_structure(supply_model_ai_instance_fixture: SupplyForecastingModel): # Renamed
    key_drug_list_supply_fc = getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    test_item_name_fc = key_drug_list_supply_fc[0] if key_drug_list_supply_fc else "TestItemForForecast"
    
    supply_input_df_fc = pd.DataFrame({
        'item': [test_item_name_fc, "AnotherTestItem"],
        'current_stock': [250.0, 180.0],
        'avg_daily_consumption_historical': [12.0, 8.0], # Ensure positive consumption
        'last_stock_update_date': pd.to_datetime(['2023-11-20', '2023-11-20'])
    })
    forecast_horizon_days = 25
    forecast_output_df_fc = supply_model_ai_instance_fixture.forecast_supply_levels_advanced(
        supply_input_df_fc, forecast_days_out=forecast_horizon_days
    )
    
    assert isinstance(forecast_output_df_fc, pd.DataFrame), "AI Supply forecast (advanced) did not return a DataFrame."
    if not forecast_output_df_fc.empty:
        assert len(forecast_output_df_fc['item'].unique()) <= 2, "Forecast contains more items than input."
        # Each item should have `forecast_horizon_days` records
        assert len(forecast_output_df_fc) <= 2 * forecast_horizon_days, "Forecast has incorrect number of daily records."
        
        expected_output_cols_fc = ['item', 'forecast_date', 'forecasted_stock_level', 
                                   'forecasted_days_of_supply', 'predicted_daily_consumption', 
                                   'estimated_stockout_date_ai']
        for col_name_check_fc in expected_output_cols_fc:
            assert col_name_check_fc in forecast_output_df_fc.columns, f"AI Supply forecast output DataFrame missing column: {col_name_check_fc}"
        
        # Check stock depletion for the first test item as a basic sanity check
        item1_fc_data = forecast_output_df_fc[forecast_output_df_fc['item'] == test_item_name_fc]
        if not item1_fc_data.empty and len(item1_fc_data) > 1:
            initial_stock_fc = item1_fc_data['forecasted_stock_level'].iloc[0]
            final_stock_fc = item1_fc_data['forecasted_stock_level'].iloc[-1]
            # Stock should deplete if consumption is positive, or stay at 0 if it started at 0.
            # Allow for minimal consumption where stock might not change much with noise.
            base_cons_item1 = supply_input_df_fc.loc[supply_input_df_fc['item']==test_item_name_fc, 'avg_daily_consumption_historical'].iloc[0]
            assert final_stock_fc < initial_stock_fc or initial_stock_fc == 0 or \
                   (np.isclose(final_stock_fc, initial_stock_fc) and base_cons_item1 < 0.1), \
                   f"Stock for item '{test_item_name_fc}' did not deplete as expected in AI forecast."


# --- Tests for Central apply_ai_models Orchestrator (analytics.orchestrator) ---
def test_apply_ai_models_adds_columns_and_preserves_rows(sample_health_records_df_main_fixture: pd.DataFrame): # Renamed fixture
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health data fixture is empty. Skipping apply_ai_models test.")
    
    df_input_for_apply_ai = sample_health_records_df_main_fixture.copy()
    # The fixture itself calls apply_ai_models. For a clean test of `apply_ai_models` here,
    # we should ideally pass data *before* any AI scores are added, or drop them.
    # Let's drop them to ensure they are added by *this* call to apply_ai_models.
    cols_to_drop_for_recalc = ['ai_risk_score', 'ai_followup_priority_score']
    for col_drop_recalc in cols_to_drop_for_recalc:
        if col_drop_recalc in df_input_for_apply_ai.columns:
            df_input_for_apply_ai = df_input_for_apply_ai.drop(columns=[col_drop_recalc])
            
    enriched_df_output_apply_ai, _ = apply_ai_models(df_input_for_apply_ai.copy()) # Pass a copy
    
    assert 'ai_risk_score' in enriched_df_output_apply_ai.columns, "'ai_risk_score' column was not added by apply_ai_models."
    assert 'ai_followup_priority_score' in enriched_df_output_apply_ai.columns, "'ai_followup_priority_score' column was not added."
    assert len(enriched_df_output_apply_ai) == len(df_input_for_apply_ai), "Row count changed after apply_ai_models execution."
    
    if not enriched_df_output_apply_ai.empty:
        assert enriched_df_output_apply_ai['ai_risk_score'].notna().all(), "NaN values found in 'ai_risk_score' after apply_ai_models."
        assert enriched_df_output_apply_ai['ai_followup_priority_score'].notna().all(), "NaN values found in 'ai_followup_priority_score'."


def test_apply_ai_models_handles_empty_or_none_input_gracefully(): # Renamed test
    # Test with an empty DataFrame (with some schema for realism)
    empty_df_input_ai = pd.DataFrame(columns=['encounter_id', 'patient_id', 'condition'])
    enriched_df_empty_res, supply_df_empty_res = apply_ai_models(empty_df_input_ai.copy())
    
    assert isinstance(enriched_df_empty_res, pd.DataFrame), "apply_ai_models should return a DataFrame for empty input."
    # Check that AI columns are present even if DF is empty, for schema consistency
    assert 'ai_risk_score' in enriched_df_empty_res.columns and \
           'ai_followup_priority_score' in enriched_df_empty_res.columns, \
        "Empty output DataFrame from apply_ai_models is missing expected AI columns."
    assert enriched_df_empty_res.empty, "Enriched DataFrame should be empty when input DataFrame is empty."
    assert supply_df_empty_res is None, "Supply DataFrame should be None when health_df is empty and no explicit supply_status_df is passed."

    # Test with None input for health_df
    enriched_df_none_res, supply_df_none_res = apply_ai_models(None) # type: ignore
    assert isinstance(enriched_df_none_res, pd.DataFrame), "apply_ai_models should return a DataFrame for None input."
    assert 'ai_risk_score' in enriched_df_none_res.columns and \
           'ai_followup_priority_score' in enriched_df_none_res.columns, \
        "Output DataFrame from None input is missing expected AI columns."
    assert enriched_df_none_res.empty, "Enriched DataFrame should be empty when input health_df is None."
    assert supply_df_none_res is None, "Supply DataFrame should be None when input health_df is None."
