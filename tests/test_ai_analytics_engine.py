# sentinel_project_root/tests/test_ai_analytics_engine.py
# Pytest tests for AI simulation logic in analytics module for Sentinel.

import pytest
import pandas as pd
import numpy as np

# --- Module Imports ---
from analytics.risk_prediction import RiskPredictionModel
from analytics.followup_prioritization import FollowUpPrioritizer
from analytics.supply_forecasting import SupplyForecastingModel
from analytics.orchestrator import apply_ai_models
from config import settings

# Fixtures are sourced from conftest.py


@pytest.fixture(scope="module")
def risk_model() -> RiskPredictionModel:
    """Provides a single instance of the RiskPredictionModel for the module."""
    return RiskPredictionModel()

@pytest.fixture(scope="module")
def priority_model() -> FollowUpPrioritizer:
    """Provides a single instance of the FollowUpPrioritizer for the module."""
    return FollowUpPrioritizer()

@pytest.fixture(scope="module")
def supply_model_ai() -> SupplyForecastingModel:
    """Provides a single instance of the AI-simulated SupplyForecastingModel."""
    return SupplyForecastingModel()


# --- Tests for RiskPredictionModel ---
def test_risk_model_condition_base_score(risk_model: RiskPredictionModel):
    """Tests that the risk model correctly assigns base scores for various condition strings."""
    key_conditions = settings.KEY_CONDITIONS_FOR_ACTION
    if not key_conditions:
        pytest.skip("No KEY_CONDITIONS_FOR_ACTION defined in settings.")
        
    for cond in key_conditions:
        expected_score = risk_model.condition_base_scores.get(cond.lower(), 0.0)
        assert risk_model._get_condition_base_score(cond) == expected_score, f"Score mismatch for '{cond}'"
        assert risk_model._get_condition_base_score(cond.upper()) == expected_score, f"Case-insensitivity failed for '{cond.upper()}'"

    # Test multi-condition string
    cond1 = key_conditions[0]
    cond2 = "Pneumonia"
    multi_cond = f"{cond1}; {cond2}"
    # FIXED: The expected score calculation now correctly uses the same conditions as the test input.
    expected_multi_score = max(risk_model._get_condition_base_score(cond1), risk_model._get_condition_base_score(cond2))
    assert risk_model._get_condition_base_score(multi_cond) == expected_multi_score, "Multi-condition score was incorrect."
    
    # Test neutral and negative scores
    assert risk_model._get_condition_base_score("UnknownCondition") == 0.0
    assert risk_model._get_condition_base_score(None) == 0.0
    assert risk_model._get_condition_base_score("") == 0.0
    if "wellness visit" in risk_model.condition_base_scores:
        assert risk_model._get_condition_base_score("Wellness Visit") < 0

def test_risk_model_predict_score_factors(risk_model: RiskPredictionModel):
    """Tests that specific high-risk factors correctly increase the risk score."""
    base_features = pd.Series({
        'condition': 'Wellness Visit', 'age': 30, 'min_spo2_pct': 99.0,
        'vital_signs_temperature_celsius': 36.8, 'fall_detected_today': '0'
    })
    base_risk = risk_model.predict_risk_score(base_features.copy())
    assert 0 <= base_risk <= 100

    features_low_spo2 = base_features.copy()
    features_low_spo2['min_spo2_pct'] = settings.ALERT_SPO2_CRITICAL_LOW_PCT - 2
    assert risk_model.predict_risk_score(features_low_spo2) > base_risk, "Critical SpO2 did not increase risk."

    features_high_fever = base_features.copy()
    features_high_fever['vital_signs_temperature_celsius'] = settings.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.5
    assert risk_model.predict_risk_score(features_high_fever) > base_risk, "High fever did not increase risk."

    features_fall = base_features.copy()
    features_fall['fall_detected_today'] = '1'
    assert risk_model.predict_risk_score(features_fall) > base_risk, "Fall detection did not increase risk."

def test_risk_model_bulk_predict_scores(risk_model: RiskPredictionModel, sample_health_records_df_main_fixture: pd.DataFrame):
    """Tests the bulk prediction method on a sample DataFrame."""
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records are empty.")
    
    df_to_score = sample_health_records_df_main_fixture.copy()
    risk_scores = risk_model.predict_bulk_risk_scores(df_to_score)
    
    assert isinstance(risk_scores, pd.Series)
    assert len(risk_scores) == len(df_to_score)
    assert risk_scores.notna().all(), "Bulk risk scores should not contain NaNs."
    assert (risk_scores >= 0).all() and (risk_scores <= 100).all(), "Risk scores are outside the 0-100 range."


# --- Tests for FollowUpPrioritizer ---
def test_priority_model_helpers(priority_model: FollowUpPrioritizer):
    """Tests the helper methods of the FollowUpPrioritizer."""
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': settings.ALERT_SPO2_CRITICAL_LOW_PCT - 1}))
    assert priority_model._is_pending_urgent_task_or_referral(pd.Series({'referral_status': 'Pending', 'condition': settings.KEY_CONDITIONS_FOR_ACTION[0]}))
    assert priority_model._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Sepsis'}))
    assert priority_model._has_significant_contextual_hazard(pd.Series({'ambient_heat_index_c': settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1}))
    assert not priority_model._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 99}))

def test_priority_model_calculate_score(priority_model: FollowUpPrioritizer):
    """Tests that specific factors correctly increase the priority score."""
    base_features = pd.Series({'ai_risk_score': 30.0})
    score_base = priority_model.calculate_priority_score(base_features.copy(), days_task_overdue=0)
    
    features_critical = base_features.copy(); features_critical['fall_detected_today'] = '1'
    score_critical = priority_model.calculate_priority_score(features_critical.copy(), days_task_overdue=0)
    assert score_critical > score_base

    score_overdue = priority_model.calculate_priority_score(base_features.copy(), days_task_overdue=5)
    assert score_overdue > score_base

def test_priority_model_generate_bulk_scores(priority_model: FollowUpPrioritizer, sample_health_records_df_main_fixture: pd.DataFrame):
    """Tests the bulk priority score generation method."""
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records are empty.")
    
    priority_scores = priority_model.generate_followup_priorities(sample_health_records_df_main_fixture.copy())
    
    assert isinstance(priority_scores, pd.Series)
    assert len(priority_scores) == len(sample_health_records_df_main_fixture)
    assert priority_scores.notna().all(), "Bulk priority scores should not contain NaNs."
    assert (priority_scores >= 0).all() and (priority_scores <= 100).all(), "Priority scores are outside the 0-100 range."


# --- Tests for SupplyForecastingModel ---
def test_supply_model_ai_forecast_output(supply_model_ai: SupplyForecastingModel):
    """Tests the output of the AI-simulated supply forecasting model."""
    item_name = "TestForecastItem"
    supply_input = pd.DataFrame({
        'item': [item_name], 'current_stock': [200.0],
        'avg_daily_consumption_historical': [10.0],
        'last_stock_update_date': pd.to_datetime(['2023-11-01'])
    })
    
    forecast_df = supply_model_ai.forecast_supply_levels_advanced(supply_input, forecast_days_out=20)
    
    assert isinstance(forecast_df, pd.DataFrame)
    if not forecast_df.empty:
        expected_cols = ['item', 'forecast_date', 'forecasted_stock_level', 'estimated_stockout_date_ai']
        assert all(col in forecast_df.columns for col in expected_cols)
        assert forecast_df['forecasted_stock_level'].iloc[-1] < supply_input['current_stock'].iloc[0]


# --- Tests for Orchestrator ---
def test_apply_ai_models_adds_cols_preserves_rows(sample_health_records_df_main_fixture: pd.DataFrame):
    """Tests that the orchestrator adds AI columns without changing row count."""
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health data is empty.")
    
    df_input = sample_health_records_df_main_fixture.drop(columns=['ai_risk_score', 'ai_followup_priority_score'], errors='ignore')
    enriched_df, _ = apply_ai_models(df_input.copy())
    
    assert 'ai_risk_score' in enriched_df.columns
    assert 'ai_followup_priority_score' in enriched_df.columns
    assert len(enriched_df) == len(df_input)
    assert enriched_df['ai_risk_score'].notna().any()

def test_apply_ai_models_empty_none_input():
    """Tests that the orchestrator handles empty and None inputs gracefully."""
    enriched_empty, _ = apply_ai_models(pd.DataFrame(columns=['patient_id']))
    assert isinstance(enriched_empty, pd.DataFrame)
    assert 'ai_risk_score' in enriched_empty.columns
    assert enriched_empty.empty

    enriched_none, _ = apply_ai_models(None)
    assert isinstance(enriched_none, pd.DataFrame)
    assert 'ai_risk_score' in enriched_none.columns
    assert enriched_none.empty
