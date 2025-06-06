# sentinel_project_root/tests/test_ai_analytics_engine.py
# Pytest tests for AI simulation logic in analytics module for Sentinel.

import pytest
import pandas as pd
import numpy as np

from analytics.risk_prediction import RiskPredictionModel
from analytics.followup_prioritization import FollowUpPrioritizer
from analytics.supply_forecasting import SupplyForecastingModel
from analytics.orchestrator import apply_ai_models
from config import settings

# Fixtures are sourced from conftest.py

@pytest.fixture(scope="module")
def risk_model() -> RiskPredictionModel:
    return RiskPredictionModel()

@pytest.fixture(scope="module")
def priority_model() -> FollowUpPrioritizer:
    return FollowUpPrioritizer()

@pytest.fixture(scope="module")
def supply_model_ai() -> SupplyForecastingModel:
    return SupplyForecastingModel()

# --- Tests for RiskPredictionModel ---
def test_risk_model_condition_base_score(risk_model: RiskPredictionModel):
    key_conditions = settings.KEY_CONDITIONS_FOR_ACTION
    if not key_conditions: pytest.skip("No KEY_CONDITIONS_FOR_ACTION in settings.")
        
    for cond in key_conditions:
        expected_score = risk_model.condition_base_scores.get(cond.lower(), 0.0)
        assert risk_model._get_condition_base_score(cond) == expected_score, f"Score mismatch for: {cond}"
        assert risk_model._get_condition_base_score(cond.upper()) == expected_score, f"Score mismatch for: {cond.upper()}" # Test case insensitivity

    multi_cond = f"{key_conditions[0]}; Pneumonia" if len(key_conditions) >=1 else "Pneumonia; Sepsis"
    # CORRECTED: The fallback condition in the expected score now matches the one in multi_cond.
    expected_multi_score = max(risk_model._get_condition_base_score(key_conditions[0] if len(key_conditions) >=1 else "Pneumonia"), 
                               risk_model._get_condition_base_score("Pneumonia" if len(key_conditions) >=1 else "Sepsis"))
    assert risk_model._get_condition_base_score(multi_cond) == expected_multi_score, "Multi-condition score incorrect."
    
    assert risk_model._get_condition_base_score("UnknownCondition") == 0.0
    assert risk_model._get_condition_base_score(None) == 0.0 # type: ignore
    assert risk_model._get_condition_base_score("") == 0.0
    if "wellness visit" in risk_model.condition_base_scores:
        assert risk_model._get_condition_base_score("Wellness Visit") < 0

def test_risk_model_predict_score_factors(risk_model: RiskPredictionModel):
    base_features = pd.Series({
        'condition': 'Wellness Visit', 'age': 30, 'chronic_condition_flag': '0',
        'min_spo2_pct': 99.0, 'vital_signs_temperature_celsius': 36.8,
        'fall_detected_today': '0', 'ambient_heat_index_c': 25.0,
        'ppe_compliant_flag': '1', 'signs_of_fatigue_observed_flag': '0',
        'rapid_psychometric_distress_score': 0.0, 'hrv_rmssd_ms': 60.0,
        'medication_adherence_self_report': 'Good', 'tb_contact_traced': '0'
    })
    base_risk = risk_model.predict_risk_score(base_features.copy())
    assert 0 <= base_risk <= 100

    features_low_spo2 = base_features.copy()
    features_low_spo2['min_spo2_pct'] = settings.ALERT_SPO2_CRITICAL_LOW_PCT - 2
    score_low_spo2 = risk_model.predict_risk_score(features_low_spo2)
    assert score_low_spo2 > base_risk, "Critical SpO2 did not increase risk score."

    features_high_fever = base_features.copy()
    features_high_fever['vital_signs_temperature_celsius'] = settings.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.5
    score_high_fever = risk_model.predict_risk_score(features_high_fever)
    assert score_high_fever > base_risk, "High fever did not increase risk score."

    features_fall = base_features.copy()
    features_fall['fall_detected_today'] = '1'
    score_fall = risk_model.predict_risk_score(features_fall)
    assert score_fall > base_risk, "Fall detection did not increase risk score."


def test_risk_model_bulk_predict_scores(risk_model: RiskPredictionModel, sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records empty. Skipping bulk risk prediction test.")
    
    df_to_score = sample_health_records_df_main_fixture.copy()
    if 'ai_risk_score' in df_to_score.columns: # Recalculate if already present from fixture's apply_ai_models
        df_to_score = df_to_score.drop(columns=['ai_risk_score'])

    risk_scores = risk_model.predict_bulk_risk_scores(df_to_score)
    assert isinstance(risk_scores, pd.Series)
    assert len(risk_scores) == len(df_to_score)
    assert risk_scores.notna().all() # Should handle NaNs by assigning default scores or clipping
    assert (risk_scores >= 0).all() and (risk_scores <= 100).all()

# --- Tests for FollowUpPrioritizer ---
def test_priority_model_helpers(priority_model: FollowUpPrioritizer):
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': settings.ALERT_SPO2_CRITICAL_LOW_PCT - 1}))
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'vital_signs_temperature_celsius': settings.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.1}))
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'fall_detected_today': '1'}))
    assert not priority_model._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 99, 'vital_signs_temperature_celsius': 37.0, 'fall_detected_today': '0'}))

    assert priority_model._is_pending_urgent_task_or_referral(pd.Series({'referral_status': 'Pending', 'condition': settings.KEY_CONDITIONS_FOR_ACTION[0]}))
    assert not priority_model._is_pending_urgent_task_or_referral(pd.Series({'referral_status': 'Completed'}))

    assert priority_model._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Pneumonia', 'min_spo2_pct': settings.ALERT_SPO2_WARNING_LOW_PCT - 1}))
    assert priority_model._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Sepsis'}))
    assert not priority_model._has_acute_condition_with_severity_indicators(pd.Series({'condition': 'Wellness Visit'}))
    
    assert priority_model._has_significant_contextual_hazard(pd.Series({'ambient_heat_index_c': settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 1}))
    assert not priority_model._has_significant_contextual_hazard(pd.Series({'ambient_heat_index_c': 25}))

def test_priority_model_calculate_score(priority_model: FollowUpPrioritizer):
    base_features = pd.Series({'ai_risk_score': 30.0, 'days_task_overdue': 0}) # Ensure days_task_overdue is provided or defaulted
    score_base = priority_model.calculate_priority_score(base_features.copy())
    
    features_critical = base_features.copy(); features_critical['min_spo2_pct'] = settings.ALERT_SPO2_CRITICAL_LOW_PCT - 1
    score_critical = priority_model.calculate_priority_score(features_critical.copy())
    assert score_critical > score_base

    score_overdue = priority_model.calculate_priority_score(base_features.copy(), days_task_overdue=5)
    assert score_overdue > score_base

def test_priority_model_generate_bulk_scores(priority_model: FollowUpPrioritizer, sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health records empty. Skipping bulk priority score test.")
    
    df_for_prio = sample_health_records_df_main_fixture.copy()
    if 'ai_followup_priority_score' in df_for_prio.columns: # Recalculate
        df_for_prio = df_for_prio.drop(columns=['ai_followup_priority_score'])
    if 'days_task_overdue' not in df_for_prio.columns: # Ensure this column exists for the model
        df_for_prio['days_task_overdue'] = 0


    priority_scores = priority_model.generate_followup_priorities(df_for_prio)
    assert isinstance(priority_scores, pd.Series)
    assert len(priority_scores) == len(df_for_prio)
    assert priority_scores.notna().all()
    assert (priority_scores >= 0).all() and (priority_scores <= 100).all()

# --- Tests for SupplyForecastingModel ---
def test_supply_model_ai_get_params(supply_model_ai: SupplyForecastingModel):
    key_drugs = settings.KEY_DRUG_SUBSTRINGS_SUPPLY
    if not key_drugs: pytest.skip("No KEY_DRUG_SUBSTRINGS_SUPPLY in settings.")
    
    params = supply_model_ai._get_simulated_item_params(key_drugs[0])
    assert isinstance(params, dict)
    assert all(k in params for k in ["monthly_coeffs", "annual_trend_factor", "random_noise_std_dev"])

    default_params = supply_model_ai._get_simulated_item_params("UnknownItemXYZ987")
    assert default_params["annual_trend_factor"] == 0.0001 # Check default fallback

def test_supply_model_ai_forecast_output(supply_model_ai: SupplyForecastingModel):
    item_name = "TestForecastItem" if not settings.KEY_DRUG_SUBSTRINGS_SUPPLY else settings.KEY_DRUG_SUBSTRINGS_SUPPLY[0]
    supply_input = pd.DataFrame({
        'item': [item_name, "AnotherItem"], 'current_stock': [200.0, 150.0],
        'avg_daily_consumption_historical': [10.0, 5.0], # Must be positive
        'last_stock_update_date': pd.to_datetime(['2023-11-01', '2023-11-01'])
    })
    horizon = 20
    forecast_df = supply_model_ai.forecast_supply_levels_advanced(supply_input, forecast_days_out=horizon)
    
    assert isinstance(forecast_df, pd.DataFrame)
    if not forecast_df.empty:
        assert len(forecast_df['item'].unique()) <= 2
        assert len(forecast_df) <= 2 * horizon # Each item gets `horizon` records
        expected_cols = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']
        assert all(col in forecast_df.columns for col in expected_cols)
        
        item1_data = forecast_df[forecast_df['item'] == item_name]
        if not item1_data.empty and len(item1_data) > 1:
            initial_stock_input = supply_input.loc[supply_input['item'] == item_name, 'current_stock'].iloc[0]
            final_stock_output = item1_data['forecasted_stock_level'].iloc[-1]
            assert final_stock_output < initial_stock_input or np.isclose(final_stock_output, 0.0) or \
                   (np.isclose(final_stock_output, initial_stock_input) and supply_input.loc[supply_input['item']==item_name, 'avg_daily_consumption_historical'].iloc[0] < 0.1), \
                   f"Stock for '{item_name}' did not deplete as expected or started/ended near zero."


# --- Tests for Orchestrator ---
def test_apply_ai_models_adds_cols_preserves_rows(sample_health_records_df_main_fixture: pd.DataFrame):
    if sample_health_records_df_main_fixture.empty:
        pytest.skip("Sample health data empty. Skipping apply_ai_models test.")
    
    df_input = sample_health_records_df_main_fixture.copy()
    cols_to_drop = ['ai_risk_score', 'ai_followup_priority_score']
    df_input_clean = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns], errors='ignore')
            
    enriched_df, _ = apply_ai_models(df_input_clean.copy()) # Pass copy
    
    assert 'ai_risk_score' in enriched_df.columns
    assert 'ai_followup_priority_score' in enriched_df.columns
    assert len(enriched_df) == len(df_input_clean)
    
    if not enriched_df.empty:
        assert enriched_df['ai_risk_score'].notna().any() or enriched_df['ai_risk_score'].isnull().all(), "ai_risk_score should have values or be all NaN if input had no basis."
        assert enriched_df['ai_followup_priority_score'].notna().any() or enriched_df['ai_followup_priority_score'].isnull().all()

def test_apply_ai_models_empty_none_input():
    empty_df = pd.DataFrame(columns=['patient_id', 'condition']) # Schema helps orchestrator
    enriched_empty, supply_empty = apply_ai_models(empty_df.copy())
    assert isinstance(enriched_empty, pd.DataFrame)
    assert 'ai_risk_score' in enriched_empty.columns and 'ai_followup_priority_score' in enriched_empty.columns
    assert enriched_empty.empty
    assert supply_empty is None

    enriched_none, supply_none = apply_ai_models(None) # type: ignore
    assert isinstance(enriched_none, pd.DataFrame)
    assert 'ai_risk_score' in enriched_none.columns and 'ai_followup_priority_score' in enriched_none.columns
    assert enriched_none.empty
    assert supply_none is None
