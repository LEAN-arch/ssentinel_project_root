# sentinel_project_root/tests/test_analytics.py
# SME PLATINUM STANDARD - ANALYTICS ENGINE TESTS

import pandas as pd
import pytest

from analytics import (apply_ai_models, calculate_followup_priority,
                       calculate_risk_score, generate_clinic_patient_alerts,
                       generate_linear_forecast)
from config import settings

# Fixtures are sourced from conftest.py

# --- Risk Prediction Model Tests ---
def test_risk_score_increases_with_risk_factors(enriched_health_records_df):
    """Tests that high-risk factors result in higher scores."""
    base_score = 10.0 # A hypothetical low base score
    
    # Test for high fever
    high_fever_df = pd.DataFrame([{'age': 30, 'temperature': 40}])
    result_df = calculate_risk_score(high_fever_df)
    assert result_df['ai_risk_score'].iloc[0] > base_score

    # Test for critical SpO2
    low_spo2_df = pd.DataFrame([{'age': 30, 'min_spo2_pct': 88}])
    result_df = calculate_risk_score(low_spo2_df)
    assert result_df['ai_risk_score'].iloc[0] > base_score

# --- Follow-up Prioritization Tests ---
def test_priority_score_increases_with_urgency(enriched_health_records_df):
    """Tests that urgent factors result in higher priority scores."""
    base_df = pd.DataFrame([{'ai_risk_score': 20, 'days_task_overdue': 0}])
    base_score = calculate_followup_priority(base_df)['ai_followup_priority_score'].iloc[0]

    # Test for overdue tasks
    overdue_df = pd.DataFrame([{'ai_risk_score': 20, 'days_task_overdue': 10}])
    overdue_score = calculate_followup_priority(overdue_df)['ai_followup_priority_score'].iloc[0]
    assert overdue_score > base_score

    # Test for pending urgent referral
    referral_df = pd.DataFrame([{'ai_risk_score': 20, 'referral_status': 'Pending', 'diagnosis': 'Tuberculosis'}])
    referral_score = calculate_followup_priority(referral_df)['ai_followup_priority_score'].iloc[0]
    assert referral_score > base_score

# --- Alerting Engine Tests ---
def test_generate_clinic_patient_alerts_structure(enriched_health_records_df):
    """Tests the structure and output of the clinic patient alert generator."""
    alerts_df = generate_clinic_patient_alerts(enriched_health_records_df)
    assert isinstance(alerts_df, pd.DataFrame)
    if not alerts_df.empty:
        expected_cols = ['patient_id', 'encounter_date', 'Alert Summary', 'Priority']
        assert all(col in alerts_df.columns for col in expected_cols)
        assert pd.api.types.is_integer_dtype(alerts_df['Priority'])

# --- Supply Forecasting Tests ---
def test_generate_linear_forecast_structure(enriched_health_records_df):
    """Tests the output structure of the linear supply forecast."""
    item = settings.KEY_SUPPLY_ITEMS[0]
    forecast_df = generate_linear_forecast(enriched_health_records_df, item_filter=[item])
    assert isinstance(forecast_df, pd.DataFrame)
    if not forecast_df.empty:
        expected_cols = ['item', 'forecast_date', 'forecasted_stock', 'days_of_supply']
        assert all(col in forecast_df.columns for col in expected_cols)
        assert forecast_df['item'].iloc[0] == item

# --- Orchestrator Tests ---
def test_apply_ai_models_pipeline(health_records_df):
    """Ensures the main AI orchestrator pipeline runs and adds required columns."""
    result_df, errors = apply_ai_models(health_records_df)
    assert isinstance(result_df, pd.DataFrame)
    assert not errors
    assert 'ai_risk_score' in result_df.columns
    assert 'ai_followup_priority_score' in result_df.columns
    assert 'priority_reasons' in result_df.columns
    assert len(result_df) == len(health_records_df)
