# sentinel_project_root/tests/test_data_processing.py
# SME PLATINUM STANDARD - DATA PROCESSING TESTS

import pandas as pd
import numpy as np
import pytest

from data_processing import (
    DataPipeline,
    enrich_health_records_with_kpis,
    enrich_zone_data_with_aggregates,
    get_cached_clinic_kpis,
    get_cached_district_kpis,
    get_cached_trend
)

# Fixtures are sourced from conftest.py

# --- DataPipeline Tests ---
def test_data_pipeline_fluent_chaining():
    """Tests the fluent, chainable interface of the DataPipeline."""
    df_dirty = pd.DataFrame({
        ' First Name ': [' John ', ' Jane '],
        'Date Of Birth': ['1990-01-15', '1985-05-20'],
        '  Value': ['100', 'N/A']
    })
    
    pipeline = DataPipeline(df_dirty)
    processed_df = (pipeline
        .clean_column_names()
        .convert_date_columns(['date_of_birth'])
        .standardize_missing_values({'value': 0})
        .get_dataframe()
    )
    
    assert list(processed_df.columns) == ['first_name', 'date_of_birth', 'value']
    assert pd.api.types.is_datetime64_any_dtype(processed_df['date_of_birth'])
    assert processed_df['value'].iloc[1] == 0

# --- Enrichment Tests ---
def test_enrich_health_records_adds_kpi_flags(health_records_df):
    """Tests that enrichment adds the correct boolean/flag columns."""
    enriched_df = enrich_health_records_with_kpis(health_records_df)
    assert 'is_rejected' in enriched_df.columns
    assert 'is_critical_and_pending' in enriched_df.columns
    assert 'is_positive' in enriched_df.columns
    assert 'is_supply_at_risk' in enriched_df.columns
    assert pd.api.types.is_integer_dtype(enriched_df['is_rejected'])

def test_enrich_zone_data_aggregates_correctly(enriched_zone_df, enriched_health_records_df):
    """Verifies that zone enrichment calculations are correct."""
    zone_a_health_df = enriched_health_records_df[enriched_health_records_df['zone_id'] == 'Zone-A']
    expected_risk = zone_a_health_df['ai_risk_score'].mean()
    
    actual_risk = enriched_zone_df.loc[enriched_zone_df['zone_id'] == 'Zone-A', 'avg_risk_score'].iloc[0]
    assert np.isclose(actual_risk, expected_risk)

# --- Aggregation Tests ---
def test_clinic_kpis_structure(enriched_health_records_df):
    """Verifies the structure of the clinic summary KPI dictionary."""
    summary = get_cached_clinic_kpis(enriched_health_records_df)
    assert isinstance(summary, dict)
    expected_keys = [
        "avg_test_tat_days", "perc_tests_within_tat",
        "sample_rejection_rate_perc", "pending_critical_tests_count"
    ]
    assert all(key in summary for key in expected_keys)
    assert isinstance(summary.get("positivity_rates"), dict)

def test_district_kpis_structure(enriched_zone_df):
    """Verifies the structure of the district summary KPI dictionary."""
    summary = get_cached_district_kpis(enriched_zone_df)
    assert isinstance(summary, dict)
    expected_keys = [
        "total_population", "total_zones", "population_weighted_avg_risk_score"
    ]
    assert all(key in summary for key in expected_keys)

def test_trend_calculation(enriched_health_records_df):
    """Tests that the trend function produces a valid time series."""
    trend_series = get_cached_trend(
        df=enriched_health_records_df,
        value_col='ai_risk_score',
        date_col='encounter_date',
        freq='W'
    )
    assert isinstance(trend_series, pd.Series)
    assert isinstance(trend_series.index, pd.DatetimeIndex)
    assert not trend_series.empty
