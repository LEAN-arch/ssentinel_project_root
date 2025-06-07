import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os
import json
import re
from pathlib import Path
import hashlib

# --- Path Setup for Imports ---
# FIXED: Use the correct `__file__` magic variable.
_current_conftest_dir = Path(__file__).resolve().parent
_project_root_dir_for_tests = _current_conftest_dir.parent
if str(_project_root_dir_for_tests) not in sys.path:
    sys.path.insert(0, str(_project_root_dir_for_tests))

# --- Critical Project Module Imports ---
try:
    from config import settings
    from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe
except ImportError as e_conftest:
    print(f"FATAL ERROR in conftest.py: Could not import core project modules.", file=sys.stderr)
    print(f"PYTHONPATH currently is: {sys.path}", file=sys.stderr)
    print(f"Attempted to add project root: {_project_root_dir_for_tests}", file=sys.stderr)
    print(f"Error details: {e_conftest}", file=sys.stderr)
    raise


@pytest.fixture(scope="session")
def sample_health_records_df_main_fixture() -> pd.DataFrame:
    """
    Provides a comprehensive, AI-enriched DataFrame of sample health records.
    """
    num_records = 75
    base_date = datetime.now() - timedelta(days=num_records // 2)
    record_dates = [base_date + timedelta(days=i) for i in range(num_records)]
    
    raw_data = {
        'encounter_id': [f'SENC_FIX_{i:03d}' for i in range(1, num_records + 1)],
        'patient_id': [f'SPID_FIX_{(i % 30):03d}' for i in range(1, num_records + 1)],
        'encounter_date': record_dates,
        'encounter_type': np.random.choice(['CHW_HOME_VISIT', 'CHW_ALERT_RESPONSE', 'CLINIC_INTAKE', 'WORKER_SELF_CHECK'], num_records),
        'age': np.random.randint(0, 96, num_records),
        'gender': np.random.choice(['Male', 'Female', 'Other'], num_records, p=[0.48, 0.48, 0.04]),
        'zone_id': [f"Zone{chr(65 + (i % 5))}" for i in range(num_records)],
        'chw_id': [f"CHW{(i % 7) + 1:03d}" for i in range(num_records)],
        'min_spo2_pct': np.random.choice([87, 90, 93, 94, 96, 98, 99, np.nan], num_records, p=[0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1]),
        'vital_signs_temperature_celsius': np.random.choice([37.2, 38.3, 39.3, 39.8, np.nan], num_records, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'fall_detected_today': np.random.choice([0, 1], num_records, p=[0.95, 0.05]),
        'condition': np.random.choice(settings.KEY_CONDITIONS_FOR_ACTION + ['Wellness Visit', 'Hypertension Check'], num_records),
        'patient_reported_symptoms': ['fever;cough;fatigue', 'none', 'headache, ache', 'diarrhea;vomiting', np.nan] * (num_records // 5),
        'test_type': np.random.choice(list(settings.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None"], num_records),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'Rejected'], num_records),
        'test_turnaround_days': [np.random.uniform(0.1, 10).round(1) if res in ['Positive', 'Negative'] else np.nan for res in np.random.choice(['Positive', 'Negative', 'Pending'], num_records)],
        'referral_status': np.random.choice(['Pending', 'Completed', 'N/A'], num_records),
        'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor', 'N/A'], num_records),
        'item': np.random.choice(settings.KEY_DRUG_SUBSTRINGS_SUPPLY, num_records),
        'item_stock_agg_zone': np.random.randint(0, 300, num_records),
        'consumption_rate_per_day': np.random.uniform(0.01, 6.0, num_records),
        'avg_daily_steps': np.random.choice([np.nan, 0] + list(range(1000, 18000, 500)), num_records),
        'days_task_overdue': np.random.choice([0, 1, 2, 5, 10, np.nan], num_records, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.1]),
        'tb_contact_traced': np.random.choice([0, 1], num_records, p=[0.8, 0.2])
    }

    df = pd.DataFrame(raw_data)
    enriched_df, _ = apply_ai_models(df.copy(), source_context="ConftestHealthFixtureAI")
    
    date_cols = [col for col in enriched_df.columns if 'date' in col]
    for col in date_cols:
        enriched_df[col] = pd.to_datetime(enriched_df[col], errors='coerce')
        
    return enriched_df


@pytest.fixture(scope="session")
def sample_iot_clinic_df_main_fixture() -> pd.DataFrame:
    """Provides a DataFrame of sample IoT sensor data for clinics."""
    num_records = 25
    base_date = datetime.now() - timedelta(days=num_records)
    timestamps = [base_date + timedelta(days=i) for i in range(num_records)]
    
    data = {
        'timestamp': timestamps,
        'clinic_id': [f"CLINIC{(i % 4) + 1:02d}" for i in range(num_records)],
        'room_name': np.random.choice(['WaitingArea_A', 'ConsultRoom_1', 'Lab_Main'], num_records),
        'zone_id': [f"Zone{chr(65 + (i % 5))}" for i in range(num_records)],
        'avg_co2_ppm': np.random.choice([550, 1200, 1700, 2450, 2800, np.nan], num_records),
        'avg_pm25': np.random.choice([7.5, 40, 20.0, 48, 65, np.nan], num_records),
        'avg_temp_celsius': np.random.uniform(18.0, 33.0, num_records).round(1),
        'waiting_room_occupancy': [np.random.randint(0, 25) if 'Waiting' in r else np.nan for r in np.random.choice(['WaitingArea_A', 'Lab_Main'], num_records)],
        'patient_throughput_per_hour': np.random.randint(1, 20, num_records)
    }
    df = pd.DataFrame(data)
    
    for col in df.columns:
        if col not in ['timestamp', 'clinic_id', 'room_name', 'zone_id']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@pytest.fixture(scope="session")
def sample_zone_data_df_main_fixture() -> pd.DataFrame:
    """Provides a DataFrame of sample zone attributes and geometries."""
    attributes = [
        {'zone_id': 'ZoneA', 'name': 'Alpha Central', 'population': 12500, 'area_sqkm': 40.5},
        {'zone_id': 'ZoneB', 'name': 'Beta South', 'population': 18200, 'area_sqkm': 150.2},
        {'zone_id': 'ZoneC', 'name': 'Gamma Hills', 'population': 7800, 'area_sqkm': 25.0},
        {'zone_id': 'ZoneD', 'name': 'Delta Coast', 'population': 14000, 'area_sqkm': 75.8},
        {'zone_id': 'ZoneE', 'name': 'Epsilon Valley', 'population': 9500, 'area_sqkm': 90.0}
    ]
    attr_df = pd.DataFrame(attributes)

    geometries = [
        {"zone_id": "ZoneA", "geometry_obj": {"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}},
        {"zone_id": "ZoneB", "geometry_obj": {"type": "Polygon", "coordinates": [[[1,0],[1,1],[2,1],[2,0],[1,0]]]}},
        {"zone_id": "ZoneC", "geometry_obj": {"type": "MultiPolygon", "coordinates": [[[[0,1],[0,2],[1,2],[1,1],[0,1]]]]}},
        {"zone_id": "ZoneD", "geometry_obj": {"type": "Polygon", "coordinates": [[[1,1],[1,2],[2,2],[2,1],[1,1]]]}},
        {"zone_id": "ZoneE", "geometry_obj": {"type": "Polygon", "coordinates": [[[-1,0],[-1,1],[0,1],[0,0],[-1,0]]]}},
    ]
    geom_df = pd.DataFrame(geometries)

    merged_df = pd.merge(attr_df, geom_df, on='zone_id', how='left')
    merged_df['crs'] = settings.DEFAULT_CRS_STANDARD
    
    return merged_df


@pytest.fixture(scope="session")
def sample_enriched_zone_df_main_fixture(
    sample_zone_data_df_main_fixture: pd.DataFrame,
    sample_health_records_df_main_fixture: pd.DataFrame,
    sample_iot_clinic_df_main_fixture: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """Provides a zone DataFrame enriched with health and IoT data aggregates."""
    if not isinstance(sample_zone_data_df_main_fixture, pd.DataFrame) or sample_zone_data_df_main_fixture.empty:
        pytest.skip("Base zone DataFrame fixture is invalid/empty.")
    
    return enrich_zone_geodata_with_health_aggregates(
        zone_df=sample_zone_data_df_main_fixture.copy(),
        health_df=sample_health_records_df_main_fixture.copy(),
        iot_df=sample_iot_clinic_df_main_fixture.copy(),
        source_context="ConftestEnrichment"
    )
