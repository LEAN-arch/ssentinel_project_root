# sentinel_project_root/tests/conftest.py
# SME PLATINUM STANDARD - PYTEST FIXTURES

import sys
from pathlib import Path

# --- Path Setup for Module Imports ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from analytics import apply_ai_models
from config import settings
from data_processing import enrich_zone_data_with_aggregates

# --- Core Data Fixtures ---

@pytest.fixture(scope="session")
def health_records_df() -> pd.DataFrame:
    """Provides a comprehensive, raw DataFrame of sample health records."""
    num_records = 150
    base_date = datetime.now() - timedelta(days=num_records)
    record_dates = [base_date + timedelta(days=i) for i in range(num_records)]
    
    raw_data = {
        'encounter_id': [f'ENC{i:04d}' for i in range(num_records)],
        'patient_id': [f'PID{(i % 50):03d}' for i in range(num_records)],
        'encounter_date': record_dates,
        'age': np.random.randint(1, 90, num_records),
        'gender': np.random.choice(['Male', 'Female'], num_records, p=[0.5, 0.5]),
        'zone_id': [f"Zone-{chr(65 + (i % 5))}" for i in range(num_records)],
        'chw_id': [f"CHW{(i % 10):03d}" for i in range(num_records)],
        'min_spo2_pct': np.random.choice([88, 93, 96, 98, 99, np.nan], num_records),
        'vital_signs_temperature_celsius': np.random.choice([37.2, 38.5, 39.6, np.nan], num_records),
        'fall_detected_today': np.random.choice([0, 1], num_records, p=[0.98, 0.02]),
        'diagnosis': np.random.choice(settings.KEY_DIAGNOSES + ['URI'], num_records),
        'patient_reported_symptoms': ['fever;cough', 'headache', 'diarrhea;vomiting', np.nan] * (num_records // 4),
        'test_type': np.random.choice(list(settings.KEY_TEST_TYPES.keys()), num_records),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'Rejected'], num_records),
        'sample_status': np.random.choice(['Completed', 'Pending', 'Rejected by Lab'], num_records),
        'test_turnaround_days': [np.random.uniform(0.5, 5) if r != 'Pending' else np.nan for r in np.random.choice(['Completed', 'Pending'], num_records)],
        'referral_status': np.random.choice(['N/A', 'Pending', 'Completed'], num_records),
        'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor'], num_records),
        'item': np.random.choice(settings.KEY_SUPPLY_ITEMS, num_records),
        'item_stock_agg_zone': np.random.randint(10, 500, num_records),
        'consumption_rate_per_day': np.random.uniform(1, 20, num_records),
        'days_task_overdue': np.random.choice([0, 0, 0, 1, 3, 10], num_records),
        'chronic_condition_flag': np.random.choice([0, 1], num_records, p=[0.8, 0.2]),
    }
    return pd.DataFrame(raw_data)

@pytest.fixture(scope="session")
def iot_records_df() -> pd.DataFrame:
    """Provides a sample DataFrame of IoT sensor data."""
    num_records = 50
    base_date = datetime.now() - timedelta(days=num_records)
    timestamps = [base_date + timedelta(hours=i*4) for i in range(num_records)]
    
    data = {
        'timestamp': timestamps,
        'room_name': np.random.choice(['Waiting Room', 'Consultation 1', 'Lab'], num_records),
        'zone_id': [f"Zone-{chr(65 + (i % 3))}" for i in range(num_records)],
        'avg_co2_ppm': np.random.uniform(450, 2500, num_records),
        'avg_pm25': np.random.uniform(5, 70, num_records),
        'avg_noise_db': np.random.uniform(50, 90, num_records),
        'waiting_room_occupancy': np.random.randint(0, 25, num_records)
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def zone_data_df() -> pd.DataFrame:
    """Provides a sample DataFrame of zone attributes and geometries."""
    attributes = [
        {'zone_id': 'Zone-A', 'zone_name': 'Alpha Central', 'population': 12500},
        {'zone_id': 'Zone-B', 'zone_name': 'Beta South', 'population': 18200},
        {'zone_id': 'Zone-C', 'zone_name': 'Gamma Hills', 'population': 7800},
        {'zone_id': 'Zone-D', 'zone_name': 'Delta Coast', 'population': 14000},
        {'zone_id': 'Zone-E', 'zone_name': 'Epsilon Valley', 'population': 9500}
    ]
    attr_df = pd.DataFrame(attributes)

    geometries = [
        {"zone_id": "Zone-A", "geometry": {"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}},
        {"zone_id": "Zone-B", "geometry": {"type": "Polygon", "coordinates": [[[1,0],[1,1],[2,1],[2,0],[1,0]]]}},
    ]
    geom_df = pd.DataFrame(geometries)
    return pd.merge(attr_df, geom_df, on='zone_id', how='left')


# --- Transformed/Enriched Fixtures ---

@pytest.fixture(scope="session")
def enriched_health_records_df(health_records_df: pd.DataFrame) -> pd.DataFrame:
    """Provides health records enriched with AI scores and KPI flags."""
    enriched, _ = apply_ai_models(health_records_df, "pytest_fixture")
    return enriched

@pytest.fixture(scope="session")
def enriched_zone_df(zone_data_df: pd.DataFrame, enriched_health_records_df: pd.DataFrame, iot_records_df: pd.DataFrame) -> pd.DataFrame:
    """Provides zone data enriched with health and IoT aggregates."""
    return enrich_zone_data_with_aggregates(
        zone_df=zone_data_df,
        health_df=enriched_health_records_df,
        iot_df=iot_records_df
    )
