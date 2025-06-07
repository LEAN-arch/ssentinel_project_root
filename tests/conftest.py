:# sentinel_project_root/tests/conftest.py
Pytest fixtures for testing the "Sentinel Health Co-Pilot" application.
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os
import json
import re # For condition name processing
from pathlib import Path # For path operations
import hashlib
--- Path Setup for Imports ---
_current_conftest_dir = Path(file).resolve().parent
_project_root_dir_for_tests = _current_conftest_dir.parent
if str(_project_root_dir_for_tests) not in sys.path:
sys.path.insert(0, str(_project_root_dir_for_tests))
--- Critical Project Module Imports ---
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
--- Fixture for Sample Health Records ---
@pytest.fixture(scope="session")
def sample_health_records_df_main_fixture() -> pd.DataFrame:
"""
Provides a comprehensive DataFrame of sample health records, AI-enriched.
"""
num_records = 75
# CORRECTED: Generate data relative to the current date to ensure it appears in default UI views.
base_date = datetime.now() - timedelta(days=num_records // 2)
record_dates = [base_date + timedelta(days=i) for i in range(num_records)]
raw_data = {
    'encounter_id': [f'SENC_FIX_{i:03d}' for i in range(1, num_records + 1)],
    'patient_id': [f'SPID_FIX_{(i % 30):03d}' for i in range(1, num_records + 1)], # ~30 unique patients
    'encounter_date': record_dates,
    'encounter_type': np.random.choice(['CHW_HOME_VISIT', 'CHW_ALERT_RESPONSE', 'CLINIC_INTAKE', 'WORKER_SELF_CHECK', 'CHW_SCHEDULED_DOTS'], num_records).tolist(),
    'age': np.random.randint(0, 96, num_records).tolist(),
    'gender': np.random.choice(['Male', 'Female', 'Other', 'Unknown'], num_records, p=[0.47,0.47,0.03,0.03]).tolist(),
    'pregnancy_status': [1 if g == 'Female' and 18 < a < 48 and np.random.rand() < 0.12 else 0 for a, g in zip(np.random.randint(0,96,num_records), np.random.choice(['Male','Female'],num_records))],
    'chronic_condition_flag': np.random.choice([0, 1, 0, 0], num_records).tolist(),
    'zone_id': [f"Zone{chr(65 + (i % 5))}" for i in range(num_records)], # ZoneA, B, C, D, E
    'clinic_id': [f"CLINIC{(i % 3) + 1:02d}" if (i % 4 != 0) else f"HUB{(i % 2) + 1:02d}" for i in range(num_records)],
    'chw_id': [f"CHW{(i % 7) + 1:03d}" if 'CHW' in et else "N/A_ClinicStaff" for i, et in enumerate(np.random.choice(['CHW_HOME_VISIT', 'CLINIC_INTAKE'], num_records))],
    'hrv_rmssd_ms': np.random.uniform(settings.STRESS_HRV_LOW_THRESHOLD_MS - 8, 90, num_records).round(1).tolist(),
    'min_spo2_pct': np.random.choice([settings.ALERT_SPO2_CRITICAL_LOW_PCT - 3, settings.ALERT_SPO2_CRITICAL_LOW_PCT, settings.ALERT_SPO2_WARNING_LOW_PCT - 1, settings.ALERT_SPO2_WARNING_LOW_PCT, 96, 98, 99, 97, np.nan], num_records, p=[0.05,0.05,0.1,0.1,0.2,0.2,0.1,0.15,0.05]).tolist(),
    'vital_signs_temperature_celsius': np.random.choice([36.1, 37.2, settings.ALERT_BODY_TEMP_FEVER_C + 0.3, settings.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.2, settings.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.3, 38.7, np.nan], num_records, p=[0.2,0.2,0.15,0.15,0.1,0.15,0.05]).tolist(),
    'max_skin_temp_celsius': lambda df_lambda: (df_lambda['vital_signs_temperature_celsius'].fillna(37.0) + np.random.uniform(-0.7, 0.7, len(df_lambda))).round(1),
    'movement_activity_level': np.random.choice([0, 1, 2, 3, 1, 2, np.nan], num_records, p=[0.1,0.2,0.3,0.1,0.1,0.15,0.05]).tolist(),
    'fall_detected_today': np.random.choice([0] * 15 + [1], num_records).tolist(), # Lower fall probability
    'ambient_heat_index_c': np.random.uniform(18, settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C + 12, num_records).round(1).tolist(),
    'ppe_compliant_flag': np.random.choice([0, 1, 1, 1, np.nan], num_records, p=[0.1,0.3,0.3,0.25,0.05]).tolist(),
    'signs_of_fatigue_observed_flag': np.random.choice([0, 0, 0, 1, 0, np.nan], num_records, p=[0.2,0.2,0.2,0.15,0.2,0.05]).tolist(),
    'rapid_psychometric_distress_score': np.random.randint(0, 15, num_records).tolist(), # Range 0-14
    'condition': np.random.choice(settings.KEY_CONDITIONS_FOR_ACTION + ['Wellness Visit', 'Minor Ailment', 'Injury - Minor', 'Hypertension Check', 'Asthma Flare-up', 'UnknownCondition'], num_records).tolist(),
    'patient_reported_symptoms': ['fever;cough;fatigue', 'none', 'headache, mild ache', 'diarrhea;vomiting', 'shortness of breath', 'skin rash, itchy', 'general weakness;dizzy', np.nan] * (num_records // 8) + ['cough'] * (num_records % 8),
    'test_type': np.random.choice(list(settings.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None", "PulseOx", "BP Check", "Urinalysis", "Blood Glucose"], num_records).tolist(),
    'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'N/A', '88', '95', '110/70', '150/95', 'Trace Protein', 'Indeterminate', 'Rejected'], num_records).tolist(),
    'test_turnaround_days': [np.random.uniform(0.1, 10).round(1) if res not in ["Pending", "N/A", "Rejected"] and not (isinstance(res,str) and res.replace('.','',1).isdigit()) else np.nan for res in np.random.choice(['Positive', 'Pending', 'N/A', '92', 'Rejected'], num_records)],
    'referral_status': np.random.choice(['Pending', 'Completed', 'Initiated', 'N/A', 'Declined by Patient', 'Attended - Outcome Pending', 'Cancelled'], num_records).tolist(),
    'referral_reason': ['Urgent Clinical Review', 'Lab Investigation', 'Specialist Consult', 'Routine AN Referral', 'Emergency Evacuation', 'None', 'Further Assessment - TB Contact', 'Mental Health Support'] * (num_records // 8) + ['Urgent Review'] * (num_records % 8),
    'referred_to_facility_id': [f"FACIL_DEST_{(i%6)+1:02d}" if stat != "N/A" else "N/A" for i, stat in enumerate(np.random.choice(['Pending', 'N/A', 'Completed'], num_records))],
    'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor', 'N/A', 'Unknown', 'Excellent', 'Missed doses today', 'Missed doses past week'], num_records).tolist(),
    'item': np.random.choice(settings.KEY_DRUG_SUBSTRINGS_SUPPLY + ['Gauze Pads', 'Surgical Gloves', 'ORS Packet', 'Needle & Syringe', 'Bandages'], num_records).tolist(),
    'quantity_dispensed': [np.random.randint(0,6) if itm in settings.KEY_DRUG_SUBSTRINGS_SUPPLY else np.random.randint(0,20) for itm in np.random.choice(settings.KEY_DRUG_SUBSTRINGS_SUPPLY + ['Gauze Pads'], num_records)],
    'item_stock_agg_zone': np.random.randint(0, 300, num_records).tolist(),
    'consumption_rate_per_day': np.random.uniform(0.001, 6.0, num_records).round(3).tolist(),
    'notes': [f"Fixture note {i}" if i % 5 != 0 else np.nan for i in range(num_records)], # Add some NaN notes
    'sample_collection_date': [d - timedelta(hours=np.random.randint(0,6), minutes=np.random.randint(0,59)) if tt not in ["None", "PulseOx", "BP Check"] else pd.NaT for d, tt in zip(record_dates, np.random.choice(list(settings.KEY_TEST_TYPES_FOR_ANALYSIS.keys()) + ["None", "PulseOx", "BP Check"], num_records))],
    'sample_registered_lab_date': lambda df_lambda: [d + timedelta(hours=np.random.randint(0,4), minutes=np.random.randint(0,45)) if pd.notna(d) else pd.NaT for d in df_lambda['sample_collection_date']],
    'sample_status': np.random.choice(['Accepted by Lab', 'Pending Collection by CHW', 'Rejected by Lab', 'N/A', 'In Transit to Lab', 'Received by Lab'], num_records).tolist(),
    'rejection_reason': [np.random.choice(['Hemolysis', 'Insufficient Quantity', 'Improper Label', 'N/A', 'Contaminated', 'Delay in Transit', 'Mislabeled', 'Expired Tube']) if s == 'Rejected by Lab' else 'N/A' for s in np.random.choice(['Accepted by Lab', 'Rejected by Lab', 'N/A'], num_records)],
    'avg_daily_steps': np.random.choice([np.nan, 0, 500] + list(range(1000, 18000, 500)), num_records).tolist(),
    'days_task_overdue': np.random.choice([0,0,0,1,2,5,10, np.nan], num_records, p=[0.4,0.2,0.1,0.05,0.05,0.05,0.05,0.1]).tolist()
}

df = pd.DataFrame({k: v for k, v in raw_data.items() if not callable(v)})
if callable(raw_data.get('max_skin_temp_celsius')):
    df['max_skin_temp_celsius'] = raw_data['max_skin_temp_celsius'](df)
if callable(raw_data.get('sample_registered_lab_date')):
    df['sample_registered_lab_date'] = raw_data['sample_registered_lab_date'](df)

enriched_df, _ = apply_ai_models(df.copy(), source_context="ConftestHealthFixtureAI")

date_cols_final = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date']
for col_dt in date_cols_final:
    if col_dt in enriched_df.columns:
        enriched_df[col_dt] = pd.to_datetime(enriched_df[col_dt], errors='coerce')
return enriched_df
Use code with caution.
@pytest.fixture(scope="session")
def sample_iot_clinic_df_main_fixture() -> pd.DataFrame:
num_records = 25
# CORRECTED: Generate data relative to the current date.
base_date = datetime.now() - timedelta(days=num_records)
timestamps = [base_date + timedelta(days=i) for i in range(num_records)]
data = {
    'timestamp': timestamps,
    'clinic_id': [f"CLINIC{(i%4)+1:02d}" for i in range(num_records)],
    'room_name': np.random.choice(['WaitingArea_A', 'ConsultRoom_1', 'Lab_Main', 'Screening_Tent_B', 'Pharmacy', 'Admin', 'Corridor_Main'], num_records).tolist(),
    'zone_id': [f"Zone{chr(65+(i%5))}" for i in range(num_records)],
    'avg_co2_ppm': np.random.choice([550, 1200, settings.ALERT_AMBIENT_CO2_HIGH_PPM + 200, settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM - 50, settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 300, 3200, np.nan], num_records, p=[0.15,0.15,0.15,0.15,0.1,0.1,0.2]).tolist(),
    'max_co2_ppm': lambda df_lambda: (df_lambda['avg_co2_ppm'].fillna(700) * np.random.uniform(1.02, 1.40, len(df_lambda))).round(0),
    'avg_pm25': np.random.choice([7.5, settings.ALERT_AMBIENT_PM25_HIGH_UGM3 + 5, 20.0, settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 - 2, settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 + 15, 70.0, np.nan], num_records, p=[0.15,0.15,0.15,0.15,0.1,0.1,0.2]).tolist(),
    'voc_index': np.random.randint(20, 400, num_records).tolist(),
    'avg_temp_celsius': np.random.uniform(18.0, 33.0, num_records).round(1).tolist(),
    'avg_humidity_rh': np.random.uniform(25, 90, num_records).round(0).tolist(),
    'avg_noise_db': np.random.choice([40, 65, settings.ALERT_AMBIENT_NOISE_HIGH_DBA - 10, settings.ALERT_AMBIENT_NOISE_HIGH_DBA + 15, 100, np.nan], num_records, p=[0.2,0.2,0.2,0.15,0.1,0.15]).tolist(),
    'waiting_room_occupancy': [np.random.randint(0, settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX + 15) if 'Waiting' in r else np.nan for r in np.random.choice(['WaitingArea_A', 'ConsultRoom_1', 'Corridor_Main'], num_records)],
    'patient_throughput_per_hour': [np.random.randint(max(0, settings.TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR - 4), 20) if 'Consult' in r or 'Screening' in r else np.nan for r in np.random.choice(['ConsultRoom_1', 'Screening_Tent_B','Lab_Main'], num_records)],
    'sanitizer_dispenses_per_hour': np.random.randint(0, 30, num_records).tolist()
}
df = pd.DataFrame({k: v for k,v in data.items() if not callable(v)})
if callable(data.get('max_co2_ppm')): df['max_co2_ppm'] = data['max_co2_ppm'](df)

num_cols_iot = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
for col in num_cols_iot:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
return df
Use code with caution.
@pytest.fixture(scope="session")
def sample_zone_data_df_main_fixture() -> pd.DataFrame:
default_crs = settings.DEFAULT_CRS_STANDARD
attributes = [
{'zone_id': 'ZoneA', 'name': 'Alpha Central', 'population': 12500, 'socio_economic_index': 0.62, 'num_clinics': 3, 'avg_travel_time_clinic_min': 15, 'area_sqkm': 40.5},
{'zone_id': 'ZoneB', 'name': 'Beta South', 'population': 18200, 'socio_economic_index': 0.38, 'num_clinics': 1, 'avg_travel_time_clinic_min': 55, 'area_sqkm': 150.2},
{'zone_id': 'ZoneC', 'name': 'Gamma Hills',  'population': 7800,  'socio_economic_index': 0.81, 'num_clinics': 1, 'avg_travel_time_clinic_min': 10, 'area_sqkm': 25.0},
{'zone_id': 'ZoneD', 'name': 'Delta Coast',  'population': 14000, 'socio_economic_index': 0.50, 'num_clinics': 2, 'avg_travel_time_clinic_min': 25, 'area_sqkm': 75.8},
{'zone_id': 'ZoneE', 'name': 'Epsilon Valley', 'population': 9500, 'socio_economic_index': 0.45, 'num_clinics': 1, 'avg_travel_time_clinic_min': 40, 'area_sqkm': 90.0},
{'zone_id': 'ZoneF', 'name': 'Foxtrot Park', 'population': 3200, 'socio_economic_index': 0.68, 'num_clinics': 0, 'avg_travel_time_clinic_min': 30, 'area_sqkm': 15.5}
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
geom_df['geometry'] = geom_df['geometry_obj'].apply(json.dumps)
merged_df = pd.merge(attr_df, geom_df[['zone_id', 'geometry', 'geometry_obj']], on='zone_id', how='left')
merged_df['crs'] = default_crs
num_attr_cols = ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min', 'area_sqkm']
for col in num_attr_cols:
    if col in merged_df.columns: merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
return merged_df
Use code with caution.
@pytest.fixture(scope="session")
def sample_enriched_zone_df_main_fixture(
sample_zone_data_df_main_fixture: pd.DataFrame,
sample_health_records_df_main_fixture: pd.DataFrame,
sample_iot_clinic_df_main_fixture: pd.DataFrame
) -> Optional[pd.DataFrame]:
if not isinstance(sample_zone_data_df_main_fixture, pd.DataFrame) or sample_zone_data_df_main_fixture.empty:
pytest.skip("Base zone DataFrame fixture invalid/empty for enriched DataFrame fixture.")
return None
return enrich_zone_geodata_with_health_aggregates(
zone_df=sample_zone_data_df_main_fixture.copy(),
health_df=sample_health_records_df_main_fixture.copy(),
iot_df=sample_iot_clinic_df_main_fixture.copy(),
source_context="ConftestEnrichment"
