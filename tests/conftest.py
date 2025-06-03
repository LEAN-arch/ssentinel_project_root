# sentinel_project_root/tests/conftest.py
# Pytest fixtures for testing the "Sentinel Health Co-Pilot" application.

import pytest
import pandas as pd
# GeoPandas import removed
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os
import json # For handling GeoJSON strings if needed
# from shapely.geometry import Polygon # Removed, as GeoPandas is removed.
                                     # If basic geometry validation is needed without GeoPandas,
                                     # it would require a different approach or a lightweight library.

# --- Path Setup for Imports ---
# Add the project root directory (parent of 'tests') to sys.path
# This allows 'config', 'data_processing', 'analytics', 'visualization' to be imported.
_current_conftest_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_for_tests = os.path.abspath(os.path.join(_current_conftest_dir, os.pardir))

if _project_root_dir_for_tests not in sys.path:
    sys.path.insert(0, _project_root_dir_for_tests)

# --- Critical Project Module Imports (from refactored structure) ---
try:
    from config import settings # Use new settings module
    from data_processing.enrichment import enrich_zone_geodata_with_health_aggregates
    from analytics.orchestrator import apply_ai_models
    from data_processing.helpers import hash_dataframe_safe # For caching DataFrames
except ImportError as e_conftest:
    # This is a critical failure for test setup.
    print(f"FATAL ERROR in conftest.py: Could not import core project modules from refactored structure.")
    print(f"PYTHONPATH currently is: {sys.path}")
    print(f"Attempted to add project root: {_project_root_dir_for_tests}")
    print(f"Error details: {e_conftest}")
    # Pytest will likely fail collection anyway, but this provides clearer context.
    raise # Re-raise the error to ensure test collection halts.


# --- Fixture for Sample Health Records ---
@pytest.fixture(scope="session")
def sample_health_records_df_main_fixture() -> pd.DataFrame: # Renamed fixture
    """
    Provides a comprehensive DataFrame of sample health records, already AI-enriched.
    """
    base_fixture_date = datetime(2023, 10, 1, 9, 30, 0)
    num_records_fixture = 70 # Increased slightly for more diversity
    record_dates_fixture = [base_fixture_date + timedelta(hours=i * 2, days=i // 6) for i in range(num_records_fixture)]

    # Use settings for configurable values
    # Wrapped in getattr for safety if a setting is somehow missing during test runs
    # (though settings.py should be robust)
    spo2_crit = getattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT', 90)
    spo2_warn = getattr(settings, 'ALERT_SPO2_WARNING_LOW_PCT', 94)
    temp_fev = getattr(settings, 'ALERT_BODY_TEMP_FEVER_C', 38.0)
    temp_high_fev = getattr(settings, 'ALERT_BODY_TEMP_HIGH_FEVER_C', 39.5)
    hrv_low = getattr(settings, 'STRESS_HRV_LOW_THRESHOLD_MS', 20)
    heat_idx_danger = getattr(settings, 'ALERT_AMBIENT_HEAT_INDEX_DANGER_C', 41)
    key_cond_list = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', ['TB', 'Malaria'])
    key_test_list = list(getattr(settings, 'KEY_TEST_TYPES_FOR_ANALYSIS', {"RDT-Malaria":{}}).keys())
    key_drug_list = getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', ["ACT", "ORS"])


    raw_data_fixture = {
        'encounter_id': [f'SENC_FIX_{i:03d}' for i in range(1, num_records_fixture + 1)],
        'patient_id': [f'SPID_FIX_{(i % 28):03d}' for i in range(1, num_records_fixture + 1)], # ~28 unique patients
        'encounter_date': record_dates_fixture,
        'encounter_type': np.random.choice(['CHW_HOME_VISIT', 'CHW_ALERT_RESPONSE', 'CLINIC_INTAKE', 'WORKER_SELF_CHECK', 'CHW_SCHEDULED_DOTS'], num_records_fixture).tolist(),
        'age': np.random.randint(0, 96, num_records_fixture).tolist(), # 0 to 95
        'gender': np.random.choice(['Male', 'Female', 'Other', 'Unknown'], num_records_fixture, p=[0.46,0.46,0.04,0.04]).tolist(),
        'pregnancy_status': [1 if g == 'Female' and 20 < a < 45 and np.random.rand() < 0.15 else 0 for a, g in zip(np.random.randint(0,96,num_records_fixture), np.random.choice(['Male','Female'],num_records_fixture))], # More realistic pregnancy
        'chronic_condition_flag': np.random.choice([0, 1, 0, 0, 0], num_records_fixture).tolist(),
        'zone_id': [f"Zone{chr(65 + (i % 5))}" for i in range(num_records_fixture)], # ZoneA, B, C, D, E
        'clinic_id': [f"CLINIC{(i % 3) + 1:02d}" if (i%4!=0) else "HUB01" for i in range(num_records_fixture)],
        'chw_id': [f"CHW{(i % 7) + 1:03d}" if 'CHW' in et else "N/A_Clinic" for i, et in enumerate(np.random.choice(['CHW_HOME_VISIT', 'CLINIC_INTAKE'], num_records_fixture))],
        
        'hrv_rmssd_ms': np.random.uniform(hrv_low - 5, 80, num_records_fixture).round(1).tolist(),
        'min_spo2_pct': np.random.choice([spo2_crit - 2, spo2_crit, spo2_warn - 1, spo2_warn, 96, 98, 99, 97], num_records_fixture).tolist(),
        'vital_signs_temperature_celsius': np.random.choice([36.1, 37.2, temp_fev + 0.4, temp_high_fev - 0.1, temp_high_fev + 0.4, 38.5], num_records_fixture).tolist(),
        'max_skin_temp_celsius': lambda df: df['vital_signs_temperature_celsius'] + np.random.uniform(-0.5, 0.5, len(df)),
        'movement_activity_level': np.random.choice([0, 1, 2, 3, 1, 2], num_records_fixture).tolist(),
        'fall_detected_today': np.random.choice([0] * 12 + [1], num_records_fixture).tolist(), # Lower fall probability
        
        'ambient_heat_index_c': np.random.uniform(20, heat_idx_danger + 10, num_records_fixture).round(1).tolist(),
        'ppe_compliant_flag': np.random.choice([0, 1, 1, 1], num_records_fixture).tolist(),
        'signs_of_fatigue_observed_flag': np.random.choice([0, 0, 0, 1, 0], num_records_fixture).tolist(),
        'rapid_psychometric_distress_score': np.random.randint(0, 12, num_records_fixture).tolist(),

        'condition': np.random.choice(key_cond_list + ['Wellness Visit', 'Minor Ailment', 'Injury - Minor', 'Hypertension Check'], num_records_fixture).tolist(),
        'patient_reported_symptoms': ['fever;cough', 'none', 'headache', 'diarrhea;vomiting', 'short breath', 'skin rash', 'general weakness;dizzy'] * (num_records_fixture // 7) + ['cough'] * (num_records_fixture % 7),
        
        'test_type': np.random.choice(key_test_list + ["None", "PulseOx", "BP Check", "Urinalysis"], num_records_fixture).tolist(),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'N/A', '88', '95', '110/70', '150/95', 'Trace Protein'], num_records_fixture).tolist(),
        'test_turnaround_days': [np.random.uniform(0, 9).round(1) if res not in ["Pending", "N/A"] and not res.replace('.','',1).isdigit() else np.nan for res in np.random.choice(['Positive', 'Pending', 'N/A', '92'], num_records_fixture)],

        'referral_status': np.random.choice(['Pending', 'Completed', 'Initiated', 'N/A', 'Declined by Patient', 'Attended - Outcome Pending'], num_records_fixture).tolist(),
        'referral_reason': ['Urgent Clinical Review', 'Lab Investigation', 'Specialist Consult', 'Routine AN Referral', 'Emergency Evacuation', 'None', 'Further Assessment Needed'] * (num_records_fixture // 7) + ['Urgent Review'] * (num_records_fixture % 7),
        'referred_to_facility_id': [f"FACIL_DEST_{(i%5)+1:02d}" if stat != "N/A" else "N/A" for i, stat in enumerate(np.random.choice(['Pending', 'N/A', 'Completed'], num_records_fixture))],
        
        'medication_adherence_self_report': np.random.choice(['Good', 'Fair', 'Poor', 'N/A', 'Unknown', 'Excellent', 'Missed doses'], num_records_fixture).tolist(),
        'item': np.random.choice(key_drug_list + ['Gauze Pads', 'Surgical Gloves', 'ORS Packet', 'Needle & Syringe'], num_records_fixture).tolist(),
        'quantity_dispensed': [np.random.randint(0,5) if itm in key_drug_list else np.random.randint(0,15) for itm in np.random.choice(key_drug_list + ['Gauze Pads'], num_records_fixture)],
        'item_stock_agg_zone': np.random.randint(0, 250, num_records_fixture).tolist(), # Represents zonal stock visible to CHW/Clinic
        'consumption_rate_per_day': np.random.uniform(0.005, 5.0, num_records_fixture).round(3).tolist(),
        
        'notes': ["Fixture generated note for encounter " + str(i) for i in range(num_records_fixture)],
        'sample_collection_date': [d - timedelta(hours=np.random.randint(0,5), minutes=np.random.randint(0,59)) if tt not in ["None", "PulseOx", "BP Check"] else pd.NaT for d, tt in zip(record_dates_fixture, np.random.choice(key_test_list + ["None", "PulseOx", "BP Check"], num_records_fixture))],
        'sample_registered_lab_date': lambda df: [d + timedelta(hours=np.random.randint(0,3), minutes=np.random.randint(0,30)) if pd.notna(d) else pd.NaT for d in df['sample_collection_date']],
        'sample_status': np.random.choice(['Accepted by Lab', 'Pending Collection by CHW', 'Rejected by Lab', 'N/A', 'In Transit to Lab'], num_records_fixture).tolist(),
        'rejection_reason': [np.random.choice(['Hemolysis', 'Insufficient Quantity', 'Improper Label', 'N/A', 'Contaminated', 'Delay in Transit']) if s == 'Rejected by Lab' else 'N/A' for s in np.random.choice(['Accepted by Lab', 'Rejected by Lab', 'N/A'], num_records_fixture)],
        'avg_daily_steps': np.random.choice([np.nan, 0] + list(range(300, 15000, 300)), num_records_fixture).tolist(), # Add some NaNs and 0s
        'days_task_overdue': np.random.choice([0,0,0,1,2,5, np.nan], num_records_fixture).tolist() # For priority model
    }
    df_fixture = pd.DataFrame({k: v for k,v in raw_data_fixture.items() if not callable(v)})
    # Apply lambda functions for dynamically calculated columns
    if callable(raw_data_fixture.get('max_skin_temp_celsius')):
        df_fixture['max_skin_temp_celsius'] = raw_data_fixture['max_skin_temp_celsius'](df_fixture) # type: ignore
    if callable(raw_data_fixture.get('sample_registered_lab_date')):
        df_fixture['sample_registered_lab_date'] = raw_data_fixture['sample_registered_lab_date'](df_fixture) # type: ignore

    # Apply AI models (this function should handle missing columns by adding defaults from its own logic)
    enriched_df_fixture, _ = apply_ai_models(df_fixture.copy(), source_context="ConftestHealthFixtureAI")
    
    # Final type coercions for consistency after all processing, if necessary
    # Example: ensure date columns are indeed datetime
    date_cols_final_fixture = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date']
    for col_dt_final in date_cols_final_fixture:
        if col_dt_final in enriched_df_fixture.columns: # Check existence before conversion
            enriched_df_fixture[col_dt_final] = pd.to_datetime(enriched_df_fixture[col_dt_final], errors='coerce')
    
    return enriched_df_fixture


@pytest.fixture(scope="session")
def sample_iot_clinic_df_main_fixture() -> pd.DataFrame: # Renamed fixture
    """Provides sample IoT data for clinic environments with varied values."""
    num_iot_records_fixture = 20 # Slightly more records
    base_iot_date_fixture = datetime(2023, 10, 20, 7, 0, 0) # Earlier start
    iot_timestamps_fixture = [base_iot_date_fixture + timedelta(hours=i*2) for i in range(num_iot_records_fixture)] # Every 2 hours
    
    # Use settings for configurable values
    co2_high = getattr(settings, 'ALERT_AMBIENT_CO2_HIGH_PPM', 1500)
    co2_very_high = getattr(settings, 'ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500)
    pm25_high = getattr(settings, 'ALERT_AMBIENT_PM25_HIGH_UGM3', 35)
    pm25_very_high = getattr(settings, 'ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 50)
    noise_high = getattr(settings, 'ALERT_AMBIENT_NOISE_HIGH_DBA', 85)
    occupancy_max = getattr(settings, 'TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX', 10)
    throughput_min = getattr(settings, 'TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR', 5)


    iot_fixture_data_vals = {
        'timestamp': iot_timestamps_fixture,
        'clinic_id': [f"CLINIC{(i%4)+1:02d}" for i in range(num_iot_records_fixture)], # More clinics
        'room_name': np.random.choice(['WaitingArea_A', 'ConsultRoom_1', 'Lab_Main', 'Screening_Tent_B', 'Pharmacy_Window', 'Admin_Office', 'Corridor_East'], num_iot_records_fixture).tolist(),
        'zone_id': [f"Zone{chr(65+(i%5))}" for i in range(num_iot_records_fixture)], # A,B,C,D,E
        'avg_co2_ppm': np.random.choice([600, 1300, co2_high + 150, co2_very_high - 100, co2_very_high + 200, 3000, np.nan], num_iot_records_fixture).tolist(), # Added NaN
        'max_co2_ppm': lambda df: (df['avg_co2_ppm'].fillna(800) * np.random.uniform(1.05, 1.35, len(df))).round(0), # Handle NaN in avg_co2
        'avg_pm25': np.random.choice([8.0, pm25_high + 7, 22.0, pm25_very_high - 3, pm25_very_high + 10, 65.0, np.nan], num_iot_records_fixture).tolist(),
        'voc_index': np.random.randint(30, 350, num_iot_records_fixture).tolist(),
        'avg_temp_celsius': np.random.uniform(19.0, 31.0, num_iot_records_fixture).round(1).tolist(),
        'avg_humidity_rh': np.random.uniform(30, 85, num_iot_records_fixture).round(0).tolist(),
        'avg_noise_db': np.random.choice([45, 68, noise_high - 7, noise_high + 12, 95, np.nan], num_iot_records_fixture).tolist(),
        'waiting_room_occupancy': [np.random.randint(1, occupancy_max + 10) if 'Waiting' in r else np.nan for r in np.random.choice(['WaitingArea_A', 'ConsultRoom_1', 'Corridor_East'], num_iot_records_fixture)],
        'patient_throughput_per_hour': [np.random.randint(max(0, throughput_min - 3), 18) if 'Consult' in r or 'Screening' in r else np.nan for r in np.random.choice(['ConsultRoom_1', 'Screening_Tent_B','Lab_Main'], num_iot_records_fixture)],
        'sanitizer_dispenses_per_hour': np.random.randint(0, 25, num_iot_records_fixture).tolist()
    }
    df_iot_fixture = pd.DataFrame({k: v for k,v in iot_fixture_data_vals.items() if not callable(v)})
    if callable(iot_fixture_data_vals.get('max_co2_ppm')):
        df_iot_fixture['max_co2_ppm'] = iot_fixture_data_vals['max_co2_ppm'](df_iot_fixture) #type: ignore
    
    # Ensure numeric columns are numeric, handling potential NaNs introduced
    for col_iot_num in ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 
                        'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 
                        'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']:
        if col_iot_num in df_iot_fixture.columns:
            df_iot_fixture[col_iot_num] = pd.to_numeric(df_iot_fixture[col_iot_num], errors='coerce')
            
    return df_iot_fixture


@pytest.fixture(scope="session")
def sample_zone_data_df_main_fixture() -> pd.DataFrame: # Renamed, returns DataFrame now
    """
    Provides a sample DataFrame representing merged zone attributes and geometries (as JSON strings).
    This simulates the output of `data_processing.loaders.load_zone_data`.
    """
    # Default CRS from settings
    default_crs_fixture = getattr(settings, 'DEFAULT_CRS_STANDARD', "EPSG:4326")

    attributes_fixture_list = [
        {'zone_id': 'ZoneA', 'name': 'Alpha Central District', 'population': 12500, 'socio_economic_index': 0.62, 'num_clinics': 3, 'avg_travel_time_clinic_min': 15, 'predominant_hazard_type': 'URBAN_HEAT_ISLAND', 'primary_livelihood': 'Services', 'water_source_main': 'Piped Network', 'area_sqkm': 40.5},
        {'zone_id': 'ZoneB', 'name': 'Beta Southern Plains', 'population': 18200, 'socio_economic_index': 0.38, 'num_clinics': 1, 'avg_travel_time_clinic_min': 55, 'predominant_hazard_type': 'SEASONAL_FLOODING', 'primary_livelihood': 'Agriculture', 'water_source_main': 'Borehole/Community Well', 'area_sqkm': 150.2},
        {'zone_id': 'ZoneC', 'name': 'Gamma Eastern Hills',  'population': 7800,  'socio_economic_index': 0.81, 'num_clinics': 1, 'avg_travel_time_clinic_min': 10, 'predominant_hazard_type': 'LANDSLIDE_RISK', 'primary_livelihood': 'Small Business/Artisanal', 'water_source_main': 'Piped (Intermittent)', 'area_sqkm': 25.0},
        {'zone_id': 'ZoneD', 'name': 'Delta Western Coast',  'population': 14000, 'socio_economic_index': 0.50, 'num_clinics': 2, 'avg_travel_time_clinic_min': 25, 'predominant_hazard_type': 'COASTAL_STORM', 'primary_livelihood': 'Fishing/Tourism', 'water_source_main': 'Piped Network', 'area_sqkm': 75.8},
        {'zone_id': 'ZoneE', 'name': 'Epsilon Northern Valley', 'population': 9500, 'socio_economic_index': 0.45, 'num_clinics': 1, 'avg_travel_time_clinic_min': 40, 'predominant_hazard_type': 'DROUGHT_RISK', 'primary_livelihood': 'Pastoralism/Agriculture', 'water_source_main': 'River/Spring Fed', 'area_sqkm': 90.0},
        {'zone_id': 'ZoneF', 'name': 'Foxtrot Industrial Park', 'population': 3200, 'socio_economic_index': 0.68, 'num_clinics': 0, 'avg_travel_time_clinic_min': 30, 'predominant_hazard_type': 'AIR_POLLUTION', 'primary_livelihood': 'Factory Work', 'water_source_main': 'Utility Tanker', 'area_sqkm': 15.5} # Zone with no clinics
    ]
    attr_df_fixture = pd.DataFrame(attributes_fixture_list)

    # Sample GeoJSON-like geometry data (as dictionaries, will be stored as JSON strings in 'geometry' column)
    # These are simplified representations. Real GeoJSON would be more complex.
    geometries_fixture_list = [
        {"zone_id": "ZoneA", "geometry_obj": {"type": "Polygon", "coordinates": [[[0,0],[0,1.1],[1.1,1.1],[1.1,0],[0,0]]]}},
        {"zone_id": "ZoneB", "geometry_obj": {"type": "Polygon", "coordinates": [[[1,0],[1,1.2],[2.2,1.2],[2.2,0],[1,0]]]}},
        {"zone_id": "ZoneC", "geometry_obj": {"type": "Polygon", "coordinates": [[[0,1],[0,2.1],[1.1,2.1],[1.1,1],[0,1]]]}},
        {"zone_id": "ZoneD", "geometry_obj": {"type": "Polygon", "coordinates": [[[1,1],[1,2.2],[2.2,2.2],[2.2,1],[1,1]]]}},
        {"zone_id": "ZoneE", "geometry_obj": {"type": "Polygon", "coordinates": [[[-1,0],[-1,1],[0,1],[0,0],[-1,0]]]}},
        {"zone_id": "ZoneF", "geometry_obj": {"type": "Polygon", "coordinates": [[[2,2],[2,3],[3,3],[3,2],[2,2]]]}}
    ]
    # Create geometry_df with 'geometry' as JSON string and 'geometry_obj' as dict
    geom_df_fixture = pd.DataFrame(geometries_fixture_list)
    geom_df_fixture['geometry'] = geom_df_fixture['geometry_obj'].apply(json.dumps) # Store as JSON string

    # Merge, simulating load_zone_data output
    merged_df_fixture = pd.merge(attr_df_fixture, geom_df_fixture[['zone_id', 'geometry', 'geometry_obj']], on='zone_id', how='left')
    
    # Add 'crs' column as expected from load_zone_data
    merged_df_fixture['crs'] = default_crs_fixture
    
    # Ensure numeric types for relevant attribute columns
    numeric_attr_cols_fixture = ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min', 'area_sqkm']
    for col_num_attr in numeric_attr_cols_fixture:
        if col_num_attr in merged_df_fixture.columns:
            merged_df_fixture[col_num_attr] = pd.to_numeric(merged_df_fixture[col_num_attr], errors='coerce')
            
    return merged_df_fixture


@pytest.fixture(scope="session")
def sample_enriched_zone_df_main_fixture( # Renamed fixture
    sample_zone_data_df_main_fixture: pd.DataFrame, # Uses the new DataFrame fixture
    sample_health_records_df_main_fixture: pd.DataFrame,
    sample_iot_clinic_df_main_fixture: pd.DataFrame
) -> Optional[pd.DataFrame]: # Returns DataFrame, not GeoDataFrame
    """
    Provides an enriched zone DataFrame by running the enrichment process on sample data.
    The output is a pandas DataFrame.
    """
    if not isinstance(sample_zone_data_df_main_fixture, pd.DataFrame) or sample_zone_data_df_main_fixture.empty:
        pytest.skip("Base zone DataFrame fixture is invalid or empty for creating enriched DataFrame fixture.")
        return None

    # Use copies to avoid modifying base fixtures if enrichment function is not purely functional
    enriched_df_result = enrich_zone_geodata_with_health_aggregates(
        zone_df=sample_zone_data_df_main_fixture.copy(), # Pass the DataFrame
        health_df=sample_health_records_df_main_fixture.copy(),
        iot_df=sample_iot_clinic_df_main_fixture.copy(),
        source_context="ConftestEnrichment/SentinelMainFixture"
    )
    return enriched_df_result


# --- Fixtures for Empty Schemas (align with new data structures) ---
@pytest.fixture
def empty_health_df_schema_fixture() -> pd.DataFrame: # Renamed
    # Reflects schema after load_health_records and AI enrichment
    cols = [
        'encounter_id', 'patient_id', 'encounter_date', 'encounter_date_obj', 'encounter_type', 
        'age', 'gender', 'pregnancy_status', 'chronic_condition_flag', 'zone_id', 'clinic_id', 'chw_id',
        'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius',
        'movement_activity_level', 'fall_detected_today', 'ambient_heat_index_c', 'ppe_compliant_flag',
        'signs_of_fatigue_observed_flag', 'rapid_psychometric_distress_score', 'condition', 
        'patient_reported_symptoms', 'test_type', 'test_result', 'test_turnaround_days',
        'sample_collection_date', 'sample_registered_lab_date', 'sample_status', 'rejection_reason',
        'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome', 'referral_outcome_date',
        'medication_adherence_self_report', 'item', 'quantity_dispensed', 'item_stock_agg_zone', 
        'consumption_rate_per_day', 'notes', 'diagnosis_code_icd10', 'physician_id', 'avg_spo2', 
        'avg_daily_steps', 'resting_heart_rate', 'avg_sleep_duration_hrs', 'sleep_score_pct', 
        'stress_level_score', 'screening_hpv_status', 'hiv_viral_load_copies_ml', 
        'key_chronic_conditions_summary', 'chw_visit', 'tb_contact_traced', 
        'patient_latitude', 'patient_longitude', 'days_task_overdue',
        'ai_risk_score', 'ai_followup_priority_score' # Added by AI engine
    ]
    return pd.DataFrame(columns=list(set(cols))) # Use set to ensure unique columns in schema

@pytest.fixture
def empty_iot_df_schema_fixture() -> pd.DataFrame: # Renamed
    cols = [
        'timestamp', 'clinic_id', 'room_name', 'zone_id', 'avg_co2_ppm', 'max_co2_ppm',
        'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_zone_data_df_schema_fixture() -> pd.DataFrame: # Renamed, represents output of load_zone_data
    cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'num_clinics',
            'avg_travel_time_clinic_min', 'predominant_hazard_type', 
            'typical_workforce_exposure_level', 'primary_livelihood', 'water_source_main', 'area_sqkm',
            'geometry', 'geometry_obj', 'crs', 'name_geojson', 'description_geojson' # Added from new loader
           ]
    return pd.DataFrame(columns=list(set(cols)))

@pytest.fixture
def empty_enriched_zone_df_schema_fixture() -> pd.DataFrame: # Renamed
    # Columns expected after `enrich_zone_geodata_with_health_aggregates`
    # This should now be a pandas DataFrame schema.
    base_cols_enriched_fixture = [
        'zone_id', 'name', 'geometry', 'geometry_obj', 'crs', 'population', # Base attributes from load_zone_data
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'total_active_key_infections', 'prevalence_per_1000',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met',
        'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score', 'population_density',
        'chw_density_per_10k' # Placeholder, needs chw_count_zone for actual calculation during enrichment
    ]
    # Add dynamic condition columns based on settings
    dynamic_condition_cols_fixture = [
        f"active_{cond_key_fixture.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" 
        for cond_key_fixture in getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])
    ]
    all_cols_final_enriched_fixture = list(set(base_cols_enriched_fixture + dynamic_condition_cols_fixture))
    return pd.DataFrame(columns=all_cols_final_enriched_fixture)


# --- Generic Plotting Data Fixtures (remain largely the same) ---
@pytest.fixture(scope="session")
def sample_series_data_plotting_fixture() -> pd.Series: # Renamed
    idx_dates_plot = pd.to_datetime(['2023-03-01', '2023-03-08', '2023-03-15', '2023-03-22', '2023-03-29', '2023-04-05'])
    return pd.Series([15.5, 18.2, 12.0, 20.7, 17.1, 22.9], index=idx_dates_plot, name="WeeklyMetricValue")

@pytest.fixture(scope="session")
def sample_bar_df_plotting_fixture() -> pd.DataFrame: # Renamed
    return pd.DataFrame({
        'category_label_plot': ['Condition Alpha', 'Condition Beta', 'Condition Gamma', 'Condition Alpha', 'Condition Delta', 'Condition Beta'],
        'value_count_plot': [25, 19, 30, 38, 12, 24],
        'grouping_col_plot': ['GroupX', 'GroupY', 'GroupX', 'GroupY', 'GroupX', 'GroupY']
    })

@pytest.fixture(scope="session")
def sample_donut_df_plotting_fixture() -> pd.DataFrame: # Renamed
    return pd.DataFrame({
        'risk_level_label_plot': ['Critical', 'Warning', 'Acceptable', 'Pending Review', 'Low'],
        'case_counts_plot': [12, 28, 65, 8, 110]
    })

@pytest.fixture(scope="session")
def sample_heatmap_df_plotting_fixture() -> pd.DataFrame: # Renamed
    rows_heatmap_plot = ['Symptom: Fever', 'Symptom: Cough', 'Symptom: Fatigue', 'Alert: Low SpO2']
    cols_heatmap_plot = ['Zone Alpha', 'Zone Beta', 'Zone Gamma', 'Zone Delta']
    data_heatmap_plot = np.array([[0.85, 0.15, 0.55, 0.30], [0.70, 0.92, 0.25, 0.45], [0.60, 0.48, 0.88, 0.65], [0.90, 0.05, 0.18, 0.75]])
    return pd.DataFrame(data_heatmap_plot, index=rows_heatmap_plot, columns=cols_heatmap_plot)

@pytest.fixture(scope="session")
def sample_map_data_df_plotting_fixture(sample_enriched_zone_df_main_fixture: pd.DataFrame) -> pd.DataFrame: # Renamed
    """ Creates a DataFrame suitable for choropleth map testing using the enriched zone data. """
    if not isinstance(sample_enriched_zone_df_main_fixture, pd.DataFrame) or sample_enriched_zone_df_main_fixture.empty:
        # Return an empty DataFrame with expected schema if base is invalid
        return pd.DataFrame(columns=['zone_id', 'name', 'sample_risk_value', 'sample_facility_count'])

    df_for_map_plot = sample_enriched_zone_df_main_fixture[['zone_id', 'name', 'population', 'area_sqkm', 'avg_risk_score']].copy()
    
    # Add some specific columns for map testing if not directly present or to ensure variance
    rng_map_plot = np.random.RandomState(456) # For reproducible random values
    if 'avg_risk_score' not in df_for_map_plot.columns or df_for_map_plot['avg_risk_score'].isnull().all():
        df_for_map_plot['avg_risk_score'] = rng_map_plot.uniform(20, 85, len(df_for_map_plot)).round(1)
    
    if 'facility_coverage_score' not in df_for_map_plot.columns: # Example if this was a direct map metric
         df_for_map_plot['facility_coverage_score'] = rng_map_plot.uniform(30, 95, len(df_for_map_plot)).round(0)
         
    return df_for_map_plot
