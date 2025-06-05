# sentinel_project_root/data_processing/loaders.py
# Functions for loading raw data sources for the Sentinel Health Co-Pilot.

import streamlit as st # Only for @st.cache_data, can be removed if caching handled elsewhere
import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Optional, Dict, List, Any
from pathlib import Path

from config import settings
from .helpers import (
    clean_column_names, 
    convert_to_numeric, 
    robust_json_load, 
    convert_date_columns, 
    standardize_missing_values, 
    hash_dataframe_safe
)

logger = logging.getLogger(__name__)

# --- Common Data Loading Utilities ---
def _load_csv_data(file_path: Path, source_context: str) -> pd.DataFrame:
    """Helper to load and minimally clean a CSV file."""
    logger.info(f"({source_context}) Attempting to load CSV data from: {file_path}")
    if not file_path.is_file():
        logger.error(f"({source_context}) CSV file not found or is not a file: {file_path}")
        # Attempt to display error in Streamlit if st is available, otherwise just log
        try:
            st.error(f"Data file missing: {file_path.name}. Please check application setup.")
        except Exception: # Streamlit not available (e.g. backend script)
            pass
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False, on_bad_lines='warn')
        if df.empty:
             logger.warning(f"({source_context}) CSV file at {file_path} is empty after loading (pd.read_csv returned empty).")
             try: st.warning(f"Data file is empty: {file_path.name}.")
             except: pass
             return pd.DataFrame()
        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} raw records from {file_path.name}. Columns: {df.columns.tolist()}")
        return df
    except pd.errors.EmptyDataError: # This specific error for when the file itself is empty (0 bytes or only headers)
        logger.warning(f"({source_context}) EmptyDataError: CSV file at {file_path} is empty.")
        try: st.warning(f"Data file is empty: {file_path.name}.")
        except: pass
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing CSV from {file_path}: {e}", exc_info=True)
        try: st.error(f"Error loading data from {file_path.name}. Details: {e}")
        except: pass
        return pd.DataFrame()

# --- Specific Data Loaders ---
# Consider moving caching to the point of use (e.g., in Streamlit pages)
# if this module is also used by non-Streamlit backend processes.
# For now, keeping @st.cache_data as it was in the original.

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_health_records(file_path_str: Optional[str] = None, source_context: str = "HealthRecordsLoader") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes health records data.
    """
    actual_file_path = Path(file_path_str if file_path_str else settings.HEALTH_RECORDS_CSV_PATH)
    df = _load_csv_data(actual_file_path, source_context)
    if df.empty:
        return pd.DataFrame()

    date_cols_to_convert = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_outcome_date']
    df = convert_date_columns(df, date_cols_to_convert)

    if 'encounter_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
        df['encounter_date_obj'] = df['encounter_date'].dt.date
    else: # Ensure column exists even if encounter_date is bad or missing
        df['encounter_date_obj'] = pd.NaT 
        if 'encounter_date' in df.columns: 
             logger.warning(f"({source_context}) 'encounter_date' column not in datetime format after conversion attempt. 'encounter_date_obj' will be NaT.")
        else:
             logger.warning(f"({source_context}) 'encounter_date' column missing. 'encounter_date_obj' created as NaT.")


    numeric_cols_with_defaults = {
        'age': np.nan, 'pregnancy_status': 0, 'chronic_condition_flag': 0,
        'hrv_rmssd_ms': np.nan, 'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'movement_activity_level': 0, 'fall_detected_today': 0,
        'ambient_heat_index_c': np.nan, 'ppe_compliant_flag': 1,
        'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': np.nan,
        'test_turnaround_days': np.nan, 'quantity_dispensed': 0,
        'item_stock_agg_zone': 0.0, 'consumption_rate_per_day': 0.001,
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, # Typically populated by AI engine later
        'avg_spo2': np.nan, 'avg_daily_steps': np.nan, 'resting_heart_rate': np.nan,
        'avg_sleep_duration_hrs': np.nan, 'sleep_score_pct': np.nan, 'stress_level_score': np.nan,
        'hiv_viral_load_copies_ml': np.nan, 'chw_visit': 0, 'tb_contact_traced': 0,
        'patient_latitude': np.nan, 'patient_longitude': np.nan, 'days_task_overdue': 0
    }
    string_cols_with_defaults = { 
        'encounter_id': "UnknownEncID", 'patient_id': "UnknownPID", 'encounter_type': "UnknownType",
        'gender': "Unknown", 'zone_id': "UnknownZone", 'clinic_id': "UnknownClinic", 'chw_id': "UnknownCHW",
        'condition': "UnknownCondition", 'patient_reported_symptoms': "Unknown", 'test_type': "UnknownTest",
        'test_result': "UnknownResult", 'sample_status': "UnknownStatus", 'rejection_reason': "N/A",
        'referral_status': "Unknown", 'referral_reason': "N/A", 'referred_to_facility_id': "N/A",
        'referral_outcome': "Unknown", 'medication_adherence_self_report': "Unknown", 'item': "UnknownItem",
        'notes': "N/A", 'diagnosis_code_icd10': "N/A", 'physician_id': "N/A",
        'screening_hpv_status': "Unknown", 'key_chronic_conditions_summary': "N/A"
    }
    
    df = standardize_missing_values(df, string_cols_with_defaults, numeric_cols_with_defaults)
    
    logger.info(f"({source_context}) Health records processed. Shape: {df.shape}")
    return df

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_iot_clinic_environment_data(file_path_str: Optional[str] = None, source_context: str = "IoTDataLoader") -> pd.DataFrame:
    actual_file_path = Path(file_path_str if file_path_str else settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH)
    df = _load_csv_data(actual_file_path, source_context)
    if df.empty:
        return pd.DataFrame()

    df = convert_date_columns(df, ['timestamp'])

    numeric_cols_with_defaults = {
        'avg_co2_ppm': np.nan, 'max_co2_ppm': np.nan, 'avg_pm25': np.nan,
        'voc_index': np.nan, 'avg_temp_celsius': np.nan, 'avg_humidity_rh': np.nan,
        'avg_noise_db': np.nan, 'waiting_room_occupancy': np.nan,
        'patient_throughput_per_hour': np.nan, 'sanitizer_dispenses_per_hour': np.nan
    }
    string_cols_with_defaults = {'clinic_id': "UnknownClinic", 'room_name': "UnknownRoom", 'zone_id': "UnknownZone"}
    
    df = standardize_missing_values(df, string_cols_with_defaults, numeric_cols_with_defaults)

    logger.info(f"({source_context}) IoT clinic environment data processed. Shape: {df.shape}")
    return df

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_zone_data(
    attributes_file_path_str: Optional[str] = None,
    geometries_file_path_str: Optional[str] = None,
    source_context: str = "ZoneDataLoader"
) -> pd.DataFrame:
    actual_attributes_path = Path(attributes_file_path_str if attributes_file_path_str else settings.ZONE_ATTRIBUTES_CSV_PATH)
    actual_geometries_path = Path(geometries_file_path_str if geometries_file_path_str else settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH)

    logger.info(f"({source_context}) Loading zone attributes from '{actual_attributes_path}' and geometries from '{actual_geometries_path}'")

    attributes_df = _load_csv_data(actual_attributes_path, f"{source_context}/Attributes")
    if attributes_df.empty and 'zone_id' not in attributes_df.columns: # If completely empty or no zone_id
        attributes_df = pd.DataFrame(columns=['zone_id']) 
    elif 'zone_id' not in attributes_df.columns:
        logger.error(f"({source_context}) 'zone_id' column missing in attributes CSV. This may cause merge issues.")
        # Attempt to proceed, merge might create NaNs or only use GeoJSON data if attributes_df has no zone_id

    geometries_data = robust_json_load(str(actual_geometries_path)) # robust_json_load expects string path
    default_crs = settings.DEFAULT_CRS_STANDARD
    geometries_list: List[Dict[str, Any]] = []

    if geometries_data and isinstance(geometries_data.get("features"), list):
        default_crs = geometries_data.get("crs", {}).get("properties", {}).get("name", default_crs)
        for feature in geometries_data['features']:
            properties = feature.get("properties", {})
            # Try common variations for zone_id in GeoJSON properties
            zone_id_prop = properties.get("zone_id", properties.get("ZONE_ID", properties.get("id", properties.get("ID"))))
            geometry = feature.get("geometry")
            if zone_id_prop and geometry: # Both must be present
                geometries_list.append({
                    "zone_id": str(zone_id_prop).strip(), # Standardize to string
                    "geometry": json.dumps(geometry), # Store geometry as JSON string
                    "geometry_obj": geometry, # Store parsed Python dict object
                    "name_geojson": str(properties.get("name", properties.get("NAME", ""))).strip(),
                    "description_geojson": str(properties.get("description", "")).strip()
                })
            else:
                logger.warning(f"({source_context}) Feature in GeoJSON missing 'zone_id' or 'geometry'. Properties: {properties}")
    else:
        logger.error(f"({source_context}) Invalid or missing GeoJSON features data from {actual_geometries_path}.")
        try: st.warning("Zone geometries data (GeoJSON) is invalid or missing. Map visualizations will not be available.")
        except: pass # Silently fail if Streamlit not available

    geometries_df: pd.DataFrame
    if not geometries_list:
        logger.warning(f"({source_context}) No valid features with zone_id and geometry found in GeoJSON: {actual_geometries_path}")
        geometries_df = pd.DataFrame(columns=['zone_id', 'geometry', 'geometry_obj', 'name_geojson', 'description_geojson'])
    else:
        geometries_df = pd.DataFrame(geometries_list)
        if 'zone_id' in geometries_df.columns: # Ensure zone_id is string for merge
            geometries_df['zone_id'] = geometries_df['zone_id'].astype(str)

    # Merge attributes and geometries
    # Ensure 'zone_id' in attributes_df is string type for merging
    if 'zone_id' in attributes_df.columns:
        attributes_df['zone_id'] = attributes_df['zone_id'].astype(str).str.strip()
        merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")
    else: # attributes_df missing zone_id, use geometries_df as base if it has data
        logger.warning(f"({source_context}) Attributes DataFrame missing 'zone_id'. Resulting zone data will be based mainly on geometries.")
        merged_df = geometries_df # Use geometries_df if attributes_df is unusable for merge

    # Consolidate 'name' column, preferring 'name' from attributes, then 'name_geojson', then from 'zone_id'
    merged_df['name'] = merged_df.get('name', pd.Series(dtype=str)).fillna(merged_df.get('name_geojson', pd.Series(dtype=str)))
    if 'zone_id' in merged_df.columns: # Ensure zone_id exists before using it for name
        merged_df['name'] = merged_df['name'].fillna("Zone " + merged_df['zone_id'].astype(str))
    else: # Fallback if no zone_id to form name
        merged_df['name'] = merged_df['name'].fillna("Unknown Zone Name")
        
    merged_df.drop(columns=['name_geojson'], errors='ignore', inplace=True) # Clean up intermediate column

    numeric_cols_zone_defaults = {
        'population': 0.0, 'socio_economic_index': np.nan, 'num_clinics': 0,
        'avg_travel_time_clinic_min': np.nan, 'area_sqkm': np.nan
    }
    string_cols_zone_defaults = {
        'predominant_hazard_type': "Unknown", 'typical_workforce_exposure_level': "Unknown",
        'primary_livelihood': "Unknown", 'water_source_main': "Unknown",
        'description_geojson': "N/A" # From GeoJSON properties if present
    }
    merged_df = standardize_missing_values(merged_df, string_cols_zone_defaults, numeric_cols_zone_defaults)
    
    merged_df['crs'] = default_crs # Add CRS info from GeoJSON or default

    # Ensure geometry_obj exists as parsed dict, even if 'geometry' string column was missing or merge failed
    if 'geometry_obj' not in merged_df.columns:
        if 'geometry' in merged_df.columns:
            merged_df['geometry_obj'] = merged_df['geometry'].apply(
                lambda x_geom: json.loads(x_geom) if isinstance(x_geom, str) and x_geom.strip().startswith('{') else None
            )
        else:
            merged_df['geometry_obj'] = None # Ensure column exists
    
    # Final check: if 'zone_id' is critical and still missing (e.g., both sources empty), create it
    if 'zone_id' not in merged_df.columns:
        logger.error(f"({source_context}) 'zone_id' is critically missing from the final merged zone data. Assigning placeholder.")
        merged_df['zone_id'] = [f"PLACEHOLDER_ZONE_{i}" for i in range(len(merged_df))]


    logger.info(f"({source_context}) Zone data loading and merging complete. Final shape: {merged_df.shape}")
    return merged_df


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_escalation_protocols(file_path_str: Optional[str] = None, source_context: str = "EscalationProtocolLoader") -> Dict[str, Any]:
    actual_file_path = file_path_str if file_path_str else settings.ESCALATION_PROTOCOLS_JSON_PATH
    logger.info(f"({source_context}) Loading escalation protocols from: {actual_file_path}")
    
    data = robust_json_load(actual_file_path)
    if data is None or not isinstance(data, dict):
        try: st.error("Escalation protocols could not be loaded or are invalid. System alerts may not function correctly.")
        except: pass
        return {"protocols": [], "contacts": {}, "message_templates": {}} 

    if "protocols" not in data or not isinstance(data["protocols"], list):
        logger.error(f"({source_context}) 'protocols' key missing or not a list in escalation protocols file.")
        try: st.warning("Escalation protocols are malformed ('protocols' list missing).")
        except: pass
        data["protocols"] = [] 

    logger.info(f"({source_context}) Escalation protocols loaded successfully: {len(data.get('protocols',[]))} protocols found.")
    return data

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_pictogram_map(file_path_str: Optional[str] = None, source_context: str = "PictogramMapLoader") -> Dict[str, str]:
    actual_file_path = file_path_str if file_path_str else settings.PICTOGRAM_MAP_JSON_PATH
    logger.info(f"({source_context}) Loading pictogram map from: {actual_file_path}")
    data = robust_json_load(actual_file_path)
    if data is None or not isinstance(data, dict):
        try: st.warning("Pictogram map could not be loaded. UI elements might be affected.")
        except: pass
        return {}
    logger.info(f"({source_context}) Pictogram map loaded with {len(data)} entries.")
    return data

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_haptic_patterns(file_path_str: Optional[str] = None, source_context: str = "HapticPatternLoader") -> Dict[str, List[int]]:
    actual_file_path = file_path_str if file_path_str else settings.HAPTIC_PATTERNS_JSON_PATH
    logger.info(f"({source_context}) Loading haptic patterns from: {actual_file_path}")
    data = robust_json_load(actual_file_path)
    if data is None or not isinstance(data, dict):
        try: st.warning("Haptic patterns could not be loaded. Device feedback may be affected.")
        except: pass
        return {}
    logger.info(f"({source_context}) Haptic patterns loaded with {len(data)} entries.")
    return data
