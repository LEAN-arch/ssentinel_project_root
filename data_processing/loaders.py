# sentinel_project_root/data_processing/loaders.py
# Functions for loading raw data sources for the Sentinel Health Co-Pilot.

import streamlit as st 
import pandas as pd
import numpy as np
import os # Retain os for os.path.exists if direct string paths are ever used
import logging
import json
from typing import Optional, Dict, List, Any
from pathlib import Path

from config import settings # This now provides absolute paths
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
def _load_csv_data(file_path_obj: Path, source_context: str) -> pd.DataFrame: # Expect Path object
    """Helper to load and minimally clean a CSV file. Uses absolute path."""
    # Ensure file_path_obj is absolute for clear logging
    abs_file_path = file_path_obj.resolve()
    logger.info(f"({source_context}) Attempting to load CSV data from absolute path: {abs_file_path}")

    if not abs_file_path.is_file():
        err_msg = f"({source_context}) CSV file NOT FOUND or is not a file at resolved absolute path: {abs_file_path}"
        logger.error(err_msg)
        try: st.error(f"Data file missing: {abs_file_path.name}. Expected at: {abs_file_path}. Please check application setup and data availability.")
        except Exception: pass # In case Streamlit context not available
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(abs_file_path, low_memory=False, on_bad_lines='warn')
        if df.empty and abs_file_path.stat().st_size > 0: # File has size but pandas reads as empty (e.g. only headers)
             logger.warning(f"({source_context}) CSV file at {abs_file_path} loaded as empty by pandas, but file has size {abs_file_path.stat().st_size} bytes. Check content/format.")
             try: st.warning(f"Data file '{abs_file_path.name}' seems to have content but loaded empty. Please check CSV format.")
             except: pass
             return pd.DataFrame()
        elif df.empty: # File is genuinely empty or read as such
             logger.warning(f"({source_context}) CSV file at {abs_file_path} is empty (or read as empty by pandas).")
             try: st.warning(f"Data file is empty: {abs_file_path.name}.")
             except: pass
             return pd.DataFrame()

        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} raw records from {abs_file_path.name}. Columns: {df.columns.tolist()}")
        return df
    except pd.errors.EmptyDataError: # Specifically for when read_csv finds an empty file
        logger.warning(f"({source_context}) EmptyDataError: CSV file at {abs_file_path} is empty.")
        try: st.warning(f"Data file is empty: {abs_file_path.name}.")
        except: pass
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing CSV from {abs_file_path}: {e}", exc_info=True)
        try: st.error(f"Error loading data from {abs_file_path.name}. Details: {e}")
        except: pass
        return pd.DataFrame()

# --- Specific Data Loaders ---
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_health_records(file_path_str: Optional[str] = None, source_context: str = "HealthRecordsLoader") -> pd.DataFrame:
    # settings.HEALTH_RECORDS_CSV_PATH is now an absolute string path from settings.py
    actual_file_path = Path(file_path_str if file_path_str else settings.HEALTH_RECORDS_CSV_PATH)
    logger.info(f"({source_context}) Preparing to load health records from: {actual_file_path}") # Log path being used
    df = _load_csv_data(actual_file_path, source_context)
    if df.empty:
        logger.warning(f"({source_context}) Health records DataFrame is empty after _load_csv_data from {actual_file_path}.")
        return pd.DataFrame()

    date_cols_to_convert = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_outcome_date']
    df = convert_date_columns(df, date_cols_to_convert)

    if 'encounter_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
        df['encounter_date_obj'] = df['encounter_date'].dt.date
    else:
        df['encounter_date_obj'] = pd.NaT 
        if 'encounter_date' in df.columns: logger.warning(f"({source_context}) 'encounter_date' not datetime. 'encounter_date_obj' is NaT.")
        else: logger.warning(f"({source_context}) 'encounter_date' missing. 'encounter_date_obj' created as NaT.")

    numeric_cols_with_defaults = {
        'age': np.nan, 'pregnancy_status': 0, 'chronic_condition_flag': 0, 'hrv_rmssd_ms': np.nan, 
        'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'movement_activity_level': 0, 'fall_detected_today': 0, 'ambient_heat_index_c': np.nan, 
        'ppe_compliant_flag': 1, 'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': np.nan,
        'test_turnaround_days': np.nan, 'quantity_dispensed': 0, 'item_stock_agg_zone': 0.0, 
        'consumption_rate_per_day': 0.001, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
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
    if df.empty: return pd.DataFrame()
    df = convert_date_columns(df, ['timestamp'])
    numeric_cols = {'avg_co2_ppm': np.nan, 'max_co2_ppm': np.nan, 'avg_pm25': np.nan, 'voc_index': np.nan, 
                    'avg_temp_celsius': np.nan, 'avg_humidity_rh': np.nan, 'avg_noise_db': np.nan, 
                    'waiting_room_occupancy': np.nan, 'patient_throughput_per_hour': np.nan, 'sanitizer_dispenses_per_hour': np.nan}
    string_cols = {'clinic_id': "UnknownClinic", 'room_name': "UnknownRoom", 'zone_id': "UnknownZone"}
    df = standardize_missing_values(df, string_cols, numeric_cols)
    logger.info(f"({source_context}) IoT data processed. Shape: {df.shape}")
    return df

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_zone_data(
    attributes_file_path_str: Optional[str] = None,
    geometries_file_path_str: Optional[str] = None,
    source_context: str = "ZoneDataLoader"
) -> pd.DataFrame:
    actual_attributes_path = Path(attributes_file_path_str if attributes_file_path_str else settings.ZONE_ATTRIBUTES_CSV_PATH)
    actual_geometries_path = Path(geometries_file_path_str if geometries_file_path_str else settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH)
    logger.info(f"({source_context}) Loading zones: Attrs='{actual_attributes_path}', Geoms='{actual_geometries_path}'")

    attributes_df = _load_csv_data(actual_attributes_path, f"{source_context}/Attributes")
    if attributes_df.empty and 'zone_id' not in attributes_df.columns: attributes_df = pd.DataFrame(columns=['zone_id'])
    elif 'zone_id' not in attributes_df.columns: logger.error(f"({source_context}) 'zone_id' missing in attributes CSV.")

    geometries_data = robust_json_load(str(actual_geometries_path))
    default_crs = settings.DEFAULT_CRS_STANDARD; geometries_list: List[Dict[str, Any]] = []
    if geometries_data and isinstance(geometries_data.get("features"), list):
        default_crs = geometries_data.get("crs", {}).get("properties", {}).get("name", default_crs)
        for feature in geometries_data['features']:
            props = feature.get("properties", {}); zid_prop = props.get("zone_id", props.get("ZONE_ID", props.get("id")))
            geom = feature.get("geometry")
            if zid_prop and geom:
                geometries_list.append({"zone_id": str(zid_prop).strip(), "geometry": json.dumps(geom), "geometry_obj": geom,
                                        "name_geojson": str(props.get("name", props.get("NAME", ""))).strip(),
                                        "description_geojson": str(props.get("description", "")).strip()})
            else: logger.warning(f"({source_context}) GeoJSON feature missing zone_id/geometry. Props: {props}")
    else: logger.error(f"({source_context}) Invalid/missing GeoJSON features: {actual_geometries_path}")

    geometries_df = pd.DataFrame(geometries_list) if geometries_list else pd.DataFrame(columns=['zone_id', 'geometry', 'geometry_obj', 'name_geojson', 'description_geojson'])
    if 'zone_id' in geometries_df.columns: geometries_df['zone_id'] = geometries_df['zone_id'].astype(str)

    if 'zone_id' in attributes_df.columns: attributes_df['zone_id'] = attributes_df['zone_id'].astype(str).str.strip()
    merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer") if 'zone_id' in attributes_df.columns and 'zone_id' in geometries_df.columns else \
                  (attributes_df if not geometries_df.empty else geometries_df) # Fallback if one is missing zone_id

    merged_df['name'] = merged_df.get('name', pd.Series(dtype=str)).fillna(merged_df.get('name_geojson', pd.Series(dtype=str)))
    merged_df['name'] = merged_df['name'].fillna("Zone " + merged_df.get('zone_id', pd.Series(dtype=str)).astype(str)) if 'zone_id' in merged_df.columns else merged_df['name'].fillna("Unknown Zone")
    merged_df.drop(columns=['name_geojson'], errors='ignore', inplace=True)

    num_cols_zone = {'population': 0.0, 'socio_economic_index': np.nan, 'num_clinics': 0, 'avg_travel_time_clinic_min': np.nan, 'area_sqkm': np.nan}
    str_cols_zone = {'predominant_hazard_type': "Unknown", 'typical_workforce_exposure_level': "Unknown", 'primary_livelihood': "Unknown", 'water_source_main': "Unknown", 'description_geojson': "N/A"}
    merged_df = standardize_missing_values(merged_df, str_cols_zone, num_cols_zone)
    merged_df['crs'] = default_crs
    if 'geometry_obj' not in merged_df.columns: merged_df['geometry_obj'] = merged_df.get('geometry', pd.Series(dtype=object)).apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('{') else None)
    if 'zone_id' not in merged_df.columns: merged_df['zone_id'] = [f"GEN_ZONE_{i}" for i in range(len(merged_df))]
    logger.info(f"({source_context}) Zone data loaded/merged. Shape: {merged_df.shape}")
    return merged_df

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_escalation_protocols(file_path_str: Optional[str] = None, source_context: str = "EscalationProtocolLoader") -> Dict[str, Any]:
    data = robust_json_load(file_path_str or settings.ESCALATION_PROTOCOLS_JSON_PATH)
    if not data or not isinstance(data, dict) or not isinstance(data.get("protocols"), list):
        logger.error(f"({source_context}) Escalation protocols invalid or missing 'protocols' list."); return {"protocols": [], "contacts": {}, "message_templates": {}}
    logger.info(f"({source_context}) Escalation protocols loaded: {len(data.get('protocols',[]))} protocols.")
    return data

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_pictogram_map(file_path_str: Optional[str] = None, source_context: str = "PictogramMapLoader") -> Dict[str, str]:
    data = robust_json_load(file_path_str or settings.PICTOGRAM_MAP_JSON_PATH)
    if not data or not isinstance(data, dict): logger.warning(f"({source_context}) Pictogram map invalid."); return {}
    logger.info(f"({source_context}) Pictogram map loaded: {len(data)} entries."); return data

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_haptic_patterns(file_path_str: Optional[str] = None, source_context: str = "HapticPatternLoader") -> Dict[str, List[int]]:
    data = robust_json_load(file_path_str or settings.HAPTIC_PATTERNS_JSON_PATH)
    if not data or not isinstance(data, dict): logger.warning(f"({source_context}) Haptic patterns invalid."); return {}
    logger.info(f"({source_context}) Haptic patterns loaded: {len(data)} entries."); return data
