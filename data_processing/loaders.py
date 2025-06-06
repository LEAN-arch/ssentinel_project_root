# sentinel_project_root/data_processing/loaders.py
# Functions for loading raw data sources for the Sentinel Health Co-Pilot.

import streamlit as st 
import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

try:
    from config import settings 
    from .helpers import ( # Relative import for helpers within the same package
        clean_column_names, 
        convert_to_numeric, 
        robust_json_load, 
        convert_date_columns, 
        standardize_missing_values, 
        hash_dataframe_safe
    )
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in loaders.py: {e}. Ensure config.py and helpers.py are accessible.")
    raise # Re-raise as loaders are fundamental

logger = logging.getLogger(__name__)

# --- Common Data Loading Utilities ---
def _load_csv_data(file_path_obj: Path, source_context: str, low_memory_setting: bool = False) -> pd.DataFrame:
    """
    Helper to load and minimally clean a CSV file. Expects an absolute Path object.
    Returns an empty DataFrame on any critical error.
    """
    abs_file_path = file_path_obj.resolve() # Ensure path is absolute for clarity
    logger.info(f"({source_context}) Attempting to load CSV data from: {abs_file_path}")

    if not abs_file_path.is_file():
        err_msg = f"({source_context}) CSV file NOT FOUND or is not a file at: {abs_file_path}"
        logger.error(err_msg)
        try: 
            st.error(f"Data file missing: '{abs_file_path.name}'. Please check application setup.")
        except Exception: pass # Silently fail if Streamlit context not available
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(abs_file_path, low_memory=low_memory_setting, on_bad_lines='warn')
        
        if df.empty:
            file_size = abs_file_path.stat().st_size
            if file_size > 0:
                 logger.warning(f"({source_context}) CSV file at {abs_file_path} loaded as empty by pandas, but file has size {file_size} bytes. Check content/format (e.g., encoding, delimiter, only headers).")
                 try: st.warning(f"Data file '{abs_file_path.name}' appears to have content but loaded empty. Please check CSV format and encoding.")
                 except Exception: pass
            else:
                 logger.warning(f"({source_context}) CSV file at {abs_file_path} is empty (0 bytes or read as such).")
                 try: st.info(f"Data file is empty: {abs_file_path.name}.")
                 except Exception: pass
            return pd.DataFrame()

        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} raw records from '{abs_file_path.name}'. Columns: {df.columns.tolist()[:10]}...")
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"({source_context}) EmptyDataError: CSV file at {abs_file_path} is effectively empty or contains only headers.")
        try: st.info(f"Data file is empty or contains only headers: {abs_file_path.name}.")
        except Exception: pass
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing CSV from {abs_file_path}: {e}", exc_info=True)
        try: st.error(f"Error loading data from '{abs_file_path.name}'. Details logged. Error: {type(e).__name__}")
        except Exception: pass
        return pd.DataFrame()

# --- Specific Data Loaders ---
def load_health_records(file_path_str: Optional[str] = None, source_context: str = "HealthRecordsLoader") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes health records data.
    Uses settings.HEALTH_RECORDS_CSV_PATH if file_path_str is None.
    """
    path_to_load_str = file_path_str
    if not path_to_load_str:
        if hasattr(settings, 'HEALTH_RECORDS_CSV_PATH'):
            path_to_load_str = settings.HEALTH_RECORDS_CSV_PATH
        else:
            logger.error(f"({source_context}) HEALTH_RECORDS_CSV_PATH not found in settings and no explicit path provided.")
            return pd.DataFrame()
            
    actual_file_path = Path(path_to_load_str)
    if not actual_file_path.is_absolute():
        logger.warning(f"({source_context}) Health records path '{path_to_load_str}' is not absolute. Attempting to resolve from DATA_SOURCES_DIR.")
        if hasattr(settings, 'DATA_SOURCES_DIR'):
            actual_file_path = (Path(settings.DATA_SOURCES_DIR) / actual_file_path).resolve()
        else:
            logger.error(f"({source_context}) DATA_SOURCES_DIR not in settings, cannot resolve relative health records path. Path used: {actual_file_path}")
            return pd.DataFrame()

    logger.info(f"({source_context}) Preparing to load health records from: {actual_file_path}")
    df = _load_csv_data(actual_file_path, source_context, low_memory_setting=True)

    if df.empty:
        logger.warning(f"({source_context}) Health records DataFrame is empty after _load_csv_data from {actual_file_path}.")
        return pd.DataFrame()

    date_cols_to_convert = getattr(settings, 'HEALTH_RECORDS_DATE_COLS', [
        'encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 
        'referral_outcome_date', 'date_of_birth'
    ])
    df = convert_date_columns(df, date_cols_to_convert)

    # CORRECTED: Use None for missing date objects to avoid dtype issues with pd.NaT.
    if 'encounter_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
        # .dt.date correctly returns python `date` objects, with NaT becoming None automatically.
        df['encounter_date_obj'] = df['encounter_date'].dt.date
    else:
        # Create a column of `None` with object dtype. This is compatible with functions
        # expecting `Optional[date]`.
        df['encounter_date_obj'] = pd.Series([None] * len(df), index=df.index, dtype=object)
        logger.warning(f"({source_context}) 'encounter_date' not datetime or missing. 'encounter_date_obj' column is None.")

    numeric_cols_config = getattr(settings, 'HEALTH_RECORDS_NUMERIC_COLS_DEFAULTS', {
        'age': np.nan, 'pregnancy_status': 0, 'chronic_condition_flag': 0, 
        'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 
        'fall_detected_today': 0, 'test_turnaround_days': np.nan, 
        'quantity_dispensed': 0, 'item_stock_agg_zone': 0.0, 
        'consumption_rate_per_day': 0.001, 'ai_risk_score': np.nan, 
        'ai_followup_priority_score': np.nan, 'avg_daily_steps': np.nan,
        'tb_contact_traced': 0
    })
    string_cols_config = getattr(settings, 'HEALTH_RECORDS_STRING_COLS_DEFAULTS', {
        'encounter_id': "UnknownEncID", 'patient_id': "UnknownPID", 'encounter_type': "UnknownType",
        'gender': "Unknown", 'zone_id': "UnknownZone", 'clinic_id': "UnknownClinic", 
        'chw_id': "UnknownCHW", 'condition': "UnknownCondition", 
        'patient_reported_symptoms': "", 'test_type': "UnknownTest",
        'test_result': "UnknownResult", 'sample_status': "UnknownStatus", 
        'referral_status': "Unknown", 'item': "UnknownItem"
    })
    
    df = standardize_missing_values(df, string_cols_config, numeric_cols_config)
    logger.info(f"({source_context}) Health records loaded and processed. Shape: {df.shape}")
    return df


def load_iot_clinic_environment_data(file_path_str: Optional[str] = None, source_context: str = "IoTDataLoader") -> pd.DataFrame:
    """Loads, cleans, and standardizes IoT clinic environment data."""
    path_to_load_str = file_path_str
    if not path_to_load_str:
        if hasattr(settings, 'IOT_CLINIC_ENVIRONMENT_CSV_PATH'):
            path_to_load_str = settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH
        else:
            logger.error(f"({source_context}) IOT_CLINIC_ENVIRONMENT_CSV_PATH not in settings and no explicit path provided.")
            return pd.DataFrame()

    actual_file_path = Path(path_to_load_str)
    if not actual_file_path.is_absolute():
        if hasattr(settings, 'DATA_SOURCES_DIR'):
            actual_file_path = (Path(settings.DATA_SOURCES_DIR) / actual_file_path).resolve()
        else:
            logger.error(f"({source_context}) DATA_SOURCES_DIR not in settings, cannot resolve relative IoT path. Path used: {actual_file_path}")
            return pd.DataFrame()
            
    logger.info(f"({source_context}) Preparing to load IoT data from: {actual_file_path}")
    df = _load_csv_data(actual_file_path, source_context)
    if df.empty: 
        logger.warning(f"({source_context}) IoT data DataFrame is empty after _load_csv_data from {actual_file_path}.")
        return pd.DataFrame()

    date_cols_iot = getattr(settings, 'IOT_DATE_COLS', ['timestamp'])
    df = convert_date_columns(df, date_cols_iot)

    numeric_cols_iot_config = getattr(settings, 'IOT_NUMERIC_COLS_DEFAULTS', {
        'avg_co2_ppm': np.nan, 'max_co2_ppm': np.nan, 'avg_pm25': np.nan, 
        'voc_index': np.nan, 'avg_temp_celsius': np.nan, 'avg_humidity_rh': np.nan, 
        'avg_noise_db': np.nan, 'waiting_room_occupancy': np.nan, 
        'patient_throughput_per_hour': np.nan, 'sanitizer_dispenses_per_hour': np.nan
    })
    string_cols_iot_config = getattr(settings, 'IOT_STRING_COLS_DEFAULTS', {
        'clinic_id': "UnknownClinic", 'room_name': "UnknownRoom", 'zone_id': "UnknownZone", 'sensor_id': "UnknownSensor"
    })
    df = standardize_missing_values(df, string_cols_iot_config, numeric_cols_iot_config)
    
    logger.info(f"({source_context}) IoT data loaded and processed. Shape: {df.shape}")
    return df


def load_zone_data(
    attributes_file_path_str: Optional[str] = None,
    geometries_file_path_str: Optional[str] = None,
    source_context: str = "ZoneDataLoader"
) -> pd.DataFrame:
    """Loads and merges zone attribute data (CSV) and zone geometries (GeoJSON)."""
    
    attr_path_str = attributes_file_path_str
    if not attr_path_str:
        if hasattr(settings, 'ZONE_ATTRIBUTES_CSV_PATH'): attr_path_str = settings.ZONE_ATTRIBUTES_CSV_PATH
        else: logger.error(f"({source_context}) ZONE_ATTRIBUTES_CSV_PATH not in settings. Cannot load attributes."); return pd.DataFrame()
    
    geom_path_str = geometries_file_path_str
    if not geom_path_str:
        if hasattr(settings, 'ZONE_GEOMETRIES_GEOJSON_FILE_PATH'): geom_path_str = settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH
        else: logger.error(f"({source_context}) ZONE_GEOMETRIES_GEOJSON_FILE_PATH not in settings. Cannot load geometries."); return pd.DataFrame()

    actual_attributes_path = Path(attr_path_str)
    if not actual_attributes_path.is_absolute():
        if hasattr(settings, 'DATA_SOURCES_DIR'): actual_attributes_path = (Path(settings.DATA_SOURCES_DIR) / actual_attributes_path).resolve()
        else: logger.error(f"({source_context}) Cannot resolve relative attributes path '{attr_path_str}'. DATA_SOURCES_DIR missing."); return pd.DataFrame()

    actual_geometries_path = Path(geom_path_str)
    if not actual_geometries_path.is_absolute():
        if hasattr(settings, 'DATA_SOURCES_DIR'): actual_geometries_path = (Path(settings.DATA_SOURCES_DIR) / actual_geometries_path).resolve()
        else: logger.error(f"({source_context}) Cannot resolve relative geometries path '{geom_path_str}'. DATA_SOURCES_DIR missing."); return pd.DataFrame()

    logger.info(f"({source_context}) Loading zone data: Attributes='{actual_attributes_path}', Geometries='{actual_geometries_path}'")

    attributes_df = _load_csv_data(actual_attributes_path, f"{source_context}/Attributes")
    if 'zone_id' in attributes_df.columns:
        attributes_df['zone_id'] = attributes_df['zone_id'].astype(str).str.strip()
    else:
        logger.error(f"({source_context}) 'zone_id' column missing in attributes CSV '{actual_attributes_path.name}'. Zone data merge will be incomplete.")

    geometries_list: List[Dict[str, Any]] = []
    default_crs_setting = getattr(settings, 'DEFAULT_CRS_STANDARD', 'EPSG:4326')
    loaded_crs = default_crs_setting
    
    geometries_data = robust_json_load(actual_geometries_path)
    if geometries_data and isinstance(geometries_data.get("features"), list):
        try:
            loaded_crs = geometries_data.get("crs", {}).get("properties", {}).get("name", default_crs_setting)
        except Exception:
            logger.debug(f"({source_context}) Could not parse CRS from GeoJSON, using default: {default_crs_setting}")

        for feature in geometries_data['features']:
            props = feature.get("properties", {}) if isinstance(feature, dict) and isinstance(feature.get("properties"), dict) else {}
            geom = feature.get("geometry") if isinstance(feature, dict) else None
            
            zid_prop_val = props.get("zone_id", props.get("ZONE_ID", props.get("id", props.get("OBJECTID"))))
            name_prop_val = props.get("name", props.get("NAME", props.get("zone_name", "")))
            
            if zid_prop_val is not None and geom:
                geometries_list.append({
                    "zone_id": str(zid_prop_val).strip(), 
                    "geometry_str": json.dumps(geom),
                    "geometry_obj": geom,
                    "name_geojson": str(name_prop_val).strip(),
                    "description_geojson": str(props.get("description", "")).strip()
                })
            else: 
                logger.debug(f"({source_context}) GeoJSON feature missing 'zone_id' (or equivalent) or 'geometry'. Properties: {list(props.keys())}")
    else: 
        logger.warning(f"({source_context}) Invalid or missing GeoJSON features in '{actual_geometries_path.name}'. No geometries will be loaded.")

    geometries_df = pd.DataFrame(geometries_list) if geometries_list else pd.DataFrame(columns=['zone_id', 'geometry_str', 'geometry_obj', 'name_geojson', 'description_geojson'])
    if 'zone_id' in geometries_df.columns: 
        geometries_df['zone_id'] = geometries_df['zone_id'].astype(str).str.strip()

    if attributes_df.empty and geometries_df.empty:
        logger.error(f"({source_context}) Both zone attributes and geometries are empty. Returning empty DataFrame.")
        return pd.DataFrame()
    elif attributes_df.empty:
        logger.warning(f"({source_context}) Zone attributes DataFrame is empty. Using only geometries data.")
        merged_df = geometries_df
    elif geometries_df.empty:
        logger.warning(f"({source_context}) Zone geometries DataFrame is empty. Using only attributes data.")
        merged_df = attributes_df
    elif 'zone_id' not in attributes_df.columns or 'zone_id' not in geometries_df.columns:
        logger.warning(f"({source_context}) 'zone_id' missing in attributes or geometries. Cannot merge. Returning attributes only.")
        merged_df = attributes_df
    else:
        merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

    if 'name' not in merged_df.columns and 'name_geojson' in merged_df.columns:
        merged_df['name'] = merged_df['name_geojson']
    elif 'name' in merged_df.columns and 'name_geojson' in merged_df.columns:
        merged_df['name'] = merged_df['name'].fillna(merged_df['name_geojson'])
    
    if 'name' in merged_df.columns and 'zone_id' in merged_df.columns:
        merged_df['name'] = merged_df['name'].fillna("Zone " + merged_df['zone_id'].astype(str))
    elif 'name' not in merged_df.columns and 'zone_id' in merged_df.columns:
         merged_df['name'] = "Zone " + merged_df['zone_id'].astype(str)
    elif 'name' not in merged_df.columns:
         merged_df['name'] = [f"UnknownZone_{i}" for i in range(len(merged_df))]

    merged_df.drop(columns=['name_geojson'], errors='ignore', inplace=True)

    zone_numeric_cols_config = getattr(settings, 'ZONE_NUMERIC_COLS_DEFAULTS', {
        'population': 0.0, 'socio_economic_index': np.nan, 'num_clinics': 0, 
        'num_chws': 0, 'avg_travel_time_clinic_min': np.nan, 'area_sqkm': np.nan
    })
    zone_string_cols_config = getattr(settings, 'ZONE_STRING_COLS_DEFAULTS', {
        'predominant_hazard_type': "Unknown", 'typical_workforce_exposure_level': "Unknown", 
        'primary_livelihood': "Unknown", 'water_source_main': "Unknown", 
        'description_geojson': "N/A"
    })
    merged_df = standardize_missing_values(merged_df, zone_string_cols_config, zone_numeric_cols_config)
    
    merged_df['crs'] = loaded_crs

    if 'geometry_obj' not in merged_df.columns:
        if 'geometry_str' in merged_df.columns:
            def safe_json_loads(x):
                if isinstance(x, str) and x.strip().startswith('{'):
                    try: return json.loads(x)
                    except json.JSONDecodeError: return None
                return None
            merged_df['geometry_obj'] = merged_df['geometry_str'].apply(safe_json_loads)
        else:
             merged_df['geometry_obj'] = None

    if 'zone_id' not in merged_df.columns:
        logger.warning(f"({source_context}) 'zone_id' column still missing after processing. Generating generic IDs.")
        merged_df['zone_id'] = [f"GEN_ZONE_IDX_{i}" for i in range(len(merged_df))]

    logger.info(f"({source_context}) Zone data loaded and merged. Final shape: {merged_df.shape}")
    return merged_df


def load_json_config_file(
    file_path_str_or_setting_attr: str, 
    default_return_value: Union[Dict, List],
    source_context: str = "JSONConfigLoader"
) -> Union[Dict, List]:
    """
    Loads a JSON configuration file. The input can be a direct file path string
    or an attribute name from `settings` that holds the file path string.
    """
    path_to_load_str = ""
    # Check if it's an attribute in settings
    if hasattr(settings, file_path_str_or_setting_attr):
        path_to_load_str = getattr(settings, file_path_str_or_setting_attr)
        logger.debug(f"({source_context}) Loading JSON from settings attribute '{file_path_str_or_setting_attr}' -> path '{path_to_load_str}'")
    # Check if it looks like a path
    elif isinstance(file_path_str_or_setting_attr, str) and (os.path.sep in file_path_str_or_setting_attr or Path(file_path_str_or_setting_attr).suffix == '.json'):
        path_to_load_str = file_path_str_or_setting_attr
        logger.debug(f"({source_context}) Loading JSON from direct path string: '{path_to_load_str}'")
    else:
        logger.error(f"({source_context}) Invalid file path or settings attribute: '{file_path_str_or_setting_attr}'. Cannot load JSON.")
        return default_return_value

    if not path_to_load_str or not isinstance(path_to_load_str, str):
        logger.error(f"({source_context}) Resolved path to load is empty or not a string for '{file_path_str_or_setting_attr}'.")
        return default_return_value

    actual_file_path = Path(path_to_load_str)
    if not actual_file_path.is_absolute():
        if hasattr(settings, 'PROJECT_ROOT_DIR'):
            actual_file_path = (Path(settings.PROJECT_ROOT_DIR) / actual_file_path).resolve()
        else:
            logger.warning(f"({source_context}) Cannot resolve relative JSON path '{actual_file_path}' as PROJECT_ROOT_DIR is not in settings. Trying current dir.")
            actual_file_path = actual_file_path.resolve()

    data = robust_json_load(actual_file_path)
    if data is None:
        logger.warning(f"({source_context}) Failed to load JSON from '{actual_file_path}'. Returning default.")
        return default_return_value
    
    if isinstance(default_return_value, dict) and not isinstance(data, dict):
        logger.warning(f"({source_context}) Loaded JSON from '{actual_file_path}' is not a dictionary as expected. Returning default.")
        return default_return_value
    if isinstance(default_return_value, list) and not isinstance(data, list):
        logger.warning(f"({source_context}) Loaded JSON from '{actual_file_path}' is not a list as expected. Returning default.")
        return default_return_value

    logger.info(f"({source_context}) Successfully loaded JSON configuration from '{actual_file_path.name}'. Type: {type(data)}")
    return data


# --- Functions to be called from pages, using the generic JSON loader ---

def load_escalation_protocols(file_path_str: Optional[str] = None, source_context: str = "EscalationProtocolLoader") -> Dict[str, Any]:
    """Loads escalation protocols. Expects a dictionary with a 'protocols' list."""
    default_protocols_struct = {"protocols": [], "contacts": {}, "message_templates": {}}
    path_or_attr = file_path_str if file_path_str else 'ESCALATION_PROTOCOLS_JSON_PATH'
    
    data = load_json_config_file(path_or_attr, default_protocols_struct, source_context)
    
    if not (isinstance(data, dict) and isinstance(data.get("protocols"), list)):
        logger.error(f"({source_context}) Loaded escalation data from '{path_or_attr}' is invalid or missing 'protocols' list. Returning default structure.")
        return default_protocols_struct
        
    logger.info(f"({source_context}) Escalation protocols processed: {len(data.get('protocols',[]))} protocols.")
    return data

def load_pictogram_map(file_path_str: Optional[str] = None, source_context: str = "PictogramMapLoader") -> Dict[str, str]:
    """Loads pictogram map. Expects a dictionary."""
    path_or_attr = file_path_str if file_path_str else 'PICTOGRAM_MAP_JSON_PATH'
    data = load_json_config_file(path_or_attr, {}, source_context)
    if not isinstance(data, dict):
        logger.warning(f"({source_context}) Pictogram map data from '{path_or_attr}' is not a dictionary. Returning empty map.")
        return {}
    logger.info(f"({source_context}) Pictogram map processed: {len(data)} entries."); 
    return data

def load_haptic_patterns(file_path_str: Optional[str] = None, source_context: str = "HapticPatternLoader") -> Dict[str, List[int]]:
    """Loads haptic patterns. Expects a dictionary where values are lists of integers."""
    path_or_attr = file_path_str if file_path_str else 'HAPTIC_PATTERNS_JSON_PATH'
    data = load_json_config_file(path_or_attr, {}, source_context)
    if not isinstance(data, dict):
        logger.warning(f"({source_context}) Haptic patterns data from '{path_or_attr}' is not a dictionary. Returning empty map.")
        return {}
    
    valid_data = {}
    for key, value in data.items():
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            valid_data[key] = value
        else:
            logger.warning(f"({source_context}) Invalid haptic pattern for key '{key}' in '{path_or_attr}'. Expected list of integers. Skipping.")
            
    logger.info(f"({source_context}) Haptic patterns processed: {len(valid_data)} valid entries."); 
    return valid_data
