# sentinel_project_root/data_processing/loaders.py
# Functions for loading raw data sources for the Sentinel Health Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Optional, Dict, List, Any

from config import settings
from .helpers import clean_column_names, convert_to_numeric, robust_json_load, convert_date_columns, standardize_missing_values

logger = logging.getLogger(__name__)

# --- Common Data Loading Utilities ---
def _load_csv_data(file_path: str, source_context: str) -> pd.DataFrame:
    """Helper to load and minimally clean a CSV file."""
    logger.info(f"({source_context}) Attempting to load CSV data from: {file_path}")
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.error(f"({source_context}) CSV file not found or is not a file: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} raw records from {os.path.basename(file_path)}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logger.error(f"({source_context}) FileNotFoundError: CSV file not found at {file_path}")
        st.error(f"Data file missing: {os.path.basename(file_path)}. Please check application setup.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.warning(f"({source_context}) EmptyDataError: CSV file at {file_path} is empty.")
        st.warning(f"Data file is empty: {os.path.basename(file_path)}.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing CSV from {file_path}: {e}", exc_info=True)
        st.error(f"Error loading data from {os.path.basename(file_path)}. Details: {e}")
        return pd.DataFrame()

# --- Specific Data Loaders ---

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_health_records(file_path: Optional[str] = None, source_context: str = "HealthRecordsLoader") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes health records data.
    """
    actual_file_path = file_path or settings.HEALTH_RECORDS_CSV_PATH
    df = _load_csv_data(actual_file_path, source_context)
    if df.empty:
        return pd.DataFrame()

    # Define standard date columns
    date_cols_to_convert = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_outcome_date']
    df = convert_date_columns(df, date_cols_to_convert)

    # Add encounter_date_obj if encounter_date is present and valid
    if 'encounter_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
        df['encounter_date_obj'] = df['encounter_date'].dt.date
    else:
        df['encounter_date_obj'] = pd.NaT # Ensure column exists even if encounter_date is bad

    # Define numeric columns and their default fill values for missing/unconvertible entries
    numeric_cols_with_defaults = {
        'age': np.nan, 'pregnancy_status': 0, 'chronic_condition_flag': 0,
        'hrv_rmssd_ms': np.nan, 'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'movement_activity_level': 0, 'fall_detected_today': 0,
        'ambient_heat_index_c': np.nan, 'ppe_compliant_flag': 1, # Default to compliant if not specified
        'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': np.nan,
        'test_turnaround_days': np.nan, 'quantity_dispensed': 0,
        'item_stock_agg_zone': 0.0, 'consumption_rate_per_day': 0.001, # Small positive default
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, # Will be populated by AI engine
        'avg_spo2': np.nan, 'avg_daily_steps': np.nan, 'resting_heart_rate': np.nan,
        'avg_sleep_duration_hrs': np.nan, 'sleep_score_pct': np.nan, 'stress_level_score': np.nan,
        'hiv_viral_load_copies_ml': np.nan, 'chw_visit': 0, 'tb_contact_traced': 0,
        'patient_latitude': np.nan, 'patient_longitude': np.nan
    }

    # Define string columns that need standardization (e.g., 'Unknown' for missing)
    string_cols_to_standardize = [
        'encounter_id', 'patient_id', 'encounter_type', 'gender', 'zone_id', 'clinic_id', 'chw_id',
        'condition', 'patient_reported_symptoms', 'test_type', 'test_result', 'sample_status',
        'rejection_reason', 'referral_status', 'referral_reason', 'referred_to_facility_id',
        'referral_outcome', 'medication_adherence_self_report', 'item', 'notes',
        'diagnosis_code_icd10', 'physician_id', 'screening_hpv_status', 'key_chronic_conditions_summary'
    ]
    
    df = standardize_missing_values(df, string_cols_to_standardize, numeric_cols_with_defaults)
    
    logger.info(f"({source_context}) Health records processed. Shape: {df.shape}")
    return df

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_iot_clinic_environment_data(file_path: Optional[str] = None, source_context: str = "IoTDataLoader") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes IoT clinic environment data.
    """
    actual_file_path = file_path or settings.IOT_CLINIC_ENVIRONMENT_CSV_PATH
    df = _load_csv_data(actual_file_path, source_context)
    if df.empty:
        return pd.DataFrame()

    date_cols_to_convert = ['timestamp']
    df = convert_date_columns(df, date_cols_to_convert)

    numeric_cols_with_defaults = {
        'avg_co2_ppm': np.nan, 'max_co2_ppm': np.nan, 'avg_pm25': np.nan,
        'voc_index': np.nan, 'avg_temp_celsius': np.nan, 'avg_humidity_rh': np.nan,
        'avg_noise_db': np.nan, 'waiting_room_occupancy': np.nan,
        'patient_throughput_per_hour': np.nan, 'sanitizer_dispenses_per_hour': np.nan
    }
    string_cols_to_standardize = ['clinic_id', 'room_name', 'zone_id']
    
    df = standardize_missing_values(df, string_cols_to_standardize, numeric_cols_with_defaults)

    logger.info(f"({source_context}) IoT clinic environment data processed. Shape: {df.shape}")
    return df

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={pd.DataFrame: hash_dataframe_safe})
def load_zone_data(
    attributes_file_path: Optional[str] = None,
    geometries_file_path: Optional[str] = None,
    source_context: str = "ZoneDataLoader"
) -> pd.DataFrame:
    """
    Loads zone attributes from CSV and zone geometries from GeoJSON, then merges them.
    This version AVOIDS GeoPandas for loading geometries. The 'geometry' column
    will store the GeoJSON geometry object (dictionary) as a string or Python dict.
    Further processing (like area calculation) will need custom logic if geometries are complex.
    For simple polygons, area can be approximated or exact if using a geometry library later.
    """
    actual_attributes_path = attributes_file_path or settings.ZONE_ATTRIBUTES_CSV_PATH
    actual_geometries_path = geometries_file_path or settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH

    logger.info(f"({source_context}) Loading zone attributes from '{actual_attributes_path}' and geometries from '{actual_geometries_path}'")

    # Load attributes
    attributes_df = _load_csv_data(actual_attributes_path, f"{source_context}/Attributes")
    if attributes_df.empty:
        st.warning("Zone attributes data is missing or empty. Map visualizations and zonal analysis will be limited.")
        # Return a DataFrame with at least zone_id and geometry if only geometries load
        # or an entirely empty DF if neither loads.
        attributes_df = pd.DataFrame(columns=['zone_id'])


    # Load geometries from GeoJSON
    geometries_data = robust_json_load(actual_geometries_path)
    if not geometries_data or 'features' not in geometries_data or not isinstance(geometries_data['features'], list):
        logger.error(f"({source_context}) Invalid or missing GeoJSON features data from {actual_geometries_path}.")
        st.warning("Zone geometries data (GeoJSON) is invalid or missing. Map visualizations will not be available.")
        # If attributes loaded, return them, otherwise an empty DF
        if 'zone_id' not in attributes_df.columns and not attributes_df.empty:
            logger.warning(f"Attributes DF for zones is missing 'zone_id'. Cannot proceed with zone data loading.")
            return pd.DataFrame()
        attributes_df['geometry'] = None # Add empty geometry column
        attributes_df['crs'] = None
        return attributes_df

    geometries_list = []
    default_crs = geometries_data.get("crs", {}).get("properties", {}).get("name", settings.DEFAULT_CRS_STANDARD)

    for feature in geometries_data['features']:
        properties = feature.get("properties", {})
        zone_id = properties.get("zone_id")
        geometry = feature.get("geometry")
        if zone_id and geometry:
            geometries_list.append({
                "zone_id": str(zone_id).strip(),
                "geometry": json.dumps(geometry), # Store geometry as JSON string
                "name_geojson": str(properties.get("name", "")).strip(), # Name from GeoJSON properties
                "description_geojson": str(properties.get("description", "")).strip()
            })
        else:
            logger.warning(f"({source_context}) Feature in GeoJSON missing 'zone_id' or 'geometry'. Properties: {properties}")

    if not geometries_list:
        logger.warning(f"({source_context}) No valid features with zone_id and geometry found in GeoJSON: {actual_geometries_path}")
        st.warning("No valid zone geometries found in GeoJSON. Map visualizations will be impaired.")
        if 'zone_id' not in attributes_df.columns and not attributes_df.empty:
             return pd.DataFrame()
        attributes_df['geometry'] = None
        attributes_df['crs'] = default_crs
        return attributes_df

    geometries_df = pd.DataFrame(geometries_list)
    geometries_df['zone_id'] = geometries_df['zone_id'].astype(str)


    # Merge attributes and geometries
    if 'zone_id' not in attributes_df.columns:
        if not attributes_df.empty:
            logger.error(f"({source_context}) 'zone_id' column missing in attributes CSV. Cannot merge with geometries.")
            st.error("Zone attributes data is missing the 'zone_id' column. Map functionality affected.")
            # Return geometries_df with an empty 'geometry' column if attributes are unusable.
            geometries_df['geometry_obj'] = geometries_df['geometry'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            geometries_df['crs'] = default_crs
            return geometries_df # Or a completely empty DF if this path is problematic
        else: # Both are essentially empty or attributes_df was initialized as empty
             geometries_df['geometry_obj'] = geometries_df['geometry'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
             geometries_df['crs'] = default_crs
             return geometries_df


    attributes_df['zone_id'] = attributes_df['zone_id'].astype(str).str.strip()
    merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

    # Handle cases where 'name' might be in attributes or geojson, prefer attributes
    if 'name' not in merged_df.columns and 'name_geojson' in merged_df.columns:
        merged_df['name'] = merged_df['name_geojson']
    elif 'name' in merged_df.columns and 'name_geojson' in merged_df.columns:
        merged_df['name'] = merged_df['name'].fillna(merged_df['name_geojson'])

    # Ensure 'name' column exists, defaulting from zone_id if necessary
    if 'name' not in merged_df.columns or merged_df['name'].isnull().all():
        merged_df['name'] = "Zone " + merged_df['zone_id'].astype(str)
    merged_df['name'] = merged_df['name'].fillna("Zone " + merged_df['zone_id'].astype(str))


    # Standardize numeric and string columns from attributes part
    numeric_cols_with_defaults_zone = {
        'population': 0.0, 'socio_economic_index': np.nan, 'num_clinics': 0,
        'avg_travel_time_clinic_min': np.nan, 'area_sqkm': np.nan
    }
    string_cols_to_standardize_zone = [
        'predominant_hazard_type', 'typical_workforce_exposure_level',
        'primary_livelihood', 'water_source_main', 'description_geojson'
    ]
    merged_df = standardize_missing_values(merged_df, string_cols_to_standardize_zone, numeric_cols_with_defaults_zone)

    # Add CRS information as a column (it's not a GeoDataFrame property anymore)
    merged_df['crs'] = default_crs
    # Add geometry as actual dict objects for easier parsing later, if it's a string
    merged_df['geometry_obj'] = merged_df['geometry'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)


    # Log counts of zones from each source and merged
    logger.info(f"({source_context}) Zones from attributes: {len(attributes_df)}, Zones from geometries: {len(geometries_df)}, Merged zones: {len(merged_df)}")
    if len(merged_df[merged_df['geometry'].isnull()]) > 0:
        logger.warning(f"({source_context}) {len(merged_df[merged_df['geometry'].isnull()])} zones have no geometry data after merge.")
    if len(merged_df[merged_df['population'].isnull() & merged_df['name'].notnull()]) > 0: # check name is notnull for zones that should have attributes
        logger.warning(f"({source_context}) Some zones with names are missing population data after merge.")


    logger.info(f"({source_context}) Zone data loading and merging complete. Final shape: {merged_df.shape}")
    return merged_df


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_escalation_protocols(file_path: Optional[str] = None, source_context: str = "EscalationProtocolLoader") -> Dict[str, Any]:
    """Loads escalation protocols from a JSON file."""
    actual_file_path = file_path or settings.ESCALATION_PROTOCOLS_JSON_PATH
    logger.info(f"({source_context}) Loading escalation protocols from: {actual_file_path}")
    
    data = robust_json_load(actual_file_path)
    if data is None or not isinstance(data, dict):
        st.error("Escalation protocols could not be loaded or are invalid. System alerts may not function correctly.")
        return {"protocols": [], "contacts": {}, "message_templates": {}} # Return empty structure

    # Basic validation (can be expanded)
    if "protocols" not in data or not isinstance(data["protocols"], list):
        logger.error(f"({source_context}) 'protocols' key missing or not a list in escalation protocols file.")
        st.warning("Escalation protocols are malformed ('protocols' list missing).")
        data["protocols"] = [] # Ensure it exists as an empty list

    logger.info(f"({source_context}) Escalation protocols loaded successfully: {len(data.get('protocols',[]))} protocols found.")
    return data

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_pictogram_map(file_path: Optional[str] = None, source_context: str = "PictogramMapLoader") -> Dict[str, str]:
    """Loads pictogram mappings from a JSON file."""
    actual_file_path = file_path or settings.PICTOGRAM_MAP_JSON_PATH
    logger.info(f"({source_context}) Loading pictogram map from: {actual_file_path}")
    data = robust_json_load(actual_file_path)
    if data is None or not isinstance(data, dict):
        st.warning("Pictogram map could not be loaded. UI elements might be affected.")
        return {}
    logger.info(f"({source_context}) Pictogram map loaded with {len(data)} entries.")
    return data

@st.cache_data(ttl=settings.CACHE_TTL_SECONDS_WEB_REPORTS)
def load_haptic_patterns(file_path: Optional[str] = None, source_context: str = "HapticPatternLoader") -> Dict[str, List[int]]:
    """Loads haptic feedback patterns from a JSON file."""
    actual_file_path = file_path or settings.HAPTIC_PATTERNS_JSON_PATH
    logger.info(f"({source_context}) Loading haptic patterns from: {actual_file_path}")
    data = robust_json_load(actual_file_path)
    if data is None or not isinstance(data, dict):
        st.warning("Haptic patterns could not be loaded. Device feedback may be affected.")
        return {}
    logger.info(f"({source_context}) Haptic patterns loaded with {len(data)} entries.")
    return data
