import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

# --- Module Imports & Setup ---
try:
    from config import settings
    from .helpers import (
        clean_column_names,
        convert_to_numeric,
        robust_json_load,
        convert_date_columns,
        standardize_missing_values,
        hash_dataframe_safe
    )
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    # FIXED: Use the correct `__name__` magic variable.
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in loaders.py: {e}. Ensure config.py and helpers.py are accessible.")
    raise

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


def _load_csv_data(file_path_obj: Path, source_context: str, low_memory_setting: bool = False) -> pd.DataFrame:
    """
    Helper to load and minimally clean a CSV file. Expects an absolute Path object.
    Returns an empty DataFrame on any critical error.
    """
    abs_file_path = file_path_obj.resolve()
    logger.info(f"({source_context}) Attempting to load CSV data from: {abs_file_path}")
    if not abs_file_path.is_file():
        logger.error(f"({source_context}) CSV file NOT FOUND at: {abs_file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(abs_file_path, low_memory=low_memory_setting, on_bad_lines='warn')
        
        if df.empty:
            logger.warning(f"({source_context}) CSV file at {abs_file_path} loaded as empty by pandas.")
            return pd.DataFrame()

        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} raw records from '{abs_file_path.name}'.")
        return df
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing CSV from {abs_file_path}: {e}", exc_info=True)
        return pd.DataFrame()


def load_health_records(file_path_str: Optional[str] = None, source_context: str = "HealthRecordsLoader") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes health records data.
    Uses settings.HEALTH_RECORDS_CSV_PATH if file_path_str is None.
    """
    path_to_load_str = file_path_str or getattr(settings, 'HEALTH_RECORDS_CSV_PATH', None)
    if not path_to_load_str:
        logger.error(f"({source_context}) HEALTH_RECORDS_CSV_PATH not found in settings and no path provided.")
        return pd.DataFrame()

    actual_file_path = Path(path_to_load_str)
    if not actual_file_path.is_absolute():
        data_dir = getattr(settings, 'DATA_SOURCES_DIR', Path('.'))
        actual_file_path = (Path(data_dir) / actual_file_path).resolve()

    df = _load_csv_data(actual_file_path, source_context, low_memory_setting=True)
    if df.empty:
        return pd.DataFrame()

    date_cols = getattr(settings, 'HEALTH_RECORDS_DATE_COLS', ['encounter_date', 'date_of_birth'])
    df = convert_date_columns(df, date_cols)

    if 'encounter_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['encounter_date']):
        df['encounter_date_obj'] = df['encounter_date'].dt.date
    else:
        df['encounter_date_obj'] = pd.Series([None] * len(df), index=df.index, dtype=object)

    numeric_cols_config = getattr(settings, 'HEALTH_RECORDS_NUMERIC_COLS_DEFAULTS', {})
    string_cols_config = getattr(settings, 'HEALTH_RECORDS_STRING_COLS_DEFAULTS', {})
    df = standardize_missing_values(df, string_cols_config, numeric_cols_config)
    
    logger.info(f"({source_context}) Health records loaded and processed. Shape: {df.shape}")
    return df


def load_iot_clinic_environment_data(file_path_str: Optional[str] = None, source_context: str = "IoTDataLoader") -> pd.DataFrame:
    """Loads, cleans, and standardizes IoT clinic environment data."""
    path_to_load_str = file_path_str or getattr(settings, 'IOT_CLINIC_ENVIRONMENT_CSV_PATH', None)
    if not path_to_load_str:
        logger.error(f"({source_context}) IOT_CLINIC_ENVIRONMENT_CSV_PATH not in settings.")
        return pd.DataFrame()
    
    actual_file_path = Path(path_to_load_str)
    if not actual_file_path.is_absolute():
        data_dir = getattr(settings, 'DATA_SOURCES_DIR', Path('.'))
        actual_file_path = (Path(data_dir) / actual_file_path).resolve()

    df = _load_csv_data(actual_file_path, source_context)
    if df.empty: return pd.DataFrame()

    df = convert_date_columns(df, getattr(settings, 'IOT_DATE_COLS', ['timestamp']))
    numeric_cols_config = getattr(settings, 'IOT_NUMERIC_COLS_DEFAULTS', {})
    string_cols_config = getattr(settings, 'IOT_STRING_COLS_DEFAULTS', {})
    df = standardize_missing_values(df, string_cols_config, numeric_cols_config)

    logger.info(f"({source_context}) IoT data loaded and processed. Shape: {df.shape}")
    return df


def load_zone_data(
    attributes_file_path_str: Optional[str] = None,
    geometries_file_path_str: Optional[str] = None,
    source_context: str = "ZoneDataLoader"
) -> pd.DataFrame:
    """Loads and merges zone attribute data (CSV) and zone geometries (GeoJSON)."""
    attr_path_str = attributes_file_path_str or getattr(settings, 'ZONE_ATTRIBUTES_CSV_PATH', None)
    geom_path_str = geometries_file_path_str or getattr(settings, 'ZONE_GEOMETRIES_GEOJSON_FILE_PATH', None)

    if not attr_path_str or not geom_path_str:
        logger.error(f"({source_context}) Zone attributes or geometries path missing.")
        return pd.DataFrame()

    data_dir = Path(getattr(settings, 'DATA_SOURCES_DIR', '.'))
    attr_path = (data_dir / attr_path_str).resolve() if not Path(attr_path_str).is_absolute() else Path(attr_path_str)
    geom_path = (data_dir / geom_path_str).resolve() if not Path(geom_path_str).is_absolute() else Path(geom_path_str)

    attributes_df = _load_csv_data(attr_path, f"{source_context}/Attributes")
    if 'zone_id' in attributes_df:
        attributes_df['zone_id'] = attributes_df['zone_id'].astype(str).str.strip()

    geometries_data = robust_json_load(geom_path)
    geometries_list = []
    if geometries_data and isinstance(geometries_data.get("features"), list):
        for feature in geometries_data['features']:
            props = feature.get("properties", {})
            geom = feature.get("geometry")
            zid = props.get("zone_id", props.get("ZONE_ID", props.get("id")))
            if zid is not None and geom:
                geometries_list.append({
                    "zone_id": str(zid).strip(),
                    "geometry_obj": geom,
                    "name_geojson": str(props.get("name", "")).strip()
                })
    
    geometries_df = pd.DataFrame(geometries_list)
    if 'zone_id' in geometries_df:
        geometries_df['zone_id'] = geometries_df['zone_id'].astype(str).str.strip()

    if attributes_df.empty and geometries_df.empty:
        return pd.DataFrame()
    elif attributes_df.empty:
        merged_df = geometries_df
    elif geometries_df.empty:
        merged_df = attributes_df
    else:
        merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

    if 'name' not in merged_df or merged_df['name'].isnull().any():
        merged_df['name'] = merged_df['name'].fillna(merged_df.get('name_geojson', ''))
    if 'name' in merged_df:
        merged_df['name'] = merged_df['name'].fillna("Zone " + merged_df['zone_id'].astype(str))
    
    merged_df.drop(columns=['name_geojson'], errors='ignore', inplace=True)
    
    numeric_cols_config = getattr(settings, 'ZONE_NUMERIC_COLS_DEFAULTS', {})
    string_cols_config = getattr(settings, 'ZONE_STRING_COLS_DEFAULTS', {})
    merged_df = standardize_missing_values(merged_df, string_cols_config, numeric_cols_config)

    if 'geometry_obj' not in merged_df: merged_df['geometry_obj'] = None
    if 'zone_id' not in merged_df: merged_df['zone_id'] = [f"GEN_ZONE_{i}" for i in range(len(merged_df))]

    logger.info(f"({source_context}) Zone data loaded and merged. Final shape: {merged_df.shape}")
    return merged_df


def load_json_config_file(
    file_path_or_setting_attr: str,
    default_return_value: Union[Dict, List],
    source_context: str = "JSONConfigLoader"
) -> Union[Dict, List]:
    """Loads a JSON config file from a settings attribute or direct path."""
    path_str = getattr(settings, file_path_or_setting_attr, file_path_or_setting_attr)
    
    if not path_str or not isinstance(path_str, str):
        logger.error(f"({source_context}) Invalid path provided for '{file_path_or_setting_attr}'.")
        return default_return_value

    path = Path(path_str)
    if not path.is_absolute():
        root_dir = getattr(settings, 'PROJECT_ROOT_DIR', Path('.'))
        path = (Path(root_dir) / path).resolve()

    data = robust_json_load(path)
    if data is None or not isinstance(data, type(default_return_value)):
        logger.warning(f"({source_context}) Failed to load or validate JSON from '{path}'. Returning default.")
        return default_return_value

    logger.info(f"({source_context}) Successfully loaded JSON from '{path.name}'.")
    return data


# --- Wrappers for JSON Configs ---
def load_escalation_protocols(file_path: Optional[str] = None, source_context: str = "EscalationLoader") -> Dict[str, Any]:
    """Loads escalation protocols JSON."""
    default = {"protocols": [], "contacts": {}, "message_templates": {}}
    return load_json_config_file(file_path or 'ESCALATION_PROTOCOLS_JSON_PATH', default, source_context)

def load_pictogram_map(file_path: Optional[str] = None, source_context: str = "PictogramLoader") -> Dict[str, str]:
    """Loads pictogram map JSON."""
    # FIXED: The body of this function was not indented.
    path_or_attr = file_path or 'PICTOGRAM_MAP_JSON_PATH'
    data = load_json_config_file(path_or_attr, {}, source_context)
    if not isinstance(data, dict):
        logger.warning(f"({source_context}) Pictogram map from '{path_or_attr}' is not a dictionary.")
        return {}
    logger.info(f"({source_context}) Pictogram map processed: {len(data)} entries.")
    return data

def load_haptic_patterns(file_path: Optional[str] = None, source_context: str = "HapticLoader") -> Dict[str, List[int]]:
    """Loads haptic patterns JSON, validating contents."""
    path_or_attr = file_path or 'HAPTIC_PATTERNS_JSON_PATH'
    data = load_json_config_file(path_or_attr, {}, source_context)
    if not isinstance(data, dict):
        return {}
    valid_data = {k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(i, int) for i in v)}
    if len(valid_data) != len(data):
        logger.warning(f"({source_context}) Some invalid haptic patterns were filtered out.")
    logger.info(f"({source_context}) Haptic patterns processed: {len(valid_data)} valid entries.")
    return valid_data
