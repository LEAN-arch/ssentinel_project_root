# ssentinel_project_root/data_processing/loaders.py
"""
Contains standardized data loading functions for the Sentinel application.
This module is responsible for reading data from sources and performing initial cleaning.
It has no dependency on any UI framework.
"""
import pandas as pd
import logging
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

# --- Module Imports & Setup ---
try:
    from config import settings
    from .helpers import (
        clean_column_names,
        convert_date_columns,
        standardize_missing_values,
        robust_json_load
    )
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in loaders.py: {e}. Ensure config and helpers are accessible.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Internal Helper Functions ---

def _resolve_data_path(file_path_str: Optional[str], setting_attr_name: str) -> Optional[Path]:
    """Resolves a file path from a string, using a settings attribute as a fallback."""
    path_str = file_path_str or getattr(settings, setting_attr_name, None)
    if not path_str:
        logger.error(f"Path not found: No path string provided and '{setting_attr_name}' is not defined in settings.")
        return None

    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        data_dir = getattr(settings, 'DATA_SOURCES_DIR', None)
        if not data_dir:
            logger.error("DATA_SOURCES_DIR is not defined in settings, cannot resolve relative path.")
            return None
        return (Path(data_dir) / path_obj).resolve()
    return path_obj.resolve()


def _load_csv_data(file_path: Path, source_context: str, low_memory: bool = False) -> pd.DataFrame:
    """
    Helper to load and minimally clean a CSV file from a resolved Path object.
    Returns an empty DataFrame on any critical error.
    """
    logger.info(f"({source_context}) Attempting to load CSV data from: {file_path}")
    if not file_path.is_file():
        logger.error(f"({source_context}) CSV file NOT FOUND at: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, low_memory=low_memory, on_bad_lines='warn')
        if df.empty:
            logger.warning(f"({source_context}) CSV file at {file_path} loaded as empty.")
            return pd.DataFrame()

        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} records from '{file_path.name}'.")
        return df
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing CSV from {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

# --- Public Data Loading Functions ---

def load_health_records(file_path_str: Optional[str] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes the primary health records data."""
    source_context = "HealthRecordsLoader"
    file_path = _resolve_data_path(file_path_str, 'HEALTH_RECORDS_CSV_PATH')
    if not file_path:
        return pd.DataFrame()

    df = _load_csv_data(file_path, source_context, low_memory=True)
    if df.empty:
        return df

    date_cols = ['encounter_date', 'date_of_birth']
    df = convert_date_columns(df, date_cols)
    
    # These configs should be defined in settings.py for full robustness
    numeric_cols_config = getattr(settings, 'HEALTH_RECORDS_NUMERIC_COLS_DEFAULTS', {})
    string_cols_config = getattr(settings, 'HEALTH_RECORDS_STRING_COLS_DEFAULTS', {})
    df = standardize_missing_values(df, string_cols_config, numeric_cols_config)
    
    logger.info(f"({source_context}) Health records loaded and processed. Final shape: {df.shape}")
    return df


def load_iot_clinic_environment_data(file_path_str: Optional[str] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes IoT clinic environment data."""
    source_context = "IoTDataLoader"
    file_path = _resolve_data_path(file_path_str, 'IOT_CLINIC_ENVIRONMENT_CSV_PATH')
    if not file_path:
        return pd.DataFrame()

    df = _load_csv_data(file_path, source_context)
    if df.empty:
        return df

    df = convert_date_columns(df, ['timestamp'])
    numeric_cols_config = getattr(settings, 'IOT_NUMERIC_COLS_DEFAULTS', {})
    string_cols_config = getattr(settings, 'IOT_STRING_COLS_DEFAULTS', {})
    df = standardize_missing_values(df, string_cols_config, numeric_cols_config)

    logger.info(f"({source_context}) IoT data loaded and processed. Final shape: {df.shape}")
    return df


def load_zone_data(
    attributes_file_path_str: Optional[str] = None,
    geometries_file_path_str: Optional[str] = None
) -> pd.DataFrame:
    """Loads and merges zone attribute data (CSV) and zone geometries (GeoJSON)."""
    source_context = "ZoneDataLoader"
    attr_path = _resolve_data_path(attributes_file_path_str, 'ZONE_ATTRIBUTES_CSV_PATH')
    geom_path = _resolve_data_path(geometries_file_path_str, 'ZONE_GEOMETRIES_GEOJSON_FILE_PATH')

    attributes_df = _load_csv_data(attr_path, f"{source_context}/Attributes") if attr_path else pd.DataFrame()
    
    geometries_list = []
    if geom_path:
        geometries_data = robust_json_load(geom_path)
        if isinstance(geometries_data, dict) and isinstance(geometries_data.get("features"), list):
            for feature in geometries_data['features']:
                props = feature.get("properties", {})
                zid = props.get("zone_id", props.get("ZONE_ID"))
                if zid is not None and feature.get("geometry"):
                    geometries_list.append({
                        "zone_id": str(zid).strip(),
                        "geometry_obj": feature["geometry"],
                        "name_geojson": str(props.get("name", "")).strip()
                    })
    geometries_df = pd.DataFrame(geometries_list)

    if 'zone_id' in attributes_df:
        attributes_df['zone_id'] = attributes_df['zone_id'].astype(str)
    if 'zone_id' in geometries_df:
        geometries_df['zone_id'] = geometries_df['zone_id'].astype(str)

    if attributes_df.empty and geometries_df.empty:
        return pd.DataFrame()
    elif attributes_df.empty:
        return geometries_df.rename(columns={'name_geojson': 'name'})
    elif geometries_df.empty:
        return attributes_df
    
    merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")
    if 'name' in merged_df and 'name_geojson' in merged_df:
        merged_df['name'] = merged_df['name'].fillna(merged_df['name_geojson'])
    
    merged_df.drop(columns=['name_geojson'], errors='ignore', inplace=True)
    
    logger.info(f"({source_context}) Zone data loaded and merged. Final shape: {merged_df.shape}")
    return merged_df


def load_json_config_file(
    path_or_setting_attr: str,
    default_return_value: Union[Dict, List]
) -> Union[Dict, List]:
    """Loads a JSON config file identified by a setting attribute or direct path."""
    source_context = "JSONConfigLoader"
    path_obj = _resolve_data_path(None, path_or_setting_attr) or _resolve_data_path(path_or_setting_attr, '')
    if not path_obj:
        return default_return_value

    data = robust_json_load(path_obj)
    if not isinstance(data, type(default_return_value)):
        logger.warning(f"({source_context}) JSON from '{path_obj.name}' is not of expected type. Returning default.")
        return default_return_value

    logger.info(f"({source_context}) Successfully loaded JSON from '{path_obj.name}'.")
    return data

def load_escalation_protocols() -> Dict[str, Any]:
    """Wrapper to load escalation protocols JSON."""
    return load_json_config_file('ESCALATION_PROTOCOLS_JSON_PATH', {})

def load_pictogram_map() -> Dict[str, str]:
    """Wrapper to load pictogram map JSON."""
    return load_json_config_file('PICTOGRAM_MAP_JSON_PATH', {})

def load_haptic_patterns() -> Dict[str, List[int]]:
    """Wrapper to load and validate haptic patterns JSON."""
    data = load_json_config_file('HAPTIC_PATTERNS_JSON_PATH', {})
    if not isinstance(data, dict): return {}
    valid_data = {k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(i, int) for i in v)}
    if len(valid_data) != len(data):
        logger.warning("Some invalid haptic patterns were filtered out during loading.")
    return valid_data
