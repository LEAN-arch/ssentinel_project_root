# sentinel_project_root/data_processing/loaders.py
# SME PLATINUM STANDARD - ROBUST & INTEGRATED DATA LOADING

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, FilePath, HttpUrl

from config import settings
from .enrichment import enrich_health_records_with_kpis
from .helpers import DataPipeline, robust_json_load

logger = logging.getLogger(__name__)

# --- Pydantic Models for Type-Safe Configuration ---

class CsvConfig(BaseModel):
    """Defines the schema for loading and processing a CSV data source."""
    path_setting: str
    date_cols: List[str] = Field(default_factory=list)
    dtype_map: Dict[str, str] = Field(default_factory=dict)
    rename_map: Dict[str, str] = Field(default_factory=dict)
    required_cols: List[str] = Field(default_factory=list)
    read_options: Dict[str, Any] = Field(default_factory=lambda: {'low_memory': False})

class JsonConfig(BaseModel):
    """Defines the schema for a JSON or GeoJSON data source."""
    path_setting: str

# --- Centralized Data Source Configuration ---

DATA_CONFIG: Dict[str, Union[CsvConfig, JsonConfig]] = {
    'health_records': CsvConfig(
        path_setting='HEALTH_RECORDS_PATH',
        date_cols=['encounter_date', 'sample_collection_date', 'test_result_date'],
        dtype_map={'patient_id': 'str', 'chw_id': 'str', 'clinic_id': 'str', 'zone_id': 'str'},
        required_cols=['patient_id', 'encounter_date', 'diagnosis', 'test_type']
    ),
    'iot_records': CsvConfig(
        path_setting='IOT_RECORDS_PATH',
        date_cols=['timestamp'],
        dtype_map={'room_id': 'str', 'sensor_id': 'str', 'zone_id': 'str'},
        required_cols=['timestamp', 'room_name', 'avg_co2_ppm']
    ),
    'zone_attributes': CsvConfig(
        path_setting='ZONE_ATTRIBUTES_PATH',
        dtype_map={'zone_id': 'str'},
        required_cols=['zone_id', 'population']
    ),
    'zone_geometries': JsonConfig(path_setting='ZONE_GEOMETRIES_PATH'),
    'escalation_protocols': JsonConfig(path_setting='ESCALATION_PROTOCOLS_PATH'),
    'pictogram_map': JsonConfig(path_setting='PICTOGRAM_MAP_PATH'),
    'haptic_patterns': JsonConfig(path_setting='HAPTIC_PATTERNS_PATH'),
}

# --- Main Loading Functions ---

def _resolve_path(path_or_setting: Union[str, Path]) -> Optional[Path]:
    """Resolves a string to a Path object, checking settings if it's not a file."""
    if isinstance(path_or_setting, Path) and path_or_setting.is_file():
        return path_or_setting
    
    path_str = str(path_or_setting)
    # Check if it's a direct file path
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
        
    # Assume it's a setting attribute
    config_path = getattr(settings, path_str, None) or getattr(settings, path_str.upper(), None)
    if config_path:
        return Path(config_path).resolve()
        
    logger.error(f"Could not resolve path for '{path_str}'. Not a file or valid setting.")
    return None

def _load_and_process_csv(config_key: str, filepath_override: Optional[str] = None) -> pd.DataFrame:
    """Generic CSV loader using the centralized configuration."""
    config = DATA_CONFIG.get(config_key)
    if not isinstance(config, CsvConfig):
        logger.error(f"Invalid CSV config key: '{config_key}'")
        return pd.DataFrame()

    path_to_load = _resolve_path(filepath_override) if filepath_override else getattr(settings, config.path_setting)
    
    if not path_to_load or not Path(path_to_load).is_file():
        logger.error(f"({config_key}) CSV file not found at: {path_to_load}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path_to_load, **config.read_options)
        
        pipeline = DataPipeline(df)
        processed_df = (pipeline
            .clean_column_names()
            .rename_columns(config.rename_map)
            .cast_column_types(config.dtype_map)
            .convert_date_columns(config.date_cols)
            .get_dataframe()
        )

        missing_cols = set(config.required_cols) - set(processed_df.columns)
        if missing_cols:
            logger.critical(f"({config_key}) Schema validation failed! Missing required columns: {missing_cols}")
            return pd.DataFrame()

        logger.info(f"({config_key}) Successfully loaded and processed {len(processed_df)} records.")
        return processed_df
    except Exception as e:
        logger.critical(f"({config_key}) Critical error loading CSV from {path_to_load}: {e}", exc_info=True)
        return pd.DataFrame()


def load_health_records(filepath_override: Optional[str] = None) -> pd.DataFrame:
    """
    Loads, cleans, and enriches the primary health records data.

    This function is the definitive source for health data. It returns a
    fully analytics-ready DataFrame, fulfilling the system's data contract.
    """
    df = _load_and_process_csv('health_records', filepath_override)
    if not df.empty:
        logger.info(f"Applying KPI enrichment to {len(df)} health records...")
        return enrich_health_records_with_kpis(df)
    return df


def load_iot_records(filepath_override: Optional[str] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes IoT clinic environment data."""
    return _load_and_process_csv('iot_records', filepath_override)


def load_zone_data(
    attributes_path: Optional[str] = None,
    geometries_path: Optional[str] = None
) -> pd.DataFrame:
    """Loads and merges zone attributes (CSV) with geometries (GeoJSON)."""
    attributes_df = _load_and_process_csv('zone_attributes', attributes_path)
    
    geo_data = load_json_asset('zone_geometries', geometries_path)
    if not geo_data or 'features' not in geo_data:
        return attributes_df

    geo_list = [
        {
            "zone_id": str(feat["properties"].get("zone_id")),
            "geometry": feat.get("geometry")
        }
        for feat in geo_data.get("features", [])
        if feat.get("properties", {}).get("zone_id") and feat.get("geometry")
    ]
    geometries_df = pd.DataFrame(geo_list)

    if attributes_df.empty: return geometries_df
    if geometries_df.empty: return attributes_df
    
    return pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

def load_json_asset(config_key: str, filepath_override: Optional[str] = None) -> Optional[Union[Dict, List]]:
    """Loads a JSON config/asset file from a path or settings attribute."""
    config = DATA_CONFIG.get(config_key)
    if not isinstance(config, JsonConfig):
        logger.error(f"Invalid JSON config key: '{config_key}'")
        return None
        
    path_to_load = _resolve_path(filepath_override) if filepath_override else getattr(settings, config.path_setting)
    
    if not path_to_load or not Path(path_to_load).is_file():
        logger.error(f"({config_key}) JSON file not found at: {path_to_load}")
        return None

    return robust_json_load(path_to_load)
