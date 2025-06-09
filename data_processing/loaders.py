# sentinel_project_root/data_processing/loaders.py
# SME PLATINUM STANDARD - ROBUST & INTEGRATED DATA LOADING

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from config import settings
from .enrichment import enrich_health_records_with_kpis
from .helpers import DataPipeline, robust_json_load

logger = logging.getLogger(__name__)

# --- Pydantic Models for Type-Safe Configuration ---
class CsvConfig(BaseModel):
    path_setting: str
    date_cols: List[str] = Field(default_factory=list)
    dtype_map: Dict[str, str] = Field(default_factory=dict)
    rename_map: Dict[str, str] = Field(default_factory=dict)
    required_cols: List[str] = Field(default_factory=list)
    read_options: Dict[str, Any] = Field(default_factory=lambda: {'low_memory': False})

class JsonConfig(BaseModel):
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
        rename_map={'name': 'zone_name', 'id': 'zone_id'},
        dtype_map={'zone_id': 'str'},
        required_cols=['zone_id', 'population']
    ),
    'zone_geometries': JsonConfig(path_setting='ZONE_GEOMETRIES_PATH'),
    'escalation_protocols': JsonConfig(path_setting='ESCALATION_PROTOCOLS_PATH'),
    'pictogram_map': JsonConfig(path_setting='PICTOGRAM_MAP_PATH'),
    'haptic_patterns': JsonConfig(path_setting='HAPTIC_PATTERNS_PATH'),
}

# --- Main Loading Functions ---
def _load_and_process_csv(config_key: str, filepath_override: Optional[str] = None) -> pd.DataFrame:
    config = DATA_CONFIG.get(config_key)
    if not isinstance(config, CsvConfig):
        logger.error(f"Invalid CSV config key: '{config_key}'")
        return pd.DataFrame()

    path_to_load = Path(filepath_override or getattr(settings, config.path_setting))
    
    if not path_to_load.is_file():
        logger.error(f"({config_key}) CSV file not found at: {path_to_load}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path_to_load, **config.read_options)
        
        processed_df = (DataPipeline(df)
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
    df = _load_and_process_csv('health_records', filepath_override)
    if not df.empty:
        logger.info(f"Applying KPI enrichment to {len(df)} health records...")
        return enrich_health_records_with_kpis(df)
    return df

def load_iot_records(filepath_override: Optional[str] = None) -> pd.DataFrame:
    return _load_and_process_csv('iot_records', filepath_override)

def load_zone_data(
    attributes_path: Optional[str] = None,
    geometries_path: Optional[str] = None
) -> pd.DataFrame:
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

    if attributes_df.empty: return geometries_df.reindex(columns=['zone_id', 'geometry'])
    if geometries_df.empty: return attributes_df
    
    return pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

def load_json_asset(config_key: str, filepath_override: Optional[str] = None) -> Optional[Union[Dict, List]]:
    config = DATA_CONFIG.get(config_key)
    if not isinstance(config, JsonConfig):
        logger.error(f"Invalid JSON config key: '{config_key}'")
        return None
        
    path_to_load = Path(filepath_override or getattr(settings, config.path_setting))
    
    if not path_to_load.is_file():
        logger.error(f"({config_key}) JSON file not found at: {path_to_load}")
        return None

    return robust_json_load(path_to_load)
