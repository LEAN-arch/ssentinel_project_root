# sentinel_project_root/data_processing/loaders.py
# SME PLATINUM STANDARD (V8 - FINAL INTEGRATED VERSION)
# This definitive version uses Pydantic models for robust configuration and
# automatically enriches the health records data upon loading to ensure a
# consistent, analytics-ready data contract across the application.

"""
Contains standardized data loading functions for the Sentinel application,
driven by a robust, model-validated configuration.
"""
import pandas as pd
import logging
from typing import Optional, Dict, List, Any, Type, Union
from pathlib import Path

# --- Platinum Standard Imports ---
from pydantic import BaseModel, Field

# --- Sentinel System Imports ---
try:
    from config import settings
    # Use the fluent DataPipeline from helpers
    from .helpers import DataPipeline, robust_json_load
    # <<< SME INTEGRATION >>> Import the enrichment function to be called automatically.
    from .enrichment import enrich_health_records_for_kpis
except ImportError as e:
    # Critical failure path if core dependencies are missing
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in loaders.py: {e}. Check dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Pydantic Models for Type-Safe, Validated Configuration ---
class CsvDataSourceConfig(BaseModel):
    """Defines the schema for loading and processing a CSV data source."""
    setting_attr: str
    date_cols: List[str] = Field(default_factory=list)
    dtype_map: Dict[str, Type] = Field(default_factory=dict)
    rename_map: Dict[str, str] = Field(default_factory=dict)
    required_cols: List[str] = Field(default_factory=list)
    read_csv_options: Dict[str, Any] = Field(default_factory=dict)

class GeoJsonDataSourceConfig(BaseModel):
    """Defines the schema for a GeoJSON data source."""
    setting_attr: str

# --- Main DataLoader Class ---
class DataLoader:
    """
    A declarative, configuration-driven class for loading and processing data sources.
    It uses model-validated configurations to ensure robust and consistent data loading.
    """
    _DATA_CONFIG: Dict[str, Union[CsvDataSourceConfig, GeoJsonDataSourceConfig]] = {
        'health_records': CsvDataSourceConfig(
            setting_attr='HEALTH_RECORDS_PATH',
            date_cols=['encounter_date', 'sample_collection_date', 'test_result_date'],
            dtype_map={'patient_id': str, 'chw_id': str, 'clinic_id': str, 'zone_id': str},
            required_cols=['patient_id', 'encounter_date', 'age', 'gender', 'test_type', 'diagnosis'],
            read_csv_options={'low_memory': False}
        ),
        'iot_environment': CsvDataSourceConfig(
            setting_attr='IOT_ENV_RECORDS_PATH',
            date_cols=['timestamp'],
            dtype_map={'room_id': str, 'sensor_id': str, 'zone_id': str},
            required_cols=['timestamp', 'room_name', 'avg_co2_ppm', 'avg_temp_celsius'],
        ),
        'zone_attributes': CsvDataSourceConfig(
            setting_attr='ZONE_ATTRIBUTES_PATH',
            rename_map={'name': 'zone_name', 'id': 'zone_id'}, # Robust renaming
            dtype_map={'zone_id': str},
            required_cols=['zone_id', 'zone_name', 'population'],
        ),
        'zone_geometries': GeoJsonDataSourceConfig(
            setting_attr='ZONE_GEOMETRIES_PATH'
        )
    }

    def _resolve_path(self, explicit_path: Optional[str], setting_attr: str) -> Optional[Path]:
        """Resolves a file path, prioritizing an explicit path over a settings attribute."""
        path_str = explicit_path or getattr(settings, setting_attr, None)
        if not path_str:
            logger.error(f"Path not found: No explicit path given and setting '{setting_attr}' is missing.")
            return None
        
        path_obj = Path(path_str)
        if not path_obj.is_absolute():
            base_dir = getattr(settings, 'DATA_SOURCES_DIR', None)
            if not base_dir:
                logger.error(f"DATA_SOURCES_DIR not in settings; cannot resolve relative path: {path_obj}")
                return None
            path_obj = Path(base_dir) / path_obj
        
        return path_obj.resolve()

    def _validate_schema(self, df: pd.DataFrame, required_cols: List[str], config_key: str) -> bool:
        """Checks if all required columns are present in the DataFrame."""
        if not required_cols: return True
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.critical(f"({config_key}) Schema Validation FAILED! Required columns missing: {sorted(list(missing))}.")
            return False
        return True

    def load_csv(self, config_key: str, file_path_str: Optional[str] = None) -> pd.DataFrame:
        """A generic method to load, clean, standardize, and validate a CSV based on its configuration."""
        config = self._DATA_CONFIG.get(config_key)
        if not isinstance(config, CsvDataSourceConfig):
            logger.error(f"Invalid or missing CSV configuration for key: '{config_key}'")
            return pd.DataFrame()

        file_path = self._resolve_path(file_path_str, config.setting_attr)
        if not file_path or not file_path.is_file():
            logger.error(f"({config_key}) CSV file NOT FOUND at: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path, **config.read_csv_options)
            if df.empty:
                logger.warning(f"({config_key}) CSV at {file_path} loaded as empty.")
                return pd.DataFrame()

            # Use the fluent DataPipeline for cleaning and date conversion
            pipeline = DataPipeline(df).clean_column_names()
            df_processed = pipeline.to_df().rename(columns=config.rename_map)

            # Resiliently apply data types
            for col, dtype in config.dtype_map.items():
                if col in df_processed.columns:
                    try:
                        df_processed[col] = df_processed[col].astype(dtype)
                    except (ValueError, TypeError):
                        logger.warning(f"({config_key}) Could not cast column '{col}' to {dtype}. Leaving as is.")

            if not self._validate_schema(df_processed, config.required_cols, config_key):
                return pd.DataFrame()

            df_final = DataPipeline(df_processed).convert_date_columns(config.date_cols).to_df()
            logger.info(f"({config_key}) Successfully loaded and processed {len(df_final)} records from '{file_path.name}'.")
            return df_final
        except Exception as e:
            logger.critical(f"({config_key}) CRITICAL ERROR loading CSV from {file_path}: {e}", exc_info=True)
            return pd.DataFrame()

# --- Singleton Instance and Public API ---
_loader = DataLoader()

def load_health_records(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads, cleans, standardizes, and enriches the primary health records data.
    This function ensures that any consumer receives a fully analytics-ready DataFrame.
    """
    df = _loader.load_csv('health_records', file_path)
    # <<< SME INTEGRATION >>> Automatically apply KPI enrichment to fulfill the data contract.
    if not df.empty:
        logger.info(f"Enriching {len(df)} health records for KPI calculations.")
        df = enrich_health_records_for_kpis(df)
    return df

def load_iot_clinic_environment_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes IoT clinic environment data."""
    return _loader.load_csv('iot_environment', file_path)

def load_zone_data(attributes_file_path: Optional[str] = None, geometries_file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads and merges zone attribute data (CSV) and zone geometries (GeoJSON)."""
    # This is a more complex operation not fitting the simple CSV loader, so it's kept separate.
    attributes_df = _loader.load_csv('zone_attributes', attributes_file_path)
    
    geom_config = _loader._DATA_CONFIG.get('zone_geometries')
    geom_path = _loader._resolve_path(geometries_file_path, geom_config.setting_attr if geom_config else '')

    geometries_df = pd.DataFrame()
    if geom_path and geom_path.is_file():
        geo_data = robust_json_load(geom_path)
        if isinstance(geo_data, dict) and 'features' in geo_data:
            geometries_list = [
                {"zone_id": str(feat["properties"].get("zone_id", feat["properties"].get("id"))).strip(), "geometry_obj": feat["geometry"]}
                for feat in geo_data.get("features", [])
                if feat.get("properties") and feat.get("geometry") and feat["properties"].get("zone_id", feat["properties"].get("id")) is not None
            ]
            if geometries_list:
                geometries_df = pd.DataFrame(geometries_list)

    if attributes_df.empty and geometries_df.empty: return pd.DataFrame()
    if attributes_df.empty: return geometries_df
    if geometries_df.empty: return attributes_df

    return pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")

def load_json_config(path_or_setting: str, default: Any = None) -> Any:
    """Loads a JSON config file from a path or a settings attribute name."""
    file_path = _loader._resolve_path(None, path_or_setting) or _loader._resolve_path(path_or_setting, '')
    if not file_path or not file_path.is_file(): return default
    
    data = robust_json_load(file_path)
    return data if data is not None else default
