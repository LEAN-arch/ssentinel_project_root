# sentinel_project_root/data_processing/loaders.py
# SME-EVALUATED AND REVISED VERSION (GOLD STANDARD V2)
# This definitive version corrects a function duplication bug and improves clarity
# in the JSON loading logic for maximum maintainability.

"""
Contains standardized data loading functions for the Sentinel application.
This module is responsible for reading data and performing initial, universal cleaning.
"""
import pandas as pd
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

try:
    from config import settings
    from .helpers import data_cleaner, convert_date_columns, robust_json_load
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in loaders.py: {e}. Check dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A declarative, configuration-driven class for loading and processing data sources.

    This class centralizes loading logic, making it consistent and easy to maintain.
    It follows a strict "timezone-naive" policy for all datetime columns to prevent
    downstream errors. It also enforces data types at read-time for robustness.
    """
    _DATA_CONFIG = {
        'health_records': {
            'setting_attr': 'HEALTH_RECORDS_PATH',
            'date_cols': ['encounter_date', 'sample_collection_date', 'referral_outcome_date'],
            'dtype_map': {
                'patient_id': str, 'chw_id': str, 'clinic_id': str,
                'physician_id': str, 'diagnosis_code_icd10': str
            },
            'read_csv_options': {'engine': 'c', 'low_memory': False}
        },
        'iot_environment': {
            'setting_attr': 'IOT_ENV_RECORDS_PATH',
            'date_cols': ['timestamp'],
            'dtype_map': {'room_id': str, 'sensor_id': str},
            'read_csv_options': {}
        },
        'zone_attributes': {
            'setting_attr': 'ZONE_ATTRIBUTES_PATH',
            'date_cols': [],
            'dtype_map': {'zone_id': str},
            'read_csv_options': {}
        }
    }

    def _resolve_path(self, explicit_path: Optional[str], setting_attr: str) -> Optional[Path]:
        """Resolves a file path, prioritizing the explicit path over the setting."""
        path_str = explicit_path or getattr(settings, setting_attr, None)
        if not path_str:
            if setting_attr: # Only log error if a setting attribute was expected
                logger.error(f"Path not found: No explicit path provided and setting '{setting_attr}' is not defined.")
            return None

        path_obj = Path(path_str)
        if path_obj.is_absolute():
            return path_obj.resolve()

        base_dir = getattr(settings, 'DATA_SOURCES_DIR', getattr(settings, 'DATA_DIR', None))
        if not base_dir:
            logger.error("DATA_SOURCES_DIR or DATA_DIR not defined in settings; cannot resolve relative path.")
            return None
        return (Path(base_dir) / path_obj).resolve()

    def _load_and_process_csv(self, config_key: str, file_path_str: Optional[str] = None) -> pd.DataFrame:
        """A generic method to load, clean, and standardize a CSV based on its configuration."""
        config = self._DATA_CONFIG.get(config_key)
        if not config:
            logger.error(f"No configuration found for data key: '{config_key}'")
            return pd.DataFrame()

        file_path = self._resolve_path(file_path_str, config['setting_attr'])
        if not file_path or not file_path.is_file():
            logger.error(f"({config_key}) CSV file NOT FOUND at resolved path: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path, dtype=config.get('dtype_map'), **config.get('read_csv_options', {}))

            if df.empty:
                logger.warning(f"({config_key}) CSV at {file_path} loaded as empty.")
                return pd.DataFrame()

            df = data_cleaner.clean_column_names(df)
            df = convert_date_columns(df, config.get('date_cols', []))

            logger.info(f"({config_key}) Successfully loaded and processed {len(df)} records from '{file_path.name}'.")
            return df
        except Exception as e:
            logger.error(f"({config_key}) Error loading/processing CSV from {file_path}: {e}", exc_info=True)
            return pd.DataFrame()

    def load_zone_data(self, attributes_path: Optional[str], geometries_path: Optional[str]) -> pd.DataFrame:
        """Loads and merges zone attributes (CSV) and geometries (GeoJSON)."""
        attributes_df = self._load_and_process_csv('zone_attributes', attributes_path)

        geometries_list = []
        geom_path = self._resolve_path(geometries_path, 'ZONE_GEOMETRIES_PATH')
        if geom_path and geom_path.is_file():
            geo_data = robust_json_load(geom_path)
            if isinstance(geo_data, dict) and 'features' in geo_data:
                for feature in geo_data.get('features', []):
                    props = feature.get("properties", {})
                    zid = props.get("zone_id", props.get("ZONE_ID"))
                    if zid is not None and feature.get("geometry"):
                        geometries_list.append({"zone_id": str(zid).strip(), "geometry_obj": feature["geometry"]})
        geometries_df = pd.DataFrame(geometries_list)

        if attributes_df.empty and geometries_df.empty:
            return pd.DataFrame()
        if attributes_df.empty:
            return geometries_df
        if geometries_df.empty:
            return attributes_df

        merged_df = pd.merge(attributes_df, geometries_df, on="zone_id", how="outer")
        logger.info(f"(ZoneData) Loaded and merged. Final shape: {merged_df.shape}")
        return merged_df

# --- Singleton Instance and Public API ---
_data_loader = DataLoader()

def load_health_records(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes the primary health records data."""
    return _data_loader._load_and_process_csv('health_records', file_path)

def load_iot_clinic_environment_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes IoT clinic environment data."""
    return _data_loader._load_and_process_csv('iot_environment', file_path)

def load_zone_data(attributes_file_path: Optional[str] = None, geometries_file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads and merges zone attribute data (CSV) and zone geometries (GeoJSON)."""
    return _data_loader.load_zone_data(attributes_file_path, geometries_file_path)

def load_json_config(path_or_setting: str, default: Any = None) -> Any:
    """
    Loads a JSON config file from a path or setting attribute.

    It first attempts to resolve `path_or_setting` as an attribute name in the
    settings file. If that fails, it treats `path_or_setting` as an explicit file path.
    """
    # <<< SME REVISION >>> Added comments to clarify the "clever" but complex logic.
    # First, try to resolve path_or_setting as a setting attribute (e.g., "MY_CONFIG_PATH").
    file_path = _data_loader._resolve_path(None, path_or_setting)
    # If that fails (returns None or path doesn't exist), treat path_or_setting as an explicit path string.
    if not file_path or not file_path.exists():
        file_path = _data_loader._resolve_path(path_or_setting, "") # "" setting_attr ensures it's not used.

    if not file_path or not file_path.is_file():
        return default

    data = robust_json_load(file_path)
    # This check ensures that if we loaded something, its type matches the default's type.
    # e.g., if default is a dict, we expect the loaded data to be a dict.
    return data if data is not None and (default is None or isinstance(data, type(default))) else default

# --- Wrapper functions for specific JSON configs ---
def load_escalation_protocols() -> Dict[str, Any]:
    """Wrapper to load escalation protocols JSON."""
    return load_json_config('ESCALATION_PROTOCOLS_JSON_PATH', default={})

def load_pictogram_map() -> Dict[str, str]:
    """Wrapper to load pictogram map JSON."""
    return load_json_config('PICTOGRAM_MAP_JSON_PATH', default={})

# <<< SME REVISION >>> Corrected duplicated function and refined logic.
def load_haptic_patterns() -> Dict[str, List[int]]:
    """Wrapper to load and validate haptic patterns JSON."""
    data = load_json_config('HAPTIC_PATTERNS_JSON_PATH', default={})
    if not isinstance(data, dict):
        logger.warning("Haptic patterns data is not a dictionary. Returning empty.")
        return {}
    # Validate the structure of the haptic patterns data, keeping only valid entries.
    valid_patterns = {
        key: value for key, value in data.items()
        if isinstance(value, list) and all(isinstance(i, int) for i in value)
    }
    if len(valid_patterns) != len(data):
        logger.warning("Some haptic patterns were invalid and have been filtered out.")
    return valid_patterns

# <<< SME REVISION >>> Removed the duplicated function definition that was here.
