# sentinel_project_root/data_processing/loaders.py
# SME-EVALUATED AND REVISED VERSION (GOLD STANDARD V6 - RESILIENT SCHEMA)
# This version introduces resilient loading by standardizing column names
# on-the-fly, permanently resolving schema mismatch errors.

"""
Contains standardized data loading functions for the Sentinel application.
This module is responsible for reading data and performing initial, universal cleaning
and schema validation.
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
    It follows a strict "timezone-naive" policy for all datetime columns and enforces
    data types and the presence of required columns for robustness.
    """
    _DATA_CONFIG = {
        'health_records': {
            'setting_attr': 'HEALTH_RECORDS_PATH',
            'date_cols': ['encounter_date'],
            'dtype_map': {
                'patient_id': str, 'chw_id': str, 'clinic_id': str,
                'physician_id': str, 'diagnosis_code_icd10': str
            },
            'required_cols': [
                'patient_id', 'encounter_date', 'age', 'gender', 'test_type',
                'test_result', 'diagnosis', 'ai_risk_score'
            ],
            'read_csv_options': {'engine': 'c', 'low_memory': False}
        },
        'iot_environment': {
            'setting_attr': 'IOT_ENV_RECORDS_PATH',
            'date_cols': ['timestamp'],
            'dtype_map': {'room_id': str, 'sensor_id': str},
            'required_cols': ['timestamp', 'room_name', 'avg_co2_ppm', 'avg_temp_celsius'],
            'read_csv_options': {}
        },
        'zone_attributes': {
            'setting_attr': 'ZONE_ATTRIBUTES_PATH',
            'date_cols': [],
            'dtype_map': {'zone_id': str},
            # <<< SME REVISION >>> This map defines how to standardize column names.
            # It tells the loader: "If you find a column named 'name', rename it to 'zone_name'".
            'rename_map': {'name': 'zone_name'},
            # We now require the FINAL, standardized name.
            'required_cols': ['zone_id', 'zone_name'],
            'read_csv_options': {}
        },
        'zone_geometries': {
            'setting_attr': 'ZONE_GEOMETRIES_PATH'
        }
    }

    def _resolve_path(self, explicit_path: Optional[str], setting_attr: str) -> Optional[Path]:
        """Resolves a file path, prioritizing the explicit path over the setting."""
        path_str = explicit_path or getattr(settings, setting_attr, None)
        if not path_str:
            if setting_attr:
                logger.error(f"Path not found: No explicit path provided and setting '{setting_attr}' is not defined.")
            return None
        path_obj = Path(path_str)
        if path_obj.is_absolute(): return path_obj.resolve()
        base_dir = getattr(settings, 'DATA_SOURCES_DIR', getattr(settings, 'DATA_DIR', None))
        if not base_dir:
            logger.error("DATA_SOURCES_DIR or DATA_DIR not defined in settings; cannot resolve relative path.")
            return None
        return (Path(base_dir) / path_obj).resolve()

    def _validate_schema(self, df: pd.DataFrame, required_cols: List[str], config_key: str) -> bool:
        """Checks if all required columns are present in the DataFrame."""
        if not required_cols: return True
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.critical(f"({config_key}) Schema Validation FAILED! Required columns are missing: {sorted(list(missing_cols))}.")
            return False
        return True

    def _load_and_process_csv(self, config_key: str, file_path_str: Optional[str] = None) -> pd.DataFrame:
        """A generic method to load, clean, standardize, and validate a CSV based on its configuration."""
        config = self._DATA_CONFIG.get(config_key)
        if not config:
            logger.error(f"No configuration found for data key: '{config_key}'"); return pd.DataFrame()

        file_path = self._resolve_path(file_path_str, config['setting_attr'])
        if not file_path or not file_path.is_file():
            logger.error(f"({config_key}) CSV file NOT FOUND at resolved path: {file_path}"); return pd.DataFrame()

        try:
            df = pd.read_csv(file_path, **config.get('read_csv_options', {}))
            if df.empty:
                logger.warning(f"({config_key}) CSV at {file_path} loaded as empty."); return pd.DataFrame()

            df = data_cleaner.clean_column_names(df)
            
            # <<< SME REVISION >>> Resiliently rename columns to a standard format.
            rename_map = config.get('rename_map', {})
            df.rename(columns=rename_map, inplace=True)
            
            # <<< SME REVISION >>> Resiliently apply data types only to columns that exist.
            dtype_map = config.get('dtype_map', {})
            for col, dtype in dtype_map.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)

            if not self._validate_schema(df, config.get('required_cols', []), config_key):
                return pd.DataFrame()

            df = convert_date_columns(df, config.get('date_cols', []))
            logger.info(f"({config_key}) Successfully loaded and processed {len(df)} records from '{file_path.name}'.")
            return df
        except Exception as e:
            logger.error(f"({config_key}) Error loading/processing CSV from {file_path}: {e}", exc_info=True)
            return pd.DataFrame()

    def load_zone_data(self, attributes_path: Optional[str], geometries_path: Optional[str]) -> pd.DataFrame:
        """Loads and merges zone attributes (CSV) and geometries (GeoJSON)."""
        attributes_df = self._load_and_process_csv('zone_attributes', attributes_path)
        # The renaming logic is now handled inside _load_and_process_csv, making this function simpler.

        geometries_list = []
        geom_config = self._DATA_CONFIG.get('zone_geometries', {})
        geom_path = self._resolve_path(geometries_path, geom_config.get('setting_attr', ''))

        if geom_path and geom_path.is_file():
            geo_data = robust_json_load(geom_path)
            if isinstance(geo_data, dict) and 'features' in geo_data:
                for feature in geo_data.get('features', []):
                    props = feature.get("properties", {})
                    zid = props.get("zone_id", props.get("ZONE_ID"))
                    if zid is not None and feature.get("geometry"):
                        geometries_list.append({"zone_id": str(zid).strip(), "geometry_obj": feature["geometry"]})
        geometries_df = pd.DataFrame(geometries_list)

        if attributes_df.empty and geometries_df.empty: return pd.DataFrame()
        if attributes_df.empty: return geometries_df
        if geometries_df.empty: return attributes_df

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
    """Loads a JSON config file from a path or setting attribute."""
    file_path = _data_loader._resolve_path(None, path_or_setting)
    if not file_path or not file_path.exists():
        file_path = _data_loader._resolve_path(path_or_setting, "")
    if not file_path or not file_path.is_file(): return default
    data = robust_json_load(file_path)
    return data if data is not None and (default is None or isinstance(data, type(default))) else default

# --- Wrapper functions for specific JSON configs ---
def load_escalation_protocols() -> Dict[str, Any]:
    return load_json_config('ESCALATION_PROTOCOLS_JSON_PATH', default={})

def load_pictogram_map() -> Dict[str, str]:
    return load_json_config('PICTOGRAM_MAP_JSON_PATH', default={})

def load_haptic_patterns() -> Dict[str, List[int]]:
    data = load_json_config('HAPTIC_PATTERNS_JSON_PATH', default={})
    if not isinstance(data, dict): return {}
    return { k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(i, int) for i in v) }
