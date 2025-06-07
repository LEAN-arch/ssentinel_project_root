# sentinel_project_root/data_processing/loaders.py
# Functions for loading and standardizing raw data sources for Sentinel.

import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

# --- Core Imports ---
try:
    from config import settings
    from .helpers import (
        clean_column_names,
        robust_json_load,
        standardize_missing_values,
        hash_dataframe_safe
    )
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in loaders.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)


def _load_csv_data(file_path: Path, source_context: str) -> pd.DataFrame:
    """Helper to load a CSV file with robust error handling and cleaning."""
    logger.info(f"({source_context}) Attempting to load CSV: {file_path.name}")
    if not file_path.is_file():
        logger.error(f"({source_context}) CSV file NOT FOUND: {file_path.resolve()}")
        st.error(f"Data file missing: '{file_path.name}'. Please check application setup.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if df.empty:
            logger.warning(f"({source_context}) CSV file loaded as empty: {file_path.name}")
            return pd.DataFrame()
        
        df = clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} records from '{file_path.name}'.")
        return df
    except Exception as e:
        logger.error(f"({source_context}) Failed to load/process CSV from {file_path.resolve()}: {e}", exc_info=True)
        st.error(f"Error loading data from '{file_path.name}'. See logs for details.")
        return pd.DataFrame()


def load_health_records(source_context: str = "HealthRecordsLoader", use_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Loads, cleans, and standardizes the main health records data."""
    file_path = _validate_path(settings.HEALTH_RECORDS_CSV_PATH)
    df = _load_csv_data(file_path, source_context)
    if df.empty: return pd.DataFrame()

    # Define standard column configurations
    date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'date_of_birth']
    numeric_cols = {
        'age': np.nan, 'pregnancy_status': 0, 'chronic_condition_flag': 0, 'min_spo2_pct': np.nan,
        'vital_signs_temperature_celsius': np.nan, 'fall_detected_today': 0, 'test_turnaround_days': np.nan,
        'quantity_dispensed': 0, 'item_stock_agg_zone': 0.0, 'consumption_rate_per_day': 0.0,
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan, 'avg_daily_steps': np.nan,
        'tb_contact_traced': 0, 'chw_daily_steps': np.nan, 'device_battery_level_pct': np.nan,
        'data_sync_latency_hours': np.nan, 'priority_score': np.nan
    }
    string_cols = {
        'encounter_id': "Unknown", 'patient_id': "Unknown", 'encounter_type': "Unknown",
        'gender': "Unknown", 'zone_id': "Unknown", 'clinic_id': "Unknown", 'chw_id': "Unknown",
        'condition': "Unknown", 'patient_reported_symptoms': "", 'test_type': "Unknown",
        'test_result': "Unknown", 'sample_status': "Unknown", 'referral_status': "Unknown",
        'item': "Unknown", 'task_id': 'Unknown'
    }

    # If specific columns are requested, only process those
    if use_cols:
        cols_to_load = [col for col in use_cols if col in df.columns]
        df = df[cols_to_load]

    # Standardize and clean the data
    df = standardize_missing_values(df, string_cols, numeric_cols)
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    logger.info(f"({source_context}) Health records loaded and processed. Shape: {df.shape}")
    return df


def load_zone_data(source_context: str = "ZoneDataLoader") -> pd.DataFrame:
    """Loads and merges zone attribute data (CSV) and zone geometries (GeoJSON)."""
    attr_path = _validate_path(settings.ZONE_ATTRIBUTES_CSV_PATH)
    geom_path = _validate_path(settings.ZONE_GEOMETRIES_GEOJSON_FILE_PATH)

    attr_df = _load_csv_data(attr_path, f"{source_context}/Attributes")
    if 'zone_id' not in attr_df.columns:
        logger.error(f"({source_context}) 'zone_id' missing from attributes CSV. Merge will fail.")
        return pd.DataFrame()
    attr_df['zone_id'] = attr_df['zone_id'].astype(str)

    geometries = []
    geojson_data = robust_json_load(geom_path)
    if isinstance(geojson_data, dict) and isinstance(geojson_data.get("features"), list):
        for feature in geojson_data['features']:
            props = feature.get("properties", {})
            geom = feature.get("geometry")
            zone_id = props.get("zone_id", props.get("ZONE_ID"))
            if zone_id and geom:
                geometries.append({
                    "zone_id": str(zone_id),
                    "geometry_obj": geom,
                    "name_geojson": props.get("name", "")
                })
    geom_df = pd.DataFrame(geometries)

    if geom_df.empty:
        logger.warning(f"({source_context}) No valid geometries loaded. Returning attributes only.")
        return attr_df

    # Merge attributes and geometries
    merged_df = pd.merge(attr_df, geom_df, on="zone_id", how="outer")
    merged_df['name'] = merged_df['name'].fillna(merged_df['name_geojson'])
    merged_df.drop(columns=['name_geojson'], inplace=True, errors='ignore')
    
    logger.info(f"({source_context}) Zone data loaded and merged. Final shape: {merged_df.shape}")
    return merged_df

def _validate_path(path_str: Union[str, Path]) -> Path:
    """Internal helper to resolve file paths from settings."""
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        return (settings.PROJECT_ROOT_DIR / path_obj).resolve()
    return path_obj.resolve()
