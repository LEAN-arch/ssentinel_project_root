# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Module Imports ---
try:
    from config import settings
    # FIXED: Call the now-implemented central function for KPIs
    from data_processing.aggregation import get_trend_data, get_clinic_environmental_summary_kpis
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in env_details.py: {e}. Ensure paths and dependencies are correct.")
    raise

logger = logging.getLogger(__name__)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)


def prepare_clinic_environmental_detail_data(
    filtered_iot_df_for_period: Optional[pd.DataFrame],
    reporting_period_context_str: str  # For logging and context in outputs
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.
    """
    module_log_prefix = "ClinicEnvDetailPrep"
    env_details: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "current_environmental_alerts_list": [],
        "hourly_avg_co2_trend": pd.Series(dtype='float64'),
        "latest_room_sensor_readings_df": pd.DataFrame(),
        "processing_notes": []
    }

    if not isinstance(filtered_iot_df_for_period, pd.DataFrame) or filtered_iot_df_for_period.empty:
        env_details["processing_notes"].append("No clinic IoT data provided for the period.")
        return env_details

    df_iot = filtered_iot_df_for_period.copy()
    if 'timestamp' not in df_iot.columns:
        env_details["processing_notes"].append("Critical: 'timestamp' column missing from IoT data.")
        return env_details

    # --- Data Cleaning ---
    df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'], errors='coerce')
    df_iot.dropna(subset=['timestamp'], inplace=True)
    if df_iot.empty:
        env_details["processing_notes"].append("No IoT records with valid timestamps found.")
        return env_details

    numeric_cols = ['avg_co2_ppm', 'avg_pm25', 'avg_noise_db', 'waiting_room_occupancy']
    for col in numeric_cols:
        if col in df_iot.columns:
            df_iot[col] = convert_to_numeric(df_iot[col], default_value=np.nan)
        else:
            df_iot[col] = np.nan

    # --- Current Environmental Alerts (using central KPI function) ---
    try:
        # FIXED: Call the now-implemented central function from aggregation.py
        env_kpis = get_clinic_environmental_summary_kpis(
            iot_df_period=df_iot,
            source_context=f"{module_log_prefix}/KPIs"
        )
        
        alerts = []
        if env_kpis.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
            alerts.append({"message": f"{env_kpis['rooms_co2_very_high_alert_latest_count']} area(s) with very high CO₂ levels.", "status_level": "HIGH_RISK"})
        elif env_kpis.get('rooms_co2_high_alert_latest_count', 0) > 0:
            alerts.append({"message": f"{env_kpis['rooms_co2_high_alert_latest_count']} area(s) with elevated CO₂.", "status_level": "MODERATE_CONCERN"})
        
        if env_kpis.get('waiting_room_high_occupancy_alert_latest_flag', False):
            alerts.append({"message": "Waiting area is currently overcrowded.", "status_level": "MODERATE_CONCERN"})

        if not alerts:
            alerts.append({"message": "Environmental parameters within acceptable limits.", "status_level": "ACCEPTABLE"})
        
        env_details["current_environmental_alerts_list"] = alerts
    except Exception as e:
        logger.error(f"({module_log_prefix}) Error generating environmental alerts: {e}", exc_info=True)
        env_details["processing_notes"].append("Could not generate current environmental alert list.")

    # --- Hourly CO2 Trend ---
    if 'avg_co2_ppm' in df_iot.columns and df_iot['avg_co2_ppm'].notna().any():
        try:
            co2_trend = get_trend_data(df=df_iot, value_col='avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
            env_details["hourly_avg_co2_trend"] = co2_trend.rename("avg_co2_ppm_hourly")
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error calculating CO2 trend: {e}", exc_info=True)
            env_details["processing_notes"].append("Failed to calculate hourly CO₂ trend.")

    # --- Latest Sensor Readings by Room ---
    key_cols = ['clinic_id', 'room_name', 'timestamp']
    if all(col in df_iot.columns for col in key_cols):
        display_cols = key_cols + [col for col in numeric_cols if col in df_iot.columns]
        try:
            latest_readings = df_iot[display_cols].sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
            env_details["latest_room_sensor_readings_df"] = latest_readings.reset_index(drop=True)
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error processing latest room readings: {e}", exc_info=True)
            env_details["processing_notes"].append("Failed to process latest room sensor readings.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished.")
    return env_details
