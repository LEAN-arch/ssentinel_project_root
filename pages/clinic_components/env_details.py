# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

try:
    from config import settings
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
    # FIXED: The parameter name is now standardized to match the calling context.
    filtered_iot_df: Optional[pd.DataFrame],
    reporting_period_context_str: str 
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.
    Returns a dictionary containing various environmental data components.
    """
    module_log_prefix = "ClinicEnvDetailPrep"
    logger.info(f"({module_log_prefix}) Preparing clinic environment details for period: {reporting_period_context_str}")

    env_details_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "current_environmental_alerts_list": [],
        "hourly_avg_co2_trend": pd.Series(dtype='float64'),
        "latest_room_sensor_readings_df": pd.DataFrame(),
        "processing_notes": []
    }

    if not isinstance(filtered_iot_df, pd.DataFrame) or filtered_iot_df.empty:
        note = f"No clinic IoT data provided for the period '{reporting_period_context_str}'."
        logger.warning(f"({module_log_prefix}) {note}")
        env_details_output["processing_notes"].append(note)
        return env_details_output

    df_iot = filtered_iot_df.copy()
    timestamp_col = 'timestamp'
    if timestamp_col not in df_iot.columns:
        critical_note = "Critical: 'timestamp' column missing from IoT data. Cannot process environmental details."
        logger.error(f"({module_log_prefix}) {critical_note}")
        env_details_output["processing_notes"].append(critical_note)
        return env_details_output
        
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_iot[timestamp_col]):
            df_iot[timestamp_col] = pd.to_datetime(df_iot[timestamp_col], errors='coerce')
        if df_iot[timestamp_col].dt.tz is not None:
            df_iot[timestamp_col] = df_iot[timestamp_col].dt.tz_localize(None)
        df_iot.dropna(subset=[timestamp_col], inplace=True)
    except Exception as e_ts:
        logger.error(f"({module_log_prefix}) Error converting 'timestamp' to datetime: {e_ts}", exc_info=True)
        env_details_output["processing_notes"].append("Error processing timestamps in IoT data. Details may be incomplete.")
        return env_details_output
    
    if df_iot.empty:
        empty_note = "No IoT records with valid timestamps found in the provided period data."
        logger.info(f"({module_log_prefix}) {empty_note}")
        env_details_output["processing_notes"].append(empty_note)
        return env_details_output

    # --- Current Environmental Alerts (Derived from latest summary KPIs for the period) ---
    try:
        env_summary_kpis = get_clinic_environmental_summary_kpis(
            iot_df_period=df_iot,
            source_context=f"{module_log_prefix}/LatestAlertsKPIs"
        )
        
        alerts_buffer_list: List[Dict[str, Any]] = []
        co2_very_high_thresh = _get_setting('ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500)
        co2_high_thresh = _get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1500)
        pm25_very_high_thresh = _get_setting('ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 50)
        pm25_high_thresh = _get_setting('ALERT_AMBIENT_PM25_HIGH_UGM3', 35)
        
        if env_summary_kpis.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"message": f"{env_summary_kpis['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {co2_very_high_thresh}ppm.", "level": "HIGH_RISK"})
        elif env_summary_kpis.get('rooms_co2_high_alert_latest_count', 0) > 0:
             alerts_buffer_list.append({"message": f"{env_summary_kpis['rooms_co2_high_alert_latest_count']} area(s) with CO2 > {co2_high_thresh}ppm.", "level": "MODERATE_CONCERN"})
        
        if env_summary_kpis.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"message": f"Poor Air Quality: PM2.5 > {pm25_very_high_thresh}µg/m³ detected.", "level": "HIGH_RISK"})
        elif env_summary_kpis.get('rooms_pm25_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"message": f"Elevated PM2.5: > {pm25_high_thresh}µg/m³ detected.", "level": "MODERATE_CONCERN"})

        if not alerts_buffer_list:
            alerts_buffer_list.append({"message": "No significant environmental alerts detected.", "level": "ACCEPTABLE"})
        
        env_details_output["current_environmental_alerts_list"] = alerts_buffer_list
    except Exception as e_env_alerts:
        logger.error(f"({module_log_prefix}) Error generating current environmental alerts: {e_env_alerts}", exc_info=True)

    # --- Hourly CO2 Trend ---
    co2_col = 'avg_co2_ppm'
    if co2_col in df_iot.columns and df_iot[co2_col].notna().any():
        try:
            co2_trend_series = get_trend_data(df=df_iot, value_col=co2_col, date_col=timestamp_col, period='H', agg_func='mean')
            env_details_output["hourly_avg_co2_trend"] = co2_trend_series
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error calculating CO2 trend: {e}", exc_info=True)
            env_details_output["processing_notes"].append("Failed to calculate hourly CO2 trend.")

    # --- Latest Sensor Readings by Room ---
    expected_numeric_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy']
    latest_reading_keys = ['clinic_id', 'room_name', timestamp_col]
    if all(col in df_iot.columns for col in latest_reading_keys):
        display_cols = latest_reading_keys + [col for col in expected_numeric_cols if col in df_iot.columns]
        try:
            latest_readings = df_iot[display_cols].sort_values(timestamp_col).drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
            env_details_output["latest_room_sensor_readings_df"] = latest_readings.reset_index(drop=True)
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error processing latest room sensor readings: {e}", exc_info=True)
            env_details_output["processing_notes"].append("Failed to process latest room sensor readings.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Notes: {len(env_details_output['processing_notes'])}")
    return env_details_output
