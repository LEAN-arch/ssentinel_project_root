# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

try:
    from config import settings
    # CORRECTED: Call the now-implemented central function for KPIs
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
    reporting_period_context_str: str # For logging and context in outputs
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
        "hourly_avg_occupancy_trend": pd.Series(dtype='float64'),
        "latest_room_sensor_readings_df": pd.DataFrame(),
        "processing_notes": []
    }

    if not isinstance(filtered_iot_df_for_period, pd.DataFrame) or filtered_iot_df_for_period.empty:
        note = f"No clinic IoT data provided for the period '{reporting_period_context_str}'."
        logger.warning(f"({module_log_prefix}) {note}")
        env_details_output["processing_notes"].append(note)
        return env_details_output

    df_iot = filtered_iot_df_for_period.copy()

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

    expected_numeric_env_cols = [
        'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 
        'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 
        'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    for col_name in expected_numeric_env_cols:
        if col_name in df_iot.columns:
            df_iot[col_name] = convert_to_numeric(df_iot[col_name], default_value=np.nan)
        else:
            df_iot[col_name] = np.nan

    # --- Current Environmental Alerts (Derived from latest summary KPIs for the period) ---
    try:
        # CORRECTED: Call the now-implemented central function from aggregation.py.
        env_summary_kpis = get_clinic_environmental_summary_kpis(
            iot_df_period=df_iot,
            source_context=f"{module_log_prefix}/LatestAlertsKPIs"
        )
        
        alerts_buffer_list: List[Dict[str, Any]] = []
        co2_very_high_thresh = _get_setting('ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500)
        co2_high_thresh = _get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1500)
        pm25_very_high_thresh = _get_setting('ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 50)
        pm25_high_thresh = _get_setting('ALERT_AMBIENT_PM25_HIGH_UGM3', 35)
        noise_high_thresh = _get_setting('ALERT_AMBIENT_NOISE_HIGH_DBA', 85)
        occupancy_max_thresh = _get_setting('TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX', 10)

        if env_summary_kpis.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"alert_type": "High CO2 Levels", "message": f"{env_summary_kpis['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {co2_very_high_thresh}ppm. Ventilation check advised.", "level": "HIGH_RISK", "icon_char": "💨"})
        elif env_summary_kpis.get('rooms_co2_high_alert_latest_count', 0) > 0:
             alerts_buffer_list.append({"alert_type": "Elevated CO2", "message": f"{env_summary_kpis['rooms_co2_high_alert_latest_count']} area(s) with CO2 > {co2_high_thresh}ppm. Monitor ventilation.", "level": "MODERATE_CONCERN", "icon_char": "💨"})
        
        if env_summary_kpis.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"alert_type": "Poor Air Quality (PM2.5)", "message": f"{env_summary_kpis['rooms_pm25_very_high_alert_latest_count']} area(s) with PM2.5 > {pm25_very_high_thresh}µg/m³. Check filtration.", "level": "HIGH_RISK", "icon_char": "🌫️"})
        elif env_summary_kpis.get('rooms_pm25_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"alert_type": "Elevated PM2.5", "message": f"{env_summary_kpis['rooms_pm25_high_alert_latest_count']} area(s) with PM2.5 > {pm25_high_thresh}µg/m³. Monitor air quality.", "level": "MODERATE_CONCERN", "icon_char": "🌫️"})
        
        if env_summary_kpis.get('rooms_noise_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"alert_type": "High Noise Levels", "message": f"{env_summary_kpis['rooms_noise_high_alert_latest_count']} area(s) with Noise > {noise_high_thresh}dBA. Mitigate sources.", "level": "MODERATE_CONCERN", "icon_char": "🔊"})
        
        if env_summary_kpis.get('waiting_room_high_occupancy_alert_latest_flag', False):
            alerts_buffer_list.append({"alert_type": "Waiting Area Overcrowding", "message": f"High Occupancy: Waiting area(s) currently > {occupancy_max_thresh} persons. Manage patient flow.", "level": "MODERATE_CONCERN", "icon_char": "👨‍👩‍👧‍👦"})
        
        if not alerts_buffer_list:
            alerts_buffer_list.append({"alert_type": "Environmental Status", "message": "No significant environmental alerts detected from latest readings in this period.", "level": "ACCEPTABLE", "icon_char":"✅"})
        
        env_details_output["current_environmental_alerts_list"] = alerts_buffer_list
    except Exception as e_env_alerts:
        logger.error(f"({module_log_prefix}) Error generating current environmental alerts: {e_env_alerts}", exc_info=True)
        env_details_output["processing_notes"].append("Could not generate current environmental alert list.")

    # --- Hourly CO2 Trend ---
    co2_col = 'avg_co2_ppm'
    if co2_col in df_iot.columns and df_iot[co2_col].notna().any():
        try:
            co2_trend_series = get_trend_data(df=df_iot, value_col=co2_col, date_col=timestamp_col, period='H', agg_func='mean', source_context=f"{module_log_prefix}/CO2Trend")
            if isinstance(co2_trend_series, pd.Series) and not co2_trend_series.empty:
                env_details_output["hourly_avg_co2_trend"] = co2_trend_series.rename("avg_co2_ppm_hourly")
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error calculating CO2 trend: {e}", exc_info=True)
            env_details_output["processing_notes"].append("Failed to calculate hourly CO2 trend.")

    # --- Hourly Waiting Room Occupancy Trend ---
    occupancy_col = 'waiting_room_occupancy'
    room_name_col = 'room_name'
    if occupancy_col in df_iot.columns and room_name_col in df_iot.columns and df_iot[occupancy_col].notna().any():
        df_waiting_iot_data = df_iot[df_iot[room_name_col].astype(str).str.contains('Waiting', case=False, na=False) & df_iot[occupancy_col].notna()]
        if not df_waiting_iot_data.empty:
            try:
                occupancy_trend_series = get_trend_data(df=df_waiting_iot_data, value_col=occupancy_col, date_col=timestamp_col, period='H', agg_func='mean', source_context=f"{module_log_prefix}/WaitingOccupancyTrend")
                if isinstance(occupancy_trend_series, pd.Series) and not occupancy_trend_series.empty:
                    env_details_output["hourly_avg_occupancy_trend"] = occupancy_trend_series.rename("avg_waiting_occupancy_hourly")
            except Exception as e:
                logger.error(f"({module_log_prefix}) Error calculating occupancy trend: {e}", exc_info=True)
                env_details_output["processing_notes"].append("Failed to calculate hourly waiting room occupancy trend.")
        
    # --- Latest Sensor Readings by Room ---
    latest_reading_key_cols = ['clinic_id', room_name_col, timestamp_col]
    if all(col in df_iot.columns for col in latest_reading_key_cols):
        cols_for_latest_display = latest_reading_key_cols + [col for col in expected_numeric_env_cols if col in df_iot.columns]
        try:
            latest_readings_df = df_iot[cols_for_latest_display].sort_values(timestamp_col).drop_duplicates(subset=['clinic_id', room_name_col], keep='last')
            if not latest_readings_df.empty:
                env_details_output["latest_room_sensor_readings_df"] = latest_readings_df.reset_index(drop=True)
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error processing latest room sensor readings: {e}", exc_info=True)
            env_details_output["processing_notes"].append("Failed to process latest room sensor readings.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Notes: {len(env_details_output['processing_notes'])}")
    return env_details_output
