# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings
from data_processing.aggregation import get_trend_data, get_clinic_environmental_summary_kpis
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def prepare_clinic_environmental_detail_data(
    filtered_iot_df_for_period: Optional[pd.DataFrame],
    iot_data_source_is_generally_available: bool,
    reporting_period_context_str: str
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.
    """
    module_log_prefix = "ClinicEnvDetailPrep"
    logger.info(f"({module_log_prefix}) Preparing clinic environment details for: {reporting_period_context_str}")

    env_details_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "current_environmental_alerts_list": [],
        "hourly_avg_co2_trend": pd.Series(dtype='float64'),
        "hourly_avg_occupancy_trend": pd.Series(dtype='float64'),
        "latest_room_sensor_readings_df": pd.DataFrame(),
        "processing_notes": []
    }

    if not isinstance(filtered_iot_df_for_period, pd.DataFrame) or filtered_iot_df_for_period.empty:
        note = f"No clinic IoT data for '{reporting_period_context_str}'." if iot_data_source_is_generally_available \
               else "IoT data source generally unavailable; env monitoring details cannot be prepared."
        logger.warning(f"({module_log_prefix}) {note}")
        env_details_output["processing_notes"].append(note)
        return env_details_output

    df_iot = filtered_iot_df_for_period.copy()

    if 'timestamp' not in df_iot.columns:
        crit_note = "Critical: 'timestamp' column missing from IoT data. Cannot process env details."
        logger.error(f"({module_log_prefix}) {crit_note}"); env_details_output["processing_notes"].append(crit_note)
        return env_details_output
        
    try:
        df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'], errors='coerce')
        df_iot.dropna(subset=['timestamp'], inplace=True)
    except Exception as e_ts:
        logger.error(f"({module_log_prefix}) Error converting 'timestamp' to datetime: {e_ts}")
        env_details_output["processing_notes"].append("Error processing timestamps in IoT data."); return env_details_output
    if df_iot.empty:
        empty_note = "No IoT records with valid timestamps in provided period."; logger.info(f"({module_log_prefix}) {empty_note}")
        env_details_output["processing_notes"].append(empty_note); return env_details_output

    num_env_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 
                    'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 
                    'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
    for col in num_env_cols:
        if col in df_iot.columns: df_iot[col] = convert_to_numeric(df_iot[col], np.nan)
        else: df_iot[col] = np.nan # Ensure column exists

    # Current Environmental Alerts (from latest readings in period)
    env_summary_kpis = get_clinic_environmental_summary_kpis(
        iot_df_period=df_iot, source_context=f"{module_log_prefix}/LatestAlertsKPIs"
    )
    alerts_buffer: List[Dict[str, Any]] = []
    if env_summary_kpis.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
        alerts_buffer.append({"alert_type": "High CO2 Levels", "message": f"{env_summary_kpis['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM}ppm. Ventilation check advised.", "level": "HIGH_RISK", "icon_char": "💨"})
    elif env_summary_kpis.get('rooms_co2_high_alert_latest_count', 0) > 0:
         alerts_buffer.append({"alert_type": "Elevated CO2", "message": f"{env_summary_kpis['rooms_co2_high_alert_latest_count']} area(s) with CO2 > {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm. Monitor ventilation.", "level": "MODERATE_CONCERN", "icon_char": "💨"})
    if env_summary_kpis.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
        alerts_buffer.append({"alert_type": "Poor Air Quality (PM2.5)", "message": f"{env_summary_kpis['rooms_pm25_very_high_alert_latest_count']} area(s) with PM2.5 > {settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3}µg/m³. Check filtration.", "level": "HIGH_RISK", "icon_char": "🌫️"})
    elif env_summary_kpis.get('rooms_pm25_high_alert_latest_count', 0) > 0:
        alerts_buffer.append({"alert_type": "Elevated PM2.5", "message": f"{env_summary_kpis['rooms_pm25_high_alert_latest_count']} area(s) with PM2.5 > {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³. Monitor air quality.", "level": "MODERATE_CONCERN", "icon_char": "🌫️"})
    if env_summary_kpis.get('rooms_noise_high_alert_latest_count', 0) > 0:
        alerts_buffer.append({"alert_type": "High Noise Levels", "message": f"{env_summary_kpis['rooms_noise_high_alert_latest_count']} area(s) with Noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA. Mitigate sources.", "level": "MODERATE_CONCERN", "icon_char": "🔊"})
    if env_summary_kpis.get('waiting_room_high_occupancy_alert_latest_flag', False):
        alerts_buffer.append({"alert_type": "Waiting Area Overcrowding", "message": f"High Occupancy: Waiting area(s) occupancy > {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX}. Manage patient flow.", "level": "MODERATE_CONCERN", "icon_char": "👨‍👩‍👧‍👦"})
    if not alerts_buffer:
        alerts_buffer.append({"alert_type": "Environmental Status", "message": "No significant environmental alerts from latest readings.", "level": "ACCEPTABLE", "icon_char":"✅"})
    env_details_output["current_environmental_alerts_list"] = alerts_buffer

    # Hourly CO2 Trend (Clinic-wide average)
    if 'avg_co2_ppm' in df_iot.columns and df_iot['avg_co2_ppm'].notna().any():
        co2_trend = get_trend_data(df=df_iot, value_col='avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean', source_context=f"{module_log_prefix}/CO2Trend")
        if isinstance(co2_trend, pd.Series) and not co2_trend.empty: env_details_output["hourly_avg_co2_trend"] = co2_trend.rename("avg_co2_ppm_hourly")
        else: env_details_output["processing_notes"].append("Could not generate hourly CO2 trend (empty series).")
    else: env_details_output["processing_notes"].append("CO2 data ('avg_co2_ppm') missing or all NaN for trend.")

    # Hourly Waiting Room Occupancy Trend
    if 'waiting_room_occupancy' in df_iot.columns and 'room_name' in df_iot.columns and df_iot['waiting_room_occupancy'].notna().any():
        df_waiting_iot = df_iot[df_iot['room_name'].astype(str).str.contains('Waiting', case=False, na=False) & df_iot['waiting_room_occupancy'].notna()]
        if not df_waiting_iot.empty:
            occupancy_trend = get_trend_data(df=df_waiting_iot, value_col='waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean', source_context=f"{module_log_prefix}/WaitingOccupancyTrend")
            if isinstance(occupancy_trend, pd.Series) and not occupancy_trend.empty: env_details_output["hourly_avg_occupancy_trend"] = occupancy_trend.rename("avg_waiting_occupancy_hourly")
            else: env_details_output["processing_notes"].append("Could not generate hourly waiting room occupancy trend (empty series).")
        else: env_details_output["processing_notes"].append("No 'waiting_room_occupancy' data in rooms named 'Waiting*' for trend.")
    else: env_details_output["processing_notes"].append("Waiting room occupancy or 'room_name' data missing/all NaN for trend.")

    # Latest Sensor Readings by Room
    latest_cols = ['clinic_id', 'room_name', 'timestamp'] + [c for c in num_env_cols if c in df_iot.columns] # Only available numeric cols
    if all(c in df_iot.columns for c in ['clinic_id', 'room_name', 'timestamp']):
        latest_readings_df = df_iot.sort_values('timestamp', na_position='first').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        if not latest_readings_df.empty: env_details_output["latest_room_sensor_readings_df"] = latest_readings_df[latest_cols].reset_index(drop=True)
        else: env_details_output["processing_notes"].append("No distinct room sensor readings found for latest point in period.")
    else:
        missing_keys = [k for k in ['clinic_id','room_name','timestamp'] if k not in df_iot.columns]
        env_details_output["processing_notes"].append(f"Essential columns for latest room readings missing: {missing_keys}. Table not generated.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Notes: {len(env_details_output['processing_notes'])}")
    return env_details_output
