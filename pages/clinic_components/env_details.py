# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime # Added datetime

try:
    from config import settings
    # Assuming get_trend_data and get_clinic_environmental_summary_kpis are robust
    from data_processing.aggregation import get_trend_data, get_clinic_environmental_summary_kpis
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__) # Basic logger if import fails early
    logger.error(f"Critical import error in env_details.py: {e}. Ensure paths and dependencies are correct.")
    raise # These dependencies are critical for this module

logger = logging.getLogger(__name__)

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


def prepare_clinic_environmental_detail_data(
    filtered_iot_df_for_period: Optional[pd.DataFrame],
    iot_data_source_is_generally_available: bool, # Flag indicating if IoT source file itself is present
    reporting_period_context_str: str # For logging and context in outputs
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.
    Returns a dictionary containing various environmental data components.
    """
    module_log_prefix = "ClinicEnvDetailPrep"
    logger.info(f"({module_log_prefix}) Preparing clinic environment details for period: {reporting_period_context_str}")

    # Initialize the output structure with default empty/NaN values
    env_details_output: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "current_environmental_alerts_list": [],
        "hourly_avg_co2_trend": pd.Series(dtype='float64'),
        "hourly_avg_occupancy_trend": pd.Series(dtype='float64'),
        "latest_room_sensor_readings_df": pd.DataFrame(),
        "processing_notes": []
    }

    if not isinstance(filtered_iot_df_for_period, pd.DataFrame) or filtered_iot_df_for_period.empty:
        note = (f"No clinic IoT data provided for the period '{reporting_period_context_str}'." 
                if iot_data_source_is_generally_available 
                else "IoT data source is generally unavailable. Environmental monitoring details cannot be prepared.")
        logger.warning(f"({module_log_prefix}) {note}")
        env_details_output["processing_notes"].append(note)
        return env_details_output

    df_iot = filtered_iot_df_for_period.copy() # Work on a copy

    # ---Timestamp Validation and Preparation---
    timestamp_col = 'timestamp'
    if timestamp_col not in df_iot.columns:
        critical_note = "Critical: 'timestamp' column missing from IoT data. Cannot process environmental details."
        logger.error(f"({module_log_prefix}) {critical_note}")
        env_details_output["processing_notes"].append(critical_note)
        return env_details_output # Cannot proceed without timestamps
        
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_iot[timestamp_col]):
            df_iot[timestamp_col] = pd.to_datetime(df_iot[timestamp_col], errors='coerce')
        if df_iot[timestamp_col].dt.tz is not None: # Ensure timezone-naive
            df_iot[timestamp_col] = df_iot[timestamp_col].dt.tz_localize(None)
        df_iot.dropna(subset=[timestamp_col], inplace=True) # Remove rows with invalid timestamps
    except Exception as e_ts:
        logger.error(f"({module_log_prefix}) Error converting 'timestamp' to datetime: {e_ts}", exc_info=True)
        env_details_output["processing_notes"].append("Error processing timestamps in IoT data. Details may be incomplete.")
        return env_details_output # Cannot reliably proceed
    
    if df_iot.empty:
        empty_note = "No IoT records with valid timestamps found in the provided period data."
        logger.info(f"({module_log_prefix}) {empty_note}")
        env_details_output["processing_notes"].append(empty_note)
        return env_details_output

    # ---Numeric Column Preparation---
    # Define expected numeric columns and ensure they exist, converting to numeric with NaN for errors
    expected_numeric_env_cols = [
        'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 
        'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 
        'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    for col_name in expected_numeric_env_cols:
        if col_name in df_iot.columns:
            df_iot[col_name] = convert_to_numeric(df_iot[col_name], default_value=np.nan)
        else:
            logger.debug(f"({module_log_prefix}) Numeric IoT column '{col_name}' not found. Will be treated as NaN.")
            df_iot[col_name] = np.nan # Ensure column exists for consistent processing

    # --- Current Environmental Alerts (Derived from latest summary KPIs for the period) ---
    try:
        # Use the already filtered df_iot for the period to get latest alerts
        env_summary_kpis = get_clinic_environmental_summary_kpis(
            iot_df_period=df_iot, # Pass the period-filtered and cleaned df_iot
            source_context=f"{module_log_prefix}/LatestAlertsKPIs"
        )
        
        alerts_buffer_list: List[Dict[str, Any]] = []
        co2_very_high_thresh = _get_setting('ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 1500)
        co2_high_thresh = _get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1000)
        pm25_very_high_thresh = _get_setting('ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 35.4)
        pm25_high_thresh = _get_setting('ALERT_AMBIENT_PM25_HIGH_UGM3', 12.0)
        noise_high_thresh = _get_setting('ALERT_AMBIENT_NOISE_HIGH_DBA', 70)
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
        
        if not alerts_buffer_list: # If no specific issues found
            alerts_buffer_list.append({"alert_type": "Environmental Status", "message": "No significant environmental alerts detected from latest readings in this period.", "level": "ACCEPTABLE", "icon_char":"✅"})
        
        env_details_output["current_environmental_alerts_list"] = alerts_buffer_list
    except Exception as e_env_alerts:
        logger.error(f"({module_log_prefix}) Error generating current environmental alerts: {e_env_alerts}", exc_info=True)
        env_details_output["processing_notes"].append("Could not generate current environmental alert list.")


    # --- Hourly CO2 Trend (Clinic-wide average) ---
    co2_col = 'avg_co2_ppm'
    if co2_col in df_iot.columns and df_iot[co2_col].notna().any():
        try:
            co2_trend_series = get_trend_data(df=df_iot, value_col=co2_col, date_col=timestamp_col, period='H', agg_func='mean', source_context=f"{module_log_prefix}/CO2Trend")
            if isinstance(co2_trend_series, pd.Series) and not co2_trend_series.empty:
                env_details_output["hourly_avg_co2_trend"] = co2_trend_series.rename("avg_co2_ppm_hourly")
            else: env_details_output["processing_notes"].append("Hourly CO2 trend calculation resulted in empty series.")
        except Exception as e_co2_trend:
            logger.error(f"({module_log_prefix}) Error calculating CO2 trend: {e_co2_trend}", exc_info=True)
            env_details_output["processing_notes"].append("Failed to calculate hourly CO2 trend.")
    else: env_details_output["processing_notes"].append(f"CO2 data ('{co2_col}') missing or all NaN. Cannot generate trend.")

    # --- Hourly Waiting Room Occupancy Trend ---
    occupancy_col = 'waiting_room_occupancy'
    room_name_col = 'room_name' # Assuming this column exists for filtering waiting rooms
    if occupancy_col in df_iot.columns and room_name_col in df_iot.columns and df_iot[occupancy_col].notna().any():
        # Filter for rows specifically related to waiting rooms
        # Ensure room_name is string for robust .str.contains
        df_iot[room_name_col] = df_iot[room_name_col].astype(str)
        df_waiting_iot_data = df_iot[df_iot[room_name_col].str.contains('Waiting', case=False, na=False) & df_iot[occupancy_col].notna()]
        
        if not df_waiting_iot_data.empty:
            try:
                occupancy_trend_series = get_trend_data(df=df_waiting_iot_data, value_col=occupancy_col, date_col=timestamp_col, period='H', agg_func='mean', source_context=f"{module_log_prefix}/WaitingOccupancyTrend")
                if isinstance(occupancy_trend_series, pd.Series) and not occupancy_trend_series.empty:
                    env_details_output["hourly_avg_occupancy_trend"] = occupancy_trend_series.rename("avg_waiting_occupancy_hourly")
                else: env_details_output["processing_notes"].append("Hourly waiting room occupancy trend resulted in empty series.")
            except Exception as e_occ_trend:
                logger.error(f"({module_log_prefix}) Error calculating occupancy trend: {e_occ_trend}", exc_info=True)
                env_details_output["processing_notes"].append("Failed to calculate hourly waiting room occupancy trend.")
        else: env_details_output["processing_notes"].append("No valid 'waiting_room_occupancy' data found in rooms identified as 'Waiting' areas.")
    else: env_details_output["processing_notes"].append(f"'{occupancy_col}' or '{room_name_col}' data missing or all NaN. Cannot generate occupancy trend.")

    # --- Latest Sensor Readings by Room ---
    # Ensure essential columns for grouping and displaying latest readings exist
    latest_reading_key_cols = ['clinic_id', room_name_col, timestamp_col] # Use defined room_name_col
    if all(col in df_iot.columns for col in latest_reading_key_cols):
        # Select only relevant columns for the latest readings display
        cols_for_latest_display = latest_reading_key_cols + [col for col in expected_numeric_env_cols if col in df_iot.columns]
        
        try:
            # Sort by timestamp to get the latest, then drop duplicates keeping the last record per room
            latest_readings_df = df_iot[cols_for_latest_display].sort_values(timestamp_col, ascending=True, na_position='first').drop_duplicates(subset=['clinic_id', room_name_col], keep='last')
            if not latest_readings_df.empty:
                env_details_output["latest_room_sensor_readings_df"] = latest_readings_df.reset_index(drop=True)
            else:
                env_details_output["processing_notes"].append("No distinct room sensor readings found to determine latest values for the period.")
        except Exception as e_latest_readings:
            logger.error(f"({module_log_prefix}) Error processing latest room sensor readings: {e_latest_readings}", exc_info=True)
            env_details_output["processing_notes"].append("Failed to process latest room sensor readings.")
    else:
        missing_key_cols_latest = [col for col in latest_reading_key_cols if col not in df_iot.columns]
        env_details_output["processing_notes"].append(f"Essential columns for latest room readings missing: {missing_key_cols_latest}. Table cannot be generated.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Number of notes: {len(env_details_output['processing_notes'])}")
    return env_details_output
