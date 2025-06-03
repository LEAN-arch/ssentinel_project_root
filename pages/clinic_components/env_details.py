# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data from clinic IoT sensors for Sentinel.
# Renamed from environment_detail_preparer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
from data_processing.aggregation import get_trend_data # For time-series trends
# For overall env summary to derive alerts, use aggregation.get_clinic_environmental_summary_kpis
from data_processing.aggregation import get_clinic_environmental_summary_kpis
from data_processing.helpers import convert_to_numeric # For data cleaning

logger = logging.getLogger(__name__)


def prepare_clinic_environmental_detail_data( # Renamed function
    filtered_iot_df_for_period: Optional[pd.DataFrame], # IoT data already filtered for the clinic and period
    iot_data_source_is_generally_available: bool, # Flag: is the IoT source file/connection generally OK?
    reporting_period_context_str: str # String describing the reporting period for context
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings
    for the clinic dashboard.

    Args:
        filtered_iot_df_for_period: DataFrame of IoT readings for the clinic and selected period.
        iot_data_source_is_generally_available: Flag indicating if IoT data source is generally available.
        reporting_period_context_str: String describing the reporting period for contextual logging/display.

    Returns:
        Dictionary containing structured environmental detail data:
            - "reporting_period": str
            - "current_environmental_alerts_list": List of alert dicts (derived from latest readings).
            - "hourly_avg_co2_trend": pd.Series (clinic-wide hourly average CO2).
            - "hourly_avg_occupancy_trend": pd.Series (hourly average for waiting rooms).
            - "latest_room_sensor_readings_df": pd.DataFrame (latest readings by room).
            - "processing_notes": List of strings for any issues or contextual info.
    """
    module_log_prefix = "ClinicEnvDetailPrep" # Renamed for clarity
    logger.info(f"({module_log_prefix}) Preparing clinic environment details for period: {reporting_period_context_str}")

    # Initialize output structure with defaults, especially for DataFrames to ensure consistent schema
    env_details_output_dict: Dict[str, Any] = {
        "reporting_period": reporting_period_context_str,
        "current_environmental_alerts_list": [], # List of dicts: {alert_type, message, level, icon}
        "hourly_avg_co2_trend": pd.Series(dtype='float64'),      # Empty Series default
        "hourly_avg_occupancy_trend": pd.Series(dtype='float64'),# Empty Series default
        "latest_room_sensor_readings_df": pd.DataFrame(),       # Empty DataFrame default
        "processing_notes": []                                  # For issues or contextual info
    }

    if not isinstance(filtered_iot_df_for_period, pd.DataFrame) or filtered_iot_df_for_period.empty:
        note_msg = ""
        if iot_data_source_is_generally_available: # Source exists, but no data for this specific period/filter
            note_msg = f"No clinic environmental IoT data found for the period '{reporting_period_context_str}' to prepare details."
            logger.info(f"({module_log_prefix}) {note_msg}")
        else: # Source itself is likely missing or misconfigured
            note_msg = "IoT data source appears generally unavailable; environmental monitoring details cannot be prepared."
            logger.warning(f"({module_log_prefix}) {note_msg}")
        env_details_output_dict["processing_notes"].append(note_msg)
        return env_details_output_dict

    df_iot_selected_period = filtered_iot_df_for_period.copy() # Work on a copy

    # --- Data Preparation and Validation ---
    # 1. Ensure 'timestamp' column exists and is valid datetime
    if 'timestamp' not in df_iot_selected_period.columns:
        critical_note_msg = "Critical: 'timestamp' column missing from IoT data. Cannot process environment details."
        logger.error(f"({module_log_prefix}) {critical_note_msg}")
        env_details_output_dict["processing_notes"].append(critical_note_msg)
        return env_details_output_dict # Cannot proceed without timestamps
        
    try:
        df_iot_selected_period['timestamp'] = pd.to_datetime(df_iot_selected_period['timestamp'], errors='coerce')
        df_iot_selected_period.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp conversion failed
    except Exception as e_ts_conv:
        logger.error(f"({module_log_prefix}) Error converting 'timestamp' column to datetime: {e_ts_conv}")
        env_details_output_dict["processing_notes"].append("Error processing timestamps in IoT data.")
        return env_details_output_dict


    if df_iot_selected_period.empty:
        empty_after_clean_note = "No IoT records with valid timestamps found in the provided period for environment details."
        logger.info(f"({module_log_prefix}) {empty_after_clean_note}")
        env_details_output_dict["processing_notes"].append(empty_after_clean_note)
        return env_details_output_dict

    # Ensure other key numeric columns are numeric
    numeric_env_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 
                        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
                        'waiting_room_occupancy', 'patient_throughput_per_hour', 
                        'sanitizer_dispenses_per_hour']
    for num_col in numeric_env_cols:
        if num_col in df_iot_selected_period.columns:
            df_iot_selected_period[num_col] = convert_to_numeric(df_iot_selected_period[num_col], default_value=np.nan)
        else: # Ensure column exists if expected, even if all NaN
            df_iot_selected_period[num_col] = np.nan


    # --- 1. Current Environmental Alerts Summary (derived from latest readings in period) ---
    # Use the centralized get_clinic_environmental_summary_kpis function
    # This function itself should find the latest reading per room within the df_iot_selected_period.
    env_summary_kpis_from_aggregation = get_clinic_environmental_summary_kpis(
        iot_df_period=df_iot_selected_period, # Pass the period-filtered and cleaned data
        source_context=f"{module_log_prefix}/LatestAlertsSummaryKPIs"
    )
    
    current_alerts_buffer: List[Dict[str, Any]] = []
    # Structure alerts based on the summary from the aggregation function
    if env_summary_kpis_from_aggregation.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
        current_alerts_buffer.append({
            "alert_type": "High CO2 Levels",
            "message": f"{env_summary_kpis_from_aggregation['rooms_co2_very_high_alert_latest_count']} area(s) with CO2 > {settings.ALERT_AMBIENT_CO2_VERY_HIGH_PPM}ppm (Very High). Immediate ventilation check advised.",
            "level": "HIGH_RISK", "icon_char": "💨" # Changed from icon to icon_char
        })
    elif env_summary_kpis_from_aggregation.get('rooms_co2_high_alert_latest_count', 0) > 0: # Check for 'high' if not 'very_high'
         current_alerts_buffer.append({
            "alert_type": "Elevated CO2 Levels",
            "message": f"{env_summary_kpis_from_aggregation['rooms_co2_high_alert_latest_count']} area(s) with CO2 > {settings.ALERT_AMBIENT_CO2_HIGH_PPM}ppm (High). Monitor ventilation.",
            "level": "MODERATE_CONCERN", "icon_char": "💨"
        })

    if env_summary_kpis_from_aggregation.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
        current_alerts_buffer.append({
            "alert_type": "Poor Air Quality (PM2.5)",
            "message": f"{env_summary_kpis_from_aggregation['rooms_pm25_very_high_alert_latest_count']} area(s) with PM2.5 > {settings.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3}µg/m³ (Very High). Check air filtration/sources.",
            "level": "HIGH_RISK", "icon_char": "🌫️"
        })
    elif env_summary_kpis_from_aggregation.get('rooms_pm25_high_alert_latest_count', 0) > 0:
        current_alerts_buffer.append({
            "alert_type": "Elevated Air Pollution (PM2.5)",
            "message": f"{env_summary_kpis_from_aggregation['rooms_pm25_high_alert_latest_count']} area(s) with PM2.5 > {settings.ALERT_AMBIENT_PM25_HIGH_UGM3}µg/m³. Monitor air quality.",
            "level": "MODERATE_CONCERN", "icon_char": "🌫️"
        })
        
    if env_summary_kpis_from_aggregation.get('rooms_noise_high_alert_latest_count', 0) > 0:
        current_alerts_buffer.append({
            "alert_type": "High Noise Levels",
            "message": f"{env_summary_kpis_from_aggregation['rooms_noise_high_alert_latest_count']} area(s) with Noise > {settings.ALERT_AMBIENT_NOISE_HIGH_DBA}dBA. Identify and mitigate noise sources.",
            "level": "MODERATE_CONCERN", "icon_char": "🔊"
        })
        
    if env_summary_kpis_from_aggregation.get('waiting_room_high_occupancy_alert_latest_flag', False):
        current_alerts_buffer.append({
            "alert_type": "Waiting Area Overcrowding",
            "message": f"High Occupancy: At least one waiting area reported occupancy > {settings.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons. Manage patient flow.",
            "level": "MODERATE_CONCERN", "icon_char": "👨‍👩‍👧‍👦"
        })
    
    if not current_alerts_buffer: # If no specific critical/warning alerts based on above logic
        current_alerts_buffer.append({
            "alert_type": "Environmental Status", 
            "message": "No significant environmental alerts identified from the latest readings in this period.", 
            "level": "ACCEPTABLE", "icon_char":"✅"
            })
    env_details_output_dict["current_environmental_alerts_list"] = current_alerts_buffer


    # --- 2. Hourly Trends for Key Environmental Metrics ---
    # CO2 Trend (Overall average for the clinic if multiple rooms/sensors)
    if 'avg_co2_ppm' in df_iot_selected_period.columns and df_iot_selected_period['avg_co2_ppm'].notna().any():
        co2_hourly_trend_series = get_trend_data(
            df=df_iot_selected_period, value_col='avg_co2_ppm', date_col='timestamp', 
            period='H', agg_func='mean', # Hourly average
            source_context=f"{module_log_prefix}/CO2HourlyTrend"
        )
        if isinstance(co2_hourly_trend_series, pd.Series) and not co2_hourly_trend_series.empty:
            env_details_output_dict["hourly_avg_co2_trend"] = co2_hourly_trend_series.rename("avg_co2_ppm_hourly")
        else:
            env_details_output_dict["processing_notes"].append("Could not generate hourly CO2 trend for the period (result was empty series).")
    else:
        env_details_output_dict["processing_notes"].append("CO2 data ('avg_co2_ppm') column missing or all NaN for trend calculation.")

    # Waiting Room Occupancy Trend (Average occupancy in designated waiting areas)
    if 'waiting_room_occupancy' in df_iot_selected_period.columns and \
       'room_name' in df_iot_selected_period.columns and \
       df_iot_selected_period['waiting_room_occupancy'].notna().any():
        
        df_waiting_areas_iot_data = df_iot_selected_period[
            df_iot_selected_period['room_name'].astype(str).str.contains('Waiting', case=False, na=False) & # Identify waiting rooms
            df_iot_selected_period['waiting_room_occupancy'].notna() # Ensure occupancy data exists
        ]
        if not df_waiting_areas_iot_data.empty:
            occupancy_hourly_trend_series = get_trend_data(
                df=df_waiting_areas_iot_data, value_col='waiting_room_occupancy', date_col='timestamp', 
                period='H', agg_func='mean', # Hourly average occupancy
                source_context=f"{module_log_prefix}/WaitingOccupancyHourlyTrend"
            )
            if isinstance(occupancy_hourly_trend_series, pd.Series) and not occupancy_hourly_trend_series.empty:
                env_details_output_dict["hourly_avg_occupancy_trend"] = occupancy_hourly_trend_series.rename("avg_waiting_occupancy_hourly")
            else:
                env_details_output_dict["processing_notes"].append("Could not generate hourly waiting room occupancy trend (result was empty series).")
        else:
            env_details_output_dict["processing_notes"].append("No data points specifically identified as 'waiting_room_occupancy' in rooms named 'Waiting*' found for trend.")
    else:
        env_details_output_dict["processing_notes"].append("Waiting room occupancy ('waiting_room_occupancy') or 'room_name' data column missing or all NaN for trend.")


    # --- 3. Latest Sensor Readings by Room (from end of selected period) ---
    # Define the desired columns for the latest readings table.
    # This helps ensure a consistent output schema for the UI.
    desired_cols_for_latest_table = [
        'clinic_id', 'room_name', 'timestamp', # Identifiers
        'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 
        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    # Filter to only columns that actually exist in the DataFrame to avoid KeyErrors
    available_cols_for_latest_display = [col for col in desired_cols_for_latest_table if col in df_iot_selected_period.columns]

    # Need 'clinic_id' and 'room_name' (and 'timestamp') to identify unique rooms' latest readings
    if 'clinic_id' in available_cols_for_latest_display and \
       'room_name' in available_cols_for_latest_display and \
       'timestamp' in available_cols_for_latest_display:
        
        # Get the absolute latest (last timestamp) reading for each unique room within the period
        df_latest_readings_by_room_result = df_iot_selected_period.sort_values('timestamp', ascending=True).drop_duplicates(
            subset=['clinic_id', 'room_name'], keep='last' # Get the last record for each room
        )
        if not df_latest_readings_by_room_result.empty:
            env_details_output_dict["latest_room_sensor_readings_df"] = df_latest_readings_by_room_result[available_cols_for_latest_display].reset_index(drop=True)
        else:
            env_details_output_dict["processing_notes"].append("No distinct room sensor readings found for the latest point in this period after initial data filtering.")
    else:
        missing_key_cols_for_latest_str = [key_col for key_col in ['clinic_id','room_name', 'timestamp'] if key_col not in available_cols_for_latest_display]
        env_details_output_dict["processing_notes"].append(f"Essential columns for identifying latest room readings are missing: {missing_key_cols_for_latest_str}. Table cannot be generated.")
        
    logger.info(f"({module_log_prefix}) Clinic environment details preparation finished. Notes recorded: {len(env_details_output_dict['processing_notes'])}")
    return env_details_output_dict
