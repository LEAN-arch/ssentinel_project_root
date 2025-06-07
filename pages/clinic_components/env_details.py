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
    # FIXED: The parameter name is now standardized to match its calling context.
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
        return env_details_output
    
    # --- Current Environmental Alerts (Derived from latest summary KPIs for the period) ---
    try:
        env_summary_kpis = get_clinic_environmental_summary_kpis(
            iot_df_period=df_iot, source_context=f"{module_log_prefix}/LatestAlertsKPIs"
        )
        alerts_buffer_list: List[Dict[str, Any]] = []
        if env_summary_kpis.get('rooms_co2_very_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"message": "High CO2 Levels Detected", "level": "HIGH_RISK"})
        elif env_summary_kpis.get('rooms_co2_high_alert_latest_count', 0) > 0:
            alerts_buffer_list.append({"message": "Elevated CO2 Levels", "level": "MODERATE_CONCERN"})
        
        if not alerts_buffer_list:
            alerts_buffer_list.append({"message": "Environmental parameters acceptable.", "level": "ACCEPTABLE"})
        env_details_output["current_environmental_alerts_list"] = alerts_buffer_list
    except Exception as e_env_alerts:
        logger.error(f"({module_log_prefix}) Error generating current environmental alerts: {e_env_alerts}", exc_info=True)

    # --- Hourly CO2 Trend ---
    if 'avg_co2_ppm' in df_iot.columns and df_iot['avg_co2_ppm'].notna().any():
        try:
            co2_trend_series = get_trend_data(df=df_iot, value_col='avg_co_ppm', date_col=timestamp_col, period='H', agg_func='mean')
            env_details_output["hourly_avg_co2_trend"] = co2_trend_series
        except Exception as e:
            logger.error(f"({module_log_prefix}) Error calculating CO2 trend: {e}", exc_info=True)
        
    return env_details_output
