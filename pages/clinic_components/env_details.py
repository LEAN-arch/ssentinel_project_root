# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data, including plot objects, for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric
    from visualization.plots import plot_annotated_line_chart, create_empty_figure
except ImportError as e:
    # This initial logger is for bootstrap errors only.
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in env_details.py: {e}. Ensure paths and dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# FIX: The missing helper function is defined at the module level to resolve the NameError.
def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely get attributes from the global settings object."""
    return getattr(settings, attr_name, default_value)

# --- Constants for Clarity and Maintainability ---
COL_TIMESTAMP = 'timestamp'
COL_ROOM_NAME = 'room_name'
COL_CLINIC_ID = 'clinic_id'
COL_CO2_PPM = 'avg_co2_ppm'
COL_PM25 = 'avg_pm25'

ALERT_LEVEL_HIGH = "HIGH_RISK"
ALERT_LEVEL_MODERATE = "MODERATE_CONCERN"
ALERT_LEVEL_ACCEPTABLE = "ACCEPTABLE"

class ClinicEnvDetailPrep:
    """
    Encapsulates logic for preparing detailed environmental data for the clinic dashboard.
    This class generates ready-to-render Plotly figure objects.
    """
    
    def __init__(self, filtered_iot_df: Optional[pd.DataFrame], reporting_period_context_str: str):
        self.df_iot = filtered_iot_df.copy() if isinstance(filtered_iot_df, pd.DataFrame) and not filtered_iot_df.empty else pd.DataFrame()
        self.reporting_period = reporting_period_context_str
        self.notes: List[str] = []
        self.module_log_prefix = self.__class__.__name__
        self._validate_and_prepare_input_df()

        # This configuration can now correctly call the _get_setting function.
        self.ALERT_CONFIG = [
            { "metric": COL_CO2_PPM, "threshold": _get_setting('ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500), "level": ALERT_LEVEL_HIGH, "message_template": "{count} area(s) with very high CO2 (> {threshold} ppm)." },
            { "metric": COL_CO2_PPM, "threshold": _get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1500), "level": ALERT_LEVEL_MODERATE, "message_template": "{count} area(s) with elevated CO2 (> {threshold} ppm)." },
            { "metric": COL_PM25, "threshold": _get_setting('ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 50), "level": ALERT_LEVEL_HIGH, "message_template": "Poor Air Quality: PM2.5 > {threshold} µg/m³ detected in {count} area(s)." },
            { "metric": COL_PM25, "threshold": _get_setting('ALERT_AMBIENT_PM25_HIGH_UGM3', 35), "level": ALERT_LEVEL_MODERATE, "message_template": "Elevated PM2.5: > {threshold} µg/m³ detected in {count} area(s)." }
        ]

    def _validate_and_prepare_input_df(self):
        """Performs initial validation and cleaning of the input DataFrame."""
        if self.df_iot.empty:
            self.notes.append(f"No clinic IoT data provided for the period '{self.reporting_period}'.")
            return

        if COL_TIMESTAMP not in self.df_iot.columns:
            self.notes.append(f"Critical: '{COL_TIMESTAMP}' column missing from IoT data. Cannot process.")
            self.df_iot = pd.DataFrame() # Invalidate DataFrame
            return
        
        try:
            if not pd.api.types.is_datetime64_any_dtype(self.df_iot[COL_TIMESTAMP]):
                self.df_iot[COL_TIMESTAMP] = pd.to_datetime(self.df_iot[COL_TIMESTAMP], errors='coerce')
            if self.df_iot[COL_TIMESTAMP].dt.tz is not None:
                self.df_iot[COL_TIMESTAMP] = self.df_iot[COL_TIMESTAMP].dt.tz_localize(None)
            
            self.df_iot.dropna(subset=[COL_TIMESTAMP], inplace=True)
            if self.df_iot.empty:
                 self.notes.append("No IoT records with valid timestamps found.")
        except Exception as e:
            logger.error(f"({self.module_log_prefix}) Error processing timestamps: {e}", exc_info=True)
            self.notes.append("Error processing timestamps; data may be incomplete.")
            self.df_iot = pd.DataFrame() # Invalidate on critical error

    def _get_latest_room_readings(self) -> pd.DataFrame:
        """Finds the single most recent sensor reading for each room."""
        if self.df_iot.empty: return pd.DataFrame()

        required_keys = [COL_CLINIC_ID, COL_ROOM_NAME, COL_TIMESTAMP]
        if not all(col in self.df_iot.columns for col in required_keys):
            self.notes.append("Missing required columns for identifying latest readings.")
            return pd.DataFrame()
        try:
            return self.df_iot.sort_values(COL_TIMESTAMP).drop_duplicates(subset=[COL_CLINIC_ID, COL_ROOM_NAME], keep='last').reset_index(drop=True)
        except Exception as e:
            logger.error(f"({self.module_log_prefix}) Error getting latest readings: {e}", exc_info=True)
            self.notes.append("Failed to isolate latest room sensor readings.")
            return pd.DataFrame()

    def _generate_environmental_alerts(self, latest_readings_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generates a list of alerts based on the latest readings and a configuration."""
        if latest_readings_df.empty: return []

        alerts = []
        sorted_alert_config = sorted(self.ALERT_CONFIG, key=lambda x: x['threshold'], reverse=True)
        checked_metrics = set()
        for config in sorted_alert_config:
            metric = config['metric']
            if metric in checked_metrics or metric not in latest_readings_df.columns: continue
            
            latest_readings_df[metric] = convert_to_numeric(latest_readings_df[metric])
            alert_df = latest_readings_df[latest_readings_df[metric] > config['threshold']]
            
            if not alert_df.empty:
                count = len(alert_df)
                message = config['message_template'].format(count=count, threshold=config['threshold'])
                alerts.append({"message": message, "level": config['level']})
                checked_metrics.add(metric)
        
        if not alerts:
            alerts.append({"message": "No significant environmental alerts detected.", "level": ALERT_LEVEL_ACCEPTABLE})
        
        return alerts

    def _prepare_co2_trend_plot(self) -> Any:
        """
        Calculates the hourly CO2 trend and returns a Plotly Figure object.
        Returns an empty figure with a message if data is unavailable.
        """
        if self.df_iot.empty or COL_CO2_PPM not in self.df_iot.columns or self.df_iot[COL_CO2_PPM].notna().sum() == 0:
            self.notes.append("CO2 data not available for trend plot.")
            return create_empty_figure("No CO₂ Trend Data Available")
        try:
            trend_data = get_trend_data(df=self.df_iot, value_col=COL_CO2_PPM, date_col=COL_TIMESTAMP, period='H', agg_func='mean')
            if trend_data.empty:
                return create_empty_figure("No CO₂ Trend Data in Selected Period")
            
            return plot_annotated_line_chart(trend_data, "Hourly Avg. CO₂ Levels", "CO₂ (ppm)")
        except Exception as e:
            logger.error(f"({self.module_log_prefix}) Error creating CO2 trend plot: {e}", exc_info=True)
            self.notes.append("Failed to generate hourly CO2 trend plot.")
            return create_empty_figure("Error Generating CO₂ Plot")

    def prepare(self) -> Dict[str, Any]:
        """
        Orchestrates the preparation of all environmental detail components.
        Returns a structured dictionary including pre-built plot objects.
        """
        logger.info(f"({self.module_log_prefix}) Preparing details for period: {self.reporting_period}")

        latest_readings_df = self._get_latest_room_readings()
        
        output = {
            "reporting_period": self.reporting_period,
            "current_environmental_alerts_list": self._generate_environmental_alerts(latest_readings_df),
            "co2_trend_plot": self._prepare_co2_trend_plot(),
            "latest_room_sensor_readings_df": latest_readings_df,
            "processing_notes": self.notes
        }

        logger.info(f"({self.module_log_prefix}) Preparation finished. Notes: {len(self.notes)}")
        return output

def prepare_clinic_environmental_detail_data(
    filtered_iot_df: Optional[pd.DataFrame],
    reporting_period_context_str: str 
) -> Dict[str, Any]:
    """
    Public factory function to instantiate and run the ClinicEnvDetailPrep class.
    """
    preparer = ClinicEnvDetailPrep(filtered_iot_df, reporting_period_context_str)
    return preparer.prepare()
