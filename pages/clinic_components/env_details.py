# sentinel_project_root/pages/clinic_components/env_details.py
# Prepares detailed environmental data and trends for the Sentinel Clinic Dashboard.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    # Use a basic logger for critical import errors, as the full config may not be available.
    logging.basicConfig(level=logging.ERROR)
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.critical(f"Fatal import error in env_details.py: {e}. Check project structure and dependencies.", exc_info=True)
    # Re-raise to prevent the application from starting in a broken state.
    raise

logger = logging.getLogger(__name__)

# --- Constants for Clarity and Maintainability ---
# These constants centralize column names, reducing typos and improving refactorability.
COL_TIMESTAMP = 'timestamp'
COL_ROOM_NAME = 'room_name'
COL_CLINIC_ID = 'clinic_id'
COL_CO2_PPM = 'avg_co2_ppm'
COL_PM25 = 'avg_pm25'
COL_NOISE_DBA = 'avg_noise_dba'
COL_OCCUPANCY = 'occupancy_count'

# Standardized alert levels for consistent use across the application.
ALERT_LEVEL_HIGH = "HIGH_RISK"
ALERT_LEVEL_MODERATE = "MODERATE_CONCERN"
ALERT_LEVEL_ACCEPTABLE = "ACCEPTABLE"


class ClinicEnvDetailPreparer:
    """
    Encapsulates logic for preparing detailed environmental data for the clinic dashboard.

    This class processes a filtered IoT DataFrame to produce actionable alerts,
    time-series trend data, and summary tables, ensuring all data is clean and
    ready for direct consumption by the Streamlit UI.
    """

    def __init__(self, filtered_iot_df: Optional[pd.DataFrame], reporting_period_context_str: str):
        """
        Initializes the preparer with IoT data for a specific period.

        Args:
            filtered_iot_df: The IoT data, pre-filtered for the reporting period.
            reporting_period_context_str: A human-readable string of the report period.
        """
        # A defensive copy prevents modifications to the original DataFrame outside this class.
        self.df_iot = filtered_iot_df.copy() if isinstance(filtered_iot_df, pd.DataFrame) and not filtered_iot_df.empty else pd.DataFrame()
        self.reporting_period = reporting_period_context_str
        self.notes: List[str] = []
        self._validate_and_prepare_input_df()

    @staticmethod
    def _get_setting(attr_name: str, default_value: Any) -> Any:
        """Safely retrieves a configuration value from the global settings object."""
        return getattr(settings, attr_name, default_value)

    def _get_alert_config(self) -> List[Dict[str, Any]]:
        """Dynamically builds the alert configuration from settings for flexibility."""
        return [
            {"metric": COL_CO2_PPM, "threshold": self._get_setting('ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500), "level": ALERT_LEVEL_HIGH, "alert_type": "Very High CO2", "message_template": "{count} area(s) with very high CO2 (> {threshold} ppm)."},
            {"metric": COL_CO2_PPM, "threshold": self._get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1500), "level": ALERT_LEVEL_MODERATE, "alert_type": "Elevated CO2", "message_template": "{count} area(s) with elevated CO2 (> {threshold} ppm)."},
            {"metric": COL_PM25, "threshold": self._get_setting('ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 50), "level": ALERT_LEVEL_HIGH, "alert_type": "Poor Air Quality", "message_template": "Poor Air Quality: PM2.5 > {threshold} µg/m³ in {count} area(s)."},
            {"metric": COL_PM25, "threshold": self._get_setting('ALERT_AMBIENT_PM25_HIGH_UGM3', 35), "level": ALERT_LEVEL_MODERATE, "alert_type": "Moderate Air Quality", "message_template": "Elevated PM2.5: > {threshold} µg/m³ in {count} area(s)."},
            {"metric": COL_NOISE_DBA, "threshold": self._get_setting('ALERT_AMBIENT_NOISE_HIGH_DBA', 70), "level": ALERT_LEVEL_MODERATE, "alert_type": "High Noise Level", "message_template": "High noise levels (> {threshold} dBA) detected in {count} area(s)."},
        ]

    def _validate_and_prepare_input_df(self):
        """
        Performs essential validation, cleaning, and type conversion on the input DataFrame.
        This method ensures data integrity before any further processing occurs.
        """
        if self.df_iot.empty:
            self.notes.append(f"No IoT data was available for the period '{self.reporting_period}'.")
            return

        if COL_TIMESTAMP not in self.df_iot.columns:
            self.notes.append(f"Data Integrity Issue: The required '{COL_TIMESTAMP}' column is missing from IoT data.")
            self.df_iot = pd.DataFrame()  # Invalidate DataFrame to halt further processing.
            return

        try:
            # Standardize timestamp column for reliable, timezone-naive processing.
            self.df_iot[COL_TIMESTAMP] = pd.to_datetime(self.df_iot[COL_TIMESTAMP], errors='coerce')
            self.df_iot.dropna(subset=[COL_TIMESTAMP], inplace=True)
            if self.df_iot.empty:
                self.notes.append("No IoT records with valid timestamps were found after cleaning.")
        except Exception as e:
            logger.error(f"Error during timestamp standardization: {e}", exc_info=True)
            self.notes.append("A technical error occurred during data preparation; results may be incomplete.")
            self.df_iot = pd.DataFrame()  # Invalidate on critical error.

    def _get_latest_room_readings(self) -> pd.DataFrame:
        """Finds the single most recent valid sensor reading for each unique room."""
        if self.df_iot.empty:
            return pd.DataFrame()

        # Ensure key columns for identification exist.
        required_keys = [COL_CLINIC_ID, COL_ROOM_NAME, COL_TIMESTAMP]
        if not all(col in self.df_iot.columns for col in required_keys):
            self.notes.append("Cannot determine latest readings; key columns (e.g., clinic_id, room_name) are missing.")
            return pd.DataFrame()

        try:
            latest = self.df_iot.sort_values(COL_TIMESTAMP).drop_duplicates(subset=[COL_CLINIC_ID, COL_ROOM_NAME], keep='last')
            
            # Select and order columns for a clean, predictable display in the dashboard UI.
            display_cols = [COL_ROOM_NAME, COL_TIMESTAMP, COL_CO2_PPM, COL_PM25, COL_NOISE_DBA, COL_OCCUPANCY]
            existing_cols = [col for col in display_cols if col in latest.columns]
            
            return latest[existing_cols].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error isolating latest room readings: {e}", exc_info=True)
            self.notes.append("Failed to generate the latest sensor readings summary.")
            return pd.DataFrame()

    def _generate_environmental_alerts(self, latest_readings_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generates a list of actionable alerts based on the latest readings against configured thresholds."""
        if latest_readings_df.empty:
            return []
            
        alerts = []
        # Sort config to check for highest thresholds first for each metric.
        alert_config = sorted(self._get_alert_config(), key=lambda x: (x['metric'], x['threshold']), reverse=True)
        
        # Tracks metrics for which an alert has been generated to avoid duplicate-category alerts (e.g., high and very high).
        processed_metrics = set()

        for config in alert_config:
            metric = config['metric']
            if metric in processed_metrics or metric not in latest_readings_df.columns:
                continue

            numeric_series = convert_to_numeric(latest_readings_df[metric])
            # Use boolean indexing on the original DataFrame to find rows exceeding the threshold.
            alert_df = latest_readings_df.loc[numeric_series > config['threshold']]

            if not alert_df.empty:
                count = len(alert_df)
                message = config['message_template'].format(count=count, threshold=config['threshold'])
                alerts.append({
                    "message": message,
                    "level": config['level'],
                    "alert_type": config['alert_type']
                })
                processed_metrics.add(metric)

        if not alerts:
            alerts.append({"message": "All monitored environmental parameters appear within acceptable limits.", "level": ALERT_LEVEL_ACCEPTABLE, "alert_type": "System Normal"})
        
        return alerts

    def _prepare_co2_trend_series(self) -> pd.Series:
        """Calculates and returns the hourly average CO2 trend as a pandas Series, ready for plotting."""
        if self.df_iot.empty or COL_CO2_PPM not in self.df_iot.columns or self.df_iot[COL_CO2_PPM].notna().sum() == 0:
            self.notes.append("CO2 trend data is unavailable for this period.")
            return pd.Series(dtype=np.float64) # Return empty series to prevent downstream errors.
            
        try:
            return get_trend_data(df=self.df_iot, value_col=COL_CO2_PPM, date_col=COL_TIMESTAMP, period='H', agg_func='mean')
        except Exception as e:
            logger.error(f"Error generating CO2 trend series: {e}", exc_info=True)
            self.notes.append("Failed to process hourly CO2 trend data due to a technical error.")
            return pd.Series(dtype=np.float64)

    def prepare(self) -> Dict[str, Any]:
        """
        Orchestrates the preparation of all environmental detail components.

        Returns:
            A structured dictionary containing data ready for the UI to consume.
        """
        logger.info(f"Starting environmental detail preparation for period: {self.reporting_period}")

        latest_readings_df = self._get_latest_room_readings()
        
        output = {
            "current_environmental_alerts_list": self._generate_environmental_alerts(latest_readings_df),
            "hourly_avg_co2_trend": self._prepare_co2_trend_series(),
            "latest_room_sensor_readings_df": latest_readings_df,
            "processing_notes": self.notes,
        }

        logger.info(f"Environmental detail preparation complete. Generated {len(self.notes)} processing notes.")
        return output


def prepare_clinic_environmental_detail_data(
    filtered_iot_df: Optional[pd.DataFrame],
    iot_data_source_is_generally_available: bool,
    reporting_period_context_str: str
) -> Dict[str, Any]:
    """
    Public factory function to instantiate and run the ClinicEnvDetailPreparer.

    This function serves as the clean, public-facing entry point for the dashboard page,
    abstracting away the internal class implementation.

    Args:
        filtered_iot_df: A DataFrame of IoT data already filtered for the desired time period.
        iot_data_source_is_generally_available: A flag indicating if the source file exists/loaded.
        reporting_period_context_str: A string describing the reporting period (e.g., "01 Jan 2023 - 31 Jan 2023").

    Returns:
        A dictionary containing structured data for the environment details tab.
    """
    preparer = ClinicEnvDetailPreparer(filtered_iot_df, reporting_period_context_str)
    prepared_data = preparer.prepare()
    
    # Add a final, high-level contextual note if the source itself is the problem.
    if not iot_data_source_is_generally_available:
        note = "The primary IoT data source appears to be unavailable or empty."
        if note not in prepared_data["processing_notes"]:
             prepared_data["processing_notes"].append(note)
        
    return prepared_data
