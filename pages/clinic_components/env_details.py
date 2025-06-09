# sentinel_project_root/pages/clinic_components/env_details.py
# SME-EVALUATED AND REVISED VERSION
# This version includes significant performance optimization for finding latest readings
# and a clearer, more robust, vectorized approach for generating alerts.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# --- Sentinel System Imports ---
try:
    from config import settings
    # Assuming the imported get_trend_data is robust. If not, a self-contained version
    # like the one in 02_clinic_dashboard.py would be necessary.
    from data_processing.aggregation import get_trend_data
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.critical(f"Fatal import error in env_details.py: {e}. Check project structure and dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Constants for Clarity and Maintainability ---
COL_TIMESTAMP = 'timestamp'
COL_ROOM_NAME = 'room_name'
COL_CLINIC_ID = 'clinic_id'
COL_CO2_PPM = 'avg_co2_ppm'
COL_PM25 = 'avg_pm25'
COL_NOISE_DBA = 'avg_noise_dba'
COL_OCCUPANCY = 'occupancy_count'

ALERT_LEVEL_HIGH = "HIGH_RISK"
ALERT_LEVEL_MODERATE = "MODERATE_CONCERN"
ALERT_LEVEL_ACCEPTABLE = "ACCEPTABLE"


class ClinicEnvDetailPreparer:
    """
    Encapsulates logic for preparing detailed environmental data for the clinic dashboard.
    This class processes a filtered IoT DataFrame to produce actionable alerts,
    time-series trend data, and summary tables.
    """

    def __init__(self, filtered_iot_df: Optional[pd.DataFrame], reporting_period_context_str: str):
        """
        Initializes the preparer with IoT data for a specific period.
        """
        self.df_iot = filtered_iot_df.copy() if isinstance(filtered_iot_df, pd.DataFrame) and not filtered_iot_df.empty else pd.DataFrame()
        self.reporting_period = reporting_period_context_str
        self.notes: List[str] = []
        self._validate_and_prepare_input_df()

    @staticmethod
    def _get_setting(attr_name: str, default_value: Any) -> Any:
        """Safely retrieves a configuration value from the global settings object."""
        return getattr(settings, attr_name, default_value)

    def _validate_and_prepare_input_df(self):
        """
        Performs essential validation, cleaning, and type conversion on the input DataFrame.
        """
        if self.df_iot.empty:
            self.notes.append(f"No IoT data was available for the period '{self.reporting_period}'.")
            return

        if COL_TIMESTAMP not in self.df_iot.columns:
            self.notes.append(f"Data Integrity Issue: Required '{COL_TIMESTAMP}' column is missing.")
            self.df_iot = pd.DataFrame()
            return

        try:
            self.df_iot[COL_TIMESTAMP] = pd.to_datetime(self.df_iot[COL_TIMESTAMP], errors='coerce')
            self.df_iot.dropna(subset=[COL_TIMESTAMP], inplace=True)
            if self.df_iot.empty:
                self.notes.append("No IoT records with valid timestamps were found after cleaning.")
        except Exception as e:
            logger.error(f"Error during timestamp standardization: {e}", exc_info=True)
            self.notes.append("A technical error occurred during data preparation.")
            self.df_iot = pd.DataFrame()

    def _get_latest_room_readings(self) -> pd.DataFrame:
        """
        Finds the single most recent valid sensor reading for each unique room.
        This method is performance-optimized using `groupby().idxmax()`.
        """
        if self.df_iot.empty:
            return pd.DataFrame()

        required_keys = [COL_CLINIC_ID, COL_ROOM_NAME, COL_TIMESTAMP]
        if not all(col in self.df_iot.columns for col in required_keys):
            self.notes.append("Cannot determine latest readings; key columns like clinic_id or room_name are missing.")
            return pd.DataFrame()

        try:
            # PERFORMANCE OPTIMIZATION: Use idxmax() to find the index of the latest entry
            # per group. This is significantly faster than sorting the entire DataFrame.
            latest_indices = self.df_iot.groupby([COL_CLINIC_ID, COL_ROOM_NAME])[COL_TIMESTAMP].idxmax()
            latest = self.df_iot.loc[latest_indices]

            display_cols = [COL_ROOM_NAME, COL_TIMESTAMP, COL_CO2_PPM, COL_PM25, COL_NOISE_DBA, COL_OCCUPANCY]
            existing_cols = [col for col in display_cols if col in latest.columns]
            
            return latest[existing_cols].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error isolating latest room readings: {e}", exc_info=True)
            self.notes.append("Failed to generate the latest sensor readings summary.")
            return pd.DataFrame()

    def _generate_environmental_alerts(self, latest_readings_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates alerts based on latest readings using a clear, vectorized approach.
        This method categorizes each room's status and then aggregates the results.
        """
        if latest_readings_df.empty:
            return []

        alerts = []
        df = latest_readings_df.copy()

        # Define alert configurations
        alert_metrics = {
            COL_CO2_PPM: {
                "thresholds": [self._get_setting('ALERT_AMBIENT_CO2_HIGH_PPM', 1500), self._get_setting('ALERT_AMBIENT_CO2_VERY_HIGH_PPM', 2500)],
                "messages": {
                    ALERT_LEVEL_MODERATE: "{count} area(s) with elevated CO2 (> {threshold} ppm).",
                    ALERT_LEVEL_HIGH: "{count} area(s) with very high CO2 (> {threshold} ppm)."
                },
                "alert_types": {ALERT_LEVEL_MODERATE: "Elevated CO2", ALERT_LEVEL_HIGH: "Very High CO2"}
            },
            COL_PM25: {
                "thresholds": [self._get_setting('ALERT_AMBIENT_PM25_HIGH_UGM3', 35), self._get_setting('ALERT_AMBIENT_PM25_VERY_HIGH_UGM3', 50)],
                "messages": {
                    ALERT_LEVEL_MODERATE: "Elevated PM2.5: > {threshold} µg/m³ in {count} area(s).",
                    ALERT_LEVEL_HIGH: "Poor Air Quality: PM2.5 > {threshold} µg/m³ in {count} area(s)."
                },
                "alert_types": {ALERT_LEVEL_MODERATE: "Moderate Air Quality", ALERT_LEVEL_HIGH: "Poor Air Quality"}
            },
            COL_NOISE_DBA: {
                "thresholds": [self._get_setting('ALERT_AMBIENT_NOISE_HIGH_DBA', 70)],
                "messages": {ALERT_LEVEL_MODERATE: "High noise levels (> {threshold} dBA) detected in {count} area(s)."},
                "alert_types": {ALERT_LEVEL_MODERATE: "High Noise Level"}
            },
        }

        # Vectorized categorization of alerts
        for metric, config in alert_metrics.items():
            if metric not in df.columns: continue
            
            series = convert_to_numeric(df[metric])
            thresholds = config['thresholds']
            
            conditions = [series > t for t in reversed(thresholds)]
            choices = [ALERT_LEVEL_HIGH, ALERT_LEVEL_MODERATE][:len(thresholds)]
            
            status_col = f"{metric}_status"
            df[status_col] = np.select(conditions, reversed(choices), default=ALERT_LEVEL_ACCEPTABLE)
            
            status_counts = df[status_col].value_counts()

            # Generate alert messages based on counts
            for level, count in status_counts.items():
                if level != ALERT_LEVEL_ACCEPTABLE and count > 0:
                    threshold_index = choices.index(level) if level in choices else -1
                    alerts.append({
                        "message": config['messages'][level].format(count=count, threshold=thresholds[threshold_index]),
                        "level": level,
                        "alert_type": config['alert_types'][level]
                    })
        
        if not alerts:
            alerts.append({"message": "All monitored environmental parameters are within acceptable limits.", "level": ALERT_LEVEL_ACCEPTABLE, "alert_type": "System Normal"})
        
        # Sort alerts by severity for display
        level_order = {ALERT_LEVEL_HIGH: 0, ALERT_LEVEL_MODERATE: 1, ALERT_LEVEL_ACCEPTABLE: 2}
        return sorted(alerts, key=lambda x: level_order.get(x['level'], 99))

    def _prepare_co2_trend_series(self) -> pd.Series:
        """Calculates the hourly average CO2 trend, ready for plotting."""
        if self.df_iot.empty or COL_CO2_PPM not in self.df_iot.columns or self.df_iot[COL_CO2_PPM].notna().sum() == 0:
            self.notes.append("CO2 trend data is unavailable for this period.")
            return pd.Series(dtype=np.float64)
            
        try:
            return get_trend_data(df=self.df_iot, value_col=COL_CO2_PPM, date_col=COL_TIMESTAMP, period='H', agg_func='mean')
        except Exception as e:
            logger.error(f"Error generating CO2 trend series: {e}", exc_info=True)
            self.notes.append("Failed to process hourly CO2 trend data.")
            return pd.Series(dtype=np.float64)

    def prepare(self) -> Dict[str, Any]:
        """
        Orchestrates the preparation of all environmental detail components.
        """
        logger.info(f"Starting environmental detail preparation for period: {self.reporting_period}")

        latest_readings_df = self._get_latest_room_readings()
        
        output = {
            "current_environmental_alerts_list": self._generate_environmental_alerts(latest_readings_df),
            "hourly_avg_co2_trend": self._prepare_co2_trend_series(),
            "latest_room_sensor_readings_df": latest_readings_df,
            "processing_notes": list(set(self.notes)), # Ensure unique notes
        }

        logger.info(f"Environmental detail preparation complete. Generated {len(output['processing_notes'])} notes.")
        return output


def prepare_clinic_environmental_detail_data(
    filtered_iot_df: Optional[pd.DataFrame],
    iot_data_source_is_generally_available: bool,
    reporting_period_context_str: str
) -> Dict[str, Any]:
    """
    Public factory function to instantiate and run the ClinicEnvDetailPreparer.
    """
    preparer = ClinicEnvDetailPreparer(filtered_iot_df, reporting_period_context_str)
    prepared_data = preparer.prepare()
    
    if not iot_data_source_is_generally_available:
        note = "The primary IoT data source appears to be unavailable or empty."
        if note not in prepared_data["processing_notes"]:
             prepared_data["processing_notes"].append(note)
        
    return prepared_data
