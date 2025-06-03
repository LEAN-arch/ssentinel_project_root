# sentinel_project_root/pages/district_components/trend_analysis.py
# Calculates district-wide health & environmental trend data for Sentinel DHO dashboards.
# Renamed from trend_calculator_district.py

import pandas as pd
import numpy as np # Not directly used for values, but good for type hints
import logging
from typing import Dict, Any, Optional, Union, Callable # Added Union, Callable
from datetime import date as date_type # For type hinting

from config import settings # Use new settings module
from data_processing.aggregation import get_trend_data # Use centralized trend utility
from data_processing.helpers import convert_to_numeric # For data cleaning

logger = logging.getLogger(__name__)


def calculate_district_wide_trends( # Renamed function
    health_df_filtered_for_period: Optional[pd.DataFrame], # Health data already filtered for the trend period
    iot_df_filtered_for_period: Optional[pd.DataFrame],    # IoT data already filtered for the trend period
    trend_start_date_context: Any, # Primarily for logging/context, as data is pre-filtered
    trend_end_date_context: Any,   # Primarily for logging/context
    reporting_period_display_str: str, # Renamed for clarity: For the 'reporting_period' key in the output dict
    disease_incidence_agg_period: str = 'W-MON', # Renamed, default: Weekly (Monday start) for disease incidence
    general_metrics_agg_period: str = 'D'      # Renamed, default: Daily for other general trends
) -> Dict[str, Any]:
    """
    Calculates district-wide health and environmental trends using provided, period-filtered data.
    Leverages the `get_trend_data` utility for robust time-series aggregation.

    Args:
        health_df_filtered_for_period: DataFrame of health records for the trend period.
        iot_df_filtered_for_period: DataFrame of IoT environmental data for the trend period.
        trend_start_date_context, trend_end_date_context: For context/logging; data should be pre-filtered.
        reporting_period_display_str: String for the 'reporting_period' key in output.
        disease_incidence_agg_period: Aggregation period for disease incidence (e.g., 'W-MON', 'M').
        general_metrics_agg_period: Aggregation period for other trends (e.g., 'D', 'W-MON').

    Returns:
        Dict[str, Any]: Dictionary containing trend Series (pd.Series) and processing notes.
                        Keys for trends include "disease_incidence_trends" (a nested dict),
                        "avg_patient_ai_risk_trend", "avg_patient_daily_steps_trend",
                        "avg_clinic_co2_trend".
    """
    module_log_prefix = "DistrictTrendCalc" # Renamed for clarity
    
    # Convert context dates to string for logging, handling potential NaT or None
    start_date_log_str = str(pd.to_datetime(trend_start_date_context, errors='coerce').date()) if trend_start_date_context else "UnknownStart"
    end_date_log_str = str(pd.to_datetime(trend_end_date_context, errors='coerce').date()) if trend_end_date_context else "UnknownEnd"

    logger.info(
        f"({module_log_prefix}) Calculating district trends. Reporting Period: {reporting_period_display_str} "
        f"(Contextual Data Period: {start_date_log_str} to {end_date_log_str})"
    )
    
    # Initialize output structure with empty Series/dict for type consistency
    output_district_trends_data: Dict[str, Any] = {
        "reporting_period": reporting_period_display_str,
        "disease_incidence_trends": {}, # Dict: {condition_display_name: pd.Series of new case counts}
        "avg_patient_ai_risk_trend": pd.Series(dtype='float64'),
        "avg_patient_daily_steps_trend": pd.Series(dtype='float64'),
        "avg_clinic_co2_trend": pd.Series(dtype='float64'), # District-wide average of clinic means from IoT
        "data_availability_notes": []
    }
    
    # --- Data Availability Checks ---
    is_health_data_available_for_trends = isinstance(health_df_filtered_for_period, pd.DataFrame) and not health_df_filtered_for_period.empty
    is_iot_data_available_for_trends = isinstance(iot_df_filtered_for_period, pd.DataFrame) and not iot_df_filtered_for_period.empty

    if not is_health_data_available_for_trends and not is_iot_data_available_for_trends:
        note_msg = "No health or IoT data provided for the selected trend period. Cannot calculate any trends."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        output_district_trends_data["data_availability_notes"].append(note_msg)
        return output_district_trends_data

    # --- 1. Disease Incidence Trends (e.g., New Cases per Week/Month) ---
    if is_health_data_available_for_trends:
        # Ensure required columns for disease trends are present
        if 'condition' in health_df_filtered_for_period.columns and \
           'patient_id' in health_df_filtered_for_period.columns and \
           'encounter_date' in health_df_filtered_for_period.columns:
            
            disease_trends_map_results: Dict[str, pd.Series] = {}
            for condition_name_from_config in settings.KEY_CONDITIONS_FOR_ACTION:
                display_name_for_trend_condition = condition_name_from_config.replace("(Severe)", "").strip() # Cleaner UI label
                
                # Filter for records matching this specific condition (case-insensitive, whole word/common variations)
                # Using regex with word boundaries (\b) for more precise matching of terms like "TB"
                condition_pattern_for_regex = r"\b" + re.escape(condition_name_from_config) + r"\b"
                condition_match_mask = health_df_filtered_for_period['condition'].astype(str).str.contains(
                    condition_pattern_for_regex, case=False, na=False, regex=True
                )
                df_for_this_condition_trend = health_df_filtered_for_period[condition_match_mask]
                
                if not df_for_this_condition_trend.empty:
                    # Assumption: For incidence trend, each unique patient_id within an aggregated period
                    # for a specific condition represents a "newly identified or active case" in that period.
                    # True epidemiological incidence (first-time diagnosis only) requires more complex state tracking.
                    incidence_trend_series_result = get_trend_data(
                        df=df_for_this_condition_trend,
                        value_col='patient_id', # Count unique patients as new/active cases
                        date_col='encounter_date', # This column is already pd.Timestamp from loader/filter
                        period=disease_incidence_agg_period,
                        agg_func='nunique', # Number of unique patients
                        source_context=f"{module_log_prefix}/IncidenceTrend/{display_name_for_trend_condition}"
                    )
                    if isinstance(incidence_trend_series_result, pd.Series) and not incidence_trend_series_result.empty:
                        disease_trends_map_results[display_name_for_trend_condition] = incidence_trend_series_result
                    else:
                        output_district_trends_data["data_availability_notes"].append(
                            f"No trend data generated for '{display_name_for_trend_condition}' incidence (result was empty series)."
                        )
                else: # No records for this specific condition in the filtered period
                     output_district_trends_data["data_availability_notes"].append(
                         f"No records found for condition '{display_name_for_trend_condition}' in the trend period."
                     )
            output_district_trends_data["disease_incidence_trends"] = disease_trends_map_results
        else: # Health data exists, but missing one or more critical columns for disease trends
            missing_cols_for_disease_trends = [
                col for col in ['condition','patient_id','encounter_date'] 
                if col not in health_df_filtered_for_period.columns
            ]
            output_district_trends_data["data_availability_notes"].append(
                f"Health data missing critical columns for disease incidence trends: {missing_cols_for_disease_trends}."
            )
    else: # No health data at all
        output_district_trends_data["data_availability_notes"].append("Health data unavailable for disease incidence trends.")


    # --- 2. Average Patient AI Risk Score Trend ---
    if is_health_data_available_for_trends:
        if 'ai_risk_score' in health_df_filtered_for_period.columns and \
           health_df_filtered_for_period['ai_risk_score'].notna().any() and \
           'encounter_date' in health_df_filtered_for_period.columns:
            
            ai_risk_trend_series_result = get_trend_data(
                df=health_df_filtered_for_period, value_col='ai_risk_score', date_col='encounter_date',
                period=general_metrics_agg_period, agg_func='mean', # Average AI risk score
                source_context=f"{module_log_prefix}/AIRiskScoreTrend"
            )
            if isinstance(ai_risk_trend_series_result, pd.Series) and not ai_risk_trend_series_result.empty:
                output_district_trends_data["avg_patient_ai_risk_trend"] = ai_risk_trend_series_result
            else:
                output_district_trends_data["data_availability_notes"].append("Could not generate AI risk score trend (result was empty series).")
        else:
            output_district_trends_data["data_availability_notes"].append(
                "AI risk score data ('ai_risk_score' or 'encounter_date') missing or all NaN for trend calculation."
            )
    else:
        output_district_trends_data["data_availability_notes"].append("Health data unavailable for AI risk score trend.")


    # --- 3. Average Patient Daily Steps Trend ---
    if is_health_data_available_for_trends:
        if 'avg_daily_steps' in health_df_filtered_for_period.columns and \
           health_df_filtered_for_period['avg_daily_steps'].notna().any() and \
           'encounter_date' in health_df_filtered_for_period.columns:

            daily_steps_trend_series_result = get_trend_data(
                df=health_df_filtered_for_period, value_col='avg_daily_steps', date_col='encounter_date',
                period=general_metrics_agg_period, agg_func='mean', # Average daily steps
                source_context=f"{module_log_prefix}/AvgDailyStepsTrend"
            )
            if isinstance(daily_steps_trend_series_result, pd.Series) and not daily_steps_trend_series_result.empty:
                output_district_trends_data["avg_patient_daily_steps_trend"] = daily_steps_trend_series_result
            else:
                output_district_trends_data["data_availability_notes"].append("Could not generate average daily steps trend (result was empty series).")
        else:
            output_district_trends_data["data_availability_notes"].append(
                "Average daily steps data ('avg_daily_steps' or 'encounter_date') missing or all NaN for trend calculation."
            )
    else:
         output_district_trends_data["data_availability_notes"].append("Health data unavailable for average daily steps trend.")


    # --- 4. Average Clinic CO2 Levels Trend (District-wide average of clinic means from IoT data) ---
    if is_iot_data_available_for_trends:
        if 'avg_co2_ppm' in iot_df_filtered_for_period.columns and \
           iot_df_filtered_for_period['avg_co2_ppm'].notna().any() and \
           'timestamp' in iot_df_filtered_for_period.columns: # IoT data uses 'timestamp'

            avg_co2_trend_series_result = get_trend_data(
                df=iot_df_filtered_for_period, value_col='avg_co2_ppm', date_col='timestamp', # Use 'timestamp' for IoT
                period=general_metrics_agg_period, agg_func='mean', # District-wide average of mean CO2 readings
                source_context=f"{module_log_prefix}/AvgClinicCO2Trend"
            )
            if isinstance(avg_co2_trend_series_result, pd.Series) and not avg_co2_trend_series_result.empty:
                output_district_trends_data["avg_clinic_co2_trend"] = avg_co2_trend_series_result
            else:
                output_district_trends_data["data_availability_notes"].append("Could not generate average clinic CO2 trend (result was empty series).")
        else:
            output_district_trends_data["data_availability_notes"].append(
                "Clinic CO2 data ('avg_co2_ppm' or 'timestamp' in IoT data) missing or all NaN for trend calculation."
            )
    else: # No IoT data
        output_district_trends_data["data_availability_notes"].append("IoT data unavailable for average clinic CO2 trend.")

    # --- Final Logging ---
    num_disease_trends_generated = len(output_district_trends_data.get("disease_incidence_trends", {}))
    num_other_trends_generated = sum(
        1 for trend_key in ["avg_patient_ai_risk_trend", "avg_patient_daily_steps_trend", "avg_clinic_co2_trend"]
        if isinstance(output_district_trends_data.get(trend_key), pd.Series) and \
           not output_district_trends_data[trend_key].empty
    )
    total_trends_generated = num_disease_trends_generated + num_other_trends_generated
    
    logger.info(
        f"({module_log_prefix}) District trends calculation complete. Generated {total_trends_generated} distinct trend series. "
        f"Notes recorded: {len(output_district_trends_data['data_availability_notes'])}"
    )
    return output_district_trends_data
