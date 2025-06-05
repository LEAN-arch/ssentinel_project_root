# sentinel_project_root/pages/district_components/trend_analysis.py
# Calculates district-wide health & environmental trend data for Sentinel DHO dashboards.

import pandas as pd
import numpy as np
import logging
import re # For condition matching
from typing import Dict, Any, Optional, Union, Callable 
from datetime import date as date_type # For type hinting

from config import settings
from data_processing.aggregation import get_trend_data
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def calculate_district_wide_trends(
    health_df_filtered_for_period: Optional[pd.DataFrame],
    iot_df_filtered_for_period: Optional[pd.DataFrame],
    trend_start_date_context: Any, # Contextual, data should be pre-filtered
    trend_end_date_context: Any,   # Contextual
    reporting_period_display_str: str,
    disease_incidence_agg_period: str = 'W-MON', # Weekly for disease incidence
    general_metrics_agg_period: str = 'D'      # Daily for other general trends
) -> Dict[str, Any]:
    """
    Calculates district-wide health and environmental trends using pre-filtered data.
    """
    module_log_prefix = "DistrictTrendCalc"
    start_log = str(pd.to_datetime(trend_start_date_context, errors='coerce').date()) if trend_start_date_context else "UnknownStart"
    end_log = str(pd.to_datetime(trend_end_date_context, errors='coerce').date()) if trend_end_date_context else "UnknownEnd"
    logger.info(f"({module_log_prefix}) Calculating district trends. Reporting Period: {reporting_period_display_str} (Data Context: {start_log} to {end_log})")
    
    output_trends: Dict[str, Any] = {
        "reporting_period": reporting_period_display_str, "disease_incidence_trends": {},
        "avg_patient_ai_risk_trend": pd.Series(dtype='float64'), "avg_patient_daily_steps_trend": pd.Series(dtype='float64'),
        "avg_clinic_co2_trend": pd.Series(dtype='float64'), "data_availability_notes": []
    }
    
    health_ok = isinstance(health_df_filtered_for_period, pd.DataFrame) and not health_df_filtered_for_period.empty
    iot_ok = isinstance(iot_df_filtered_for_period, pd.DataFrame) and not iot_df_filtered_for_period.empty

    if not health_ok and not iot_ok:
        note = "No health or IoT data for selected trend period. Cannot calculate trends."
        logger.warning(f"({module_log_prefix}) {note}"); output_trends["data_availability_notes"].append(note)
        return output_trends

    # Disease Incidence Trends
    if health_ok:
        df_health = health_df_filtered_for_period # Already filtered for period
        if all(c in df_health.columns for c in ['condition', 'patient_id', 'encounter_date']):
            disease_trends: Dict[str, pd.Series] = {}
            for cond_name_cfg in settings.KEY_CONDITIONS_FOR_ACTION:
                disp_name_cond = cond_name_cfg.replace("(Severe)", "").strip()
                # More precise regex matching: \b for word boundaries
                cond_pattern_re = r"\b" + re.escape(cond_name_cfg) + r"\b"
                cond_mask_re = df_health['condition'].astype(str).str.contains(cond_pattern_re, case=False, na=False, regex=True)
                df_cond_trend = df_health[cond_mask_re]
                if not df_cond_trend.empty:
                    inc_trend = get_trend_data(df=df_cond_trend, value_col='patient_id', date_col='encounter_date',
                                               period=disease_incidence_agg_period, agg_func='nunique',
                                               source_context=f"{module_log_prefix}/Incidence/{disp_name_cond}")
                    if isinstance(inc_trend, pd.Series) and not inc_trend.empty: disease_trends[disp_name_cond] = inc_trend
                    else: output_trends["data_availability_notes"].append(f"No trend data for '{disp_name_cond}' incidence (empty series).")
                else: output_trends["data_availability_notes"].append(f"No records for '{disp_name_cond}' in trend period.")
            output_trends["disease_incidence_trends"] = disease_trends
        else:
            missing_cols_dis = [c for c in ['condition','patient_id','encounter_date'] if c not in df_health.columns]
            output_trends["data_availability_notes"].append(f"Health data missing cols for disease trends: {missing_cols_dis}.")
    else: output_trends["data_availability_notes"].append("Health data unavailable for disease incidence trends.")

    # Avg Patient AI Risk Score Trend
    if health_ok:
        if all(c in df_health.columns for c in ['ai_risk_score', 'encounter_date']) and df_health['ai_risk_score'].notna().any():
            risk_trend = get_trend_data(df=df_health, value_col='ai_risk_score', date_col='encounter_date',
                                        period=general_metrics_agg_period, agg_func='mean', source_context=f"{module_log_prefix}/AIRiskTrend")
            if isinstance(risk_trend, pd.Series) and not risk_trend.empty: output_trends["avg_patient_ai_risk_trend"] = risk_trend
            else: output_trends["data_availability_notes"].append("Could not generate AI risk score trend (empty series).")
        else: output_trends["data_availability_notes"].append("AI risk score or encounter_date data missing/all NaN for trend.")
    else: output_trends["data_availability_notes"].append("Health data unavailable for AI risk score trend.")

    # Avg Patient Daily Steps Trend
    if health_ok:
        if all(c in df_health.columns for c in ['avg_daily_steps', 'encounter_date']) and df_health['avg_daily_steps'].notna().any():
            steps_trend = get_trend_data(df=df_health, value_col='avg_daily_steps', date_col='encounter_date',
                                         period=general_metrics_agg_period, agg_func='mean', source_context=f"{module_log_prefix}/AvgStepsTrend")
            if isinstance(steps_trend, pd.Series) and not steps_trend.empty: output_trends["avg_patient_daily_steps_trend"] = steps_trend
            else: output_trends["data_availability_notes"].append("Could not generate avg daily steps trend (empty series).")
        else: output_trends["data_availability_notes"].append("Avg daily steps or encounter_date data missing/all NaN for trend.")
    else: output_trends["data_availability_notes"].append("Health data unavailable for avg daily steps trend.")

    # Avg Clinic CO2 Levels Trend
    if iot_ok:
        df_iot = iot_df_filtered_for_period # Already filtered
        if all(c in df_iot.columns for c in ['avg_co2_ppm', 'timestamp']) and df_iot['avg_co2_ppm'].notna().any():
            co2_trend_iot = get_trend_data(df=df_iot, value_col='avg_co2_ppm', date_col='timestamp',
                                       period=general_metrics_agg_period, agg_func='mean', source_context=f"{module_log_prefix}/AvgCO2Trend")
            if isinstance(co2_trend_iot, pd.Series) and not co2_trend_iot.empty: output_trends["avg_clinic_co2_trend"] = co2_trend_iot
            else: output_trends["data_availability_notes"].append("Could not generate avg clinic CO2 trend (empty series).")
        else: output_trends["data_availability_notes"].append("Clinic CO2 ('avg_co2_ppm' or 'timestamp') data missing/all NaN for trend.")
    else: output_trends["data_availability_notes"].append("IoT data unavailable for avg clinic CO2 trend.")

    num_dis_trends = len(output_trends.get("disease_incidence_trends", {}))
    num_other_trends = sum(1 for k in ["avg_patient_ai_risk_trend", "avg_patient_daily_steps_trend", "avg_clinic_co2_trend"] if isinstance(output_trends.get(k), pd.Series) and not output_trends[k].empty)
    logger.info(f"({module_log_prefix}) District trends calculation complete. Generated {num_dis_trends + num_other_trends} trends. Notes: {len(output_trends['data_availability_notes'])}")
    return output_trends
