# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Generates supply forecast overview data for the Clinic Console.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, timedelta, datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__) 
    logger_init.error(f"Critical import error in supply_forecast.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Helper to safely get attributes from settings."""
    return getattr(settings, attr_name, default_value)


def generate_simple_supply_forecast(
    df_historical_consumption: pd.DataFrame,
    forecast_horizon_days: int,
    items_to_forecast: List[str],
    log_prefix: str = "SimpleSupplyForecast"
) -> pd.DataFrame:
    """
    A simplified supply forecasting model.
    Estimates future stock based on latest stock and average daily consumption.
    """
    logger.info(f"({log_prefix}) Generating simple forecast. Horizon: {forecast_horizon_days} days, Items: {items_to_forecast}")
    
    output_columns = [
        'item', 'current_stock_level', 'avg_daily_consumption_rate', 
        'days_of_supply_remaining', 'estimated_stockout_date', 
        'forecast_date_horizon'
    ]

    if not isinstance(df_historical_consumption, pd.DataFrame) or df_historical_consumption.empty or not items_to_forecast:
        return pd.DataFrame(columns=output_columns)

    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in df_historical_consumption.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_historical_consumption.columns]
        logger.error(f"({log_prefix}) Missing required columns for forecasting: {missing}. Aborting.")
        return pd.DataFrame(columns=output_columns)

    df_hist = df_historical_consumption.copy()

    try:
        df_hist['encounter_date'] = pd.to_datetime(df_hist['encounter_date'], errors='coerce')
        df_hist.dropna(subset=['encounter_date', 'item'], inplace=True)
        df_hist['item_stock_agg_zone'] = convert_to_numeric(df_hist['item_stock_agg_zone'], default_value=0.0)
        df_hist['consumption_rate_per_day'] = convert_to_numeric(df_hist['consumption_rate_per_day'], default_value=0.001)
        df_hist.loc[df_hist['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001
    except Exception as e:
        logger.error(f"({log_prefix}) Error preparing historical data: {e}", exc_info=True)
        return pd.DataFrame(columns=output_columns)

    if df_hist.empty:
        return pd.DataFrame(columns=output_columns)
        
    latest_item_records = df_hist.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    
    forecast_results_list = []
    today = pd.Timestamp.now().normalize()

    for item_name in items_to_forecast:
        latest_item_record = latest_item_records[latest_item_records['item'] == item_name]
        
        if latest_item_record.empty:
            forecast_results_list.append({
                "item": item_name, "current_stock_level": 0.0, "avg_daily_consumption_rate": 0.0,
                "days_of_supply_remaining": 0.0, "estimated_stockout_date": "N/A (No History)", "forecast_date_horizon": "N/A"
            })
            continue

        latest_record = latest_item_record.iloc[0]
        last_recorded_stock = float(latest_record['item_stock_agg_zone'])
        last_update_date = latest_record['encounter_date']
        
        avg_daily_consumption = float(df_hist[df_hist['item'] == item_name]['consumption_rate_per_day'].mean())
        if avg_daily_consumption <= 0:
            avg_daily_consumption = 0.001

        # CORRECTED LOGIC: Project stock to today before calculating DOS.
        days_since_update = (today - last_update_date).days
        consumption_since_update = max(0, days_since_update) * avg_daily_consumption
        current_stock_level = max(0, last_recorded_stock - consumption_since_update)

        days_of_supply = current_stock_level / avg_daily_consumption if avg_daily_consumption > 0 else float('inf')
        
        stockout_date_str = "N/A"
        if np.isfinite(days_of_supply):
            try:
                # Calculate from today, not from the last record date.
                stockout_date = today + pd.to_timedelta(days_of_supply, unit='D')
                stockout_date_str = stockout_date.strftime('%Y-%m-%d')
            except (OverflowError, ValueError): 
                stockout_date_str = ">5Y"
        else:
            stockout_date_str = "N/A (>5Y)"

        forecast_horizon_date_str = (today + pd.Timedelta(days=forecast_horizon_days)).strftime('%Y-%m-%d')
        
        forecast_results_list.append({
            "item": item_name,
            "current_stock_level": round(current_stock_level, 1),
            "avg_daily_consumption_rate": round(avg_daily_consumption, 3),
            "days_of_supply_remaining": round(days_of_supply, 1) if np.isfinite(days_of_supply) else "N/A",
            "estimated_stockout_date": stockout_date_str,
            "forecast_date_horizon": forecast_horizon_date_str
        })
    
    return pd.DataFrame(forecast_results_list, columns=output_columns)


def prepare_clinic_supply_forecast_overview_data(
    full_historical_health_df: Optional[pd.DataFrame],
    current_period_context_str: str, 
    use_ai_supply_forecasting_model: bool = False,
    log_prefix: str = "ClinicSupplyForecastPrep" 
) -> Dict[str, Any]:
    """
    Prepares an overview of supply forecasts.
    """
    forecast_days_out = int(_get_setting('DEFAULT_SUPPLY_FORECAST_HORIZON_DAYS', 30))
    forecast_output: Dict[str, Any] = {
        "forecast_items_overview_list": [], 
        "forecast_model_type_used": "AI Advanced (Simulated)" if use_ai_supply_forecasting_model else "Simple Aggregate",
        "forecast_horizon_days": forecast_days_out,
        "data_processing_notes": []
    }

    if not isinstance(full_historical_health_df, pd.DataFrame) or full_historical_health_df.empty:
        forecast_output["data_processing_notes"].append("Historical health data is empty; cannot generate forecasts.")
        return forecast_output

    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in full_historical_health_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in full_historical_health_df.columns]
        forecast_output["data_processing_notes"].append(f"Missing required columns for forecast: {missing}.")
        return forecast_output

    key_drug_substrings = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if not key_drug_substrings:
        forecast_output["data_processing_notes"].append("KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings.")
        return forecast_output

    key_drug_pattern = '|'.join(re.escape(s.strip()) for s in key_drug_substrings if s.strip())
    if not key_drug_pattern:
        forecast_output["data_processing_notes"].append("KEY_DRUG_SUBSTRINGS_SUPPLY contains only empty strings.")
        return forecast_output

    all_items_in_history = full_historical_health_df['item'].dropna().unique()
    items_to_forecast = [item for item in all_items_in_history if re.search(key_drug_pattern, item, re.IGNORECASE)]

    if not items_to_forecast:
        forecast_output["data_processing_notes"].append("No items matching configured KEY_DRUG_SUBSTRINGS_SUPPLY found in data.")
        return forecast_output

    forecast_detail_df = pd.DataFrame()
    if use_ai_supply_forecasting_model:
        forecast_output["data_processing_notes"].append("AI Advanced Forecasting Model is currently simulated.")
        logger.warning(f"({log_prefix}) AI forecasting selected; using simple forecast as placeholder.")
    
    try:
        forecast_detail_df = generate_simple_supply_forecast(
            full_historical_health_df, forecast_days_out, items_to_forecast, log_prefix
        )
    except Exception as e:
        logger.error(f"({log_prefix}) Error in forecast generation: {e}", exc_info=True)
        forecast_output["data_processing_notes"].append(f"Error in forecast generation: {e}")

    if not forecast_detail_df.empty:
        if 'days_of_supply_remaining' in forecast_detail_df.columns:
            def dos_to_numeric(dos):
                return pd.to_numeric(dos, errors='coerce') if not isinstance(dos, str) else np.inf

            dos_numeric = forecast_detail_df['days_of_supply_remaining'].apply(dos_to_numeric)
            critical_thresh = _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
            warning_thresh = _get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)

            conditions = [dos_numeric < critical_thresh, dos_numeric < warning_thresh, dos_numeric >= warning_thresh]
            choices = ["Critical Low", "Warning Low", "Sufficient"]
            forecast_detail_df['stock_status'] = np.select(conditions, choices, default="Unknown")
        
        forecast_output["forecast_items_overview_list"] = forecast_detail_df.to_dict('records')

    logger.info(f"({log_prefix}) Supply forecast prep complete. Items in overview: {len(forecast_output['forecast_items_overview_list'])}")
    return forecast_output
