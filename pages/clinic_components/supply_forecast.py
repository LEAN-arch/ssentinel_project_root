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

# Helper to safely get attributes from settings
def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)


# --- Placeholder/Simple Forecasting Model ---
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
        logger.warning(f"({log_prefix}) Insufficient historical data or no items specified for forecasting.")
        return pd.DataFrame(columns=output_columns)

    # Ensure required columns are present
    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in df_historical_consumption.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_historical_consumption.columns]
        logger.error(f"({log_prefix}) Missing required columns for forecasting: {missing}. Aborting simple forecast.")
        return pd.DataFrame(columns=output_columns)

    df_hist = df_historical_consumption.copy()

    # Prepare data types
    try:
        df_hist['encounter_date'] = pd.to_datetime(df_hist['encounter_date'], errors='coerce')
        # CORRECTED: Ensure datetime is timezone-naive before calculations
        if pd.api.types.is_datetime64_any_dtype(df_hist['encounter_date']) and df_hist['encounter_date'].dt.tz is not None:
            df_hist['encounter_date'] = df_hist['encounter_date'].dt.tz_localize(None)

        df_hist.dropna(subset=['encounter_date', 'item'], inplace=True) # Essential for grouping and latest record
        df_hist['item_stock_agg_zone'] = convert_to_numeric(df_hist['item_stock_agg_zone'], default_value=0.0)
        df_hist['consumption_rate_per_day'] = convert_to_numeric(df_hist['consumption_rate_per_day'], default_value=0.001) # Small default to avoid /0
        df_hist.loc[df_hist['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001
    except Exception as e_prep:
        logger.error(f"({log_prefix}) Error preparing historical data for simple forecast: {e_prep}", exc_info=True)
        return pd.DataFrame(columns=output_columns)

    if df_hist.empty:
        logger.warning(f"({log_prefix}) No valid historical data after cleaning for forecasting.")
        return pd.DataFrame(columns=output_columns)
        
    latest_item_records = df_hist.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    
    forecast_results_list = []
    today = pd.Timestamp.now().normalize() # This is already tz-naive

    for item_name in items_to_forecast:
        latest_item_record = latest_item_records[latest_item_records['item'] == item_name]
        
        if latest_item_record.empty:
            logger.debug(f"({log_prefix}) No historical data found for item: {item_name}")
            forecast_results_list.append({
                "item": item_name, "current_stock_level": 0.0,
                "avg_daily_consumption_rate": 0.0, "days_of_supply_remaining": 0.0,
                "estimated_stockout_date": "N/A (No History)", 
                "forecast_date_horizon": "N/A"
            })
            continue

        latest_record_series = latest_item_record.iloc[0]
        last_recorded_stock = float(latest_record_series['item_stock_agg_zone'])
        last_update_date = latest_record_series['encounter_date'] # This is now tz-naive
        
        item_specific_hist_df = df_hist[df_hist['item'] == item_name]
        avg_daily_item_consumption = float(item_specific_hist_df['consumption_rate_per_day'].mean())
        if avg_daily_item_consumption <= 0: avg_daily_item_consumption = 0.001

        days_since_update = (today - last_update_date).days
        current_item_stock = max(0, last_recorded_stock - (days_since_update * avg_daily_item_consumption))
        
        days_of_supply_val = current_item_stock / avg_daily_item_consumption if avg_daily_item_consumption > 0 else float('inf')
        
        estimated_stockout_date_str = "N/A"
        if np.isfinite(days_of_supply_val):
            try:
                stockout_datetime = today + pd.to_timedelta(days_of_supply_val, unit='D')
                estimated_stockout_date_str = stockout_datetime.strftime('%Y-%m-%d')
            except (OverflowError, ValueError): 
                estimated_stockout_date_str = ">5Y"
        else:
             estimated_stockout_date_str = "N/A (>5Y)"

        forecast_date_horizon_str = (today + pd.Timedelta(days=forecast_horizon_days)).strftime('%Y-%m-%d')
        
        forecast_results_list.append({
            "item": item_name,
            "current_stock_level": round(current_item_stock, 1),
            "avg_daily_consumption_rate": round(avg_daily_item_consumption, 3),
            "days_of_supply_remaining": round(days_of_supply_val, 1) if np.isfinite(days_of_supply_val) else "N/A (>5Y)",
            "estimated_stockout_date": estimated_stockout_date_str,
            "forecast_date_horizon": forecast_date_horizon_str
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
        forecast_output["data_processing_notes"].append("Historical health data is empty. Cannot generate supply forecasts.")
        return forecast_output

    required_supply_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in full_historical_health_df.columns for col in required_supply_cols):
        missing = [col for col in required_supply_cols if col not in full_historical_health_df.columns]
        forecast_output["data_processing_notes"].append(f"Missing required columns for forecast: {missing}.")
        return forecast_output

    key_drug_substrings_list = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if not key_drug_substrings_list:
        forecast_output["data_processing_notes"].append("KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings.")
        return forecast_output

    key_drug_pattern_re = '|'.join([re.escape(s.strip()) for s in key_drug_substrings_list if s.strip()])
    if not key_drug_pattern_re:
        forecast_output["data_processing_notes"].append("KEY_DRUG_SUBSTRINGS_SUPPLY contains only empty strings.")
        return forecast_output

    all_items_in_history = full_historical_health_df['item'].dropna().unique()
    items_to_forecast_final = [item for item in all_items_in_history if re.search(key_drug_pattern_re, item, re.IGNORECASE)]

    if not items_to_forecast_final:
        forecast_output["data_processing_notes"].append(f"No items matching configured keywords found.")
        return forecast_output
    
    forecast_detail_df = pd.DataFrame()
    if use_ai_supply_forecasting_model:
        forecast_output["data_processing_notes"].append("AI Advanced Forecasting Model is currently simulated.")
    
    try:
        forecast_detail_df = generate_simple_supply_forecast(
            full_historical_health_df, forecast_days_out, items_to_forecast_final, log_prefix
        )
    except Exception as e:
        logger.error(f"({log_prefix}) Error in forecast generation: {e}", exc_info=True)
        forecast_output["data_processing_notes"].append(f"Error in forecast generation: {e}")

    if not forecast_detail_df.empty:
        if 'days_of_supply_remaining' in forecast_detail_df.columns:
            def dos_to_numeric(dos): return pd.to_numeric(dos, errors='coerce')
            
            dos_numeric = forecast_detail_df['days_of_supply_remaining'].apply(dos_to_numeric)
            critical_thresh = _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
            warning_thresh = _get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)

            conditions = [dos_numeric < critical_thresh, dos_numeric < warning_thresh, dos_numeric.notna()]
            choices = ["Critical Low", "Warning Low", "Sufficient"]
            forecast_detail_df['stock_status'] = np.select(conditions, choices, default="Unknown")
        
        forecast_output["forecast_items_overview_list"] = forecast_detail_df.to_dict('records')

    logger.info(f"({log_prefix}) Supply forecast prep complete. Items: {len(forecast_output['forecast_items_overview_list'])}")
    return forecast_output
