# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Generates supply forecast overview data for the Clinic Console.

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date as date_type, timedelta, datetime # Added datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
    # Assuming generate_simple_supply_forecast is defined elsewhere, e.g., in analytics
    # from analytics.forecasting import generate_simple_supply_forecast 
    # For now, we'll use a placeholder for generate_simple_supply_forecast if it's not imported
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__) # Use different name to avoid conflict
    logger_init.error(f"Critical import error in supply_forecast.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# --- Placeholder for actual forecasting model ---
def generate_simple_supply_forecast(
    df_historical_consumption: pd.DataFrame,
    forecast_horizon_days: int,
    items_to_forecast: List[str],
    log_prefix: str = "SimpleSupplyForecast"
) -> pd.DataFrame:
    """
    Placeholder for a simple supply forecasting model.
    Replace with your actual forecasting logic.
    This version returns an empty DataFrame or a DataFrame with dummy forecast data.
    """
    logger.info(f"({log_prefix}) Called with horizon {forecast_horizon_days} for items: {items_to_forecast}")
    if df_historical_consumption.empty or not items_to_forecast:
        logger.warning(f"({log_prefix}) Insufficient data or no items to forecast.")
        return pd.DataFrame(columns=['item', 'forecast_date', 'predicted_consumption', 'predicted_stock_level', 'days_of_supply_remaining', 'estimated_stockout_date'])

    # Dummy forecast logic:
    # For each item, assume average daily consumption from history and project forward.
    # This is very basic and should be replaced with a real model.
    forecast_results = []
    
    # Ensure 'item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day' exist and are correct type
    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in df_historical_consumption.columns for col in required_cols):
        logger.error(f"({log_prefix}) Missing one or more required columns for forecasting: {required_cols}")
        return pd.DataFrame(columns=['item', 'forecast_date', 'predicted_consumption', 'predicted_stock_level', 'days_of_supply_remaining', 'estimated_stockout_date'])

    df_hist = df_historical_consumption.copy()
    df_hist['encounter_date'] = pd.to_datetime(df_hist['encounter_date'], errors='coerce')
    df_hist.dropna(subset=['encounter_date'], inplace=True)
    df_hist['item_stock_agg_zone'] = convert_to_numeric(df_hist['item_stock_agg_zone'], 0.0)
    df_hist['consumption_rate_per_day'] = convert_to_numeric(df_hist['consumption_rate_per_day'], 0.001) # Avoid 0 for division
    df_hist.loc[df_hist['consumption_rate_per_day'] <=0, 'consumption_rate_per_day'] = 0.001


    latest_date_in_data = df_hist['encounter_date'].max()
    if pd.NaT is latest_date_in_data:
        logger.warning(f"({log_prefix}) No valid dates in historical data for forecasting.")
        return pd.DataFrame(columns=['item', 'forecast_date', 'predicted_consumption', 'predicted_stock_level', 'days_of_supply_remaining', 'estimated_stockout_date'])


    for item_name in items_to_forecast:
        item_hist_df = df_hist[df_hist['item'] == item_name].sort_values(by='encounter_date')
        if item_hist_df.empty:
            logger.debug(f"({log_prefix}) No historical data for item: {item_name}")
            continue

        latest_record = item_hist_df.iloc[-1]
        current_stock = float(latest_record['item_stock_agg_zone'])
        avg_daily_consumption = float(item_hist_df['consumption_rate_per_day'].mean()) # Simple average
        if avg_daily_consumption <= 0: avg_daily_consumption = 0.001 # Avoid issues with zero consumption

        estimated_stockout_date_val = None
        days_remaining_val = current_stock / avg_daily_consumption if avg_daily_consumption > 0 else float('inf')
        if days_remaining_val != float('inf') and days_remaining_val >=0 :
            try:
                estimated_stockout_date_val = (latest_date_in_data + pd.to_timedelta(days_remaining_val, unit='D')).strftime('%Y-%m-%d')
            except OverflowError: # Handle very large days_remaining_val
                estimated_stockout_date_val = ">10Y" # Placeholder for very far stockout
        
        # Create a single entry representing the current status and very near-term forecast
        forecast_results.append({
            "item": item_name,
            "forecast_date": (latest_date_in_data + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), # Forecast for next day
            "current_stock_level": round(current_stock,1),
            "avg_daily_consumption_rate": round(avg_daily_consumption, 2),
            "days_of_supply_remaining": round(days_remaining_val,1) if days_remaining_val != float('inf') else "N/A (>10Y)",
            "estimated_stockout_date": estimated_stockout_date_val if estimated_stockout_date_val else "N/A"
        })
    
    return pd.DataFrame(forecast_results)


def prepare_clinic_supply_forecast_overview_data(
    full_historical_health_df: Optional[pd.DataFrame],
    current_period_context_str: str, # For logging context
    use_ai_supply_forecasting_model: bool = False,
    # Consider adding parameters for items_to_forecast, forecast_horizon_days
    log_prefix: str = "ClinicSupplyForecastPrep" 
) -> Dict[str, Any]:
    """
    Prepares an overview of supply forecasts, using either a simple model or a placeholder for an AI model.
    """
    module_log_prefix = log_prefix # Consistent naming
    
    # Get forecast_days_out from settings or use a default
    forecast_days_out = getattr(settings, 'DEFAULT_SUPPLY_FORECAST_HORIZON_DAYS', 30) 

    logger.info(f"({module_log_prefix}) Preparing supply forecast. Model: {'AI Advanced (Simulated)' if use_ai_supply_forecasting_model else 'Simple Aggregate'}, Horizon: {forecast_days_out} days, Items: Auto-detect from key list.")
    
    forecast_results: Dict[str, Any] = {
        "forecast_items_overview_list": [], # List of dicts for st.dataframe
        "forecast_model_type_used": "AI Advanced (Simulated)" if use_ai_supply_forecasting_model else "Simple Aggregate",
        "forecast_horizon_days": forecast_days_out,
        "data_processing_notes": []
    }

    if not isinstance(full_historical_health_df, pd.DataFrame) or full_historical_health_df.empty:
        note = "Historical health data insufficient for supply forecasts (empty or invalid DataFrame)."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_results["data_processing_notes"].append(note)
        return forecast_results

    required_supply_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in full_historical_health_df.columns for col in required_supply_cols):
        missing_cols = [col for col in required_supply_cols if col not in full_historical_health_df.columns]
        note = f"Historical health data insufficient. Missing required columns for supply forecast: {missing_cols}."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_results["data_processing_notes"].append(note)
        return forecast_results

    df_supply_hist = full_historical_health_df[required_supply_cols].copy()
    df_supply_hist['encounter_date'] = pd.to_datetime(df_supply_hist['encounter_date'], errors='coerce')
    df_supply_hist.dropna(subset=['item', 'encounter_date'], inplace=True)
    
    # Convert numeric columns, ensuring defaults that make sense for calculations
    df_supply_hist['item_stock_agg_zone'] = convert_to_numeric(df_supply_hist['item_stock_agg_zone'], default_value=0.0)
    # For consumption rate, a small positive default if it's NaN or zero, to avoid division by zero later
    df_supply_hist['consumption_rate_per_day'] = convert_to_numeric(df_supply_hist['consumption_rate_per_day'], default_value=0.001)
    df_supply_hist.loc[df_supply_hist['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001


    if df_supply_hist.empty:
        note = "No valid historical records after cleaning for supply forecasting."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_results["data_processing_notes"].append(note)
        return forecast_results

    # Determine items to forecast: use KEY_DRUG_SUBSTRINGS_SUPPLY from settings
    key_drug_substrings = getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if not key_drug_substrings:
        note = "KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings. Cannot determine items for forecast."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_results["data_processing_notes"].append(note)
        return forecast_results
        
    # Find unique items in historical data that match any of the key drug substrings
    # This is more robust than assuming exact matches.
    key_drug_pattern = '|'.join(re.escape(s.strip()) for s in key_drug_substrings if s.strip())
    if not key_drug_pattern: # If pattern is empty after stripping
        note = "KEY_DRUG_SUBSTRINGS_SUPPLY contains only empty strings. Cannot determine items for forecast."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_results["data_processing_notes"].append(note)
        return forecast_results

    all_items_in_hist = df_supply_hist['item'].astype(str).str.strip().unique()
    final_items_to_forecast = [item for item in all_items_in_hist if re.search(key_drug_pattern, item, re.IGNORECASE)]

    if not final_items_to_forecast:
        note = "No items matching KEY_DRUG_SUBSTRINGS_SUPPLY found in historical data."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_results["data_processing_notes"].append(note)
        return forecast_results
    
    logger.info(f"({module_log_prefix}) Final items for forecast: {final_items_to_forecast}")


    if use_ai_supply_forecasting_model:
        forecast_results["data_processing_notes"].append("AI Advanced Forecasting Model is simulated / placeholder.")
        # Placeholder for AI model call - for now, it might return empty or dummy data
        # ai_forecast_df = call_your_ai_forecasting_model(df_supply_hist, forecast_days_out, final_items_to_forecast)
        # For demonstration, let's use the simple forecast as a stand-in if AI is selected but not implemented
        logger.warning(f"({module_log_prefix}) AI forecasting selected but using simple forecast as placeholder.")
        try:
            forecast_detail_df = generate_simple_supply_forecast(df_supply_hist, forecast_days_out, final_items_to_forecast, f"{module_log_prefix}/AI_PlaceholderSimpleLinear")
            if isinstance(forecast_detail_df, pd.DataFrame) and not forecast_detail_df.empty:
                forecast_results["forecast_items_overview_list"] = forecast_detail_df.to_dict(orient='records')
        except Exception as e_ai_placeholder_forecast:
            logger.error(f"({module_log_prefix}) Error in AI placeholder (simple) forecast: {e_ai_placeholder_forecast}", exc_info=True)
            forecast_results["data_processing_notes"].append(f"Error in AI placeholder forecast: {str(e_ai_placeholder_forecast)}")

    else: # Simple Linear / Aggregate Forecasting
        logger.info(f"({module_log_prefix}) Initiating Simple Aggregate Supply Forecasting for: {final_items_to_forecast}")
        try:
            # Now forecast_days_out is defined from settings or local default earlier in this function
            forecast_detail_df = generate_simple_supply_forecast(df_supply_hist, forecast_days_out, final_items_to_forecast, f"{module_log_prefix}/SimpleAggregate")
            if isinstance(forecast_detail_df, pd.DataFrame) and not forecast_detail_df.empty:
                 # Ensure the columns match what the dashboard expects for st.dataframe
                expected_cols = ["item", "current_stock_level", "avg_daily_consumption_rate", "days_of_supply_remaining", "estimated_stockout_date"]
                cols_to_display = [col for col in expected_cols if col in forecast_detail_df.columns]
                if len(cols_to_display) < len(expected_cols):
                    logger.warning(f"({module_log_prefix}) Simple forecast output missing some expected columns. Available: {forecast_detail_df.columns.tolist()}")
                
                # Add status based on days_of_supply_remaining
                def get_stock_status(dos):
                    if pd.isna(dos) or not isinstance(dos, (int,float)): return "Unknown"
                    critical_dos = getattr(settings, 'CRITICAL_SUPPLY_DAYS_REMAINING', 7)
                    warning_dos = getattr(settings, 'WARNING_SUPPLY_DAYS_REMAINING', 14)
                    if dos < critical_dos: return "Critical Low"
                    if dos < warning_dos: return "Warning Low"
                    return "Sufficient"

                if 'days_of_supply_remaining' in forecast_detail_df.columns:
                    # Convert to numeric before applying status logic
                    forecast_detail_df['dos_numeric_temp'] = convert_to_numeric(forecast_detail_df['days_of_supply_remaining'], default_value=np.nan)
                    forecast_detail_df['stock_status'] = forecast_detail_df['dos_numeric_temp'].apply(get_stock_status)
                    forecast_detail_df.drop(columns=['dos_numeric_temp'], inplace=True, errors='ignore')
                    cols_to_display.insert(1, 'stock_status') # Insert status after item

                forecast_results["forecast_items_overview_list"] = forecast_detail_df[list(set(cols_to_display))].to_dict(orient='records') # Use set to ensure unique cols
            else:
                forecast_results["data_processing_notes"].append("Simple forecast did not return any detailed results.")
        except Exception as e_simple_forecast:
            logger.error(f"({module_log_prefix}) Simple forecast processing error: {e_simple_forecast}", exc_info=True)
            forecast_results["data_processing_notes"].append(f"Error in simple forecast processing: {str(e_simple_forecast)}")

    num_items_forecasted = len(forecast_results["forecast_items_overview_list"])
    logger.info(f"({module_log_prefix}) Supply forecast prep complete. Items in overview: {num_items_forecasted}")
    return forecast_results
