# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Generates supply forecast overview data for the Clinic Console.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union # Added Union for date types
from datetime import date as date_type, timedelta, datetime

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric # Ensure this is robust
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
    forecast_horizon_days: int, # This is now passed correctly
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
        'forecast_date_horizon' # Added for clarity on what the forecast represents
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

    df_hist = df_historical_consumption.copy() # Work on a copy

    # Prepare data types
    try:
        df_hist['encounter_date'] = pd.to_datetime(df_hist['encounter_date'], errors='coerce')
        df_hist.dropna(subset=['encounter_date', 'item'], inplace=True) # Essential for grouping and latest record
        df_hist['item_stock_agg_zone'] = convert_to_numeric(df_hist['item_stock_agg_zone'], default_value=0.0)
        df_hist['consumption_rate_per_day'] = convert_to_numeric(df_hist['consumption_rate_per_day'], default_value=0.001) # Small default to avoid /0
        # Ensure consumption is not zero or negative for DOS calculation
        df_hist.loc[df_hist['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001
    except Exception as e_prep:
        logger.error(f"({log_prefix}) Error preparing historical data for simple forecast: {e_prep}", exc_info=True)
        return pd.DataFrame(columns=output_columns)

    if df_hist.empty:
        logger.warning(f"({log_prefix}) No valid historical data after cleaning for forecasting.")
        return pd.DataFrame(columns=output_columns)
        
    latest_date_in_overall_data = df_hist['encounter_date'].max()
    if pd.NaT is latest_date_in_overall_data: # Check if any valid date exists at all
        logger.warning(f"({log_prefix}) No valid dates in historical data. Cannot determine latest record for forecasting.")
        return pd.DataFrame(columns=output_columns)

    forecast_results_list = []
    for item_name in items_to_forecast:
        item_specific_hist_df = df_hist[df_hist['item'] == item_name].sort_values(by='encounter_date')
        
        if item_specific_hist_df.empty:
            logger.debug(f"({log_prefix}) No historical data found for item: {item_name}")
            forecast_results_list.append({
                "item": item_name, "current_stock_level": 0.0,
                "avg_daily_consumption_rate": 0.0, "days_of_supply_remaining": 0.0,
                "estimated_stockout_date": "N/A (No History)", 
                "forecast_date_horizon": (latest_date_in_overall_data + pd.Timedelta(days=forecast_horizon_days)).strftime('%Y-%m-%d') if pd.notna(latest_date_in_overall_data) else "N/A"
            })
            continue

        latest_item_record = item_specific_hist_df.iloc[-1]
        current_item_stock = float(latest_item_record['item_stock_agg_zone'])
        
        # Calculate average daily consumption based on available history for that item
        # Consider a more robust way if consumption rates vary wildly or are sparse
        avg_daily_item_consumption = float(item_specific_hist_df['consumption_rate_per_day'].mean())
        if avg_daily_item_consumption <= 0: # If mean is still zero (e.g. only one record with 0.001)
            avg_daily_item_consumption = 0.001 # Fallback to small positive number

        days_of_supply_val = current_item_stock / avg_daily_item_consumption if avg_daily_item_consumption > 0 else float('inf')
        
        estimated_stockout_date_str = "N/A"
        if days_of_supply_val != float('inf') and days_of_supply_val >= 0 :
            try:
                stockout_datetime = latest_item_record['encounter_date'] + pd.to_timedelta(days_of_supply_val, unit='D')
                estimated_stockout_date_str = stockout_datetime.strftime('%Y-%m-%d')
            except OverflowError: 
                estimated_stockout_date_str = ">5Y" # Indicate very far stockout
            except Exception as e_dos_date:
                logger.warning(f"({log_prefix}) Error calculating stockout date for {item_name}: {e_dos_date}")
                estimated_stockout_date_str = "Error Calc"
        elif days_of_supply_val == float('inf'):
             estimated_stockout_date_str = "N/A (>5Y or zero consumption)"


        forecast_date_horizon_str = "N/A"
        if pd.notna(latest_item_record['encounter_date']):
            forecast_date_horizon_str = (latest_item_record['encounter_date'] + pd.Timedelta(days=forecast_horizon_days)).strftime('%Y-%m-%d')
        
        forecast_results_list.append({
            "item": item_name,
            "current_stock_level": round(current_item_stock, 1),
            "avg_daily_consumption_rate": round(avg_daily_item_consumption, 3), # More precision for rate
            "days_of_supply_remaining": round(days_of_supply_val, 1) if days_of_supply_val != float('inf') else "N/A (>5Y)",
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
    module_log_prefix = log_prefix 
    
    forecast_days_out = int(_get_setting('DEFAULT_SUPPLY_FORECAST_HORIZON_DAYS', 30))

    logger.info(f"({module_log_prefix}) Preparing supply forecast. Model: {'AI Advanced (Simulated)' if use_ai_supply_forecasting_model else 'Simple Aggregate'}, Horizon: {forecast_days_out} days.")
    
    forecast_output: Dict[str, Any] = {
        "forecast_items_overview_list": [], 
        "forecast_model_type_used": "AI Advanced (Simulated)" if use_ai_supply_forecasting_model else "Simple Aggregate",
        "forecast_horizon_days": forecast_days_out,
        "data_processing_notes": []
    }

    if not isinstance(full_historical_health_df, pd.DataFrame) or full_historical_health_df.empty:
        note = "Historical health data is empty or invalid. Cannot generate supply forecasts."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_output["data_processing_notes"].append(note)
        return forecast_output

    required_supply_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in full_historical_health_df.columns for col in required_supply_cols):
        missing_cols = [col for col in required_supply_cols if col not in full_historical_health_df.columns]
        note = f"Missing required columns for supply forecast: {missing_cols}. Cannot generate forecasts."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_output["data_processing_notes"].append(note)
        return forecast_output

    # Prepare a clean DataFrame for historical supply analysis
    df_supply_hist_clean = full_historical_health_df[required_supply_cols].copy()
    try:
        df_supply_hist_clean['encounter_date'] = pd.to_datetime(df_supply_hist_clean['encounter_date'], errors='coerce')
        df_supply_hist_clean.dropna(subset=['item', 'encounter_date'], inplace=True) # Item name and date are crucial
        df_supply_hist_clean['item'] = df_supply_hist_clean['item'].astype(str).str.strip()
        df_supply_hist_clean['item_stock_agg_zone'] = convert_to_numeric(df_supply_hist_clean['item_stock_agg_zone'], default_value=0.0)
        df_supply_hist_clean['consumption_rate_per_day'] = convert_to_numeric(df_supply_hist_clean['consumption_rate_per_day'], default_value=0.001)
        df_supply_hist_clean.loc[df_supply_hist_clean['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001 # Avoid non-positive rates
    except Exception as e_prep_supply:
        note = f"Error during supply data preparation: {e_prep_supply}"
        logger.error(f"({module_log_prefix}) {note}", exc_info=True)
        forecast_output["data_processing_notes"].append(note)
        return forecast_output


    if df_supply_hist_clean.empty:
        note = "No valid historical records after cleaning for supply forecasting."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_output["data_processing_notes"].append(note)
        return forecast_output

    # Determine items to forecast based on settings
    key_drug_substrings_list = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if not isinstance(key_drug_substrings_list, list) or not key_drug_substrings_list:
        note = "KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings or is empty. Cannot auto-detect items for forecast."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_output["data_processing_notes"].append(note)
        return forecast_output # Or forecast for all items if desired, but usually too noisy
        
    key_drug_pattern_re = '|'.join(re.escape(s.strip()) for s in key_drug_substrings_list if s.strip())
    if not key_drug_pattern_re:
        note = "KEY_DRUG_SUBSTRINGS_SUPPLY contains only empty strings after stripping. Cannot determine items."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_output["data_processing_notes"].append(note)
        return forecast_output

    all_items_in_history = df_supply_hist_clean['item'].unique()
    items_to_forecast_final = [item for item in all_items_in_history if re.search(key_drug_pattern_re, item, re.IGNORECASE)]

    if not items_to_forecast_final:
        note = f"No items matching configured KEY_DRUG_SUBSTRINGS_SUPPLY ({key_drug_substrings_list}) found in historical data."
        logger.warning(f"({module_log_prefix}) {note}")
        forecast_output["data_processing_notes"].append(note)
        return forecast_output
    
    logger.info(f"({module_log_prefix}) Identified {len(items_to_forecast_final)} items for forecasting: {items_to_forecast_final[:5]}...")

    # --- Forecasting ---
    forecast_detail_df = pd.DataFrame() # Initialize
    if use_ai_supply_forecasting_model:
        forecast_output["data_processing_notes"].append("AI Advanced Forecasting Model is currently simulated using simple aggregation.")
        logger.warning(f"({module_log_prefix}) AI forecasting selected; using simple forecast as placeholder/simulation.")
        # Replace with actual AI model call when available
        # For now, simulate with simple forecast
        try:
            forecast_detail_df = generate_simple_supply_forecast(
                df_supply_hist_clean, forecast_days_out, items_to_forecast_final, 
                f"{module_log_prefix}/AI_PlaceholderSimple"
            )
        except Exception as e_ai_placeholder:
            logger.error(f"({module_log_prefix}) Error in AI placeholder (simple) forecast simulation: {e_ai_placeholder}", exc_info=True)
            forecast_output["data_processing_notes"].append(f"Error in AI placeholder forecast sim: {str(e_ai_placeholder)}")
    else: # Simple Aggregate Forecasting
        logger.info(f"({module_log_prefix}) Initiating Simple Aggregate Supply Forecasting for identified items.")
        try:
            forecast_detail_df = generate_simple_supply_forecast(
                df_supply_hist_clean, forecast_days_out, items_to_forecast_final, 
                f"{module_log_prefix}/SimpleAggregate"
            )
        except Exception as e_simple_forecast_call:
            logger.error(f"({module_log_prefix}) Error calling simple forecast processing: {e_simple_forecast_call}", exc_info=True)
            forecast_output["data_processing_notes"].append(f"Error in simple forecast processing: {str(e_simple_forecast_call)}")

    if isinstance(forecast_detail_df, pd.DataFrame) and not forecast_detail_df.empty:
        # Define expected columns for the overview table in the UI
        expected_overview_cols = ["item", "current_stock_level", "avg_daily_consumption_rate", 
                                  "days_of_supply_remaining", "estimated_stockout_date", "stock_status"]
        
        # Add stock_status based on days_of_supply_remaining
        if 'days_of_supply_remaining' in forecast_detail_df.columns:
            # Convert 'days_of_supply_remaining' to numeric for status calculation, handling "N/A (>5Y)"
            def dos_to_numeric(dos_val):
                if isinstance(dos_val, (int, float)): return dos_val
                if isinstance(dos_val, str) and "N/A" in dos_val: return float('inf') # Treat N/A or very long as infinite
                try: return float(dos_val)
                except ValueError: return np.nan

            forecast_detail_df['dos_numeric_for_status'] = forecast_detail_df['days_of_supply_remaining'].apply(dos_to_numeric)
            
            critical_dos_thresh = _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
            warning_dos_thresh = _get_setting('WARNING_SUPPLY_DAYS_REMAINING', 14)

            def get_stock_status_from_dos(dos_numeric):
                if pd.isna(dos_numeric): return "Unknown"
                if dos_numeric == float('inf'): return "Sufficient (High)"
                if dos_numeric < critical_dos_thresh: return "Critical Low"
                if dos_numeric < warning_dos_thresh: return "Warning Low"
                return "Sufficient"
            
            forecast_detail_df['stock_status'] = forecast_detail_df['dos_numeric_for_status'].apply(get_stock_status_from_dos)
            forecast_detail_df.drop(columns=['dos_numeric_for_status'], inplace=True, errors='ignore')
        else:
            forecast_detail_df['stock_status'] = "Unknown" # Add column if DOS was missing
            forecast_output["data_processing_notes"].append("'days_of_supply_remaining' missing from forecast details, status set to Unknown.")

        # Ensure all expected columns are present for the final list of dicts
        final_overview_list = []
        for _, row in forecast_detail_df.iterrows():
            item_dict = {}
            for col in expected_overview_cols:
                item_dict[col] = row.get(col, "N/A" if col != "current_stock_level" else 0.0) # Sensible defaults
            final_overview_list.append(item_dict)
        forecast_output["forecast_items_overview_list"] = final_overview_list
    else:
        forecast_output["data_processing_notes"].append("Forecasting did not return any detailed results or returned non-DataFrame.")

    num_items_in_overview = len(forecast_output["forecast_items_overview_list"])
    logger.info(f"({module_log_prefix}) Supply forecast prep complete. Items in overview: {num_items_in_overview}")
    return forecast_output
