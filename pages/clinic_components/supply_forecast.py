# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Prepares supply forecast data for medical items at a clinic for Sentinel.
# Renamed from supply_forecast_generator.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
# Import forecasting models from analytics module
from analytics.supply_forecasting import generate_simple_supply_forecast, SupplyForecastingModel
from data_processing.helpers import convert_to_numeric # Local import

logger = logging.getLogger(__name__)


def prepare_clinic_supply_forecast_overview_data( # Renamed function
    clinic_historical_health_df_for_supply: Optional[pd.DataFrame], # Full historical data with item usage/stock
    reporting_period_context_str: str, # Contextual string for reporting period (not directly used in calc)
    forecast_horizon_days: int = 30, # Renamed, default forecast horizon
    use_ai_supply_forecasting_model: bool = False, # Renamed flag
    items_list_to_forecast: Optional[List[str]] = None # Renamed, specific items to forecast
) -> Dict[str, Any]:
    """
    Prepares supply forecast data using either a simple linear or an AI-simulated model.
    Provides both a detailed forecast DataFrame and a summarized overview list.

    Args:
        clinic_historical_health_df_for_supply: DataFrame of health records containing item usage,
                                                stock levels (e.g., 'item_stock_agg_zone'), and
                                                consumption rates (e.g., 'consumption_rate_per_day').
                                                This DataFrame should span enough history for rate calculation.
        reporting_period_context_str: String describing the current reporting context (for logging).
        forecast_horizon_days: Number of days into the future to forecast.
        use_ai_supply_forecasting_model: Boolean flag to select the AI simulation model.
        items_list_to_forecast: Optional list of specific item names. If None, forecasts for
                                items matching KEY_DRUG_SUBSTRINGS_SUPPLY from settings.

    Returns:
        Dict[str, Any]: Contains the forecast model type used, a detailed forecast DataFrame (optional),
                        a summary list per item (overview), and processing notes.
                        Keys: "reporting_context", "forecast_model_type_used",
                              "forecast_detail_df", "forecast_items_overview_list",
                              "data_processing_notes".
    """
    module_log_prefix = "ClinicSupplyForecastPrep" # Renamed for clarity
    chosen_model_type_description = "AI-Simulated" if use_ai_supply_forecasting_model else "Simple Linear"
    logger.info(
        f"({module_log_prefix}) Preparing clinic supply forecast. Model: {chosen_model_type_description}, "
        f"Horizon: {forecast_horizon_days} days, Items: {'Specific List Provided' if items_list_to_forecast else 'Auto-detect from Config/Data'}."
    )

    # Initialize output structure
    forecast_output_data_dict: Dict[str, Any] = {
        "reporting_context": reporting_period_context_str,
        "forecast_model_type_used": chosen_model_type_description,
        "forecast_detail_df": pd.DataFrame(),      # Default to empty DF
        "forecast_items_overview_list": [], # List of dicts for summary
        "data_processing_notes": []
    }

    # --- Input Validation ---
    # Required columns for *any* forecast (simple or AI status input)
    # AI model takes current_supply_levels_df which is derived from historical.
    # Simple model takes historical_health_df directly.
    required_hist_cols_for_supply = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    
    if not isinstance(clinic_historical_health_df_for_supply, pd.DataFrame) or \
       clinic_historical_health_df_for_supply.empty or \
       not all(col in clinic_historical_health_df_for_supply.columns for col in required_hist_cols_for_supply):
        
        missing_cols_details = [col for col in required_hist_cols_for_supply if col not in (clinic_historical_health_df_for_supply.columns if isinstance(clinic_historical_health_df_for_supply, pd.DataFrame) else [])]
        error_msg = (f"Historical health data is insufficient for supply forecasts. "
                     f"Required columns: {required_hist_cols_for_supply}. "
                     f"Missing or DataFrame empty. Missing cols found: {missing_cols_details if missing_cols_details else 'None, DF likely empty'}.")
        logger.error(f"({module_log_prefix}) {error_msg}")
        forecast_output_data_dict["data_processing_notes"].append(error_msg)
        return forecast_output_data_dict

    df_supply_hist_cleaned = clinic_historical_health_df_for_supply.copy() # Work on a copy

    # --- Data Cleaning and Preparation of Historical Data ---
    df_supply_hist_cleaned['encounter_date'] = pd.to_datetime(df_supply_hist_cleaned['encounter_date'], errors='coerce')
    df_supply_hist_cleaned.dropna(subset=['encounter_date', 'item'], inplace=True) # Item name and date are critical

    # Ensure stock and consumption are numeric, with safe defaults
    df_supply_hist_cleaned['item_stock_agg_zone'] = convert_to_numeric(df_supply_hist_cleaned.get('item_stock_agg_zone'), default_value=0.0)
    df_supply_hist_cleaned['consumption_rate_per_day'] = convert_to_numeric(df_supply_hist_cleaned.get('consumption_rate_per_day'), default_value=1e-7) # Tiny positive default
    # Ensure consumption rate is strictly positive to avoid division by zero or illogical forecasts
    df_supply_hist_cleaned.loc[df_supply_hist_cleaned['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-7

    if df_supply_hist_cleaned.empty: # Check after cleaning critical date/item columns
        note_msg = "No valid historical records remaining after cleaning for supply forecast preparation."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        forecast_output_data_dict["data_processing_notes"].append(note_msg)
        return forecast_output_data_dict

    # --- Determine the list of items to forecast ---
    final_items_for_forecasting: List[str]
    if items_list_to_forecast and isinstance(items_list_to_forecast, list) and len(items_list_to_forecast) > 0:
        final_items_for_forecasting = list(set(items_list_to_forecast)) # Ensure unique items
        logger.debug(f"({module_log_prefix}) Using provided specific item list for forecast: {final_items_for_forecasting}")
    elif settings.KEY_DRUG_SUBSTRINGS_SUPPLY: # If no specific list, try to find key drugs from config
        all_unique_items_from_data = df_supply_hist_cleaned['item'].dropna().unique()
        final_items_for_forecasting = [
            item_name_data for item_name_data in all_unique_items_from_data
            if any(drug_substr_cfg.lower() in str(item_name_data).lower() for drug_substr_cfg in settings.KEY_DRUG_SUBSTRINGS_SUPPLY)
        ]
        if not final_items_for_forecasting and len(all_unique_items_from_data) > 0: # Fallback if no key drugs match
            num_fallback = min(5, len(all_unique_items_from_data)) # Forecast a small sample
            final_items_for_forecasting = np.random.choice(all_unique_items_from_data, num_fallback, replace=False).tolist() if len(all_unique_items_from_data) > num_fallback else all_unique_items_from_data.tolist()
            note_msg = f"No key drug substrings matched items in data; forecasting for {len(final_items_for_forecasting)} sample items: {final_items_for_forecasting}"
            logger.info(f"({module_log_prefix}) {note_msg}")
            forecast_output_data_dict["data_processing_notes"].append(note_msg)
    else: # No specific list, no key substrings in config -> fallback to a sample of all unique items
        all_unique_items_from_data = df_supply_hist_cleaned['item'].dropna().unique()
        num_fallback = min(5, len(all_unique_items_from_data))
        final_items_for_forecasting = np.random.choice(all_unique_items_from_data, num_fallback, replace=False).tolist() if len(all_unique_items_from_data) > num_fallback else all_unique_items_from_data.tolist()
        if final_items_for_forecasting:
            note_msg = f"No specific items or key drug config; forecasting for {len(final_items_for_forecasting)} sample items: {final_items_for_forecasting}"
            logger.info(f"({module_log_prefix}) {note_msg}")
            forecast_output_data_dict["data_processing_notes"].append(note_msg)

    if not final_items_for_forecasting: # Final check if no items could be determined
        note_msg = "No items were ultimately determined for supply forecasting after all checks."
        logger.warning(f"({module_log_prefix}) {note_msg}")
        forecast_output_data_dict["data_processing_notes"].append(note_msg)
        return forecast_output_data_dict
    
    logger.info(f"({module_log_prefix}) Final items selected for forecast: {final_items_for_forecasting}")


    # --- Generate Forecast based on Selected Model ---
    df_forecast_results_detail: Optional[pd.DataFrame] = None

    if use_ai_supply_forecasting_model:
        logger.info(f"({module_log_prefix}) Initiating AI-Simulated Supply Forecasting Model for: {final_items_for_forecasting}")
        ai_supply_model = SupplyForecastingModel() # Instantiate AI model from analytics module
        
        # AI model expects a DataFrame of current status: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
        # This status is derived from the latest record for each item in the historical data.
        df_latest_item_status_for_ai_input = df_supply_hist_cleaned.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        # Filter this latest status DF for only the items selected for forecasting
        df_latest_item_status_for_ai_input = df_latest_item_status_for_ai_input[
            df_latest_item_status_for_ai_input['item'].isin(final_items_for_forecasting)
        ]
        
        # Rename columns to match AI model's expected input
        df_input_to_ai_model = df_latest_item_status_for_ai_input.rename(columns={
            'item_stock_agg_zone': 'current_stock',
            'consumption_rate_per_day': 'avg_daily_consumption_historical',
            'encounter_date': 'last_stock_update_date'
        })[['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']] # Ensure correct columns

        if not df_input_to_ai_model.empty:
            try:
                df_forecast_results_detail = ai_supply_model.forecast_supply_levels_advanced(
                    current_supply_levels_df=df_input_to_ai_model,
                    forecast_days_out=forecast_horizon_days
                    # item_filter_list is implicitly handled by df_input_to_ai_model preparation
                )
            except Exception as e_ai_fc:
                logger.error(f"({module_log_prefix}) Error during AI supply forecast execution: {e_ai_fc}", exc_info=True)
                forecast_output_data_dict["data_processing_notes"].append(f"AI forecast failed: {str(e_ai_fc)}")
                df_forecast_results_detail = pd.DataFrame() # Empty on error
        else:
            msg = f"No current status data found for AI forecasting the selected items: {final_items_for_forecasting}"
            logger.warning(f"({module_log_prefix}) {msg}")
            forecast_output_data_dict["data_processing_notes"].append(msg)
            df_forecast_results_detail = pd.DataFrame()
    
    else: # Use simple linear forecast (from analytics.supply_forecasting)
        logger.info(f"({module_log_prefix}) Initiating Simple Linear Supply Forecasting for: {final_items_for_forecasting}")
        try:
            df_forecast_results_detail = generate_simple_supply_forecast(
                health_df_for_supply=df_supply_hist_cleaned, # Pass the cleaned historical data
                forecast_days_out=forecast_horizon_days,
                item_filter_list=final_items_for_forecasting, # Pass specific items to filter
                source_context=f"{module_log_prefix}/SimpleLinearForecast"
            )
        except Exception as e_simple_fc:
            logger.error(f"({module_log_prefix}) Error during Simple Linear supply forecast execution: {e_simple_fc}", exc_info=True)
            forecast_output_data_dict["data_processing_notes"].append(f"Simple forecast failed: {str(e_simple_fc)}")
            df_forecast_results_detail = pd.DataFrame() # Empty on error

    # --- Process and Summarize the Generated Forecast DataFrame ---
    if isinstance(df_forecast_results_detail, pd.DataFrame) and not df_forecast_results_detail.empty:
        # Store the detailed daily/periodic forecast (optional for UI, but good for debugging/deeper analysis)
        forecast_output_data_dict["forecast_detail_df"] = df_forecast_results_detail.sort_values(by=['item', 'forecast_date']).reset_index(drop=True)
        
        # Create a summarized overview list for easier display in UI (one entry per item)
        overview_summary_list_items: List[Dict[str, Any]] = []
        
        # For summary, group by item. Initial stock and consumption are from the start of forecast.
        # Stockout date is the single estimated date for that item.
        # The forecast_detail_df should have a consistent 'estimated_stockout_date_ai' or 'estimated_stockout_date_linear' column.
        stockout_date_col_name = 'estimated_stockout_date_ai' if use_ai_supply_forecasting_model else 'estimated_stockout_date_linear'

        # Get unique items from the forecast results
        items_in_forecast_results = df_forecast_results_detail['item'].unique()

        for item_name_val_fc in items_in_forecast_results:
            item_specific_forecast_df = df_forecast_results_detail[df_forecast_results_detail['item'] == item_name_val_fc]
            if item_specific_forecast_df.empty: continue

            # Get initial stock and consumption rate used at the start of this item's forecast
            # For simple model, these are explicit columns. For AI, 'predicted_daily_consumption' is for each day.
            if not use_ai_supply_forecasting_model and 'initial_stock_at_forecast_start' in item_specific_forecast_df.columns:
                initial_stock_val_fc = item_specific_forecast_df['initial_stock_at_forecast_start'].iloc[0]
                base_consumption_val_fc = item_specific_forecast_df['base_consumption_rate_per_day'].iloc[0]
            else: # For AI model, or if simple model output varies, derive from first day of its forecast
                  # Need to find the original input current_stock for this item from df_latest_item_status_for_ai_input (if AI)
                  # or df_supply_hist_cleaned (if simple, though it should be in output)
                
                # Fallback: use first day's stock and predicted consumption for AI
                initial_stock_val_fc = item_specific_forecast_df['forecasted_stock_level'].iloc[0] # This is stock *after* first day's consumption
                # A better 'initial_stock' for AI would be to add back the first day's predicted consumption,
                # or ideally, pass it through from the input `current_supply_levels_df` to the AI forecaster's output.
                # For now, this is an approximation.
                base_consumption_val_fc = item_specific_forecast_df['predicted_daily_consumption'].iloc[0] if 'predicted_daily_consumption' in item_specific_forecast_df.columns else 1e-7
            
            if pd.isna(base_consumption_val_fc) or base_consumption_val_fc <= 1e-8: base_consumption_val_fc = 1e-7 # Final safety for DivByZero

            initial_dos_val_fc = (initial_stock_val_fc / base_consumption_val_fc) if base_consumption_val_fc > 1e-8 else np.inf
            
            # Get the single estimated stockout date for this item (should be same for all its rows)
            estimated_stockout_dt_val_fc = pd.NaT
            if stockout_date_col_name in item_specific_forecast_df.columns:
                # Get the first non-NaT stockout date for this item
                first_valid_stockout_date = item_specific_forecast_df[stockout_date_col_name].dropna().iloc[0] if not item_specific_forecast_df[stockout_date_col_name].dropna().empty else pd.NaT
                estimated_stockout_dt_val_fc = pd.to_datetime(first_valid_stockout_date, errors='coerce')

            overview_summary_list_items.append({
                "item_name": item_name_val_fc,
                "current_stock_on_hand_at_forecast_start": float(initial_stock_val_fc),
                "avg_daily_consumption_rate_used": float(base_consumption_val_fc),
                "initial_days_of_supply_estimated": round(initial_dos_val_fc, 1) if np.isfinite(initial_dos_val_fc) else "Adequate (>Forecast Period)",
                "estimated_stockout_date": estimated_stockout_dt_val_fc.strftime('%Y-%m-%d') if pd.notna(estimated_stockout_dt_val_fc) else "Beyond Forecast"
            })
        forecast_output_data_dict["forecast_items_overview_list"] = overview_summary_list_items
        
    elif not forecast_output_data_dict["data_processing_notes"]: # If forecast DF is empty but no specific error notes yet
        forecast_output_data_dict["data_processing_notes"].append(
            "Supply forecast could not be generated or resulted in no data with the selected model and available historical records."
        )

    num_items_in_overview_list = len(forecast_output_data_dict.get('forecast_items_overview_list',[]))
    logger.info(f"({module_log_prefix}) Clinic supply forecast data preparation complete. Items in overview: {num_items_in_overview_list}")
    return forecast_output_data_dict
