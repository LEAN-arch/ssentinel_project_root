# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Prepares supply forecast data for medical items at a clinic for Sentinel.

import pandas as pd
import numpy as np
import logging
import re # For item matching
from typing import Dict, Any, Optional, List

from config import settings
from analytics.supply_forecasting import generate_simple_supply_forecast, SupplyForecastingModel
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)


def prepare_clinic_supply_forecast_overview_data(
    clinic_historical_health_df_for_supply: Optional[pd.DataFrame],
    reporting_period_context_str: str,
    forecast_horizon_days: int = 30,
    use_ai_supply_forecasting_model: bool = False,
    items_list_to_forecast: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepares supply forecast data using either simple linear or AI-simulated model.
    """
    module_log_prefix = "ClinicSupplyForecastPrep"
    model_type_desc = "AI-Simulated" if use_ai_supply_forecasting_model else "Simple Linear"
    logger.info(
        f"({module_log_prefix}) Preparing supply forecast. Model: {model_type_desc}, "
        f"Horizon: {forecast_horizon_days} days, Items: {'Specific List' if items_list_to_forecast else 'Auto-detect'}."
    )

    output_data: Dict[str, Any] = {
        "reporting_context": reporting_period_context_str,
        "forecast_model_type_used": model_type_desc,
        "forecast_detail_df": pd.DataFrame(),
        "forecast_items_overview_list": [],
        "data_processing_notes": []
    }

    required_hist_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not isinstance(clinic_historical_health_df_for_supply, pd.DataFrame) or \
       clinic_historical_health_df_for_supply.empty or \
       not all(col in clinic_historical_health_df_for_supply.columns for col in required_hist_cols):
        missing = [c for c in required_hist_cols if not isinstance(clinic_historical_health_df_for_supply, pd.DataFrame) or c not in clinic_historical_health_df_for_supply.columns]
        err_msg = f"Historical health data insufficient for supply forecasts. Missing/Empty DF or cols: {missing if missing else 'DF Empty'}."
        logger.error(f"({module_log_prefix}) {err_msg}"); output_data["data_processing_notes"].append(err_msg)
        return output_data

    df_supply_hist = clinic_historical_health_df_for_supply.copy()
    df_supply_hist['encounter_date'] = pd.to_datetime(df_supply_hist['encounter_date'], errors='coerce')
    df_supply_hist.dropna(subset=['encounter_date', 'item'], inplace=True)
    df_supply_hist['item_stock_agg_zone'] = convert_to_numeric(df_supply_hist.get('item_stock_agg_zone'), 0.0)
    df_supply_hist['consumption_rate_per_day'] = convert_to_numeric(df_supply_hist.get('consumption_rate_per_day'), 1e-7)
    df_supply_hist.loc[df_supply_hist['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-7
    if df_supply_hist.empty:
        note = "No valid historical records after cleaning for supply forecast prep."
        logger.warning(f"({module_log_prefix}) {note}"); output_data["data_processing_notes"].append(note)
        return output_data

    # Determine items to forecast
    final_items: List[str] = []
    all_items_in_data = df_supply_hist['item'].dropna().astype(str).unique()
    if items_list_to_forecast and isinstance(items_list_to_forecast, list) and len(items_list_to_forecast) > 0:
        final_items = list(set(items_list_to_forecast))
    elif settings.KEY_DRUG_SUBSTRINGS_SUPPLY:
        final_items = [item for item in all_items_in_data if any(re.search(re.escape(sub), item, re.IGNORECASE) for sub in settings.KEY_DRUG_SUBSTRINGS_SUPPLY)]
        if not final_items and len(all_items_in_data) > 0: # Fallback to sample if no key drugs match
            num_fallback = min(5, len(all_items_in_data))
            final_items = np.random.choice(all_items_in_data, num_fallback, replace=False).tolist() if len(all_items_in_data) > num_fallback else all_items_in_data.tolist()
            output_data["data_processing_notes"].append(f"No key drugs matched; forecasting for {len(final_items)} sample items.")
    else: # Fallback if no specific list and no key substrings in config
        num_fallback = min(5, len(all_items_in_data))
        final_items = np.random.choice(all_items_in_data, num_fallback, replace=False).tolist() if len(all_items_in_data) > num_fallback else all_items_in_data.tolist()
        if final_items: output_data["data_processing_notes"].append(f"No specific items/key drug config; forecasting for {len(final_items)} sample items.")

    if not final_items:
        note = "No items determined for supply forecasting."; logger.warning(f"({module_log_prefix}) {note}")
        output_data["data_processing_notes"].append(note); return output_data
    logger.info(f"({module_log_prefix}) Final items for forecast: {final_items}")

    # Generate Forecast
    df_forecast_detail: Optional[pd.DataFrame] = None
    if use_ai_supply_forecasting_model:
        logger.info(f"({module_log_prefix}) Initiating AI-Simulated Supply Forecasting for: {final_items}")
        ai_model = SupplyForecastingModel()
        latest_status_ai = df_supply_hist.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['item'], keep='last')
        latest_status_ai = latest_status_ai[latest_status_ai['item'].isin(final_items)]
        df_input_ai = latest_status_ai.rename(columns={
            'item_stock_agg_zone': 'current_stock',
            'consumption_rate_per_day': 'avg_daily_consumption_historical',
            'encounter_date': 'last_stock_update_date'
        })[['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']]
        if not df_input_ai.empty:
            try: df_forecast_detail = ai_model.forecast_supply_levels_advanced(df_input_ai, forecast_days_out)
            except Exception as e: logger.error(f"({module_log_prefix}) AI forecast error: {e}", exc_info=True); output_data["data_processing_notes"].append(f"AI forecast failed: {e}")
        else: output_data["data_processing_notes"].append(f"No current status data for AI forecast of: {final_items}")
    else:
        logger.info(f"({module_log_prefix}) Initiating Simple Linear Supply Forecasting for: {final_items}")
        try: df_forecast_detail = generate_simple_supply_forecast(df_supply_hist, forecast_days_out, final_items, f"{module_log_prefix}/SimpleLinear")
        except Exception as e: logger.error(f"({module_log_prefix}) Simple forecast error: {e}", exc_info=True); output_data["data_processing_notes"].append(f"Simple forecast failed: {e}")

    # Process and Summarize Forecast
    if isinstance(df_forecast_detail, pd.DataFrame) and not df_forecast_detail.empty:
        output_data["forecast_detail_df"] = df_forecast_detail.sort_values(by=['item', 'forecast_date']).reset_index(drop=True)
        overview_list: List[Dict[str, Any]] = []
        stockout_col = 'estimated_stockout_date_ai' if use_ai_supply_forecasting_model else 'estimated_stockout_date_linear'
        
        for item_name in df_forecast_detail['item'].unique():
            item_df = df_forecast_detail[df_forecast_detail['item'] == item_name]
            if item_df.empty: continue

            initial_stock, base_cons = np.nan, np.nan
            if not use_ai_supply_forecasting_model and 'initial_stock_at_forecast_start' in item_df.columns:
                initial_stock = item_df['initial_stock_at_forecast_start'].iloc[0]
                base_cons = item_df['base_consumption_rate_per_day'].iloc[0]
            else: # For AI model, or if simple model output varies
                # Attempt to get original current_stock from the input data if AI model was used
                if use_ai_supply_forecasting_model and 'df_input_ai' in locals() and not df_input_ai.empty: # Check if df_input_ai was defined
                    item_input_status = df_input_ai[df_input_ai['item'] == item_name]
                    if not item_input_status.empty:
                         initial_stock = item_input_status['current_stock'].iloc[0]
                         base_cons = item_input_status['avg_daily_consumption_historical'].iloc[0] # Base historical for AI model
                if pd.isna(initial_stock): # Fallback if not found above
                    initial_stock = item_df['forecasted_stock_level'].iloc[0] + item_df['predicted_daily_consumption'].iloc[0] if 'predicted_daily_consumption' in item_df.columns else item_df['forecasted_stock_level'].iloc[0]
                if pd.isna(base_cons) and 'predicted_daily_consumption' in item_df.columns:
                    base_cons = item_df['predicted_daily_consumption'].iloc[0] # Use first day's prediction as proxy
                elif pd.isna(base_cons):
                    base_cons = 1e-7 # Absolute fallback

            if pd.isna(base_cons) or base_cons <= 1e-8: base_cons = 1e-7
            initial_dos = (initial_stock / base_cons) if base_cons > 1e-8 and pd.notna(initial_stock) else np.inf
            
            stockout_dt_val = pd.NaT
            if stockout_col in item_df.columns:
                first_valid_stockout = item_df[stockout_col].dropna().iloc[0] if not item_df[stockout_col].dropna().empty else pd.NaT
                stockout_dt_val = pd.to_datetime(first_valid_stockout, errors='coerce')

            overview_list.append({
                "item_name": item_name,
                "current_stock_on_hand_at_forecast_start": float(initial_stock) if pd.notna(initial_stock) else np.nan,
                "avg_daily_consumption_rate_used": float(base_cons),
                "initial_days_of_supply_estimated": round(initial_dos, 1) if np.isfinite(initial_dos) else "Adequate",
                "estimated_stockout_date": stockout_dt_val.strftime('%Y-%m-%d') if pd.notna(stockout_dt_val) else "Beyond Forecast"
            })
        output_data["forecast_items_overview_list"] = overview_list
        
    elif not output_data["data_processing_notes"]: # Forecast empty, no specific error notes yet
        output_data["data_processing_notes"].append("Supply forecast resulted in no data with selected model/available records.")

    logger.info(f"({module_log_prefix}) Supply forecast prep complete. Items in overview: {len(output_data.get('forecast_items_overview_list',[]))}")
    return output_data
