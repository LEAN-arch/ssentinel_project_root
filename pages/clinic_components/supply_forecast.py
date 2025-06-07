# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Generates supply forecast overview data for the Clinic Console.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List

# --- Module Imports ---
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
    df_historical: pd.DataFrame,
    forecast_horizon_days: int,
    items_to_forecast: List[str],
    log_prefix: str = "SimpleSupplyForecast"
) -> pd.DataFrame:
    """
    A simplified supply forecasting model.
    Estimates future stock based on the latest stock and average daily consumption.
    """
    output_cols = ['item', 'current_stock_level', 'avg_daily_consumption_rate', 'days_of_supply_remaining', 'estimated_stockout_date']

    if not isinstance(df_historical, pd.DataFrame) or df_historical.empty or not items_to_forecast:
        return pd.DataFrame(columns=output_cols)

    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in df_historical.columns for col in required_cols):
        logger.error(f"({log_prefix}) Missing required columns for forecasting. Aborting.")
        return pd.DataFrame(columns=output_cols)

    df = df_historical.copy()
    
    # Ensure datetime is timezone-naive before calculations
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(df['encounter_date']) and df['encounter_date'].dt.tz is not None:
        df['encounter_date'] = df['encounter_date'].dt.tz_localize(None)
    
    df.dropna(subset=['encounter_date', 'item'], inplace=True)
    df['item_stock_agg_zone'] = convert_to_numeric(df['item_stock_agg_zone'], default_value=0.0)
    df['consumption_rate_per_day'] = convert_to_numeric(df['consumption_rate_per_day'], default_value=0.001)
    df.loc[df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 0.001

    if df.empty: return pd.DataFrame(columns=output_cols)
        
    latest_records = df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    forecasts = []
    today = pd.Timestamp.now().normalize()

    for item in items_to_forecast:
        record = latest_records[latest_records['item'] == item]
        if record.empty:
            forecasts.append({"item": item, "days_of_supply_remaining": 0.0, "estimated_stockout_date": "No History"})
            continue

        row = record.iloc[0]
        days_since_update = (today - row['encounter_date']).days
        current_stock = max(0, row['item_stock_agg_zone'] - (days_since_update * row['consumption_rate_per_day']))
        
        dos = current_stock / row['consumption_rate_per_day'] if row['consumption_rate_per_day'] > 0 else float('inf')
        
        stockout_date = "N/A"
        if np.isfinite(dos):
            try:
                stockout_date = (today + pd.to_timedelta(dos, unit='D')).strftime('%Y-%m-%d')
            except (OverflowError, ValueError): 
                stockout_date = ">5Y"
        
        forecasts.append({
            "item": item,
            "current_stock_level": round(current_stock, 1),
            "avg_daily_consumption_rate": round(row['consumption_rate_per_day'], 3),
            "days_of_supply_remaining": round(dos, 1) if np.isfinite(dos) else "N/A",
            "estimated_stockout_date": stockout_date
        })
    
    return pd.DataFrame(forecasts, columns=output_cols)


def prepare_clinic_supply_forecast_overview_data(
    full_historical_health_df: Optional[pd.DataFrame],
    current_period_context_str: str,
    use_ai_supply_forecasting_model: bool = False
) -> Dict[str, Any]:
    """Prepares an overview of supply forecasts."""
    forecast_days = int(_get_setting('DEFAULT_SUPPLY_FORECAST_HORIZON_DAYS', 30))
    output: Dict[str, Any] = {
        "forecast_items_overview_list": [],
        "forecast_model_type_used": "AI Advanced (Simulated)" if use_ai_supply_forecasting_model else "Simple Aggregate",
        "processing_notes": []
    }

    if not isinstance(full_historical_health_df, pd.DataFrame) or full_historical_health_df.empty:
        output["processing_notes"].append("Historical health data is empty.")
        return output

    key_drugs = _get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
    if not key_drugs:
        output["processing_notes"].append("KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings.")
        return output

    drug_pattern = '|'.join([re.escape(s) for s in key_drugs if s])
    if not drug_pattern:
        output["processing_notes"].append("No valid drug keywords in settings.")
        return output

    all_items = full_historical_health_df['item'].dropna().unique()
    items_to_forecast = [item for item in all_items if re.search(drug_pattern, item, re.IGNORECASE)]

    if not items_to_forecast:
        output["processing_notes"].append("No items matching keywords found in data.")
        return output
    
    forecast_df = pd.DataFrame()
    if use_ai_supply_forecasting_model:
        # Placeholder for future AI model call
        output["processing_notes"].append("AI Advanced Forecasting is currently simulated by the simple model.")
    
    try:
        forecast_df = generate_simple_supply_forecast(full_historical_health_df, forecast_days, items_to_forecast)
    except Exception as e:
        logger.error(f"Error in forecast generation: {e}", exc_info=True)
        output["processing_notes"].append(f"Error during forecast: {e}")

    if not forecast_df.empty:
        if 'days_of_supply_remaining' in forecast_df.columns:
            dos_numeric = pd.to_numeric(forecast_df['days_of_supply_remaining'], errors='coerce')
            critical_thresh = _get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
            warning_thresh = _get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)
            conditions = [dos_numeric < critical_thresh, dos_numeric < warning_thresh, dos_numeric.notna()]
            choices = ["Critical Low", "Warning Low", "Sufficient"]
            forecast_df['stock_status'] = np.select(conditions, choices, default="Unknown")
        
        output["forecast_items_overview_list"] = forecast_df.to_dict('records')

    return output
