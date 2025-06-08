# sentinel_project_root/analytics/supply_forecasting.py
# Contains models for forecasting medical supply levels.

import pandas as pd
import numpy as np
import logging
from typing import Optional, List

try:
    from config import settings
except ImportError:
    logging.warning("Could not import settings. Using fallback values for supply forecasting.")
    class FallbackSettings:
        DEFAULT_FORECAST_DAYS_OUT = 30
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

class SimpleForecastingModel:
    """A simple linear consumption model for supply forecasting."""
    def _prepare_latest_status(self, source_df: pd.DataFrame, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Gets the most recent status for each item."""
        df = source_df.copy()
        if item_filter:
            df = df[df['item'].isin(item_filter)]
        
        latest = df.sort_values('encounter_date', ascending=False).drop_duplicates('item')
        if latest.empty: return pd.DataFrame()
        
        # --- DEFINITIVE FIX FOR TypeError ---
        # Make both datetime objects timezone-naive before subtraction.
        today = pd.Timestamp.now().tz_localize(None).normalize()
        latest['encounter_date'] = pd.to_datetime(latest['encounter_date']).dt.tz_localize(None)
        
        latest['days_since_update'] = (today - latest['encounter_date']).dt.days.clip(lower=0)
        latest['consumption_rate_per_day'] = latest['consumption_rate_per_day'].clip(lower=0.001)
        return latest

    def forecast(self, source_df: pd.DataFrame, forecast_days: int, item_filter: Optional[List[str]]) -> pd.DataFrame:
        """Generates a day-by-day forecast for specified items."""
        latest_status = self._prepare_latest_status(source_df, item_filter)
        if latest_status.empty: return pd.DataFrame()

        all_forecasts = []
        for _, row in latest_status.iterrows():
            forecasts = []
            current_stock, consumption_rate = row['item_stock_agg_zone'], row['consumption_rate_per_day']
            days_offset = row['days_since_update']
            
            dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=forecast_days, freq='D')
            days_elapsed = np.arange(days_offset, days_offset + forecast_days)
            
            forecasted_stock = current_stock - (consumption_rate * days_elapsed)
            days_of_supply = forecasted_stock / consumption_rate if consumption_rate > 0 else 0
            
            item_df = pd.DataFrame({
                'item': row['item'],
                'forecast_date': dates,
                'forecasted_stock_level': np.maximum(0, forecasted_stock),
                'forecasted_days_of_supply': np.maximum(0, days_of_supply)
            })
            all_forecasts.append(item_df)
            
        return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

SIMPLE_FORECASTER = SimpleForecastingModel()

def generate_simple_supply_forecast(source_df: pd.DataFrame, forecast_days_out: int = settings.DEFAULT_FORECAST_DAYS_OUT, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Public factory function for the simple supply forecasting model."""
    if not isinstance(source_df, pd.DataFrame) or source_df.empty: return pd.DataFrame()
    return SIMPLE_FORECASTER.forecast(source_df, forecast_days_out, item_filter)

def forecast_supply_levels_advanced(source_df: pd.DataFrame, item_filter: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Simulates a call to a more advanced AI/ML forecasting model."""
    logger.info("Executing AI-Assisted (Simulated) supply forecast.")
    return generate_simple_supply_forecast(source_df, item_filter=item_filter)
