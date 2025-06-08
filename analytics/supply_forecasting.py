# sentinel_project_root/analytics/supply_forecasting.py
# Contains models for forecasting medical supply levels.

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any

try:
    from config import settings
except ImportError:
    # Fallback for standalone execution or testing
    logging.warning("Could not import settings. Using fallback values for supply forecasting.")
    class FallbackSettings:
        DEFAULT_FORECAST_DAYS_OUT = 30
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

class SimpleForecastingModel:
    """
    A simple linear consumption model for supply forecasting.
    This implementation is vectorized for high performance.
    """
    def _prepare_latest_status(self, source_df: pd.DataFrame, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Gets the most recent status for each item."""
        df = source_df.copy()
        if item_filter:
            df = df[df['item'].isin(item_filter)]
        
        # The loader now guarantees 'encounter_date' is tz-naive and datetime
        df.dropna(subset=['encounter_date'], inplace=True)
        if df.empty:
            return pd.DataFrame()

        latest = df.sort_values('encounter_date', ascending=False).drop_duplicates('item')
        if latest.empty:
            return pd.DataFrame()
        
        # --- DEFINITIVE FIX FOR TypeError ---
        # Get the current time as a timezone-naive timestamp.
        today = pd.Timestamp.now().normalize()
        
        # Subtraction is now safe as both are tz-naive.
        latest['days_since_update'] = (today - latest['encounter_date']).dt.days.clip(lower=0)
        
        # Ensure consumption rate is a small positive number to avoid division by zero
        latest['consumption_rate_per_day'] = latest['consumption_rate_per_day'].clip(lower=0.001)
        return latest

    def forecast(self, source_df: pd.DataFrame, forecast_days: int, item_filter: Optional[List[str]]) -> pd.DataFrame:
        """Generates a day-by-day forecast for specified items."""
        latest_status = self._prepare_latest_status(source_df, item_filter)
        if latest_status.empty:
            return pd.DataFrame()

        all_forecasts = []
        today_naive = pd.Timestamp.now().normalize()

        for _, row in latest_status.iterrows():
            # Vectorized calculation for a single item's forecast
            dates = pd.date_range(start=today_naive, periods=forecast_days, freq='D')
            days_elapsed = np.arange(row['days_since_update'], row['days_since_update'] + forecast_days)
            
            forecasted_stock = row['item_stock_agg_zone'] - (row['consumption_rate_per_day'] * days_elapsed)
            days_of_supply = forecasted_stock / row['consumption_rate_per_day'] if row['consumption_rate_per_day'] > 0 else 0
            
            item_df = pd.DataFrame({
                'item': row['item'],
                'forecast_date': dates,
                'forecasted_stock_level': np.maximum(0, forecasted_stock),
                'forecasted_days_of_supply': np.maximum(0, days_of_supply)
            })
            all_forecasts.append(item_df)
            
        return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

# --- Singleton instance and public factory functions ---
SIMPLE_FORECASTER = SimpleForecastingModel()

def generate_simple_supply_forecast(
    source_df: pd.DataFrame, 
    forecast_days_out: int = getattr(settings, 'DEFAULT_FORECAST_DAYS_OUT', 30), 
    item_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """Public factory function for the simple supply forecasting model."""
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return pd.DataFrame()
    return SIMPLE_FORECASTER.forecast(source_df, forecast_days_out, item_filter)


def forecast_supply_levels_advanced(
    source_df: pd.DataFrame, 
    item_filter: Optional[List[str]] = None, 
    **kwargs
) -> pd.DataFrame:
    """
    Simulates a call to a more advanced AI/ML forecasting model.
    In a real scenario, this would involve a more complex model (e.g., SARIMA, Prophet).
    For this simulation, it delegates to the simple model.
    """
    logger.info("Executing AI-Assisted (Simulated) supply forecast.")
    return generate_simple_supply_forecast(source_df, item_filter=item_filter)
