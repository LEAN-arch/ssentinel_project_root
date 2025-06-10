# sentinel_project_root/analytics/supply_forecasting.py
# SME PLATINUM STANDARD - ADVANCED SUPPLY FORECASTING (V5 - GENERIC REFACTOR)

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import settings

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
logger = logging.getLogger(__name__)

def generate_linear_forecast(source_df: pd.DataFrame, forecast_days: int = 30, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
    # ... [This function remains correct for its specific use case] ...
    if not isinstance(source_df, pd.DataFrame) or source_df.empty: return pd.DataFrame()
    df = source_df.copy();
    if item_filter: df = df[df['item'].isin(item_filter)]
    latest_status = df.sort_values('encounter_date', ascending=False).drop_duplicates('item')
    if latest_status.empty: return pd.DataFrame()
    all_forecasts = []
    today = pd.Timestamp.now().normalize()
    for _, row in latest_status.iterrows():
        rate = max(row.get('consumption_rate_per_day', 0.001), 0.001)
        stock = row.get('item_stock_agg_zone', 0)
        dates = pd.date_range(start=today, periods=forecast_days, freq='D')
        forecasted_stock = stock - (rate * np.arange(forecast_days))
        item_df = pd.DataFrame({'item': row['item'], 'forecast_date': dates, 'forecasted_stock': np.maximum(0, forecasted_stock)})
        item_df['days_of_supply'] = item_df['forecasted_stock'] / rate
        all_forecasts.append(item_df)
    return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

def generate_prophet_forecast(
    history_df: pd.DataFrame,
    forecast_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates an advanced forecast for a SINGLE time series using Prophet.

    Args:
        history_df (pd.DataFrame): A DataFrame with two columns: 'ds' (datestamp) and 'y' (numeric value).
        forecast_days (int): The number of days to forecast into the future. Defaults to settings.
    
    Returns:
        A DataFrame with the forecast, including trend and uncertainty intervals.
    """
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet library not installed. Cannot generate forecast.")
        return pd.DataFrame()

    if not isinstance(history_df, pd.DataFrame) or history_df.empty or list(history_df.columns) != ['ds', 'y']:
        logger.error(f"Invalid input for generate_prophet_forecast. Expected a DataFrame with columns ['ds', 'y']. Got: {list(history_df.columns)}")
        return pd.DataFrame()

    # Ensure ds column is datetime and timezone-naive
    history_df['ds'] = pd.to_datetime(history_df['ds']).dt.tz_localize(None)
    
    if len(history_df.dropna()) < 5:
        logger.warning("Not enough data points (<5) to generate a reliable forecast.")
        return pd.DataFrame()

    try:
        days_out = forecast_days or settings.ANALYTICS.prophet_forecast_days
        model = Prophet(
            changepoint_prior_scale=settings.ANALYTICS.prophet_changepoint_prior_scale,
            seasonality_prior_scale=settings.ANALYTICS.prophet_seasonality_prior_scale,
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False
        )
        model.fit(history_df)
        future = model.make_future_dataframe(periods=days_out)
        forecast = model.predict(future)
        
        # Rename columns for clarity and consistency
        forecast.rename(columns={
            'ds': 'forecast_date',
            'yhat': 'predicted_value',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound',
        }, inplace=True)
        
        return forecast[['forecast_date', 'predicted_value', 'lower_bound', 'upper_bound']]
        
    except Exception as e:
        logger.error(f"Prophet forecast failed: {e}", exc_info=True)
        return pd.DataFrame()
