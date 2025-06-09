# sentinel_project_root/analytics/supply_forecasting.py
# SME PLATINUM STANDARD - ADVANCED SUPPLY FORECASTING

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import settings

# Prophet is an optional dependency for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
logger = logging.getLogger(__name__)


def generate_linear_forecast(
    source_df: pd.DataFrame,
    forecast_days: int = 30,
    item_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generates a simple, day-by-day linear consumption forecast.
    """
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return pd.DataFrame()
        
    df = source_df.copy()
    if item_filter:
        df = df[df['item'].isin(item_filter)]

    # Get the most recent status for each item
    latest_status = df.sort_values('encounter_date', ascending=False).drop_duplicates('item')
    if latest_status.empty:
        return pd.DataFrame()

    all_forecasts = []
    today = pd.Timestamp.now().normalize()

    for _, row in latest_status.iterrows():
        rate = max(row.get('consumption_rate_per_day', 0.001), 0.001)
        stock = row.get('item_stock_agg_zone', 0)
        
        dates = pd.date_range(start=today, periods=forecast_days, freq='D')
        forecasted_stock = stock - (rate * np.arange(forecast_days))
        
        item_df = pd.DataFrame({
            'item': row['item'],
            'forecast_date': dates,
            'forecasted_stock': np.maximum(0, forecasted_stock),
        })
        item_df['days_of_supply'] = item_df['forecasted_stock'] / rate
        all_forecasts.append(item_df)
        
    return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()


def generate_prophet_forecast(
    source_df: pd.DataFrame,
    item_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generates an advanced forecast using Meta's Prophet model, accounting for
    trends and seasonality.
    """
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet library not installed. Falling back to linear forecast.")
        return generate_linear_forecast(source_df, settings.ANALYTICS.prophet_forecast_days, item_filter)

    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return pd.DataFrame()

    df = source_df.copy()
    if item_filter:
        df = df[df['item'].isin(item_filter)]

    all_forecasts = []
    for item_name, group in df.groupby('item'):
        if len(group) < 5:  # Not enough data to forecast
            logger.warning(f"Skipping forecast for '{item_name}': not enough data points ({len(group)}).")
            continue
            
        # Prepare data for Prophet: requires 'ds' and 'y' columns
        history = group[['encounter_date', 'consumption_rate_per_day']].copy()
        history.rename(columns={'encounter_date': 'ds', 'consumption_rate_per_day': 'y'}, inplace=True)
        history = history.dropna().sort_values('ds')
        
        if len(history) < 5: continue # Check again after dropping NAs

        try:
            # Initialize and fit the model
            model = Prophet(
                changepoint_prior_scale=settings.ANALYTICS.prophet_changepoint_prior_scale,
                seasonality_prior_scale=settings.ANALYTICS.prophet_seasonality_prior_scale,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(history)

            # Create future dataframe and predict
            future = model.make_future_dataframe(periods=settings.ANALYTICS.prophet_forecast_days)
            forecast = model.predict(future)
            
            # Combine forecast with latest stock info
            latest_stock = group.sort_values('encounter_date').iloc[-1]['item_stock_agg_zone']
            
            # Project stock levels based on forecasted consumption
            forecast['projected_consumption'] = forecast['yhat'].clip(lower=0).cumsum()
            forecast['forecasted_stock'] = latest_stock - forecast['projected_consumption']
            forecast.loc[forecast['forecasted_stock'] < 0, 'forecasted_stock'] = 0
            
            # Add item name and select relevant columns
            forecast['item'] = item_name
            
            # Only keep future predictions
            future_forecast = forecast[forecast['ds'] > history['ds'].max()]
            
            all_forecasts.append(future_forecast[['ds', 'item', 'yhat', 'yhat_lower', 'yhat_upper', 'forecasted_stock']])
        
        except Exception as e:
            logger.error(f"Prophet forecast failed for item '{item_name}': {e}", exc_info=True)
            
    if not all_forecasts:
        return pd.DataFrame()

    final_df = pd.concat(all_forecasts, ignore_index=True)
    # Rename columns for clarity in the UI
    final_df.rename(columns={
        'ds': 'forecast_date',
        'yhat': 'predicted_daily_consumption',
        'yhat_lower': 'consumption_lower_bound',
        'yhat_upper': 'consumption_upper_bound',
    }, inplace=True)

    return final_df
