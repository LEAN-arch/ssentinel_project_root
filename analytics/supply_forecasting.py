# sentinel_project_root/analytics/supply_forecasting.py
# SME PLATINUM STANDARD - ADVANCED SUPPLY FORECASTING (V4 - DEFINITIVE FIX)

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
    # ... [function body is correct and remains unchanged] ...
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

def generate_prophet_forecast(source_df: pd.DataFrame, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet library not installed. Falling back to linear forecast.")
        return generate_linear_forecast(source_df, settings.ANALYTICS.prophet_forecast_days, item_filter)

    if not isinstance(source_df, pd.DataFrame) or source_df.empty: return pd.DataFrame()

    df = source_df.copy()
    if item_filter: df = df[df['item'].isin(item_filter)]

    all_forecasts = []
    for item_name, group in df.groupby('item'):
        if len(group) < 5: continue
            
        history = group[['encounter_date', 'consumption_rate_per_day']].copy()
        history.rename(columns={'encounter_date': 'ds', 'consumption_rate_per_day': 'y'}, inplace=True)
        history = history.dropna().sort_values('ds')
        
        # SME FIX: Remove timezone information from the 'ds' column before fitting.
        history['ds'] = history['ds'].dt.tz_localize(None)
        
        if len(history) < 5: continue

        try:
            model = Prophet(changepoint_prior_scale=settings.ANALYTICS.prophet_changepoint_prior_scale, seasonality_prior_scale=settings.ANALYTICS.prophet_seasonality_prior_scale, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(history)
            future = model.make_future_dataframe(periods=settings.ANALYTICS.prophet_forecast_days)
            forecast = model.predict(future)
            latest_stock = group.sort_values('encounter_date').iloc[-1]['item_stock_agg_zone']
            forecast['projected_consumption'] = forecast['yhat'].clip(lower=0).cumsum()
            forecast['forecasted_stock'] = latest_stock - forecast['projected_consumption']
            forecast.loc[forecast['forecasted_stock'] < 0, 'forecasted_stock'] = 0
            forecast['item'] = item_name
            future_forecast = forecast[forecast['ds'] > history['ds'].max()]
            all_forecasts.append(future_forecast[['ds', 'item', 'yhat', 'yhat_lower', 'yhat_upper', 'forecasted_stock']])
        except Exception as e:
            logger.error(f"Prophet forecast failed for item '{item_name}': {e}", exc_info=True)
            
    if not all_forecasts: return pd.DataFrame()

    final_df = pd.concat(all_forecasts, ignore_index=True)
    final_df.rename(columns={'ds': 'forecast_date', 'yhat': 'predicted_daily_consumption', 'yhat_lower': 'consumption_lower_bound', 'yhat_upper': 'consumption_upper_bound'}, inplace=True)
    return final_df
