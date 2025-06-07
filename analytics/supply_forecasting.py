# sentinel_project_root/analytics/supply_forecasting.py
"""
Contains efficient, vectorized supply forecasting models for Sentinel.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in supply_forecasting.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class BaseForecastingModel:
    """Base class for forecasting models, handling data prep and vectorization."""
    
    def __init__(self):
        self.model_name = "Base"

    def _prepare_latest_status(self, df: pd.DataFrame, item_filter: Optional[List[str]]) -> pd.DataFrame:
        """Validates, cleans, and gets the latest status for each item."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"({self.model_name}) Missing required columns for forecast.")
            return pd.DataFrame()

        clean_df = df.copy()
        if item_filter:
            clean_df = clean_df[clean_df['item'].isin(item_filter)]

        clean_df['encounter_date'] = pd.to_datetime(clean_df['encounter_date'], errors='coerce')
        clean_df.dropna(subset=['item', 'encounter_date'], inplace=True)
        clean_df['item_stock_agg_zone'] = convert_to_numeric(clean_df['item_stock_agg_zone'], 0.0)
        clean_df['consumption_rate_per_day'] = convert_to_numeric(clean_df['consumption_rate_per_day'], 1e-7)
        clean_df.loc[clean_df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-7

        latest = clean_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        
        today = pd.Timestamp('today').normalize()
        latest['days_since_update'] = (today - latest['encounter_date']).dt.days.clip(lower=0)
        latest['initial_stock'] = (latest['item_stock_agg_zone'] - latest['days_since_update'] * latest['consumption_rate_per_day']).clip(lower=0)
        
        return latest[['item', 'initial_stock', 'consumption_rate_per_day']]

    def _predict_consumption(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for consumption prediction. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement _predict_consumption.")

    def forecast(self, source_df: pd.DataFrame, forecast_days: int = 30, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Generates a vectorized supply forecast."""
        latest_status = self._prepare_latest_status(source_df, item_filter)
        if latest_status.empty:
            return pd.DataFrame()

        today = pd.Timestamp('today').normalize()
        date_range = pd.to_datetime([today + pd.Timedelta(days=i) for i in range(forecast_days)])
        
        forecast_grid = pd.MultiIndex.from_product([latest_status['item'].unique(), date_range], names=['item', 'forecast_date']).to_frame(index=False)
        forecast_df = pd.merge(forecast_grid, latest_status, on='item', how='left')

        forecast_df = self._predict_consumption(forecast_df)

        forecast_df['cumulative_consumption'] = forecast_df.groupby('item')['predicted_daily_consumption'].cumsum()
        forecast_df['forecasted_stock_level'] = (forecast_df['initial_stock'] - forecast_df['cumulative_consumption']).clip(lower=0)
        
        forecast_df['forecasted_days_of_supply'] = (forecast_df['forecasted_stock_level'] / forecast_df['predicted_daily_consumption']).replace([np.inf, -np.inf], np.nan)
        
        stockout_dates = forecast_df[forecast_df['forecasted_stock_level'] <= 0].groupby('item')['forecast_date'].min()
        forecast_df = pd.merge(forecast_df, stockout_dates.rename('estimated_stockout_date'), on='item', how='left')

        logger.info(f"({self.model_name}) Forecast complete: {len(forecast_df)} records generated.")
        return forecast_df

class SimpleLinearForecaster(BaseForecastingModel):
    """Generates a simple, linear forecast using the last known consumption rate."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "Simple Linear"

    def _predict_consumption(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """For the simple model, predicted consumption is the constant base rate."""
        forecast_df['predicted_daily_consumption'] = forecast_df['consumption_rate_per_day']
        return forecast_df

class AIAssistedForecaster(BaseForecastingModel):
    """Simulates an AI-driven forecast with seasonality, trend, and noise."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "AI-Assisted (Simulated)"
        self._rng = np.random.default_rng(settings.RANDOM_SEED)
        self.item_params = self._initialize_item_params()
        logger.info(f"({self.model_name}) Initialized with params for {len(self.item_params)} item types.")

    def _initialize_item_params(self) -> Dict[str, Dict]:
        params = {}
        for item_key in settings.KEY_DRUG_SUBSTRINGS_SUPPLY:
            params[item_key.lower()] = {
                "seasonality": self._rng.uniform(0.8, 1.2, 12),
                "trend": self._rng.uniform(0.0005, 0.005),
                "noise_std_dev": self._rng.uniform(0.03, 0.10)
            }
        if "act" in params:
            params["act"]["seasonality"] = np.array([0.7, 0.7, 0.8, 1.0, 1.4, 1.7, 1.8, 1.6, 1.2, 0.9, 0.8, 0.7])
        return params

    def _get_params_for_item(self, item_name: str) -> Dict:
        item_lower = item_name.lower()
        for key, params in self.item_params.items():
            if key in item_lower:
                return params
        return {"seasonality": np.ones(12), "trend": 0.0001, "noise_std_dev": 0.05}

    def _predict_consumption(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts daily consumption using 'AI' factors in a vectorized way."""
        params_df = forecast_df['item'].apply(self._get_params_for_item).apply(pd.Series)
        forecast_df = pd.concat([forecast_df, params_df], axis=1)

        today = pd.Timestamp('today').normalize()
        forecast_df['day_of_forecast'] = (forecast_df['forecast_date'] - today).dt.days
        forecast_df['month_of_forecast'] = forecast_df['forecast_date'].dt.month
        
        seasonality_factors = forecast_df.apply(lambda row: row['seasonality'][row['month_of_forecast'] - 1], axis=1)
        trend_factors = (1 + forecast_df['trend']) ** forecast_df['day_of_forecast']
        
        noise_factors = self._rng.normal(loc=1.0, scale=forecast_df['noise_std_dev'], size=len(forecast_df))
        
        forecast_df['predicted_daily_consumption'] = (
            forecast_df['consumption_rate_per_day'] * seasonality_factors * trend_factors * noise_factors.clip(0.5, 1.5)
        ).clip(lower=1e-7)

        return forecast_df

# --- Public Factory Functions for backward compatibility and ease of use ---

AI_FORECASTER = AIAssistedForecaster()
SIMPLE_FORECASTER = SimpleLinearForecaster()

def forecast_supply_levels_advanced(
    source_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list_optional: Optional[List[str]] = None
) -> pd.DataFrame:
    """Public function to run the AI-assisted forecast."""
    return AI_FORECASTER.forecast(source_df, forecast_days_out, item_filter_list_optional)

def generate_simple_supply_forecast(
    source_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None, **_kwargs
) -> pd.DataFrame:
    """Public function to run the simple linear forecast."""
    return SIMPLE_FORECASTER.forecast(source_df, forecast_days_out, item_filter_list)
