# ssentinel_project_root/analytics/supply_forecasting.py
"""
Contains efficient, vectorized supply forecasting models.
This is the core analytics engine for supply chain management.
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

# --- Base Class with Vectorized Forecasting Logic ---

class BaseForecastingModel:
    """Base class for forecasting models, handling data prep and vectorization."""
    
    def __init__(self):
        self.model_name = "Base"

    def _prepare_latest_status(self, df: pd.DataFrame, item_filter: Optional[List[str]]) -> pd.DataFrame:
        """Validates, cleans, and gets the latest status for each supply item."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"({self.model_name}) Missing required columns for forecast: {required_cols}")
            return pd.DataFrame()

        clean_df = df.copy()
        if item_filter:
            clean_df = clean_df[clean_df['item'].isin(item_filter)]

        clean_df['encounter_date'] = pd.to_datetime(clean_df['encounter_date'], errors='coerce')
        clean_df.dropna(subset=['item', 'encounter_date'], inplace=True)
        
        # Use robust numeric conversion
        clean_df['item_stock_agg_zone'] = convert_to_numeric(clean_df['item_stock_agg_zone'], 0.0)
        clean_df['consumption_rate_per_day'] = convert_to_numeric(clean_df['consumption_rate_per_day'], 1e-6) # Small default to avoid division by zero
        # Ensure consumption rate is a small positive number if zero or negative
        clean_df.loc[clean_df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-6

        latest = clean_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        
        today = pd.Timestamp('today').normalize()
        latest['days_since_update'] = (today - latest['encounter_date']).dt.days.clip(lower=0)
        # Adjust initial stock based on days since last update
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
        date_range = pd.date_range(start=today, periods=forecast_days, freq='D')
        
        # Create a grid of all items for all future dates
        forecast_grid = pd.MultiIndex.from_product([latest_status['item'].unique(), date_range], names=['item', 'forecast_date']).to_frame(index=False)
        forecast_df = pd.merge(forecast_grid, latest_status, on='item', how='left')

        # Let the specific model implementation predict daily consumption
        forecast_df = self._predict_consumption(forecast_df)

        # Vectorized calculation of stock levels over time
        forecast_df['cumulative_consumption'] = forecast_df.groupby('item')['predicted_daily_consumption'].cumsum()
        forecast_df['forecasted_stock_level'] = (forecast_df['initial_stock'] - forecast_df['cumulative_consumption']).clip(lower=0)
        
        logger.info(f"({self.model_name}) Forecast complete for {len(latest_status)} items over {forecast_days} days.")
        return forecast_df

# --- Concrete Model Implementations ---

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
    """Simulates an AI-driven forecast with seasonality and trend."""
    def __init__(self):
        super().__init__()
        self.model_name = "AI-Assisted (Simulated)"
        self._rng = np.random.default_rng(getattr(settings, 'RANDOM_SEED', 42))
        self.item_params = self._initialize_item_params()

    def _initialize_item_params(self) -> Dict[str, Dict]:
        """Creates simulated AI parameters for different item types."""
        params = {}
        for item_key in getattr(settings, 'KEY_DRUG_SUBSTRINGS_SUPPLY', []):
            params[item_key.lower()] = {
                "seasonality": self._rng.uniform(0.8, 1.2, 12), # Monthly factors
                "trend": self._rng.uniform(0.0005, 0.002),
                "noise_std_dev": self._rng.uniform(0.05, 0.15)
            }
        # Example of setting specific seasonality for a known item
        if "act" in params:
            params["act"]["seasonality"] = np.array([0.7,0.7,0.8,1.0,1.4,1.7,1.8,1.6,1.2,0.9,0.8,0.7])
        return params

    def _get_params_for_item(self, item_name: str) -> Dict:
        item_lower = item_name.lower()
        for key, params in self.item_params.items():
            if key in item_lower:
                return params
        return {"seasonality": np.ones(12), "trend": 0.0, "noise_std_dev": 0.05}

    def _predict_consumption(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts daily consumption using 'AI' factors in a vectorized way."""
        params_df = forecast_df['item'].apply(self._get_params_for_item).apply(pd.Series)
        
        # This can be slow on very large DFs, but is necessary for this simulation logic.
        # A true ML model might have a more direct vectorized prediction method.
        forecast_df['day_of_forecast'] = (forecast_df['forecast_date'] - pd.Timestamp('today').normalize()).dt.days
        forecast_df['month'] = forecast_df['forecast_date'].dt.month
        
        seasonal_factors = params_df.apply(lambda row: row['seasonality'][forecast_df['month'] - 1], axis=1).iloc[0]
        trend_factors = (1 + params_df['trend'].values[:, np.newaxis]) ** forecast_df['day_of_forecast'].values
        noise = self._rng.normal(1.0, params_df['noise_std_dev'].mean(), len(forecast_df))

        forecast_df['predicted_daily_consumption'] = (
            forecast_df['consumption_rate_per_day'] * seasonal_factors.values.flatten() * trend_factors.flatten() * noise
        ).clip(lower=1e-6)

        return forecast_df

# --- Public Interface (Preserved for backward compatibility) ---
AI_FORECASTER = AIAssistedForecaster()
SIMPLE_FORECASTER = SimpleLinearForecaster()

def forecast_supply_levels_advanced(source_df: pd.DataFrame, forecast_days_out: int = 30, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Public function to run the AI-assisted forecast."""
    return AI_FORECASTER.forecast(source_df, forecast_days_out, item_filter)

def generate_simple_supply_forecast(source_df: pd.DataFrame, forecast_days_out: int = 30, item_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Public function to run the simple linear forecast."""
    return SIMPLE_FORECASTER.forecast(source_df, forecast_days_out, item_filter)
