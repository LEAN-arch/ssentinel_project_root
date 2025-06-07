# sentinel_project_root/analytics/supply_forecasting.py
"""
Contains supply forecasting models (AI-simulated and simple linear) for Sentinel.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from datetime import date, timedelta

# --- Module Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in supply_forecasting.py: {e}. Ensure paths are correct.", exc_info=True)
    raise

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


class SupplyForecastingModel:
    """
    Simulates an AI-driven supply forecasting model that incorporates factors like
    seasonality, trends, and stochastic noise for more "realistic" forecasts.
    """
    # FIXED: Renamed `init` to the correct Python constructor `__init__`.
    def __init__(self):
        """Initializes the model with pseudo-random, reproducible parameters."""
        self._rng_params_init = np.random.RandomState(settings.RANDOM_SEED)
        self.item_specific_params: Dict[str, Dict[str, Any]] = {}
        
        for item_key_config in settings.KEY_DRUG_SUBSTRINGS_SUPPLY:
            item_key_lower = item_key_config.lower()
            self.item_specific_params[item_key_lower] = {
                "monthly_coeffs": (np.ones(12) * self._rng_params_init.uniform(0.80, 1.20, 12)).round(3).tolist(),
                "annual_trend_factor": self._rng_params_init.uniform(0.0005, 0.005),
                "random_noise_std_dev": self._rng_params_init.uniform(0.03, 0.10)
            }
        
        act_key_lower = "act"
        if act_key_lower in self.item_specific_params:
            self.item_specific_params[act_key_lower]["monthly_coeffs"] = [0.7, 0.7, 0.8, 1.0, 1.4, 1.7, 1.8, 1.6, 1.2, 0.9, 0.8, 0.7]
            self.item_specific_params[act_key_lower]["annual_trend_factor"] = 0.0020
        
        logger.info(f"AI-Simulated SupplyForecastingModel initialized with params for {len(self.item_specific_params)} item types.")

    def _get_simulated_item_params(self, item_name_input: str) -> Dict[str, Any]:
        """Retrieves simulated parameters for an item, with a fallback to defaults."""
        item_name_lower = str(item_name_input).strip().lower()
        
        if item_name_lower in self.item_specific_params:
            return self.item_specific_params[item_name_lower]
            
        for key_substr, params in self.item_specific_params.items():
            if key_substr in item_name_lower:
                return params
        
        return {"monthly_coeffs": [1.0] * 12, "annual_trend_factor": 0.0001, "random_noise_std_dev": 0.05}

    def _predict_daily_consumption_ai_sim(
        self, base_avg_daily_consumption: float, item_name: str,
        forecast_date: pd.Timestamp, day_in_forecast: int
    ) -> float:
        """Simulates predicting daily consumption using 'AI' factors."""
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 1e-7:
            return 1e-7

        params = self._get_simulated_item_params(item_name)
        
        seasonality = params["monthly_coeffs"][forecast_date.month - 1]
        trend = (1 + params["annual_trend_factor"]) ** day_in_forecast
        
        noise_seed = (settings.RANDOM_SEED + hash(item_name) + forecast_date.toordinal() + day_in_forecast) % (2**32)
        rng_noise = np.random.RandomState(noise_seed)
        noise = np.clip(rng_noise.normal(loc=1.0, scale=params["random_noise_std_dev"]), 0.5, 1.5)

        predicted_consumption = base_avg_daily_consumption * seasonality * trend * noise
        return max(1e-7, predicted_consumption)

    def forecast_supply_levels_advanced(
        self, current_supply_levels_df: pd.DataFrame, 
        forecast_days_out: int = 30,
        item_filter_list_optional: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generates an AI-simulated supply forecast."""
        output_cols = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']

        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            return pd.DataFrame(columns=output_cols)

        required_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if not all(col in current_supply_levels_df.columns for col in required_cols):
            logger.error(f"Missing required columns for AI forecast: {list(set(required_cols) - set(current_supply_levels_df.columns))}")
            return pd.DataFrame(columns=output_cols)

        df = current_supply_levels_df.copy()
        if item_filter_list_optional:
            df = df[df['item'].isin(item_filter_list_optional)]

        df['last_stock_update_date'] = pd.to_datetime(df['last_stock_update_date'], errors='coerce')
        df['current_stock'] = convert_to_numeric(df['current_stock'], default_value=0.0)
        df['avg_daily_consumption_historical'] = convert_to_numeric(df['avg_daily_consumption_historical'], default_value=1e-7)
        df.loc[df['avg_daily_consumption_historical'] <= 0, 'avg_daily_consumption_historical'] = 1e-7
        df.dropna(subset=['item', 'last_stock_update_date'], inplace=True)

        if df.empty: return pd.DataFrame(columns=output_cols)

        all_forecasts = []
        for _, row in df.iterrows():
            running_stock = max(0.0, row['current_stock'])
            stockout_date = pd.NaT
            item_forecasts = []

            for day_offset in range(forecast_days_out):
                target_date = row['last_stock_update_date'] + pd.Timedelta(days=day_offset + 1)
                predicted_cons = self._predict_daily_consumption_ai_sim(row['avg_daily_consumption_historical'], row['item'], target_date, day_offset + 1)
                
                stock_before = running_stock
                running_stock = max(0.0, running_stock - predicted_cons)
                
                if pd.isna(stockout_date) and stock_before > 0 and running_stock <= 0:
                    fraction = (stock_before / predicted_cons) if predicted_cons > 1e-8 else 0.0
                    stockout_date = row['last_stock_update_date'] + pd.Timedelta(days=day_offset + fraction)

                item_forecasts.append({
                    'item': row['item'], 'forecast_date': target_date, 'forecasted_stock_level': running_stock,
                    'forecasted_days_of_supply': (running_stock / predicted_cons) if predicted_cons > 1e-8 else np.inf,
                    'predicted_daily_consumption': predicted_cons, 'estimated_stockout_date_ai': stockout_date
                })
            all_forecasts.extend(item_forecasts)

        if not all_forecasts: return pd.DataFrame(columns=output_cols)

        final_df = pd.DataFrame(all_forecasts)
        final_df['estimated_stockout_date_ai'] = pd.to_datetime(final_df['estimated_stockout_date_ai'], errors='coerce')
        logger.info(f"AI-simulated supply forecast complete: {len(final_df)} records generated.")
        return final_df[output_cols]


def generate_simple_supply_forecast(
    health_df_for_supply: Optional[pd.DataFrame],
    forecast_days_out: int = 30,
    item_filter_list: Optional[List[str]] = None,
    source_context: str = "SimpleSupplyForecast"
) -> pd.DataFrame:
    """Generates a simple, linear supply forecast based on the last known consumption rate."""
    output_cols = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'estimated_stockout_date_linear', 'initial_stock_at_forecast_start', 'base_consumption_rate_per_day']
    
    if not isinstance(health_df_for_supply, pd.DataFrame) or health_df_for_supply.empty:
        return pd.DataFrame(columns=output_cols)

    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if not all(col in health_df_for_supply.columns for col in required_cols):
        logger.error(f"Missing required columns for simple forecast: {list(set(required_cols) - set(health_df_for_supply.columns))}")
        return pd.DataFrame(columns=output_cols)

    df = health_df_for_supply[required_cols].copy()
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    df.dropna(subset=['encounter_date', 'item'], inplace=True)
    df['item_stock_agg_zone'] = convert_to_numeric(df['item_stock_agg_zone'], default_value=0.0)
    df['consumption_rate_per_day'] = convert_to_numeric(df['consumption_rate_per_day'], default_value=1e-7)
    df.loc[df['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-7

    if item_filter_list:
        df = df[df['item'].isin(item_filter_list)]
    if df.empty: return pd.DataFrame(columns=output_cols)

    latest_status = df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    all_forecasts = []
    forecast_start_date = pd.Timestamp(date.today())

    for _, row in latest_status.iterrows():
        days_since_update = max(0, (forecast_start_date - row['encounter_date']).days)
        initial_stock = max(0.0, row['item_stock_agg_zone'] - (row['consumption_rate_per_day'] * days_since_update))
        
        dos = (initial_stock / row['consumption_rate_per_day']) if row['consumption_rate_per_day'] > 1e-8 else np.inf
        stockout_date = forecast_start_date + pd.to_timedelta(dos, unit='D') if np.isfinite(dos) else pd.NaT

        for day_offset in range(forecast_days_out):
            current_date = forecast_start_date + pd.Timedelta(days=day_offset)
            running_stock = max(0.0, initial_stock - (row['consumption_rate_per_day'] * day_offset))
            current_dos = (running_stock / row['consumption_rate_per_day']) if row['consumption_rate_per_day'] > 1e-8 else np.inf
            
            all_forecasts.append({
                'item': row['item'], 'forecast_date': current_date,
                'forecasted_stock_level': running_stock, 'forecasted_days_of_supply': current_dos,
                'estimated_stockout_date_linear': stockout_date,
                'initial_stock_at_forecast_start': initial_stock,
                'base_consumption_rate_per_day': row['consumption_rate_per_day']
            })

    if not all_forecasts: return pd.DataFrame(columns=output_cols)

    final_df = pd.DataFrame(all_forecasts)
    final_df['estimated_stockout_date_linear'] = pd.to_datetime(final_df['estimated_stockout_date_linear'], errors='coerce')
    logger.info(f"Simple linear supply forecast complete: {len(final_df)} records generated.")
    return final_df[output_cols]
