# sentinel_project_root/analytics/supply_forecasting.py
# Contains supply forecasting models (AI-simulated and simple linear) for Sentinel.

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from datetime import date, timedelta

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)

class SupplyForecastingModel:
    """
    Simulates an AI-driven supply forecasting model that might incorporate factors like
    seasonality, trends, and stochastic noise for more "realistic" (though still simulated) forecasts.
    """
    def __init__(self):
        # Use a fixed seed based on settings.RANDOM_SEED for consistent pseudo-random parameter generation
        self._rng_params_init = np.random.RandomState(settings.RANDOM_SEED)
        
        self.item_specific_params: Dict[str, Dict[str, Any]] = {}
        base_monthly_coeffs = np.array([1.0] * 12) # Base: no seasonality
        
        for item_key_config in settings.KEY_DRUG_SUBSTRINGS_SUPPLY:
            item_key_lower = item_key_config.lower() # Store and match keys in lower case
            # Use the instance-level RNG for consistent item parameter generation
            self.item_specific_params[item_key_lower] = {
                "monthly_coeffs": (base_monthly_coeffs * self._rng_params_init.uniform(0.80, 1.20, 12)).round(3).tolist(),
                "annual_trend_factor": self._rng_params_init.uniform(0.0005, 0.005),
                "random_noise_std_dev": self._rng_params_init.uniform(0.03, 0.10)
            }
        
        act_key_lower = "act" # Example: Specific parameters for a known seasonal item like "ACT"
        if act_key_lower in self.item_specific_params: # Check if "act" is part of configured drugs
            # Simulate higher consumption during typical malaria seasons
            self.item_specific_params[act_key_lower]["monthly_coeffs"] = [0.7,0.7,0.8,1.0,1.4,1.7,1.8,1.6,1.2,0.9,0.8,0.7]
            self.item_specific_params[act_key_lower]["annual_trend_factor"] = 0.0020 # Slightly higher trend for ACT
        
        logger.info(f"AI-Simulated SupplyForecastingModel initialized with pseudo-params for {len(self.item_specific_params)} item types.")

    def _get_simulated_item_params(self, item_name_input: str) -> Dict[str, Any]:
        """Retrieves simulated parameters for an item, with a fallback to defaults."""
        item_name_lower_case = str(item_name_input).strip().lower()
        
        # Try direct match first
        if item_name_lower_case in self.item_specific_params:
            return self.item_specific_params[item_name_lower_case]
            
        # Try partial match using substrings
        for key_drug_substr_lower, params_val in self.item_specific_params.items():
            if key_drug_substr_lower in item_name_lower_case:
                return params_val
        
        logger.debug(f"No specific AI sim params for item '{item_name_input}', using generic defaults.")
        return {
            "monthly_coeffs": [1.0] * 12, "annual_trend_factor": 0.0001,
            "random_noise_std_dev": 0.05
        }

    def _predict_daily_consumption_ai_sim(
        self, base_avg_daily_consumption_hist: float, item_name: str,
        forecast_target_date: pd.Timestamp, day_number_in_forecast: int
    ) -> float:
        """Simulates predicting daily consumption using 'AI' factors."""
        if pd.isna(base_avg_daily_consumption_hist) or base_avg_daily_consumption_hist <= 1e-7:
            return 1e-7 # Return a tiny positive consumption

        item_sim_params = self._get_simulated_item_params(item_name)
        
        month_idx = forecast_target_date.month - 1 # 0-indexed for list
        seasonality_multiplier = item_sim_params["monthly_coeffs"][month_idx]
        trend_multiplier = (1 + item_sim_params["annual_trend_factor"]) ** day_number_in_forecast
        
        # Create a new RNG instance for noise for each prediction to ensure variability,
        # but seed it consistently for reproducibility across runs of the same forecast.
        # Hash incorporates item_name, target_date, and day_number to make seed unique per prediction point.
        noise_seed = (settings.RANDOM_SEED + 
                      hash(item_name) + 
                      forecast_target_date.toordinal() + 
                      day_number_in_forecast) % (2**32) # Keep seed within valid range
        rng_noise_predict = np.random.RandomState(noise_seed)
        noise_multiplier = rng_noise_predict.normal(loc=1.0, scale=item_sim_params["random_noise_std_dev"])
        noise_multiplier = np.clip(noise_multiplier, 0.5, 1.5) # Cap noise effect

        predicted_cons = base_avg_daily_consumption_hist * seasonality_multiplier * trend_multiplier * noise_multiplier
        return max(1e-7, predicted_cons) # Ensure consumption is non-negative

    def forecast_supply_levels_advanced(
        self, current_supply_levels_df: pd.DataFrame, 
        forecast_days_out: int = 30,
        item_filter_list_optional: Optional[List[str]] = None
    ) -> pd.DataFrame:
        module_log_prefix = "AISupplyForecastAdvanced"
        output_df_columns = ['item', 'forecast_date', 'forecasted_stock_level',
                             'forecasted_days_of_supply', 'predicted_daily_consumption',
                             'estimated_stockout_date_ai']

        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            logger.warning(f"({module_log_prefix}) Input current_supply_levels_df is empty or invalid.")
            return pd.DataFrame(columns=output_df_columns)

        required_input_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        missing_cols_input = [col for col in required_input_cols if col not in current_supply_levels_df.columns]
        if missing_cols_input:
            logger.error(f"({module_log_prefix}) Missing required columns in current_supply_levels_df: {missing_cols_input}.")
            return pd.DataFrame(columns=output_df_columns)

        df_input_cleaned = current_supply_levels_df.copy()
        if item_filter_list_optional: # Apply item filter if provided
            df_input_cleaned = df_input_cleaned[df_input_cleaned['item'].isin(item_filter_list_optional)]
            if df_input_cleaned.empty:
                 logger.warning(f"({module_log_prefix}) No items remaining after applying item_filter_list. Forecast aborted.")
                 return pd.DataFrame(columns=output_df_columns)

        df_input_cleaned['last_stock_update_date'] = pd.to_datetime(df_input_cleaned['last_stock_update_date'], errors='coerce')
        df_input_cleaned['current_stock'] = convert_to_numeric(df_input_cleaned['current_stock'], default_value=0.0)
        df_input_cleaned['avg_daily_consumption_historical'] = convert_to_numeric(df_input_cleaned['avg_daily_consumption_historical'], default_value=1e-7)
        df_input_cleaned.loc[df_input_cleaned['avg_daily_consumption_historical'] <= 0, 'avg_daily_consumption_historical'] = 1e-7
        df_input_cleaned.dropna(subset=['item', 'last_stock_update_date'], inplace=True)

        if df_input_cleaned.empty:
            logger.warning(f"({module_log_prefix}) No valid data rows remaining after cleaning input for AI forecast.")
            return pd.DataFrame(columns=output_df_columns)

        all_forecast_records: List[Dict[str, Any]] = []
        for _, initial_status_row in df_input_cleaned.iterrows():
            item_name_forecast = initial_status_row['item']
            current_stock_level = max(0.0, initial_status_row['current_stock'])
            base_hist_consumption = initial_status_row['avg_daily_consumption_historical']
            forecast_start_date = initial_status_row['last_stock_update_date']

            running_stock_level = current_stock_level
            calculated_stockout_date_for_item: Optional[pd.Timestamp] = pd.NaT
            item_daily_forecasts: List[Dict[str, Any]] = []

            for day_offset in range(forecast_days_out):
                target_forecast_date = forecast_start_date + pd.Timedelta(days=day_offset + 1)
                predicted_consumption_today = self._predict_daily_consumption_ai_sim(
                    base_hist_consumption, item_name_forecast, target_forecast_date, day_offset + 1
                )
                stock_before_consumption_today = running_stock_level
                running_stock_level = max(0.0, running_stock_level - predicted_consumption_today)
                days_of_supply_remaining = (running_stock_level / predicted_consumption_today) if predicted_consumption_today > 1e-8 else np.inf
                
                if pd.isna(calculated_stockout_date_for_item) and stock_before_consumption_today > 0 and running_stock_level <= 0:
                    fraction_of_day_to_stockout = (stock_before_consumption_today / predicted_consumption_today) if predicted_consumption_today > 1e-8 else 0.0
                    calculated_stockout_date_for_item = forecast_start_date + pd.Timedelta(days=day_offset + fraction_of_day_to_stockout)

                item_daily_forecasts.append({
                    'item': item_name_forecast, 'forecast_date': target_forecast_date,
                    'forecasted_stock_level': running_stock_level,
                    'forecasted_days_of_supply': days_of_supply_remaining,
                    'predicted_daily_consumption': predicted_consumption_today,
                    'estimated_stockout_date_ai': calculated_stockout_date_for_item
                })
            
            if pd.isna(calculated_stockout_date_for_item) and current_stock_level > 0 and item_daily_forecasts:
                avg_predicted_consumption_in_period = pd.Series([d['predicted_daily_consumption'] for d in item_daily_forecasts]).mean()
                if avg_predicted_consumption_in_period > 1e-8:
                    estimated_days_to_stockout_beyond = current_stock_level / avg_predicted_consumption_in_period
                    calculated_stockout_date_for_item = forecast_start_date + pd.to_timedelta(estimated_days_to_stockout_beyond, unit='D')
                for record in item_daily_forecasts: # Update NaT stockout dates
                    if pd.isna(record['estimated_stockout_date_ai']):
                        record['estimated_stockout_date_ai'] = calculated_stockout_date_for_item
            all_forecast_records.extend(item_daily_forecasts)

        if not all_forecast_records:
            logger.warning(f"({module_log_prefix}) No forecast records generated after processing all items.")
            return pd.DataFrame(columns=output_df_columns)

        final_forecast_df = pd.DataFrame(all_forecast_records)
        final_forecast_df['estimated_stockout_date_ai'] = pd.to_datetime(final_forecast_df['estimated_stockout_date_ai'], errors='coerce')
        logger.info(f"({module_log_prefix}) AI-simulated supply forecast complete: {len(final_forecast_df)} records for {df_input_cleaned['item'].nunique()} items.")
        return final_forecast_df[output_df_columns]


def generate_simple_supply_forecast(
    health_df_for_supply: Optional[pd.DataFrame],
    forecast_days_out: int = 30,
    item_filter_list: Optional[List[str]] = None,
    source_context: str = "SimpleSupplyForecast"
) -> pd.DataFrame:
    logger.info(f"({source_context}) Generating simple linear supply forecast. Horizon: {forecast_days_out} days.")
    output_cols_simple = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply',
                          'estimated_stockout_date_linear', 'initial_stock_at_forecast_start',
                          'base_consumption_rate_per_day']

    if not isinstance(health_df_for_supply, pd.DataFrame) or health_df_for_supply.empty:
        logger.warning(f"({source_context}) Input health_df_for_supply is empty or invalid.")
        return pd.DataFrame(columns=output_cols_simple)

    required_cols_simple = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    missing_cols = [col for col in required_cols_simple if col not in health_df_for_supply.columns]
    if missing_cols:
        logger.error(f"({source_context}) Missing required columns in health_df_for_supply: {missing_cols}.")
        return pd.DataFrame(columns=output_cols_simple)

    df_supply_src = health_df_for_supply[required_cols_simple].copy()
    df_supply_src['encounter_date'] = pd.to_datetime(df_supply_src['encounter_date'], errors='coerce')
    df_supply_src.dropna(subset=['encounter_date', 'item'], inplace=True)
    df_supply_src['item_stock_agg_zone'] = convert_to_numeric(df_supply_src['item_stock_agg_zone'], default_value=0.0)
    df_supply_src['consumption_rate_per_day'] = convert_to_numeric(df_supply_src['consumption_rate_per_day'], default_value=1e-7)
    df_supply_src.loc[df_supply_src['consumption_rate_per_day'] <= 0, 'consumption_rate_per_day'] = 1e-7

    if item_filter_list:
        df_supply_src = df_supply_src[df_supply_src['item'].isin(item_filter_list)]
    if df_supply_src.empty:
        logger.info(f"({source_context}) No valid data rows after cleaning/filtering for simple forecast.")
        return pd.DataFrame(columns=output_cols_simple)

    latest_item_status_df = df_supply_src.sort_values('encounter_date', na_position='first').drop_duplicates(subset=['item'], keep='last')
    all_forecast_records_simple: List[Dict[str, Any]] = []
    effective_forecast_start_date = pd.Timestamp(date.today()) # Forecast from "today"

    for _, item_row in latest_item_status_df.iterrows():
        item_name_simple = item_row['item']
        initial_stock_level = max(0.0, item_row['item_stock_agg_zone'])
        base_daily_consumption_rate = item_row['consumption_rate_per_day']
        
        initial_dos_at_start = (initial_stock_level / base_daily_consumption_rate) if base_daily_consumption_rate > 1e-8 else np.inf
        estimated_stockout_dt_linear: Optional[pd.Timestamp] = pd.NaT
        if np.isfinite(initial_dos_at_start):
            estimated_stockout_dt_linear = effective_forecast_start_date + pd.to_timedelta(initial_dos_at_start, unit='D')

        for day_idx in range(forecast_days_out):
            current_forecast_date = effective_forecast_start_date + pd.Timedelta(days=day_idx)
            # Stock level at the *start* of `current_forecast_date`
            running_stock_level_simple = max(0.0, initial_stock_level - (base_daily_consumption_rate * day_idx))
            current_dos = (running_stock_level_simple / base_daily_consumption_rate) if base_daily_consumption_rate > 1e-8 else np.inf
            
            all_forecast_records_simple.append({
                'item': item_name_simple, 'forecast_date': current_forecast_date,
                'forecasted_stock_level': running_stock_level_simple,
                'forecasted_days_of_supply': current_dos,
                'estimated_stockout_date_linear': estimated_stockout_dt_linear,
                'initial_stock_at_forecast_start': initial_stock_level,
                'base_consumption_rate_per_day': base_daily_consumption_rate
            })

    if not all_forecast_records_simple:
        logger.warning(f"({source_context}) No simple forecast records generated.")
        return pd.DataFrame(columns=output_cols_simple)

    final_simple_forecast_df = pd.DataFrame(all_forecast_records_simple)
    final_simple_forecast_df['estimated_stockout_date_linear'] = pd.to_datetime(final_simple_forecast_df['estimated_stockout_date_linear'], errors='coerce')
    logger.info(f"({source_context}) Simple linear supply forecast complete: {len(final_simple_forecast_df)} records for {latest_item_status_df['item'].nunique()} items.")
    return final_simple_forecast_df[output_cols_simple]
