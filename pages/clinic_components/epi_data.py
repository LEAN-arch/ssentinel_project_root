# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Generates supply forecast overview data for the Clinic Console.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Tuple

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in supply_forecast.py: {e}. Ensure paths/dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# --- Constants ---
COL_ITEM = 'item'
COL_DATE = 'encounter_date'
COL_STOCK = 'item_stock_agg_zone'
COL_CONSUMPTION = 'consumption_rate_per_day'

# --- Forecasting Strategies ---

class SimpleAggregateForecaster:
    """A vectorized, efficient implementation of the simple supply forecasting model."""
    
    def __init__(self, historical_df: pd.DataFrame, items_to_forecast: List[str]):
        self.df = historical_df
        self.items_to_forecast = items_to_forecast
        self.model_name = "Simple Aggregate"

    def forecast(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculates future stock based on latest stock and average daily consumption.
        Ensures all requested items have a row in the output.
        """
        output_cols = [COL_ITEM, 'current_stock_level', 'avg_daily_consumption_rate', 'days_of_supply_remaining', 'estimated_stockout_date']
        
        if not self.items_to_forecast:
            return pd.DataFrame(columns=output_cols), ["No items identified for forecasting."]

        # Create a base DataFrame to ensure all items are included in the final output.
        results_df = pd.DataFrame({COL_ITEM: self.items_to_forecast})
        
        if self.df.empty:
            notes = ["Historical data is empty; cannot generate forecasts."]
        else:
            # 1. Get the single most recent record for each relevant item
            latest_records = self.df[self.df[COL_ITEM].isin(self.items_to_forecast)] \
                .sort_values(COL_DATE) \
                .drop_duplicates(subset=[COL_ITEM], keep='last')
            
            if not latest_records.empty:
                today = pd.Timestamp.now().normalize()
                
                # 2. Vectorized Calculations on available data
                latest_records['days_since_update'] = (today - latest_records[COL_DATE]).dt.days
                latest_records['current_stock_level'] = np.maximum(0, latest_records[COL_STOCK] - (latest_records['days_since_update'] * latest_records[COL_CONSUMPTION]))
                latest_records['days_of_supply_remaining'] = latest_records['current_stock_level'] / latest_records[COL_CONSUMPTION]
                
                def calculate_stockout_date(dos):
                    if not np.isfinite(dos) or dos < 0: return "Indefinite"
                    try: return (today + pd.to_timedelta(dos, unit='D')).strftime('%Y-%m-%d')
                    except (OverflowError, ValueError): return ">5 Years"

                latest_records['estimated_stockout_date'] = latest_records['days_of_supply_remaining'].apply(calculate_stockout_date)
                latest_records['days_of_supply_remaining'] = latest_records['days_of_supply_remaining'].replace([np.inf, -np.inf], np.nan)
                
                # 3. Merge calculated data back to the complete list
                results_df = pd.merge(results_df, latest_records, on=COL_ITEM, how='left')
            notes = []

        # 4. Fill default values for items that had no historical data
        fill_values = {
            'current_stock_level': 0.0,
            'avg_daily_consumption_rate': 0.0,
            'days_of_supply_remaining': 0.0,
            'estimated_stockout_date': "No History"
        }
        results_df.fillna(value=fill_values, inplace=True)
        results_df.rename(columns={COL_CONSUMPTION: 'avg_daily_consumption_rate'}, inplace=True)

        return results_df[output_cols], notes

class AIAdvancedForecaster(SimpleAggregateForecaster):
    """Placeholder for a more advanced forecasting model. Currently simulates by using the simple model."""
    def __init__(self, historical_df: pd.DataFrame, items_to_forecast: List[str]):
        super().__init__(historical_df, items_to_forecast)
        self.model_name = "AI Advanced (Simulated)"

    def forecast(self) -> Tuple[pd.DataFrame, List[str]]:
        forecast_df, _ = super().forecast()
        notes = ["AI Advanced Forecasting is currently simulated by the simple model."]
        return forecast_df, notes

# --- Main Preparation Orchestrator ---

class ClinicSupplyForecastPreparer:
    """Orchestrates the supply forecast data preparation process."""
    def __init__(self, full_historical_df: pd.DataFrame, use_ai_model: bool):
        self.use_ai_model = use_ai_model
        self.df_historical = self._prepare_clean_dataframe(full_historical_df)
        self.notes: List[str] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)
    
    def _prepare_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.notes.append("Historical health data is empty.")
            return pd.DataFrame()

        required_cols = [COL_ITEM, COL_DATE, COL_STOCK, COL_CONSUMPTION]
        if not all(col in df.columns for col in required_cols):
            self.notes.append("Missing one or more required columns for forecasting.")
            return pd.DataFrame()
        
        clean_df = df.copy()
        clean_df[COL_DATE] = pd.to_datetime(clean_df[COL_DATE], errors='coerce').dt.tz_localize(None)
        clean_df.dropna(subset=[COL_DATE, COL_ITEM], inplace=True)
        clean_df[COL_STOCK] = convert_to_numeric(clean_df[COL_STOCK], default_value=0.0)
        clean_df[COL_CONSUMPTION] = convert_to_numeric(clean_df[COL_CONSUMPTION], default_value=0.001)
        clean_df.loc[clean_df[COL_CONSUMPTION] <= 0, COL_CONSUMPTION] = 0.001
        return clean_df

    def _find_items_to_forecast(self) -> List[str]:
        key_drugs = self._get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
        if not key_drugs:
            self.notes.append("KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings.")
            return []
        
        drug_pattern = '|'.join([re.escape(s) for s in key_drugs if s])
        if not drug_pattern or COL_ITEM not in self.df_historical.columns: return []

        all_items = self.df_historical[COL_ITEM].dropna().unique()
        items = [item for item in all_items if re.search(drug_pattern, item, re.IGNORECASE)]
        
        if not items:
            self.notes.append("No items matching keywords found in data.")
        return items

    def _add_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'days_of_supply_remaining' not in df.columns: return df
        
        dos_numeric = pd.to_numeric(df['days_of_supply_remaining'], errors='coerce')
        critical_thresh = self._get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        warning_thresh = self._get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)
        
        conditions = [dos_numeric < critical_thresh, dos_numeric < warning_thresh, dos_numeric.notna()]
        choices = ["Critical Low", "Warning Low", "Sufficient"]
        df['stock_status'] = np.select(conditions, choices, default="Unknown")
        return df

    def prepare(self) -> Dict[str, Any]:
        items_to_forecast = self._find_items_to_forecast()
        
        if self.use_ai_model:
            forecaster = AIAdvancedForecaster(self.df_historical, items_to_forecast)
        else:
            forecaster = SimpleAggregateForecaster(self.df_historical, items_to_forecast)
        
        forecast_df, model_notes = forecaster.forecast()
        self.notes.extend(model_notes)
        final_df = self._add_status_column(forecast_df)

        return {
            "forecast_items_overview_list": final_df.to_dict('records'),
            "forecast_model_type_used": forecaster.model_name,
            "processing_notes": self.notes
        }

def prepare_clinic_supply_forecast_overview_data(
    full_historical_health_df: Optional[pd.DataFrame],
    current_period_context_str: str,
    use_ai_supply_forecasting_model: bool = False
) -> Dict[str, Any]:
    """Factory function to prepare an overview of supply forecasts."""
    preparer = ClinicSupplyForecastPreparer(full_historical_health_df, use_ai_supply_forecasting_model)
    return preparer.prepare()
