# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Prepares supply forecast overview data by calling the core analytics models.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List

try:
    from config import settings
    from analytics.supply_forecasting import forecast_supply_levels_advanced, generate_simple_supply_forecast
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in supply_forecast.py: {e}. Check paths/dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Constants for Clarity ---
UI_OUTPUT_COLS = ['item', 'days_of_supply_remaining', 'estimated_stockout_date', 'stock_status']


class ClinicSupplyForecastPreparer:
    """Orchestrates the supply forecast data preparation process for the UI."""

    def __init__(self, historical_health_df: Optional[pd.DataFrame], use_ai_model: bool):
        self.use_ai_model = use_ai_model
        self.df_historical = historical_health_df if isinstance(historical_health_df, pd.DataFrame) else pd.DataFrame()
        self.notes: List[str] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        """Safely retrieves a configuration value from the global settings object."""
        return getattr(settings, attr_name, default_value)

    def _find_items_to_forecast(self) -> List[str]:
        """Identifies relevant supply items based on configured keywords."""
        if self.df_historical.empty or 'item' not in self.df_historical.columns:
            return []

        key_drugs = self._get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
        if not key_drugs:
            self.notes.append("Configuration Error: KEY_DRUG_SUBSTRINGS_SUPPLY is not defined in settings.")
            return []
        
        pattern = '|'.join(re.escape(s) for s in key_drugs if s)
        if not pattern: return []

        all_items = self.df_historical['item'].dropna().unique()
        items = [item for item in all_items if re.search(pattern, item, re.IGNORECASE)]
        
        if not items:
            self.notes.append("No supply items in the data matched the configured keywords.")
        return sorted(items)

    def _calculate_stockout_dates(self, forecast_df: pd.DataFrame) -> pd.Series:
        """Calculates the first date an item is forecasted to run out."""
        if forecast_df.empty: return pd.Series(dtype='datetime64[ns]')
        
        stockout_rows = forecast_df[forecast_df['forecasted_days_of_supply'] < 1].sort_values('forecast_date').drop_duplicates('item', keep='first')
        
        return stockout_rows.set_index('item')['forecast_date'] if not stockout_rows.empty else pd.Series(dtype='datetime64[ns]')

    def _add_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a 'stock_status' column based on days of supply remaining."""
        if df.empty or 'days_of_supply_remaining' not in df.columns:
            return df.assign(stock_status="Unknown")

        dos = pd.to_numeric(df['days_of_supply_remaining'], errors='coerce')
        critical = self._get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        warning = self._get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)

        conditions = [dos < critical, dos < warning, dos.notna()]
        choices = ["Critical Low", "Warning Low", "Sufficient"]
        df['stock_status'] = np.select(conditions, choices, default="Unknown")
        return df

    def prepare(self) -> Dict[str, Any]:
        """Main method to generate the complete forecast overview."""
        items_to_forecast = self._find_items_to_forecast()
        if not items_to_forecast:
            self.notes.append("Forecast halted; no items identified for analysis.")
            return {"forecast_items_overview_list": [], "forecast_model_type_used": "N/A", "processing_notes": self.notes}

        model_name, forecast_func = ("AI-Assisted (Simulated)", forecast_supply_levels_advanced) if self.use_ai_model else ("Simple Linear", generate_simple_supply_forecast)
        
        try:
            detailed_forecast_df = forecast_func(source_df=self.df_historical, item_filter=items_to_forecast)
        except Exception as e:
            logger.error(f"Forecasting model '{model_name}' failed: {e}", exc_info=True)
            self.notes.append(f"A critical error occurred in the '{model_name}' model.")
            detailed_forecast_df = pd.DataFrame()

        # Start with a scaffold to ensure all requested items are in the final output.
        final_df = pd.DataFrame({'item': items_to_forecast})
        
        if isinstance(detailed_forecast_df, pd.DataFrame) and not detailed_forecast_df.empty:
            current_status = detailed_forecast_df.sort_values('forecast_date').drop_duplicates('item', keep='first')
            final_df = pd.merge(final_df, current_status[['item', 'forecasted_days_of_supply']], on='item', how='left')
            
            stockout_dates = self._calculate_stockout_dates(detailed_forecast_df)
            final_df = pd.merge(final_df, stockout_dates.rename('estimated_stockout_date'), on='item', how='left')
        else:
             self.notes.append("Forecasting model did not produce data.")

        # --- DEFINITIVE FIX FOR KeyError / AttributeError ---
        # Rename column *after* merge and checks
        final_df.rename(columns={'forecasted_days_of_supply': 'days_of_supply_remaining'}, inplace=True)
        final_df = self._add_status_column(final_df)
        
        # Ensure the required columns exist before trying to format them.
        if 'days_of_supply_remaining' in final_df.columns:
            final_df['days_of_supply_remaining'] = final_df['days_of_supply_remaining'].round(1).fillna(0.0)
        else:
            final_df['days_of_supply_remaining'] = 0.0

        if 'estimated_stockout_date' in final_df.columns:
            final_df['estimated_stockout_date'] = pd.to_datetime(final_df['estimated_stockout_date']).dt.strftime('%Y-%m-%d').fillna('N/A')
        else:
            final_df['estimated_stockout_date'] = 'N/A'

        final_df['stock_status'] = final_df.get('stock_status', "Unknown").fillna("Unknown")
        
        # Reindex to ensure final output has a consistent schema and order
        final_df = final_df.reindex(columns=UI_OUTPUT_COLS).fillna({
            'item': 'Unknown Item',
            'days_of_supply_remaining': 0.0, 
            'estimated_stockout_date': 'N/A', 
            'stock_status': 'Unknown'
        })
        
        return {
            "forecast_items_overview_list": final_df.to_dict('records'),
            "forecast_model_type_used": model_name,
            "processing_notes": self.notes
        }

def prepare_clinic_supply_forecast_overview_data(
    historical_health_df: Optional[pd.DataFrame],
    use_ai_supply_forecasting_model: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Factory function to prepare an overview of supply forecasts for the UI."""
    preparer = ClinicSupplyForecastPreparer(
        historical_health_df=historical_health_df,
        use_ai_model=use_ai_supply_forecasting_model
    )
    return preparer.prepare()
