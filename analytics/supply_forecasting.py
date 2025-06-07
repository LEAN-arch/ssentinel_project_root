# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Prepares supply forecast overview data by calling the core analytics models.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List

try:
    from config import settings
    # This import is now safe and will not cause a circular dependency.
    from analytics.supply_forecasting import forecast_supply_levels_advanced, generate_simple_supply_forecast
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in supply_forecast.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class ClinicSupplyForecastPreparer:
    """Orchestrates the supply forecast data preparation process for the UI."""
    def __init__(self, full_historical_df: pd.DataFrame, use_ai_model: bool):
        self.use_ai_model = use_ai_model
        self.df_historical = full_historical_df if isinstance(full_historical_df, pd.DataFrame) else pd.DataFrame()
        self.notes: List[str] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)
    
    def _find_items_to_forecast(self) -> List[str]:
        if self.df_historical.empty or 'item' not in self.df_historical.columns: return []
        key_drugs = self._get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
        if not key_drugs: return []
        drug_pattern = '|'.join([re.escape(s) for s in key_drugs if s])
        if not drug_pattern: return []
        all_items = self.df_historical['item'].dropna().unique()
        return [item for item in all_items if re.search(drug_pattern, item, re.IGNORECASE)]

    def _summarize_forecast_for_ui(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty: return pd.DataFrame()
        summary = forecast_df.sort_values('forecast_date').drop_duplicates(subset=['item'], keep='first')
        summary = summary.rename(columns={'forecasted_days_of_supply': 'days_of_supply_remaining'})
        return summary

    def _add_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'days_of_supply_remaining' not in df.columns: return df
        dos = pd.to_numeric(df['days_of_supply_remaining'], errors='coerce')
        critical = self._get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        warning = self._get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)
        conditions = [dos < critical, dos < warning, dos.notna()]
        choices = ["Critical Low", "Warning Low", "Sufficient"]
        df['stock_status'] = np.select(conditions, choices, default="Unknown")
        return df

    def prepare(self) -> Dict[str, Any]:
        if self.df_historical.empty:
            self.notes.append("Historical data is empty.")
            return {"forecast_items_overview_list": [], "forecast_model_type_used": "N/A", "processing_notes": self.notes}

        items_to_forecast = self._find_items_to_forecast()
        if not items_to_forecast:
            self.notes.append("No items matching forecast keywords found in data.")
            return {"forecast_items_overview_list": [], "forecast_model_type_used": "N/A", "processing_notes": self.notes}

        if self.use_ai_model:
            model_name, forecast_func = "AI-Assisted (Simulated)", forecast_supply_levels_advanced
        else:
            model_name, forecast_func = "Simple Linear", generate_simple_supply_forecast
            
        detailed_forecast_df = forecast_func(source_df=self.df_historical, item_filter=items_to_forecast)
        
        summary_df = self._summarize_forecast_for_ui(detailed_forecast_df)
        final_df = self._add_status_column(summary_df)

        return {
            "forecast_items_overview_list": final_df.to_dict('records'),
            "forecast_model_type_used": model_name,
            "processing_notes": self.notes
        }

def prepare_clinic_supply_forecast_overview_data(
    full_historical_health_df: Optional[pd.DataFrame],
    **_kwargs
) -> Dict[str, Any]:
    """Factory function to prepare an overview of supply forecasts for the UI."""
    use_ai = _kwargs.get('use_ai_supply_forecasting_model', False)
    preparer = ClinicSupplyForecastPreparer(full_historical_health_df, use_ai)
    return preparer.prepare()
