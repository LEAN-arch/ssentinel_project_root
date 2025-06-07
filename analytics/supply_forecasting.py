# sentinel_project_root/pages/clinic_components/supply_forecast.py
# Prepares supply forecast overview data by calling the core analytics models.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List

try:
    from config import settings
    # This correctly imports the refactored functions from the core analytics module.
    from analytics.supply_forecasting import forecast_supply_levels_advanced, generate_simple_supply_forecast
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in supply_forecast.py: {e}. Check paths/dependencies.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# --- Main Preparation Orchestrator ---

class ClinicSupplyForecastPreparer:
    """Orchestrates the supply forecast data preparation process for the UI."""

    def __init__(self, full_historical_df: pd.DataFrame, use_ai_model: bool):
        self.use_ai_model = use_ai_model
        self.df_historical = full_historical_df if isinstance(full_historical_df, pd.DataFrame) else pd.DataFrame()
        self.notes: List[str] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)
    
    def _find_items_to_forecast(self) -> List[str]:
        """Identifies relevant supply items based on settings."""
        if self.df_historical.empty or 'item' not in self.df_historical.columns:
            return []
            
        key_drugs = self._get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
        if not key_drugs:
            self.notes.append("KEY_DRUG_SUBSTRINGS_SUPPLY not defined in settings.")
            return []
        
        drug_pattern = '|'.join([re.escape(s) for s in key_drugs if s])
        if not drug_pattern: return []

        all_items = self.df_historical['item'].dropna().unique()
        items = [item for item in all_items if re.search(drug_pattern, item, re.IGNORECASE)]
        
        if not items:
            self.notes.append("No items matching keywords found in data.")
        return items

    def _summarize_forecast_for_ui(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a detailed daily forecast and returns a one-row-per-item summary
        showing the current status (Days of Supply).
        """
        if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
            return pd.DataFrame()

        # Get the status for the first day of the forecast (i.e., "today")
        summary = forecast_df.sort_values('forecast_date').drop_duplicates(subset=['item'], keep='first')
        
        # Rename column for UI consistency
        summary.rename(columns={'forecasted_days_of_supply': 'days_of_supply_remaining'}, inplace=True)
        return summary

    def _add_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a 'stock_status' column based on days of supply remaining."""
        if df.empty or 'days_of_supply_remaining' not in df.columns: return df
        
        dos_numeric = pd.to_numeric(df['days_of_supply_remaining'], errors='coerce')
        critical_thresh = self._get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        warning_thresh = self._get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)
        
        conditions = [dos_numeric < critical_thresh, dos_numeric < warning_thresh, dos_numeric.notna()]
        choices = ["Critical Low", "Warning Low", "Sufficient"]
        df['stock_status'] = np.select(conditions, choices, default="Unknown")
        return df

    def prepare(self) -> Dict[str, Any]:
        """Main method to generate the complete forecast overview."""
        if self.df_historical.empty:
            self.notes.append("Historical data is empty, cannot generate forecast.")
            return {"forecast_items_overview_list": [], "forecast_model_type_used": "N/A", "processing_notes": self.notes}

        items_to_forecast = self._find_items_to_forecast()
        if not items_to_forecast:
            return {"forecast_items_overview_list": [], "forecast_model_type_used": "N/A", "processing_notes": self.notes}

        # --- Call the correct centralized analytics model ---
        if self.use_ai_model:
            model_name = "AI-Assisted (Simulated)"
            detailed_forecast_df = forecast_supply_levels_advanced(
                source_df=self.df_historical, item_filter_list_optional=items_to_forecast
            )
        else:
            model_name = "Simple Linear"
            detailed_forecast_df = generate_simple_supply_forecast(
                source_df=self.df_historical, item_filter_list=items_to_forecast
            )
        
        summary_df = self._summarize_forecast_for_ui(detailed_forecast_df)
        final_df = self._add_status_column(summary_df)

        return {
            "forecast_items_overview_list": final_df.to_dict('records'),
            "forecast_model_type_used": model_name,
            "processing_notes": self.notes
        }

def prepare_clinic_supply_forecast_overview_data(
    full_historical_health_df: Optional[pd.DataFrame],
    current_period_context_str: str, # Kept for signature compatibility, though not used in this version
    use_ai_supply_forecasting_model: bool = False
) -> Dict[str, Any]:
    """
    Factory function to prepare an overview of supply forecasts for the UI.
    Instantiates and runs the main preparer class.
    """
    preparer = ClinicSupplyForecastPreparer(
        full_historical_df=full_historical_health_df,
        use_ai_model=use_ai_supply_forecasting_model
    )
    return preparer.prepare()
