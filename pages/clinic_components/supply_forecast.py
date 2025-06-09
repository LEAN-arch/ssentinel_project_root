# sentinel_project_root/pages/clinic_components/supply_forecast.py
# SME PLATINUM STANDARD (V2 - ARCHITECTURAL REFACTOR)
# This version refactors the component into a more declarative and robust
# architecture using:
# 1. A fluent `.pipe()` based data pipeline for clarity.
# 2. An Enum for extensible, type-safe model selection.
# 3. Pydantic models for a self-validating, explicit output contract.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field # <<< SME REVISION V2

# --- Sentinel System Imports ---
try:
    from config import settings
    from analytics.supply_forecasting import forecast_supply_levels_advanced, generate_simple_supply_forecast
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in supply_forecast.py: {e}.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# <<< SME REVISION V2 >>> Use an Enum for extensible model selection.
class ForecastModelType(str, Enum):
    SIMPLE_TREND = "Simple Linear Trend"
    AI_ASSISTED = "AI-Assisted (Prophet/ARIMA)"

# <<< SME REVISION V2 >>> Use Pydantic models for a robust output contract.
class ForecastItem(BaseModel):
    item: str
    days_of_supply_remaining: float
    estimated_stockout_date: str
    stock_status: str

class ForecastResult(BaseModel):
    forecast_items_overview: List[ForecastItem] = Field(alias="forecast_items_overview_list")
    forecast_model_type_used: str
    processing_notes: List[str]

# --- Main Preparer Class ---
class ClinicSupplyForecastPreparer:
    """Orchestrates the supply forecast data preparation process for the UI using a fluent pipeline."""
    
    _MODEL_MAP = {
        ForecastModelType.SIMPLE_TREND: generate_simple_supply_forecast,
        ForecastModelType.AI_ASSISTED: forecast_supply_levels_advanced,
    }

    def __init__(self, historical_health_df: pd.DataFrame, model_type: ForecastModelType):
        self.df_historical = historical_health_df
        self.model_type = model_type
        self.notes: List[str] = []

    def _get_setting(self, attr_name: str, default_value: Any) -> Any:
        return getattr(settings, attr_name, default_value)

    # --- Pipeline Stage 1: Identify Items ---
    def _find_items_to_forecast(self) -> List[str]:
        # ... (This method's logic is already excellent and remains unchanged) ...
        if self.df_historical.empty or 'item' not in self.df_historical.columns: return []
        key_drugs = self._get_setting('KEY_DRUG_SUBSTRINGS_SUPPLY', [])
        if not key_drugs:
            self.notes.append("Config Warning: 'KEY_DRUG_SUBSTRINGS_SUPPLY' not in settings.")
            return []
        pattern = '|'.join(re.escape(s) for s in key_drugs if s)
        if not pattern: return []
        items = [i for i in self.df_historical['item'].dropna().unique() if re.search(pattern, i, re.IGNORECASE)]
        if not items: self.notes.append("No items matched forecasting keywords.")
        return sorted(items)

    # --- Pipeline Stage 2: Run Forecast Model ---
    def _run_forecast_model(self, items: List[str]) -> pd.DataFrame:
        if not items: return pd.DataFrame()
        forecast_func = self._MODEL_MAP[self.model_type]
        try:
            logger.info(f"Running '{self.model_type.value}' forecast for {len(items)} items.")
            return forecast_func(source_df=self.df_historical, item_filter=items)
        except Exception as e:
            logger.error(f"Model '{self.model_type.value}' failed: {e}", exc_info=True)
            self.notes.append(f"A critical error occurred in the '{self.model_type.value}' model.")
            return pd.DataFrame()

    # --- Pipeline Stage 3: Summarize Forecasts ---
    @staticmethod
    def _summarize_forecast_results(df_detailed: pd.DataFrame) -> pd.DataFrame:
        if df_detailed.empty: return pd.DataFrame()
        
        def summarize_item(df_item: pd.DataFrame) -> pd.Series:
            current_dos = df_item['forecasted_days_of_supply'].iloc[0]
            stockout_date = df_item.loc[df_item['forecasted_days_of_supply'] < 1, 'forecast_date'].iloc[0] if not df_item[df_item['forecasted_days_of_supply'] < 1].empty else pd.NaT
            return pd.Series({'days_of_supply_remaining': current_dos, 'estimated_stockout_date': stockout_date})
            
        return df_detailed.sort_values(['item', 'forecast_date']).groupby('item').apply(summarize_item).reset_index()

    # --- Pipeline Stage 4: Add Status and Format for UI ---
    def _format_for_ui(self, df_summary: pd.DataFrame, all_items: List[str]) -> pd.DataFrame:
        # Merge to ensure all items are present, even if forecast failed for some
        df_final = pd.merge(pd.DataFrame({'item': all_items}), df_summary, on='item', how='left')

        # Add status column
        dos = pd.to_numeric(df_final['days_of_supply_remaining'], errors='coerce')
        critical = self._get_setting('CRITICAL_SUPPLY_DAYS_REMAINING', 7)
        warning = self._get_setting('LOW_SUPPLY_DAYS_REMAINING', 14)
        conditions = [dos < critical, dos < warning, dos.notna()]
        choices = ["Critical Low", "Warning Low", "Sufficient"]
        df_final['stock_status'] = np.select(conditions, choices, default="Unknown")

        # Format columns
        df_final['days_of_supply_remaining'] = df_final['days_of_supply_remaining'].round(1)
        df_final['estimated_stockout_date'] = pd.to_datetime(df_final['estimated_stockout_date']).dt.strftime('%Y-%m-%d')
        
        # Fill NaNs and select final columns for a clean UI contract
        final_cols = ['item', 'days_of_supply_remaining', 'estimated_stockout_date', 'stock_status']
        return df_final.reindex(columns=final_cols).fillna({
            'days_of_supply_remaining': 0.0, 'estimated_stockout_date': 'N/A', 'stock_status': 'Unknown'
        })

    # --- Main Orchestration Method ---
    def prepare(self) -> ForecastResult:
        """Main method to generate the complete forecast overview using a fluent pipeline."""
        items_to_forecast = self._find_items_to_forecast()
        
        # <<< SME REVISION V2 >>> Use a fluent .pipe() based pipeline.
        forecast_df = (
            self._run_forecast_model(items_to_forecast)
            .pipe(self._summarize_forecast_results)
            .pipe(self._format_for_ui, all_items=items_to_forecast)
        )
        
        if items_to_forecast and forecast_df.empty:
            self.notes.append("Forecasting model did not produce any data.")

        return ForecastResult(
            forecast_items_overview_list=forecast_df.to_dict('records'),
            forecast_model_type_used=self.model_type.value,
            processing_notes=list(set(self.notes))
        )

# --- Public API Function ---
def prepare_clinic_supply_forecast_overview_data(
    historical_health_df: Optional[pd.DataFrame],
    use_ai_supply_forecasting_model: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Public function to prepare the supply forecast overview data."""
    model_type = ForecastModelType.AI_ASSISTED if use_ai_supply_forecasting_model else ForecastModelType.SIMPLE_TREND
    preparer = ClinicSupplyForecastPreparer(
        historical_health_df=historical_health_df if isinstance(historical_health_df, pd.DataFrame) else pd.DataFrame(),
        model_type=model_type
    )
    # Return as a dictionary to maintain compatibility with existing UI consumers.
    return preparer.prepare().model_dump(by_alias=True)
