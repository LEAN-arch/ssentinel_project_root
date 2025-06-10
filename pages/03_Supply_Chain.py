# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - SUPPLY CHAIN DASHBOARD (V4 - DEFINITIVE PLOT FIX)

import logging

import pandas as pd
import streamlit as st

from analytics import generate_prophet_forecast
from config import settings
from data_processing import load_health_records
# SME FIX: Import the correct plotting function for a simple line chart.
from visualization import create_empty_figure, plot_forecast_chart, plot_line_chart

st.set_page_config(page_title="Supply Chain", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600, show_spinner="Loading supply data...")
def get_data() -> pd.DataFrame:
    """Loads and caches the health records which contain supply data."""
    return load_health_records()

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def get_supply_forecasts(df: pd.DataFrame, items: list, days: int) -> pd.DataFrame:
    """
    Loops through selected items and calls the generic prophet forecaster for each.
    """
    all_forecasts = []
    for item in items:
        item_df = df[df['item'] == item].copy()
        if not item_df.empty and 'consumption_rate_per_day' in item_df.columns:
            history = item_df[['encounter_date', 'consumption_rate_per_day']].rename(columns={'encounter_date': 'ds', 'consumption_rate_per_day': 'y'})
            
            forecast = generate_prophet_forecast(history, forecast_days=days)
            
            if not forecast.empty:
                latest_stock = item_df.sort_values('encounter_date').iloc[-1]['item_stock_agg_zone']
                forecast['item'] = item
                forecast['projected_consumption'] = forecast['predicted_value'].clip(lower=0).cumsum()
                forecast['forecasted_stock'] = latest_stock - forecast['projected_consumption']
                forecast.loc[forecast['forecasted_stock'] < 0, 'forecasted_stock'] = 0
                all_forecasts.append(forecast)

    return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

def main():
    st.title("📦 Supply Chain & Logistics Console")
    st.markdown("Monitor stock levels, analyze consumption, and forecast supply needs using AI-powered models.")
    st.divider()

    full_df = get_data()
    if full_df.empty or 'item' not in full_df.columns:
        st.error("No supply data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Forecasting Controls")
        all_items = sorted(full_df['item'].dropna().unique())
        default_items = [item for item in settings.KEY_SUPPLY_ITEMS if item in all_items]
        selected_items = st.multiselect("Select Items to Forecast:", options=all_items, default=default_items[:3])
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7)

    st.header(f"📈 Consumption & Stock Forecast ({forecast_days} Days Ahead)")
    if not selected_items:
        st.info("Select one or more items from the sidebar to generate a forecast.")
    else:
        forecast_df = get_supply_forecasts(full_df, selected_items, forecast_days)

        if forecast_df.empty:
            st.warning("Could not generate a forecast for the selected items. There may not be enough historical data (at least 5 data points per item are required).")
        else:
            item_to_plot = st.selectbox("View Detailed Forecast For:", options=selected_items)
            
            item_forecast_df = forecast_df[forecast_df['item'] == item_to_plot]
            
            # --- Consumption Forecast Plot ---
            st.subheader(f"Forecasted Daily Consumption: {item_to_plot}")
            fig_consumption = plot_forecast_chart(
                item_forecast_df, 
                title=f"Consumption Forecast: {item_to_plot}", 
                y_title="Units per Day"
            )
            st.plotly_chart(fig_consumption, use_container_width=True)

            # --- Stock Level Forecast Plot ---
            st.subheader(f"Projected Stock Level: {item_to_plot}")
            
            # SME FIX: Use the correct plot type for a simple time series.
            # `plot_line_chart` is designed for this and does not require
            # the 'lower_bound' and 'upper_bound' columns.
            stock_series = item_forecast_df.set_index('forecast_date')['forecasted_stock']
            fig_stock = plot_line_chart(
                stock_series,
                title=f"Stock Level Forecast: {item_to_plot}",
                y_title="Units on Hand"
            )
            st.plotly_chart(fig_stock, use_container_width=True)

if __name__ == "__main__":
    main()
