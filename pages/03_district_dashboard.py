# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - SUPPLY CHAIN DASHBOARD

import logging

import pandas as pd
import streamlit as st

from analytics import generate_linear_forecast, generate_prophet_forecast
from config import settings
from data_processing import load_health_records
from visualization import (create_empty_figure, plot_bar_chart,
                           plot_forecast_chart)

# --- Page Setup ---
st.set_page_config(page_title="Supply Chain", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)

# --- Data Loading ---
@st.cache_data(ttl=settings.WEB_CACHE_TTL_SECONDS, show_spinner="Loading supply data...")
def get_data() -> pd.DataFrame:
    return load_health_records()

# --- Main Page ---
st.title("📦 Supply Chain & Logistics Console")
st.markdown("Monitor stock levels, analyze consumption, and forecast supply needs.")
st.divider()

full_df = get_data()

if full_df.empty or 'item' not in full_df.columns:
    st.error("No supply data available. Dashboard cannot be rendered.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")
    all_items = sorted(full_df['item'].dropna().unique())
    selected_items = st.multiselect("Select Items to Forecast:", options=all_items, default=settings.KEY_SUPPLY_ITEMS[:3])
    
    use_prophet = st.toggle("Use AI-Assisted Forecast (Prophet)", value=True, help="Uses Prophet for advanced forecasting with seasonality. Slower but more accurate. If off, uses a simple linear trend.")

# --- Forecasting ---
st.header("📈 Supply & Consumption Forecast")
if not selected_items:
    st.info("Select one or more items from the sidebar to generate a forecast.")
else:
    with st.spinner("Generating forecasts..."):
        if use_prophet:
            forecast_df = generate_prophet_forecast(full_df, item_filter=selected_items)
            st.caption("Using AI-Assisted (Prophet) model.")
        else:
            forecast_df = generate_linear_forecast(full_df, item_filter=selected_items)
            st.caption("Using simple linear trend model.")

    if forecast_df.empty:
        st.warning("Could not generate a forecast for the selected items. There may not be enough historical data.")
    else:
        # Stock Level Forecast Plot
        fig_stock = plot_forecast_chart(
            forecast_df,
            title="Projected Stock Levels",
            y_title="Units on Hand"
        )
        st.plotly_chart(fig_stock, use_container_width=True)
        
        # Current Status Table
        st.subheader("Current Supply Status")
        latest_status = forecast_df.sort_values('forecast_date').drop_duplicates('item', keep='first')
        latest_status['Days of Supply'] = latest_status['forecasted_stock'] / latest_status['predicted_daily_consumption'].clip(lower=0.1)
        
        status_display = latest_status[['item', 'forecasted_stock', 'predicted_daily_consumption', 'Days of Supply']].copy()
        status_display.rename(columns={
            'item': 'Item',
            'forecasted_stock': 'Current Stock',
            'predicted_daily_consumption': 'Forecasted Daily Use',
        }, inplace=True)
        
        st.dataframe(status_display, hide_index=True, use_container_width=True,
            column_config={
                "Current Stock": st.column_config.NumberColumn(format="%d"),
                "Forecasted Daily Use": st.column_config.NumberColumn(format="%.1f"),
                "Days of Supply": st.column_config.ProgressColumn(
                    format="%.0f days",
                    min_value=0, max_value=max(30, status_display['Days of Supply'].max())
                )
            }
        )

st.divider()
st.caption(settings.APP_FOOTER_TEXT)
