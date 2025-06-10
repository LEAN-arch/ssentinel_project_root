# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - INTEGRATED LOGISTICS & CAPACITY DASHBOARD (V6 - FINAL)

import logging

import pandas as pd
import streamlit as st

from analytics import generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import create_empty_figure, plot_forecast_chart, plot_line_chart

# --- Page Setup ---
st.set_page_config(page_title="Logistics & Capacity", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)


# --- Supply Category Definitions ---
SUPPLY_CATEGORIES = {
    "Medications": {
        "items": settings.KEY_SUPPLY_ITEMS,
        "data_col": "item",
        "rate_col": "consumption_rate_per_day",
        "stock_col": "item_stock_agg_zone"
    },
    "Diagnostic Tests": {
        "items": list(settings.KEY_TEST_TYPES.keys()),
        "data_col": "test_type",
        "rate_col": "test_consumption_rate",
        "stock_col": "test_kit_stock"
    },
    "Clinic Supplies": {
        "items": ["Gloves", "Syringes", "Masks"],
        "data_col": "item",
        "rate_col": "consumption_rate_per_day",
        "stock_col": "item_stock_agg_zone"
    }
}


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and enriches all data for the dashboard."""
    health_df = load_health_records()
    iot_df = load_iot_records()
    
    # Simulate consumption data for tests for demonstration purposes
    if 'test_type' in health_df.columns:
        # Calculate daily consumption rate for each test type
        daily_test_counts = health_df.groupby([health_df['encounter_date'].dt.date, 'test_type']).size().reset_index(name='daily_count')
        avg_daily_consumption = daily_test_counts.groupby('test_type')['daily_count'].mean()
        
        health_df['test_consumption_rate'] = health_df['test_type'].map(avg_daily_consumption).fillna(0)
        
        # SME FIX: Ensure the lambda handles non-string (e.g., NaN) values gracefully.
        health_df['test_kit_stock'] = health_df['test_type'].map(
            lambda x: 1000 if isinstance(x, str) and "Malaria" in x else 500
        ).fillna(500) # Ensure no NaNs in stock column
    
    return health_df, iot_df

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def get_supply_forecasts(df: pd.DataFrame, category_config: dict, items: list, days: int) -> pd.DataFrame:
    """Loops through selected items and calls the generic prophet forecaster for each."""
    all_forecasts = []
    data_col, rate_col, stock_col = category_config["data_col"], category_config["rate_col"], category_config["stock_col"]

    for item in items:
        item_df = df[df[data_col] == item].copy()
        if not item_df.empty and rate_col in item_df.columns:
            history = item_df[['encounter_date', rate_col]].rename(columns={'encounter_date': 'ds', rate_col: 'y'})
            
            forecast = generate_prophet_forecast(history, forecast_days=days)
            
            if not forecast.empty:
                latest_stock = item_df.sort_values('encounter_date').iloc[-1][stock_col]
                forecast['item'] = item
                forecast['projected_consumption'] = forecast['predicted_value'].clip(lower=0).cumsum()
                forecast['forecasted_stock'] = latest_stock - forecast['projected_consumption']
                forecast.loc[forecast['forecasted_stock'] < 0, 'forecasted_stock'] = 0
                all_forecasts.append(forecast)

    return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

# --- Main Page Execution ---
def main():
    st.title("📦 Logistics & Capacity Console")
    st.markdown("Monitor and forecast stock levels for critical supplies and predict clinic occupancy using AI-powered models.")
    st.divider()

    health_df, iot_df = get_data()

    with st.sidebar:
        st.header("Dashboard Controls")
        
        selected_category = st.radio("Select Supply Category:", list(SUPPLY_CATEGORIES.keys()), horizontal=True)
        category_config = SUPPLY_CATEGORIES[selected_category]
        
        all_items_in_cat = sorted(health_df[category_config["data_col"]].dropna().unique())
        available_items = [item for item in category_config["items"] if item in all_items_in_cat]
        selected_items = st.multiselect(f"Select {selected_category} to Forecast:", options=available_items, default=available_items[:3])
        
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header(f"📈 {selected_category} Forecast")
        if not selected_items:
            st.info(f"Select one or more {selected_category.lower()} from the sidebar to generate a forecast.")
        else:
            forecast_df = get_supply_forecasts(health_df, category_config, selected_items, forecast_days)
            if forecast_df.empty:
                st.warning("Could not generate a forecast. There may not be enough historical data (at least 5 data points per item are required).")
            else:
                item_to_plot = st.selectbox("View Detailed Forecast For:", options=selected_items)
                item_forecast_df = forecast_df[forecast_df['item'] == item_to_plot]
                
                st.subheader(f"Forecasted Daily Consumption: {item_to_plot}")
                fig_consumption = plot_forecast_chart(item_forecast_df, title=f"Consumption Forecast: {item_to_plot}", y_title="Units per Day")
                st.plotly_chart(fig_consumption, use_container_width=True)

                st.subheader(f"Projected Stock Level: {item_to_plot}")
                stock_series = item_forecast_df.set_index('forecast_date')['forecasted_stock']
                fig_stock = plot_line_chart(stock_series, title=f"Stock Level Forecast: {item_to_plot}", y_title="Units on Hand")
                st.plotly_chart(fig_stock, use_container_width=True)

    with col2:
        st.header("Forecasting Clinic Occupancy")
        if iot_df.empty or 'waiting_room_occupancy' not in iot_df.columns:
            st.warning("No IoT data available to forecast clinic occupancy.")
        else:
            occupancy_hist = iot_df[['timestamp', 'waiting_room_occupancy']].rename(columns={'timestamp': 'ds', 'waiting_room_occupancy': 'y'})
            occupancy_fc = generate_prophet_forecast(occupancy_hist, forecast_days)

            if not occupancy_fc.empty:
                fig_occupancy = plot_forecast_chart(occupancy_fc, title="Forecasted Waiting Room Occupancy", y_title="Number of Patients")
                st.plotly_chart(fig_occupancy, use_container_width=True)
            else:
                st.info("Not enough data to generate an occupancy forecast.")

if __name__ == "__main__":
    main()
