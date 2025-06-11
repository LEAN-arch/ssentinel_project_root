# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - INTEGRATED LOGISTICS & CAPACITY DASHBOARD (V7 - FINAL)

import logging
# --- SME EXPANSION: Add necessary imports for new modules ---
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analytics import generate_prophet_forecast
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import create_empty_figure, plot_forecast_chart, plot_line_chart

# --- Page Setup ---
st.set_page_config(page_title="Logistics & Capacity", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)


# --- SME EXPANSION: Mock AI Function for New Module ---
def calculate_reorder_points(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    MOCK AI FUNCTION: Calculates dynamic reorder points and safety stock.
    This function processes the forecast output to generate actionable advice.
    """
    if forecast_df.empty:
        return pd.DataFrame()
    
    # Create a summary df from the detailed forecast df
    summary_list = []
    for item in forecast_df['item'].unique():
        item_df = forecast_df[forecast_df['item'] == item]
        if not item_df.empty:
            # Reconstruct current stock from the forecast's starting point
            # The first row's forecasted_stock is after the first day's consumption has been removed.
            # So, current stock is the first forecasted_stock + the first day's projected consumption.
            start_stock = item_df.iloc[0]['forecasted_stock'] + item_df.iloc[0]['projected_consumption']
            # Use the mean of the 'predicted_value' for daily consumption
            predicted_daily_consumption = item_df['predicted_value'].mean()
            summary_list.append({'item': item, 'current_stock': start_stock, 'predicted_daily_consumption': predicted_daily_consumption})
    
    summary_df = pd.DataFrame(summary_list)
    if summary_df.empty:
        return pd.DataFrame()

    avg_lead_time, service_level_z = 14, 1.645 # Simulate lead time and service level
    reorder_data = summary_df.copy()
    # Simulate demand volatility (standard deviation of daily use)
    reorder_data['demand_volatility'] = reorder_data['predicted_daily_consumption'] * 0.3 # Assume 30% volatility
    # Calculate Safety Stock and Reorder Point
    reorder_data['safety_stock'] = (service_level_z * reorder_data['demand_volatility'] * np.sqrt(avg_lead_time)).round()
    reorder_data['reorder_point'] = (reorder_data['predicted_daily_consumption'] * avg_lead_time + reorder_data['safety_stock']).round()
    # Determine status
    reorder_data['status'] = np.where(reorder_data['current_stock'] <= reorder_data['reorder_point'], 'Reorder Now', 'OK')
    return reorder_data


# --- SME EXPANSION: New Rendering Component for the Tab ---
def render_reorder_analysis(forecast_df: pd.DataFrame):
    st.header("🎯 Reorder Advisor")
    st.markdown("This AI module analyzes demand volatility and lead times to recommend optimal inventory policies.")
    if forecast_df.empty:
        st.info("Run a successful forecast in the main panel to generate reorder recommendations.")
        return
    reorder_df = calculate_reorder_points(forecast_df)
    st.dataframe(
        reorder_df[['item', 'current_stock', 'safety_stock', 'reorder_point', 'status']].style.apply(
            lambda row: ['background-color: #f8d7da' if row.status == 'Reorder Now' else '' for _ in row], axis=1
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "item": "Item",
            "current_stock": "Current Stock",
            "safety_stock": "AI Safety Stock",
            "reorder_point": "AI Reorder Point",
            "status": "Status"
        }
    )


# --- Supply Category Definitions ---
SUPPLY_CATEGORIES = {
    "Medications": {
        "items": settings.KEY_SUPPLY_ITEMS,
        "source_df": "health_df",
        "data_col": "item",
        "rate_col": "consumption_rate_per_day",
        "stock_col": "item_stock_agg_zone",
        "date_col": "encounter_date"
    },
    "Diagnostic Tests": {
        "items": list(settings.KEY_TEST_TYPES.keys()),
        "source_df": "test_consumption_df", # Use the new, dedicated DataFrame
        "data_col": "test_type",
        "rate_col": "daily_tests_conducted",
        "stock_col": "test_kit_stock",
        "date_col": "encounter_date"
    }
}


# --- Data Loading & Caching ---
@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> dict:
    """Loads and enriches all data for the dashboard, returning a dictionary of DataFrames."""
    health_df = load_health_records()
    iot_df = load_iot_records()
    
    # --- SME FIX: Create a proper time series for test consumption ---
    test_consumption_df = pd.DataFrame()
    if not health_df.empty and 'test_type' in health_df.columns:
        # 1. Count tests per day, per type
        daily_test_counts = health_df.dropna(subset=['test_type']).groupby([
            health_df['encounter_date'].dt.date,
            'test_type'
        ]).size().reset_index(name='daily_tests_conducted')
        daily_test_counts['encounter_date'] = pd.to_datetime(daily_test_counts['encounter_date'])
        
        # 2. Reindex to ensure a complete date range for Prophet
        if not daily_test_counts.empty:
            test_types = daily_test_counts['test_type'].unique()
            date_range = pd.date_range(start=daily_test_counts['encounter_date'].min(), end=daily_test_counts['encounter_date'].max(), freq='D')
            multi_index = pd.MultiIndex.from_product([date_range, test_types], names=['encounter_date', 'test_type'])
            
            test_consumption_df = daily_test_counts.set_index(['encounter_date', 'test_type']).reindex(multi_index, fill_value=0).reset_index()
            test_consumption_df['test_kit_stock'] = test_consumption_df['test_type'].map(lambda x: 1000 if "Malaria" in x else 500)
    
    return {"health_df": health_df, "iot_df": iot_df, "test_consumption_df": test_consumption_df}

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def get_supply_forecasts(df: pd.DataFrame, category_config: dict, items: list, days: int) -> pd.DataFrame:
    """Loops through selected items and calls the generic prophet forecaster for each."""
    all_forecasts = []
    data_col, rate_col, stock_col, date_col = category_config["data_col"], category_config["rate_col"], category_config["stock_col"], category_config["date_col"]

    for item in items:
        item_df = df[df[data_col] == item].copy()
        if not item_df.empty and rate_col in item_df.columns:
            history = item_df[[date_col, rate_col]].rename(columns={date_col: 'ds', rate_col: 'y'})
            
            forecast = generate_prophet_forecast(history, forecast_days=days)
            
            if not forecast.empty:
                # For stock, we need to find the latest recorded value, not just any.
                latest_entry = item_df.sort_values(date_col).iloc[-1]
                latest_stock = latest_entry[stock_col]
                
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

    all_data = get_data()
    health_df = all_data['health_df']
    iot_df = all_data['iot_df']

    if health_df.empty:
        st.error("No health data available. Dashboard cannot be rendered."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        
        selected_category = st.radio("Select Supply Category:", list(SUPPLY_CATEGORIES.keys()), horizontal=True)
        category_config = SUPPLY_CATEGORIES[selected_category]
        
        source_df_for_list = all_data[category_config["source_df"]]
        all_items_in_cat = sorted(source_df_for_list[category_config["data_col"]].dropna().unique())
        available_items = [item for item in category_config["items"] if item in all_items_in_cat]
        selected_items = st.multiselect(f"Select {selected_category} to Forecast:", options=available_items, default=available_items[:3])
        
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7)

    col1, col2 = st.columns(2, gap="large")
    
    # --- SME FIX: This variable must be defined here to pass to the new module ---
    forecast_df = pd.DataFrame()

    with col1:
        st.header(f"📈 {selected_category} Forecast")
        if not selected_items:
            st.info(f"Select one or more {selected_category.lower()} from the sidebar to generate a forecast.")
        else:
            source_df_for_fc = all_data[category_config["source_df"]]
            forecast_df = get_supply_forecasts(source_df_for_fc, category_config, selected_items, forecast_days)
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

    # --- SME EXPANSION: Add new module in a new section at the bottom ---
    st.divider()
    render_reorder_analysis(forecast_df)


if __name__ == "__main__":
    main()
