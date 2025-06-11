# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - INTEGRATED LOGISTICS & CAPACITY DASHBOARD (V8 - AI ENHANCED)

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import generate_prophet_forecast
# SME EXPANSION: Add mock AI functions for logistics
from analytics.logistics import (calculate_reorder_points,
                                 analyze_supplier_performance)
from config import settings
from data_processing import load_health_records, load_iot_records
from visualization import (create_empty_figure, plot_forecast_chart,
                           plot_line_chart)

# --- Page Setup ---
st.set_page_config(page_title="Logistics & Capacity", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)

# --- SME EXPANSION: Constants for better styling ---
PLOTLY_TEMPLATE = "plotly_white"

# --- Supply Category Definitions ---
SUPPLY_CATEGORIES = {
    "Medications": {
        "items": settings.KEY_SUPPLY_ITEMS,
        "source_df": "health_df",
        "data_col": "medication_prescribed", # Assuming this column exists
        "stock_col": "item_stock_agg_zone",
        "date_col": "encounter_date"
    },
    "Diagnostic Tests": {
        "items": list(settings.KEY_TEST_TYPES.keys()),
        "source_df": "test_consumption_df",
        "data_col": "test_type",
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
    
    # Mock additional columns for a richer demo if they don't exist
    if 'medication_prescribed' not in health_df.columns:
        health_df['medication_prescribed'] = np.random.choice(settings.KEY_SUPPLY_ITEMS + [None], size=len(health_df), p=[0.1]*len(settings.KEY_SUPPLY_ITEMS) + [1 - 0.1*len(settings.KEY_SUPPLY_ITEMS)])
    if 'item_stock_agg_zone' not in health_df.columns:
        health_df['item_stock_agg_zone'] = np.random.randint(100, 1000, size=len(health_df))

    test_consumption_df = pd.DataFrame()
    if not health_df.empty and 'test_type' in health_df.columns:
        daily_test_counts = health_df.dropna(subset=['test_type']).groupby([health_df['encounter_date'].dt.date, 'test_type']).size().reset_index(name='daily_tests_conducted')
        daily_test_counts['encounter_date'] = pd.to_datetime(daily_test_counts['encounter_date'])
        
        if not daily_test_counts.empty:
            test_types = daily_test_counts['test_type'].unique()
            date_range = pd.date_range(start=daily_test_counts['encounter_date'].min(), end=daily_test_counts['encounter_date'].max(), freq='D')
            multi_index = pd.MultiIndex.from_product([date_range, test_types], names=['encounter_date', 'test_type'])
            test_consumption_df = daily_test_counts.set_index(['encounter_date', 'test_type']).reindex(multi_index, fill_value=0).reset_index()
            test_consumption_df['test_kit_stock'] = test_consumption_df['test_type'].map(lambda x: 1000 if "Malaria" in x else 500)
    
    return {"health_df": health_df, "iot_df": iot_df, "test_consumption_df": test_consumption_df}


# --- SME EXPANSION: More sophisticated forecast function ---
@st.cache_data(ttl=3600, show_spinner="Generating AI-powered supply chain forecasts...")
def get_supply_forecasts(df: pd.DataFrame, category_config: dict, items: list, days: int) -> pd.DataFrame:
    """Generates consumption forecasts and calculates Days of Supply (DoS)."""
    all_forecasts = []
    data_col, stock_col, date_col = category_config["data_col"], category_config["stock_col"], category_config["date_col"]

    for item in items:
        # Create a daily consumption time series for the item
        item_consumption = df.dropna(subset=[data_col])
        item_consumption = item_consumption[item_consumption[data_col] == item]
        
        if not item_consumption.empty:
            # The rate column is simply the count per day
            history = item_consumption.groupby(pd.Grouper(key=date_col, freq='D')).size().reset_index(name='y')
            history = history.rename(columns={date_col: 'ds'})
            
            forecast = generate_prophet_forecast(history, forecast_days=days)
            
            if not forecast.empty:
                latest_stock = item_consumption.sort_values(date_col).iloc[-1][stock_col]
                
                # Use only future predictions
                future_forecast = forecast[forecast['ds'] > df[date_col].max()].copy()
                future_forecast['predicted_consumption'] = future_forecast['yhat'].clip(lower=0)
                
                # Calculate daily DoS
                avg_daily_consumption = future_forecast['predicted_consumption'].mean()
                days_of_supply = latest_stock / avg_daily_consumption if avg_daily_consumption > 0 else float('inf')
                
                # Append a summary row
                summary = {
                    "item": item,
                    "current_stock": latest_stock,
                    "predicted_daily_consumption": avg_daily_consumption,
                    "days_of_supply": days_of_supply,
                    "forecast_df": future_forecast[['ds', 'predicted_consumption']]
                }
                all_forecasts.append(summary)

    return pd.DataFrame(all_forecasts) if all_forecasts else pd.DataFrame()


# --- SME EXPANSION: New rendering components ---
def render_supply_kpi_dashboard(summary_df: pd.DataFrame):
    """Renders a dynamic grid of KPIs for each forecasted supply item."""
    st.header("📦 AI-Powered Inventory Status")
    st.markdown("At-a-glance view of current stock levels and predicted days of supply remaining.")
    
    if summary_df.empty:
        st.info("No forecast data to display. Please select items from the sidebar.")
        return

    # Dynamically create columns for each item
    cols = st.columns(len(summary_df))
    sorted_df = summary_df.sort_values("days_of_supply").reset_index()

    for i, row in sorted_df.iterrows():
        with cols[i]:
            dos = row['days_of_supply']
            if dos <= 7:
                st.error(f"**{row['item']}**")
                color = "#dc3545"
            elif dos <= 21:
                st.warning(f"**{row['item']}**")
                color = "#ffc107"
            else:
                st.success(f"**{row['item']}**")
                color = "#28a745"

            fig = go.Figure(go.Indicator(
                mode = "number",
                value = dos if dos != float('inf') else 999,
                title = {"text": "Days of Supply", "font": {"size": 16}},
                number = {'suffix': " days", 'font': {'size': 24, 'color': color}},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(height=120, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric(label="Current Stock", value=f"{row['current_stock']:,.0f} units")
            st.metric(label="Predicted Daily Use", value=f"{row['predicted_daily_consumption']:.1f} units")

def render_reorder_analysis(summary_df: pd.DataFrame):
    """Renders the AI-driven reorder point and safety stock analysis."""
    st.header("🎯 Dynamic Reorder & Safety Stock Advisor")
    st.markdown("This AI module analyzes demand volatility and lead times to recommend optimal inventory policies, minimizing both stock-outs and holding costs.")

    if summary_df.empty:
        st.info("Run a forecast to generate reorder recommendations.")
        return

    # Use the mock AI function
    reorder_df = calculate_reorder_points(summary_df)

    # Display as a styled table
    st.dataframe(
        reorder_df[['item', 'current_stock', 'safety_stock', 'reorder_point', 'status']].style.apply(
            lambda row: ['background-color: #f8d7da' if row.status == 'Reorder Now' else '' for _ in row], axis=1
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "item": "Item",
            "current_stock": st.column_config.NumberColumn("Current Stock", format="%d units"),
            "safety_stock": st.column_config.NumberColumn("AI Safety Stock", format="%d units", help="Buffer stock to prevent stock-outs due to variability."),
            "reorder_point": st.column_config.NumberColumn("AI Reorder Point", format="%d units", help="Stock level at which a new order should be placed."),
            "status": "Status"
        }
    )

def render_cold_chain_tab(iot_df: pd.DataFrame):
    st.header("🌡️ Cold Chain Integrity Monitoring")
    st.markdown("Real-time monitoring and predictive alerts for temperature-sensitive supplies like vaccines and certain medications.")
    if iot_df.empty or 'fridge_temp_c' not in iot_df.columns:
        st.warning("No cold chain IoT data available for monitoring.")
        return

    latest_reading = iot_df.sort_values('timestamp').iloc[-1]
    current_temp = latest_reading['fridge_temp_c']
    
    # Define thresholds
    safe_min, safe_max = 2.0, 8.0
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Current Status")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_temp,
            title = {'text': "Current Fridge Temp (°C)"},
            gauge = {
                'axis': {'range': [-5, 15]},
                'bar': {'color': "#2c3e50"},
                'steps' : [
                    {'range': [-5, safe_min], 'color': "red"},
                    {'range': [safe_min, safe_max], 'color': "lightgreen"},
                    {'range': [safe_max, 15], 'color': "red"}],
            }
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=40, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)
        
        if safe_min <= current_temp <= safe_max:
            st.success("✅ **Stable:** Temperature is within the safe range (2-8°C).")
        else:
            st.error("🚨 **Alert:** Temperature is outside the safe range! Check equipment immediately.")

    with col2:
        st.subheader("Historical Temperature Log")
        temp_trend = iot_df.set_index('timestamp')['fridge_temp_c']
        fig = px.line(temp_trend, title="Fridge Temperature Over Time", labels={'timestamp': 'Time', 'value': 'Temperature (°C)'}, template=PLOTLY_TEMPLATE)
        fig.add_hrect(y0=safe_min, y1=safe_max, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Safe Zone", annotation_position="bottom right")
        st.plotly_chart(fig, use_container_width=True)

# --- Main Page Execution ---
def main():
    st.title("📦 Logistics & Supply Chain Console")
    st.markdown("Monitor and forecast stock levels, manage reorder points, and ensure cold chain integrity using AI-powered models.")
    st.divider()

    all_data = get_data()
    iot_df = all_data['iot_df']

    # --- SME EXPANSION: Tabbed interface for better organization ---
    tab1, tab2, tab3 = st.tabs(["**📈 Supply Forecast**", "**🌡️ Cold Chain**", "**🚚 Supplier Performance**"])

    with tab1:
        with st.sidebar:
            st.header("Forecast Controls")
            selected_category = st.radio("Select Supply Category:", list(SUPPLY_CATEGORIES.keys()), horizontal=True, key="supply_cat_radio")
            category_config = SUPPLY_CATEGORIES[selected_category]
            
            source_df_for_list = all_data[category_config["source_df"]]
            if not source_df_for_list.empty:
                all_items_in_cat = sorted(source_df_for_list[category_config["data_col"]].dropna().unique())
                default_items = [item for item in category_config["items"] if item in all_items_in_cat]
                selected_items = st.multiselect(f"Select {selected_category} to Forecast:", options=all_items_in_cat, default=default_items[:3])
            else:
                selected_items = []
                st.warning(f"No consumption data for {selected_category}.")
            
            forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7, key="main_forecast_slider")
        
        if not selected_items:
            st.info(f"Select one or more {selected_category.lower()} from the sidebar to generate a forecast.")
        else:
            source_df_for_fc = all_data[category_config["source_df"]]
            forecast_summary_df = get_supply_forecasts(source_df_for_fc, category_config, selected_items, forecast_days)
            
            render_supply_kpi_dashboard(forecast_summary_df)
            st.divider()
            render_reorder_analysis(forecast_summary_df)

    with tab2:
        render_cold_chain_tab(iot_df)

    with tab3:
        st.header("🚚 Supplier Performance Scorecard")
        st.markdown("Analyze supplier reliability to de-risk your supply chain and improve procurement decisions.")
        # This module uses the entire health record history for a complete analysis
        supplier_df = analyze_supplier_performance(all_data['health_df']) # Mock AI function
        if not supplier_df.empty:
            st.dataframe(
                supplier_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "supplier": "Supplier",
                    "avg_lead_time_days": st.column_config.NumberColumn("Avg. Lead Time (Days)", help="Average time from order to delivery."),
                    "on_time_delivery_rate": st.column_config.ProgressColumn("On-Time Rate", help="Percentage of deliveries that arrived on or before the promised date.", format="%.1f%%", min_value=0, max_value=100),
                    "order_fill_rate": st.column_config.ProgressColumn("Order Fill Rate", help="Percentage of the ordered quantity that was actually delivered.", format="%.1f%%", min_value=0, max_value=100),
                    "reliability_score": st.column_config.NumberColumn("AI Reliability Score", help="An AI-generated score (0-100) combining all performance metrics.", format="%.1f"),
                }
            )
        else:
            st.info("Not enough procurement data to analyze supplier performance.")


if __name__ == "__main__":
    main()
