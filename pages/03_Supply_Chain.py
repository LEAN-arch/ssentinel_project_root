# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - INTEGRATED LOGISTICS & CAPACITY DASHBOARD (V7 - AI ENHANCED)

import logging
from datetime import timedelta
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

# --- SME EXPANSION: Constants for better styling ---
PLOTLY_TEMPLATE = "plotly_white"

# --- SME EXPANSION: Mock AI Functions for New Modules ---
def calculate_reorder_points(forecast_summary_df: pd.DataFrame) -> pd.DataFrame:
    """MOCK AI FUNCTION: Calculates dynamic reorder points and safety stock."""
    if forecast_summary_df.empty:
        return pd.DataFrame()
    
    # Simulate lead time and service level for calculations
    avg_lead_time = 14 # days
    service_level_z = 1.645 # Corresponds to a 95% service level
    
    reorder_data = forecast_summary_df.copy()
    
    # Simulate demand volatility (standard deviation of daily use)
    reorder_data['demand_volatility'] = reorder_data['predicted_daily_consumption'] * 0.3 # Assume 30% volatility
    
    # Calculate Safety Stock and Reorder Point
    reorder_data['safety_stock'] = (service_level_z * reorder_data['demand_volatility'] * np.sqrt(avg_lead_time)).round()
    reorder_data['reorder_point'] = (reorder_data['predicted_daily_consumption'] * avg_lead_time + reorder_data['safety_stock']).round()
    
    # Determine status
    reorder_data['status'] = np.where(reorder_data['current_stock'] <= reorder_data['reorder_point'], 'Reorder Now', 'OK')
    
    return reorder_data

def analyze_supplier_performance(health_df: pd.DataFrame) -> pd.DataFrame:
    """MOCK AI FUNCTION: Simulates analyzing supplier performance data."""
    # This is a complete simulation as we don't have supplier data in health_df.
    suppliers = ["PharmaCo Inc.", "Global Med Supplies", "HealthCare Direct"]
    data = []
    for supplier in suppliers:
        data.append({
            "supplier": supplier,
            "avg_lead_time_days": np.random.uniform(7, 25),
            "on_time_delivery_rate": np.random.uniform(75, 99),
            "order_fill_rate": np.random.uniform(90, 100),
            "reliability_score": np.random.uniform(80, 98)
        })
    return pd.DataFrame(data)


# --- Supply Category Definitions ---
# SME FIX: Corrected the data_col for medications to match mock data generation
SUPPLY_CATEGORIES = {
    "Medications": {
        "items": settings.KEY_SUPPLY_ITEMS,
        "source_df": "health_df",
        "data_col": "medication_prescribed",
        "rate_col": "consumption_rate_per_day", # Will be calculated on-the-fly
        "stock_col": "item_stock_agg_zone",
        "date_col": "encounter_date"
    },
    "Diagnostic Tests": {
        "items": list(settings.KEY_TEST_TYPES.keys()),
        "source_df": "test_consumption_df",
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

    # SME EXPANSION: Mock data for demonstration if columns are missing
    if not health_df.empty:
        if 'medication_prescribed' not in health_df.columns:
            health_df['medication_prescribed'] = np.random.choice(settings.KEY_SUPPLY_ITEMS + [None], size=len(health_df), p=[0.1]*len(settings.KEY_SUPPLY_ITEMS) + [1 - 0.1*len(settings.KEY_SUPPLY_ITEMS)])
        if 'item_stock_agg_zone' not in health_df.columns:
            health_df['item_stock_agg_zone'] = health_df['medication_prescribed'].apply(lambda x: np.random.randint(500, 2000) if x else 0)
        if 'test_kit_stock' not in health_df.columns:
            health_df['test_kit_stock'] = health_df['test_type'].apply(lambda x: np.random.randint(200, 1000) if x else 0)
    
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


# --- SME FIX: This function now correctly processes consumption data for ANY category ---
@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def get_supply_forecasts(source_df: pd.DataFrame, category_config: dict, items: list, days: int) -> tuple[pd.DataFrame, dict]:
    """Generates consumption forecasts, calculates Days of Supply (DoS), and returns detailed forecast data."""
    all_summaries = []
    all_item_forecasts = {}
    data_col, stock_col, date_col = category_config["data_col"], category_config["stock_col"], category_config["date_col"]
    rate_col = category_config["rate_col"] # Get rate column for logic

    for item in items:
        item_df = source_df.dropna(subset=[data_col])
        item_df = item_df[item_df[data_col] == item]
        
        if not item_df.empty:
            # If the rate column isn't pre-calculated, calculate it now
            if rate_col not in item_df.columns:
                 history = item_df.groupby(pd.Grouper(key=date_col, freq='D')).size().reset_index(name='y')
                 history = history.rename(columns={date_col: 'ds'})
            else:
                 history = item_df[[date_col, rate_col]].rename(columns={date_col: 'ds', rate_col: 'y'})

            if len(history[history['y'] > 0]) < 2: continue

            forecast = generate_prophet_forecast(history, forecast_days=days)
            
            if 'yhat' in forecast.columns:
                all_item_forecasts[item] = pd.merge(history, forecast, on='ds', how='outer')
                
                latest_stock = item_df.sort_values(by=date_col, ascending=False).iloc[0][stock_col]
                future_forecast = forecast[forecast['ds'] > history['ds'].max()].copy()
                avg_daily_consumption = future_forecast['yhat'].clip(lower=0).mean()
                days_of_supply = latest_stock / avg_daily_consumption if avg_daily_consumption > 0 else float('inf')
                
                all_summaries.append({
                    "item": item,
                    "current_stock": latest_stock,
                    "predicted_daily_consumption": avg_daily_consumption,
                    "days_of_supply": days_of_supply,
                })

    return pd.DataFrame(all_summaries), all_item_forecasts


# --- SME EXPANSION: New Rendering Components ---
def render_supply_kpi_dashboard(summary_df: pd.DataFrame):
    """Renders a dynamic grid of KPIs for each forecasted supply item."""
    st.subheader("Key Item Inventory Status")
    st.markdown("At-a-glance view of current stock levels and predicted **Days of Supply (DoS)** remaining.")
    
    if summary_df.empty:
        st.warning("Could not generate forecasts for the selected items. They may have insufficient historical consumption data.")
        return

    cols = st.columns(len(summary_df))
    sorted_df = summary_df.sort_values("days_of_supply").reset_index()

    for i, row in sorted_df.iterrows():
        with cols[i]:
            dos = row['days_of_supply']
            if dos <= 7: st.error(f"**{row['item']}**"); color = "#dc3545"
            elif dos <= 21: st.warning(f"**{row['item']}**"); color = "#ffc107"
            else: st.success(f"**{row['item']}**"); color = "#28a745"

            fig = go.Figure(go.Indicator(
                mode = "number", value = dos if dos != float('inf') else 999,
                title = {"text": "Days of Supply", "font": {"size": 16}},
                number = {'suffix': " days", 'font': {'size': 24, 'color': color}},
            ))
            fig.update_layout(height=120, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric(label="Current Stock", value=f"{row['current_stock']:,.0f} units")
            st.metric(label="Predicted Daily Use", value=f"{row['predicted_daily_consumption']:.1f} units")

def render_reorder_analysis(summary_df: pd.DataFrame):
    st.subheader("🎯 Dynamic Reorder & Safety Stock Advisor")
    st.markdown("This AI module analyzes demand volatility and lead times to recommend optimal inventory policies, minimizing both stock-outs and holding costs.")

    if summary_df.empty:
        st.info("Run a successful forecast to generate reorder recommendations.")
        return

    reorder_df = calculate_reorder_points(summary_df)
    st.dataframe(
        reorder_df[['item', 'current_stock', 'safety_stock', 'reorder_point', 'status']].style.apply(
            lambda row: ['background-color: #f8d7da' if row.status == 'Reorder Now' else '' for _ in row], axis=1
        ),
        use_container_width=True, hide_index=True,
        column_config={
            "item": "Item", "current_stock": "Current Stock", "safety_stock": "AI Safety Stock",
            "reorder_point": "AI Reorder Point", "status": "Status"
        }
    )

def render_cold_chain_tab(iot_df: pd.DataFrame):
    st.header("🌡️ Cold Chain Integrity Monitoring")
    st.markdown("Real-time monitoring and predictive alerts for temperature-sensitive supplies like vaccines and certain medications.")
    if iot_df.empty or 'fridge_temp_c' not in iot_df.columns:
        st.warning("No cold chain IoT data available for monitoring."); return

    latest_reading = iot_df.sort_values('timestamp').iloc[-1]
    current_temp = latest_reading['fridge_temp_c']
    safe_min, safe_max = 2.0, 8.0
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Current Status")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=current_temp, title={'text': "Current Fridge Temp (°C)"},
            gauge={'axis': {'range': [-5, 15]}, 'bar': {'color': "#2c3e50"}, 'steps': [{'range': [-5, safe_min], 'color': "red"}, {'range': [safe_min, safe_max], 'color': "lightgreen"}, {'range': [safe_max, 15], 'color': "red"}]}
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=40, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)
        if safe_min <= current_temp <= safe_max: st.success("✅ **Stable:** Temperature is within the safe range (2-8°C).")
        else: st.error("🚨 **Alert:** Temperature is outside the safe range! Check equipment immediately.")
    with col2:
        st.subheader("Historical Temperature Log")
        temp_trend = iot_df.set_index('timestamp')['fridge_temp_c']
        fig = px.line(temp_trend, title="Fridge Temperature Over Time", labels={'timestamp': 'Time', 'value': 'Temperature (°C)'}, template=PLOTLY_TEMPLATE)
        fig.add_hrect(y0=safe_min, y1=safe_max, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Safe Zone", annotation_position="bottom right")
        st.plotly_chart(fig, use_container_width=True)


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
        selected_items = st.multiselect(f"Select {selected_category} to Forecast:", options=available_items, default=available_items[:2])
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7)

    # --- SME EXPANSION: Create a tabbed interface for better organization ---
    tab1, tab2, tab3, tab4 = st.tabs(["**📈 Supply Forecast**", "**🎯 Reorder Advisor**", "**🌡️ Cold Chain**", "**🚚 Supplier Performance**"])
    
    # --- Generate forecast data once for use in multiple tabs ---
    source_df_for_fc = all_data[category_config["source_df"]]
    forecast_summary_df, all_item_forecasts = get_supply_forecasts(source_df_for_fc, category_config, selected_items, forecast_days)

    with tab1:
        st.header(f"📈 {selected_category} Consumption & Stock Forecast")
        render_supply_kpi_dashboard(forecast_summary_df)
        st.divider()

        if not selected_items or not all_item_forecasts:
            st.info(f"Select one or more items from the sidebar to generate a detailed forecast.")
        else:
            item_to_plot = st.selectbox("View Detailed Forecast For:", options=list(all_item_forecasts.keys()))
            if item_to_plot:
                item_forecast_df = all_item_forecasts[item_to_plot]
                
                st.subheader(f"Forecasted Daily Consumption: {item_to_plot}")
                fig_consumption = plot_forecast_chart(item_forecast_df, title=f"Consumption Forecast: {item_to_plot}", y_title="Units per Day")
                st.plotly_chart(fig_consumption, use_container_width=True)

                st.subheader(f"Projected Stock Level: {item_to_plot}")
                # Create the stock forecast on the fly from the summary and detailed forecast
                start_stock = forecast_summary_df[forecast_summary_df['item'] == item_to_plot]['current_stock'].iloc[0]
                item_forecast_df['projected_consumption'] = item_forecast_df['yhat'].clip(lower=0).cumsum()
                item_forecast_df['forecasted_stock'] = start_stock - item_forecast_df['projected_consumption']
                item_forecast_df.loc[item_forecast_df['forecasted_stock'] < 0, 'forecasted_stock'] = 0
                stock_series = item_forecast_df.set_index('ds')['forecasted_stock']

                fig_stock = plot_line_chart(stock_series, title=f"Stock Level Forecast: {item_to_plot}", y_title="Units on Hand")
                st.plotly_chart(fig_stock, use_container_width=True)

    # --- SME EXPANSION: New Tabs for AI modules ---
    with tab2:
        render_reorder_analysis(forecast_summary_df)

    with tab3:
        render_cold_chain_tab(iot_df)

    with tab4:
        st.header("🚚 Supplier Performance Scorecard")
        st.markdown("Analyze supplier reliability to de-risk your supply chain and improve procurement decisions.")
        supplier_df = analyze_supplier_performance(health_df)
        if not supplier_df.empty:
            st.dataframe(
                supplier_df, use_container_width=True, hide_index=True,
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
    
    st.divider()
    
    # --- Original Clinic Occupancy Module (Maintained) ---
    with st.container():
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
