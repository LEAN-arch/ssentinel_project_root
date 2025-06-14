# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - INTEGRATED LOGISTICS & CAPACITY DASHBOARD (V7 - FINAL, CORRECTLY EXPANDED AND FIXED)

import logging
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

# --- SME EXPANSION: Mock AI Functions for New Modules ---
def calculate_reorder_points(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """MOCK AI FUNCTION: Calculates dynamic reorder points and safety stock."""
    if forecast_df.empty: return pd.DataFrame()
    summary_list = []
    for item in forecast_df['item'].unique():
        item_df = forecast_df[forecast_df['item'] == item]
        if not item_df.empty:
            start_stock = item_df.iloc[0]['forecasted_stock'] + item_df.iloc[0]['projected_consumption']
            predicted_daily_consumption = item_df['predicted_value'].mean()
            summary_list.append({'item': item, 'current_stock': start_stock, 'predicted_daily_consumption': predicted_daily_consumption})
    summary_df = pd.DataFrame(summary_list)
    if summary_df.empty: return pd.DataFrame()
    avg_lead_time, service_level_z = 14, 1.645
    reorder_data = summary_df.copy()
    reorder_data['demand_volatility'] = reorder_data['predicted_daily_consumption'] * 0.3
    reorder_data['safety_stock'] = (service_level_z * reorder_data['demand_volatility'] * np.sqrt(avg_lead_time)).round()
    reorder_data['reorder_point'] = (reorder_data['predicted_daily_consumption'] * avg_lead_time + reorder_data['safety_stock']).round()
    reorder_data['status'] = np.where(reorder_data['current_stock'] <= reorder_data['reorder_point'], 'Reorder Now', 'OK')
    return reorder_data

def analyze_supplier_performance(health_df: pd.DataFrame) -> pd.DataFrame:
    """MOCK AI FUNCTION: Simulates analyzing supplier performance data."""
    suppliers = ["PharmaCo Inc.", "Global Med Supplies", "HealthCare Direct"]
    data = [{"supplier": s, "avg_lead_time_days": np.random.uniform(7, 25), "on_time_delivery_rate": np.random.uniform(75, 99), "order_fill_rate": np.random.uniform(90, 100), "reliability_score": np.random.uniform(80, 98)} for s in suppliers]
    return pd.DataFrame(data)

# --- SME EXPANSION: New Rendering Components for Tabs ---
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
        use_container_width=True, hide_index=True,
        column_config={ "item": "Item", "current_stock": "Current Stock", "safety_stock": "AI Safety Stock", "reorder_point": "AI Reorder Point", "status": "Status" }
    )

def render_cold_chain_tab(iot_df: pd.DataFrame):
    st.header("🌡️ Cold Chain Integrity")
    st.markdown("Real-time monitoring for temperature-sensitive supplies.")
    if iot_df.empty or 'fridge_temp_c' not in iot_df.columns:
        st.warning("No live cold chain IoT data available. Displaying mock data for demonstration.", icon="⚠️")
        end_date, start_date = pd.to_datetime(date.today()), pd.to_datetime(date.today() - timedelta(days=5))
        iot_date_range = pd.to_datetime(pd.date_range(start_date, end_date, periods=24*5))
        iot_df = pd.DataFrame({'timestamp': iot_date_range, 'fridge_temp_c': np.random.normal(loc=5.0, scale=2.5, size=24*5)})
        
    latest_reading, safe_min, safe_max = iot_df.sort_values('timestamp').iloc[-1], 2.0, 8.0
    current_temp = latest_reading['fridge_temp_c']
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Current Status")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=current_temp, title={'text': "Fridge Temp (°C)"},
            gauge={'axis': {'range': [-5, 15]}, 'bar': {'color': "#2c3e50"}, 'steps': [{'range': [-5, safe_min], 'color': "red"}, {'range': [safe_min, safe_max], 'color': "lightgreen"}, {'range': [safe_max, 15], 'color': "red"}]}))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Historical Temperature Log")
        fig = px.line(iot_df, x='timestamp', y='fridge_temp_c', title="Fridge Temperature Over Time", template="plotly_white")
        fig.add_hrect(y0=safe_min, y1=safe_max, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Safe Zone")
        st.plotly_chart(fig, use_container_width=True)

def render_supplier_performance_tab(health_df: pd.DataFrame):
    st.header("🚚 Supplier Performance")
    st.markdown("Analyze supplier reliability to de-risk your supply chain.")
    supplier_df = analyze_supplier_performance(health_df)
    st.dataframe(supplier_df, use_container_width=True, hide_index=True)

# --- Original, Unaltered Code Blocks (with minimal fixes for robustness) ---
SUPPLY_CATEGORIES = {
    "Medications": { "items": settings.KEY_SUPPLY_ITEMS, "source_df": "health_df", "data_col": "item", "rate_col": "consumption_rate_per_day", "stock_col": "item_stock_agg_zone", "date_col": "encounter_date"},
    "Diagnostic Tests": { "items": list(settings.KEY_TEST_TYPES.keys()), "source_df": "test_consumption_df", "data_col": "test_type", "rate_col": "daily_tests_conducted", "stock_col": "test_kit_stock", "date_col": "encounter_date"}
}

@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> dict:
    health_df, iot_df = load_health_records(), load_iot_records()
    if health_df.empty:
        st.warning("Live health data is empty. Generating realistic mock data for demonstration.", icon="⚠️")
        num_records, end_date = 500, date.today()
        start_date = end_date - timedelta(days=60)
        date_range = pd.to_datetime(pd.date_range(start_date, end_date))
        health_df = pd.DataFrame({'encounter_date': np.random.choice(date_range, num_records)})
        health_df['item'] = np.random.choice(settings.KEY_SUPPLY_ITEMS, num_records)
        health_df['consumption_rate_per_day'] = np.random.randint(5, 25, num_records)
        health_df['item_stock_agg_zone'] = np.random.randint(500, 3000, num_records)
        health_df['test_type'] = np.random.choice(list(settings.KEY_TEST_TYPES.keys()), num_records)

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

@st.cache_data(ttl=3600, show_spinner="Generating AI-powered forecasts...")
def get_supply_forecasts(df: pd.DataFrame, category_config: dict, items: list, days: int) -> pd.DataFrame:
    all_forecasts = []
    data_col, rate_col, stock_col, date_col = category_config["data_col"], category_config["rate_col"], category_config["stock_col"], category_config["date_col"]
    for item in items:
        item_df = df[df[data_col] == item].copy()
        if not item_df.empty and rate_col in item_df.columns:
            history = item_df[[date_col, rate_col]].rename(columns={date_col: 'ds', rate_col: 'y'})
            forecast = generate_prophet_forecast(history, forecast_days=days)
            if not forecast.empty:
                latest_entry = item_df.sort_values(date_col).iloc[-1]
                latest_stock = latest_entry[stock_col]
                # --- DEFINITIVE FIX: Rename columns to match what the original plotting functions expect ---
                forecast.rename(columns={'ds': 'forecast_date', 'yhat': 'predicted_value'}, inplace=True)
                forecast['item'] = item
                forecast['projected_consumption'] = forecast['predicted_value'].clip(lower=0).cumsum()
                forecast['forecasted_stock'] = latest_stock - forecast['projected_consumption']
                forecast.loc[forecast['forecasted_stock'] < 0, 'forecasted_stock'] = 0
                all_forecasts.append(forecast)
    return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

def main():
    st.title("📦 Logistics & Capacity Console")
    st.markdown("Monitor and forecast stock levels for critical supplies and predict clinic occupancy using AI-powered models.")
    st.divider()
    all_data = get_data()
    health_df, iot_df = all_data['health_df'], all_data['iot_df']
    if health_df.empty: st.error("No health data available."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        selected_category = st.radio("Select Supply Category:", list(SUPPLY_CATEGORIES.keys()), horizontal=True, key="supply_cat_radio")
        category_config = SUPPLY_CATEGORIES[selected_category]
        source_df_for_list = all_data[category_config["source_df"]]
        all_items_in_cat = sorted(source_df_for_list[category_config["data_col"]].dropna().unique())
        available_items = [item for item in category_config["items"] if item in all_items_in_cat]
        selected_items = st.multiselect(f"Select {selected_category} to Forecast:", options=available_items, default=available_items[:3])
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7, key="forecast_days_slider")

    col1, col2 = st.columns(2, gap="large")
    forecast_df = pd.DataFrame()

    with col1:
        st.header(f"📈 {selected_category} Forecast")
        if not selected_items:
            st.info(f"Select one or more {selected_category.lower()} from the sidebar.")
        else:
            source_df_for_fc = all_data[category_config["source_df"]]
            forecast_df = get_supply_forecasts(source_df_for_fc, category_config, selected_items, forecast_days)
            if forecast_df.empty:
                st.warning("Could not generate a forecast.")
            else:
                item_to_plot = st.selectbox("View Detailed Forecast For:", options=selected_items)
                item_forecast_df = forecast_df[forecast_df['item'] == item_to_plot]
                if not item_forecast_df.empty:
                    st.subheader(f"Forecasted Daily Consumption: {item_to_plot}")
                    fig_consumption = plot_forecast_chart(item_forecast_df, title=f"Consumption Forecast: {item_to_plot}", y_title="Units per Day")
                    st.plotly_chart(fig_consumption, use_container_width=True)
                    st.subheader(f"Projected Stock Level: {item_to_plot}")
                    stock_series = item_forecast_df.set_index('forecast_date')['forecasted_stock']
                    fig_stock = plot_line_chart(stock_series, title=f"Stock Level Forecast: {item_to_plot}", y_title="Units on Hand")
                    st.plotly_chart(fig_stock, use_container_width=True)
                else:
                    st.info(f"No forecast available for {item_to_plot}.")
    with col2:
        st.header("Forecasting Clinic Occupancy")
        if iot_df.empty or 'waiting_room_occupancy' not in iot_df.columns:
            st.warning("No IoT data available for forecast.")
        else:
            occupancy_hist = iot_df[['timestamp', 'waiting_room_occupancy']].rename(columns={'timestamp': 'ds', 'waiting_room_occupancy': 'y'})
            occupancy_fc = generate_prophet_forecast(occupancy_hist, forecast_days)
            if not occupancy_fc.empty:
                # Rename for compatibility with the original plotting function
                occupancy_fc.rename(columns={'ds': 'forecast_date', 'yhat': 'predicted_value'}, inplace=True)
                fig_occupancy = plot_forecast_chart(occupancy_fc, title="Forecasted Waiting Room Occupancy", y_title="Number of Patients")
                st.plotly_chart(fig_occupancy, use_container_width=True)
            else:
                st.info("Not enough data to generate an occupancy forecast.")

    st.divider()
    tab_reorder, tab_coldchain, tab_supplier = st.tabs(["🎯 Reorder Advisor", "🌡️ Cold Chain Integrity", "🚚 Supplier Performance"])
    with tab_reorder: render_reorder_analysis(forecast_df)
    with tab_coldchain: render_cold_chain_tab(iot_df)
    with tab_supplier: render_supplier_performance_tab(health_df)

if __name__ == "__main__":
    main()
