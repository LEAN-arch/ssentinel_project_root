# sentinel_project_root/pages/03_Supply_Chain.py
# SME PLATINUM STANDARD - INTEGRATED LOGISTICS & CAPACITY DASHBOARD (V13 - DEMO-READY)

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

PLOTLY_TEMPLATE = "plotly_white"

# --- Mock AI Functions ---
def calculate_reorder_points(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty: return pd.DataFrame()
    avg_lead_time, service_level_z = 14, 1.645
    reorder_data = summary_df.copy()
    reorder_data['demand_volatility'] = reorder_data['predicted_daily_consumption'] * 0.3
    reorder_data['safety_stock'] = (service_level_z * reorder_data['demand_volatility'] * np.sqrt(avg_lead_time)).round()
    reorder_data['reorder_point'] = (reorder_data['predicted_daily_consumption'] * avg_lead_time + reorder_data['safety_stock']).round()
    reorder_data['status'] = np.where(reorder_data['current_stock'] <= reorder_data['reorder_point'], 'Reorder Now', 'OK')
    return reorder_data

def analyze_supplier_performance(health_df: pd.DataFrame) -> pd.DataFrame:
    suppliers = ["PharmaCo Inc.", "Global Med Supplies", "HealthCare Direct"]
    data = []
    for supplier in suppliers:
        data.append({
            "supplier": supplier, "avg_lead_time_days": np.random.uniform(7, 25),
            "on_time_delivery_rate": np.random.uniform(75, 99), "order_fill_rate": np.random.uniform(90, 100),
            "reliability_score": np.random.uniform(80, 98)
        })
    return pd.DataFrame(data)

# --- Supply Category Definitions ---
SUPPLY_CATEGORIES = {
    "Medications": {
        "items": settings.KEY_SUPPLY_ITEMS,
        "source_df_name": "health_df", "data_col": "medication_prescribed",
        "stock_col": "item_stock_agg_zone", "date_col": "encounter_date"
    },
    "Diagnostic Tests": {
        "items": list(settings.KEY_TEST_TYPES.keys()),
        "source_df_name": "health_df", "data_col": "test_type",
        "stock_col": "test_kit_stock", "date_col": "encounter_date"
    }
}

# --- Data Loading & Caching ---
@st.cache_data(ttl=3600, show_spinner="Loading all operational data...")
def get_data() -> dict:
    health_df = load_health_records()
    iot_df = load_iot_records()
    
    # --- SME FIX: Generate rich mock data if loaded data is insufficient for demo ---
    if health_df.empty or len(health_df['encounter_date'].dt.date.unique()) < 10:
        st.warning("Live data is insufficient for a full demonstration. Generating realistic mock data.", icon="⚠️")
        num_records = 2000
        end_date = date.today()
        start_date = end_date - timedelta(days=90)
        date_range = pd.to_datetime(pd.date_range(start_date, end_date))
        
        mock_health_data = {
            'encounter_date': np.random.choice(date_range, num_records),
            'medication_prescribed': np.random.choice(settings.KEY_SUPPLY_ITEMS + [None], num_records, p=[0.1]*len(settings.KEY_SUPPLY_ITEMS) + [1 - 0.1*len(settings.KEY_SUPPLY_ITEMS)]),
            'test_type': np.random.choice(list(settings.KEY_TEST_TYPES.keys()) + [None], num_records, p=[0.2]*len(settings.KEY_TEST_TYPES) + [1 - 0.2*len(settings.KEY_TEST_TYPES)])
        }
        health_df = pd.DataFrame(mock_health_data)
        health_df['item_stock_agg_zone'] = health_df['medication_prescribed'].apply(lambda x: np.random.randint(500, 2000) if x else 0)
        health_df['test_kit_stock'] = health_df['test_type'].apply(lambda x: np.random.randint(200, 1000) if x else 0)
    
    if iot_df.empty or 'fridge_temp_c' not in iot_df.columns:
        num_iot_records = 24 * 90 # 90 days of hourly data
        iot_date_range = pd.to_datetime(pd.date_range(start_date, end_date, periods=num_iot_records))
        mock_iot_data = {
            'timestamp': iot_date_range,
            'fridge_temp_c': np.random.normal(5.0, 1.5, num_iot_records), # Centered on 5°C
            'waiting_room_occupancy': np.random.randint(0, 25, num_iot_records)
        }
        iot_df = pd.DataFrame(mock_iot_data)
        
    return {"health_df": health_df, "iot_df": iot_df}


@st.cache_data(ttl=3600, show_spinner="Generating AI-powered supply chain forecasts...")
def get_supply_forecasts(source_df: pd.DataFrame, category_config: dict, items: list, days: int) -> tuple[pd.DataFrame, dict]:
    all_summaries, all_item_forecasts = [], {}
    data_col, stock_col, date_col = category_config["data_col"], category_config["stock_col"], category_config["date_col"]

    for item in items:
        item_df = source_df.dropna(subset=[data_col])
        item_df = item_df[item_df[data_col] == item]
        
        if not item_df.empty:
            history = item_df.groupby(pd.Grouper(key=date_col, freq='D')).size().reset_index(name='y')
            history = history.rename(columns={date_col: 'ds'})

            if len(history[history['y'] > 0]) < 2: continue

            forecast = generate_prophet_forecast(history, forecast_days=days)
            
            if 'yhat' in forecast.columns:
                all_item_forecasts[item] = pd.merge(history, forecast, on='ds', how='outer')
                latest_stock = item_df.sort_values(by=date_col, ascending=False).iloc[0][stock_col]
                future_forecast = forecast[forecast['ds'] > history['ds'].max()].copy()
                avg_daily_consumption = future_forecast['yhat'].clip(lower=0).mean()
                days_of_supply = latest_stock / avg_daily_consumption if avg_daily_consumption > 0 else float('inf')
                
                all_summaries.append({
                    "item": item, "current_stock": latest_stock,
                    "predicted_daily_consumption": avg_daily_consumption, "days_of_supply": days_of_supply,
                })
    return pd.DataFrame(all_summaries), all_item_forecasts

# --- Rendering Components ---
def render_supply_kpi_dashboard(summary_df: pd.DataFrame):
    st.subheader("Key Item Inventory Status")
    if summary_df.empty:
        st.warning("Could not generate forecasts for the selected items. They may have insufficient historical consumption data (at least 2 days of activity are required).")
        return
    cols = st.columns(len(summary_df))
    for i, row in summary_df.sort_values("days_of_supply").iterrows():
        with cols[i]:
            dos = row['days_of_supply']
            if dos <= 7: st.error(f"**{row['item']}**"); color = "#dc3545"
            elif dos <= 21: st.warning(f"**{row['item']}**"); color = "#ffc107"
            else: st.success(f"**{row['item']}**"); color = "#28a745"
            st.metric(label="Predicted Days of Supply", value=f"{dos:.1f}" if dos != float('inf') else "∞")
            st.metric(label="Current Stock", value=f"{row['current_stock']:,.0f} units")

def render_reorder_analysis(summary_df: pd.DataFrame):
    st.header("🎯 Dynamic Reorder & Safety Stock Advisor")
    st.markdown("This AI module analyzes demand volatility and lead times to recommend optimal inventory policies.")
    if summary_df.empty: st.info("Run a successful forecast to generate reorder recommendations."); return
    reorder_df = calculate_reorder_points(summary_df)
    st.dataframe(reorder_df[['item', 'current_stock', 'safety_stock', 'reorder_point', 'status']], use_container_width=True, hide_index=True)

def render_cold_chain_tab(iot_df: pd.DataFrame):
    st.header("🌡️ Cold Chain Integrity Monitoring")
    if iot_df.empty: st.warning("No cold chain IoT data available for monitoring."); return
    latest_reading, safe_min, safe_max = iot_df.sort_values('timestamp').iloc[-1], 2.0, 8.0
    current_temp = latest_reading['fridge_temp_c']
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Current Status")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=current_temp, title={'text': "Fridge Temp (°C)"}, gauge={'axis': {'range': [-5, 15]}, 'bar': {'color': "#2c3e50"}, 'steps': [{'range': [-5, safe_min], 'color': "red"}, {'range': [safe_min, safe_max], 'color': "lightgreen"}, {'range': [safe_max, 15], 'color': "red"}]}))
        fig.update_layout(height=250, margin=dict(t=40, b=40, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)
        if safe_min <= current_temp <= safe_max: st.success("✅ **Stable**")
        else: st.error("🚨 **Alert:** Temp. out of range!")
    with col2:
        st.subheader("Historical Temperature Log")
        fig = px.line(iot_df, x='timestamp', y='fridge_temp_c', title="Fridge Temperature Over Time", template=PLOTLY_TEMPLATE)
        fig.add_hrect(y0=safe_min, y1=safe_max, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Safe Zone")
        st.plotly_chart(fig, use_container_width=True)

# --- Main Page Execution ---
def main():
    st.title("📦 Logistics & Capacity Console")
    st.markdown("Monitor and forecast stock levels, manage reorder points, and predict clinic occupancy using AI-powered models.")
    st.divider()

    all_data = get_data()
    health_df = all_data['health_df']
    iot_df = all_data['iot_df']

    if health_df.empty: st.error("No health data available."); st.stop()

    with st.sidebar:
        st.header("Dashboard Controls")
        selected_category = st.radio("Select Supply Category:", list(SUPPLY_CATEGORIES.keys()), horizontal=True)
        category_config = SUPPLY_CATEGORIES[selected_category]
        source_df_for_list = all_data["health_df"] # Always use health_df for the list
        all_items_in_cat = sorted(source_df_for_list[category_config["data_col"]].dropna().unique())
        available_items = [item for item in category_config["items"] if item in all_items_in_cat]
        selected_items = st.multiselect(f"Select {selected_category} to Forecast:", options=available_items, default=available_items[:3])
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, 7)

    source_df_for_fc = all_data["health_df"]
    forecast_summary_df, all_item_forecasts = get_supply_forecasts(source_df_for_fc, category_config, selected_items, forecast_days)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header(f"📈 {selected_category} Forecast")
        if not selected_items:
            st.info(f"Select one or more {selected_category.lower()} to forecast.")
        else:
            render_supply_kpi_dashboard(forecast_summary_df)
            st.divider()
            st.subheader("Detailed Forecast Analysis")
            if not all_item_forecasts:
                st.info("Select items with sufficient data to see detailed plots.")
            else:
                item_to_plot = st.selectbox("View Detailed Plot For:", options=list(all_item_forecasts.keys()))
                if item_to_plot and item_to_plot in all_item_forecasts:
                    item_forecast_df = all_item_forecasts[item_to_plot]
                    fig_consumption = plot_forecast_chart(item_forecast_df, title=f"Consumption Forecast: {item_to_plot}", y_title="Units per Day")
                    st.plotly_chart(fig_consumption, use_container_width=True)
                    start_stock_series = forecast_summary_df[forecast_summary_df['item'] == item_to_plot]['current_stock']
                    if not start_stock_series.empty:
                        start_stock = start_stock_series.iloc[0]
                        item_forecast_df['projected_consumption'] = item_forecast_df['yhat'].clip(lower=0).cumsum()
                        item_forecast_df['forecasted_stock'] = start_stock - item_forecast_df['projected_consumption']
                        item_forecast_df.loc[item_forecast_df['forecasted_stock'] < 0, 'forecasted_stock'] = 0
                        stock_series = item_forecast_df.set_index('ds')['forecasted_stock']
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

    st.divider()
    tab2, tab3, tab4 = st.tabs(["**🎯 Reorder Advisor**", "**🌡️ Cold Chain Integrity**", "**🚚 Supplier Performance**"])
    with tab2: render_reorder_analysis(forecast_summary_df)
    with tab3: render_cold_chain_tab(iot_df)
    with tab4:
        st.header("🚚 Supplier Performance Scorecard")
        st.markdown("Analyze supplier reliability to de-risk your supply chain.")
        supplier_df = analyze_supplier_performance(health_df)
        if not supplier_df.empty:
            st.dataframe(supplier_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough procurement data to analyze supplier performance.")

if __name__ == "__main__":
    main()
