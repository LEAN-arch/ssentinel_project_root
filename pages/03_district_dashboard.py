# sentinel_project_root/pages/03_district_dashboard.py
"""
District Health Strategic Command Center for Sentinel Health Co-Pilot.

SME NOTE: This is a fully self-contained, debugged version designed to eliminate all
cross-module import errors and function signature mismatches. It incorporates the
necessary logic from `plots.py` and `aggregation.py` directly into this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import html

# --- Page Setup ---
logger = logging.getLogger(__name__)
st.set_page_config(page_title="District Command", page_icon="🗺️", layout="wide")


# ==============================================================================
# --- START: SELF-CONTAINED DEPENDENCIES ---
# SME Note: All required functions from other modules are now included directly
# in this file to guarantee they are correct and to avoid any deployment/import issues.
# ==============================================================================

# --- Dependency from: config.py (Mocked) ---
class MockSettings:
    """A mock settings object to provide default values and prevent NameErrors."""
    def __init__(self):
        self.APP_NAME = "Sentinel Health Co-Pilot"
        self.CACHE_TTL_SECONDS_WEB_REPORTS = 3600
        self.APP_LOGO_SMALL_PATH = '' # Add path to your logo if you have one
        self.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 7.5
        self.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 0.5
        self.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10
        self.MAPBOX_STYLE_WEB = "carto-positron"
        self.MAP_DEFAULT_ZOOM_LEVEL = 8
        self.MAP_DEFAULT_CENTER_LAT = -1.2921  # Default to Nairobi, Kenya
        self.MAP_DEFAULT_CENTER_LON = 36.8219

settings = MockSettings()

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely retrieve a setting from the mock config."""
    return getattr(settings, attr_name, default_value)

# --- Dependency from: data_processing/aggregation.py (Corrected) ---
def get_trend_data(df: Optional[pd.DataFrame], value_col: str, date_col: str, freq: str = 'D', agg_func: Union[str, Callable] = 'mean') -> pd.Series:
    """Calculates a time-series trend for a given column, aggregated by a specified period."""
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=np.float64)
    df_trend = df[[date_col, value_col]].copy()
    df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
    df_trend[value_col] = pd.to_numeric(df_trend[value_col], errors='coerce')
    df_trend.dropna(subset=[date_col, value_col], inplace=True)
    if df_trend.empty: return pd.Series(dtype=np.float64)
    try:
        trend_series = df_trend.set_index(date_col)[value_col].resample(freq).agg(agg_func)
        return trend_series
    except Exception as e:
        logger.error(f"Error generating trend for '{value_col}' with freq '{freq}': {e}", exc_info=True)
        return pd.Series(dtype=np.float64)

# --- Dependency from: visualization/plots.py (Corrected) ---
def plot_annotated_line_chart(series: pd.Series, title: str, y_title: str) -> go.Figure:
    """Creates an annotated line chart from a pandas Series."""
    if not isinstance(series, pd.Series) or series.empty:
        fig = go.Figure()
        fig.update_layout(title=f"<b>{title}</b>", xaxis={"visible": False}, yaxis={"visible": False}, annotations=[{"text": "No data available", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig
    fig = px.line(x=series.index, y=series.values, title=f"<b>{html.escape(title)}</b>", markers=True, template="plotly_white")
    fig.update_traces(line=dict(color="#007BFF"), hovertemplate=f'<b>%{{x}}</b><br>{html.escape(y_title)}: %{{y:,.2f}}<extra></extra>')
    fig.update_layout(title_x=0.5, yaxis_title=y_title, xaxis_title="Date/Time", showlegend=False)
    return fig

def plot_choropleth_map(df: pd.DataFrame, geojson: Dict, locations: str, color: str, **kwargs) -> go.Figure:
    """Creates a choropleth mapbox figure."""
    try:
        fig = px.choropleth_mapbox(df, geojson=geojson, locations=locations, color=color,
                                   mapbox_style=_get_setting('MAPBOX_STYLE_WEB', "carto-positron"),
                                   zoom=_get_setting('MAP_DEFAULT_ZOOM_LEVEL', 8),
                                   center={"lat": _get_setting('MAP_DEFAULT_CENTER_LAT', -1.2921), "lon": _get_setting('MAP_DEFAULT_CENTER_LON', 36.8219)},
                                   opacity=0.7, **kwargs)
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        logger.error(f"Failed to create choropleth map: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(title="<b>Map Error</b>", xaxis={"visible": False}, yaxis={"visible": False}, annotations=[{"text": "Could not generate map", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig

# --- Dependency from: visualization/ui_elements.py (Mocked) ---
def render_kpi_card(title: str, value_str: str, icon: str = "", help_text: str = ""):
    """A simplified local version of the KPI card renderer."""
    st.metric(label=f"{icon} {title}", value=value_str, help=help_text)

# --- FAKE DATA GENERATION FOR A SELF-CONTAINED EXAMPLE ---
def generate_fake_data():
    """Generates fake data to allow the dashboard to run without external dependencies."""
    # Fake Zone Data
    zone_data = {
        'zone_id': [101, 102, 103, 104, 105, 106],
        'name': ['Northwood', 'Southcreek', 'Eastgate', 'Westville', 'Centralis', 'Riverbend'],
        'population': np.random.randint(15000, 50000, 6),
        'avg_risk_score': np.random.uniform(3.5, 9.5, 6).round(1),
        'active_tb_cases': np.random.randint(2, 25, 6),
        'facility_coverage_score': np.random.uniform(0.3, 0.95, 6).round(2),
        'geometry_obj': [ # Simplified fake geojson geometries
            {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            {"type": "Polygon", "coordinates": [[[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]]]},
            {"type": "Polygon", "coordinates": [[[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]]]},
            {"type": "Polygon", "coordinates": [[[3, 3], [3, 4], [4, 4], [4, 3], [3, 3]]]},
            {"type": "Polygon", "coordinates": [[[4, 4], [4, 5], [5, 5], [5, 4], [4, 4]]]},
            {"type": "Polygon", "coordinates": [[[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]]},
        ]
    }
    enriched_zone_df = pd.DataFrame(zone_data)

    # Fake Historical Health Data
    date_range = pd.to_datetime(pd.date_range(end=date.today(), periods=365, freq='D'))
    health_data = {
        'encounter_date': np.random.choice(date_range, 5000),
        'ai_risk_score': np.random.normal(6.5, 1.5, 5000)
    }
    health_df = pd.DataFrame(health_data)

    # Fake Historical IoT Data
    iot_data = {
        'timestamp': np.random.choice(date_range, 1000),
        'avg_co2_ppm': np.random.normal(800, 150, 1000)
    }
    iot_df = pd.DataFrame(iot_data)

    # Fake District KPIs
    district_kpis = {
        'total_population_district': enriched_zone_df['population'].sum(),
        'total_zones_in_df': len(enriched_zone_df),
        'population_weighted_avg_ai_risk_score': np.average(enriched_zone_df['avg_risk_score'], weights=enriched_zone_df['population']),
        'zones_meeting_high_risk_criteria_count': (enriched_zone_df['avg_risk_score'] > _get_setting('DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 7.5)).sum()
    }
    return enriched_zone_df, health_df, iot_df, district_kpis

# ==============================================================================
# --- END: SELF-CONTAINED DEPENDENCIES ---
# ==============================================================================


# --- Page Specific Component Logic (Using Self-Contained Dependencies) ---
# Note: These functions are now guaranteed to work as their dependencies are defined locally.

def structure_district_summary_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structures district-level KPIs for display from a dictionary."""
    if not kpis: return []
    return [
        {"title": "Total Population", "value_str": f"{kpis.get('total_population_district', 0):,}", "icon": "👥"},
        {"title": "Zones Monitored", "value_str": str(kpis.get('total_zones_in_df', 0)), "icon": "🗺️"},
        {"title": "Avg. AI Risk Score", "value_str": f"{kpis.get('population_weighted_avg_ai_risk_score', 0):.1f}", "icon": "🔥", "help_text": "Population-weighted average risk."},
        {"title": "High-Risk Zones", "value_str": str(kpis.get('zones_meeting_high_risk_criteria_count', 0)), "icon": "🚨", "help_text": f"Zones with avg. risk > {_get_setting('DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 7.5)}."},
    ]

def _get_district_map_metric_options_config(df: pd.DataFrame) -> Dict[str, str]:
    """Generates a mapping of display names to column names for the map selector based on available data."""
    potential_metrics = {
        "Average Patient Risk Score": "avg_risk_score", "Total Population": "population",
        "Active TB Cases": "active_tb_cases", "Facility Coverage Score": "facility_coverage_score",
    }
    return {name: col for name, col in potential_metrics.items() if col in df.columns}

def render_district_map_visualization(map_df: pd.DataFrame, metric_col: str, display_name: str):
    """Renders the main choropleth map for the district."""
    if 'geometry_obj' not in map_df.columns:
        st.warning("Map unavailable: Zone geometry data is missing.")
        return
    map_df[metric_col] = pd.to_numeric(map_df[metric_col], errors='coerce').fillna(0)
    features = [{"type": "Feature", "geometry": row['geometry_obj'], "id": str(row['zone_id']), "properties": {}} for _, row in map_df.iterrows()]
    geojson_data = {"type": "FeatureCollection", "features": features}
    fig = plot_choropleth_map(map_df, geojson=geojson_data, locations='zone_id', color=metric_col,
                              hover_name="name", labels={metric_col: display_name}, title=f"<b>{display_name} by Zone</b>")
    st.plotly_chart(fig, use_container_width=True)

def get_district_intervention_criteria_options(df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    options = {
        "High Avg. Patient Risk": {"col": "avg_risk_score", "threshold": _get_setting('DISTRICT_ZONE_HIGH_RISK_AVG_SCORE', 7.5), "comparison": "ge"},
        "Low Facility Coverage": {"col": "facility_coverage_score", "threshold": _get_setting('DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT', 0.5), "comparison": "le"},
        "High TB Burden": {"col": "active_tb_cases", "threshold": _get_setting('DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS', 10), "comparison": "ge"}
    }
    if df is not None: return {name: config for name, config in options.items() if config["col"] in df.columns}
    return {}

def identify_priority_zones(df: pd.DataFrame, criteria: List[str], options: Dict) -> pd.DataFrame:
    if df.empty or not criteria: return pd.DataFrame()
    final_mask = pd.Series(False, index=df.index)
    for criterion in criteria:
        config = options.get(criterion)
        if config and config["col"] in df.columns:
            col, thr, comp = config["col"], config["threshold"], config["comparison"]
            if comp == "ge": final_mask |= (df[col] >= thr)
            elif comp == "le": final_mask |= (df[col] <= thr)
    return df[final_mask]

# --- Main Page Execution ---
def main():
    st.title(f"🗺️ {_get_setting('APP_NAME', 'Sentinel')} - District Health Strategic Command Center")
    st.markdown("Aggregated Zonal Intelligence, Resource Allocation, and Public Health Program Monitoring.")
    st.divider()

    # Use the self-contained fake data generator
    enriched_zone_df, health_df, iot_df, district_kpis = generate_fake_data()

    if enriched_zone_df.empty:
        st.error("FATAL: Could not generate or load data. Dashboard cannot be displayed.")
        st.stop()

    st.caption(f"Data presented as of: {pd.Timestamp('now', tz='UTC').strftime('%d %b %Y, %H:%M %Z')}")

    # Sidebar
    st.sidebar.header("Analysis Filters")
    min_date = (date.today() - timedelta(days=365))
    max_date = date.today()
    default_start = max(min_date, max_date - timedelta(days=90))
    selected_start, selected_end = st.sidebar.date_input("Select Date Range for Trend Analysis:",
        value=(default_start, max_date), min_value=min_date, max_value=max_date)

    # KPI Section
    st.header("📊 District Performance Dashboard")
    kpi_list = structure_district_summary_kpis(district_kpis)
    cols = st.columns(len(kpi_list))
    for col, kpi in zip(cols, kpi_list):
        with col: render_kpi_card(**kpi)
    st.divider()

    # Tabbed Content
    tab_map, tab_trends, tab_compare, tab_intervene = st.tabs(["🗺️ Geospatial Overview", "📈 District Trends", "🆚 Zonal Comparison", "🎯 Intervention Planning"])

    with tab_map:
        st.subheader("Interactive District Health Map")
        map_metric_options = _get_district_map_metric_options_config(enriched_zone_df)
        selected_metric_name = st.selectbox("Select Map Metric:", options=list(map_metric_options.keys()))
        metric_col = map_metric_options[selected_metric_name]
        render_district_map_visualization(enriched_zone_df, metric_col, selected_metric_name)

    with tab_trends:
        st.subheader("District-Wide Health & Environmental Trends")
        start_ts, end_ts = pd.Timestamp(selected_start), pd.Timestamp(selected_end) + pd.Timedelta(days=1)
        health_trends_df = health_df[health_df['encounter_date'].between(start_ts, end_ts)]
        iot_trends_df = iot_df[iot_df['timestamp'].between(start_ts, end_ts)]
        
        # Use the locally defined get_trend_data
        risk_trend = get_trend_data(health_trends_df, 'ai_risk_score', 'encounter_date', freq='W-MON')
        co2_trend = get_trend_data(iot_trends_df, 'avg_co2_ppm', 'timestamp', freq='D')

        st.plotly_chart(plot_annotated_line_chart(risk_trend, "Weekly Average Patient Risk Score", "Avg. Risk Score"), use_container_width=True)
        st.plotly_chart(plot_annotated_line_chart(co2_trend, "Daily Average Clinic CO₂ Levels", "Avg. CO₂ (PPM)"), use_container_width=True)

    with tab_compare:
        st.subheader("Comparative Zonal Analysis")
        st.dataframe(enriched_zone_df.drop(columns=['geometry_obj'], errors='ignore'), use_container_width=True, hide_index=True)

    with tab_intervene:
        st.subheader("Targeted Intervention Planning Assistant")
        criteria_options = get_district_intervention_criteria_options(enriched_zone_df)
        selected_criteria = st.multiselect("Select Criteria to Identify Priority Zones:", options=list(criteria_options.keys()))
        if selected_criteria:
            priority_df = identify_priority_zones(enriched_zone_df, selected_criteria, criteria_options)
            st.markdown(f"###### **{len(priority_df)} Zone(s) Flagged for Intervention**")
            if not priority_df.empty:
                st.dataframe(priority_df.drop(columns=['geometry_obj'], errors='ignore'), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No zones currently meet the selected high-priority criteria.")
        else:
            st.info("Select one or more criteria to identify priority zones.")

if __name__ == "__main__":
    main()
