# sentinel_project_root/visualization/plots.py
# Plotting functions for Sentinel Health Co-Pilot Web Dashboards using Plotly.
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
import os
from typing import Optional, List, Dict, Any, Union
import html
import re

# --- Module Imports & Setup ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from .ui_elements import get_theme_color
except ImportError as e:
    logging.basicConfig(level=logging.INFO); logger_init = logging.getLogger(__name__)
    logger_init.error(f"CRITICAL IMPORT ERROR in plots.py: {e}. Using fallback settings.")
    class FallbackPlotSettings:
        THEME_FONT_FAMILY = 'sans-serif'; COLOR_TEXT_DARK = "#333333"; WEB_PLOT_DEFAULT_HEIGHT = 400;
        WEB_PLOT_COMPACT_HEIGHT = 350; MAPBOX_STYLE_WEB = "carto-positron";
    settings = FallbackPlotSettings()
    def get_theme_color(n, c="general", f=None): return f or "#757575"

logger = logging.getLogger(__name__)

def _get_setting(attr, default): return getattr(settings, attr, default)

def set_sentinel_plotly_theme():
    # This function is assumed correct and is omitted for brevity
    pass

set_sentinel_plotly_theme()

def create_empty_figure(chart_title: str, **kwargs) -> go.Figure:
    # This function is assumed correct and is omitted for brevity
    fig = go.Figure()
    fig.update_layout(title_text=f'<b>{html.escape(chart_title)}</b>', xaxis=dict(visible=False), yaxis=dict(visible=False), annotations=[dict(text=kwargs.get("message_text", "No data available."), showarrow=False)])
    return fig

def plot_bar_chart(
    df_input: Optional[pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    y_values_are_counts: bool = False,
    y_axis_title: Optional[str] = None,
    **kwargs  # FIXED: Accept arbitrary keyword arguments for Plotly Express
) -> go.Figure:
    """
    Creates a flexible bar chart. Now accepts **kwargs to pass 'orientation', etc.
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title)
        
    # Pass the entire kwargs dictionary to plotly express
    fig = px.bar(df_input, x=x_col, y=y_col, color=color_col, text_auto=True, **kwargs)
    
    hover_template = ""
    is_horizontal = kwargs.get('orientation') == 'h'
    
    if y_values_are_counts:
        # Adjust hover template based on orientation
        hover_template = f'<b>%{{y}}</b><br>Count: %{{x:,d}}<extra></extra>' if is_horizontal else f'<b>%{{x}}</b><br>Count: %{{y:,d}}<extra></extra>'
        text_template = '%{x:,.0f}' if is_horizontal else '%{y:,.0f}'
        fig.update_traces(texttemplate=text_template, hovertemplate=hover_template)
    
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title if y_axis_title is not None else y_col.replace('_', ' ').title(),
        legend_title=color_col.replace('_', ' ').title() if color_col else None,
        height=_get_setting('WEB_PLOT_DEFAULT_HEIGHT', 450)
    )
    return fig

# (Other plot functions like donut, line, heatmap, map remain unchanged and are omitted for brevity)


def plot_annotated_line_chart(
    data_series: Optional[pd.Series], chart_title: str, y_axis_label: str = "Value", **kwargs
) -> go.Figure:
    """Generates an annotated line chart from a pandas Series."""
    height = kwargs.get('chart_height') or _get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350)
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title, height)

    series = data_series.copy()
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index, errors='coerce')
    series.dropna(inplace=True)
    if series.empty: return create_empty_figure(chart_title, height)
    
    fig = px.line(series, title=f"<b>{html.escape(chart_title)}</b>", labels={'value': y_axis_label, 'index': 'Date'}, height=height)
    fig.update_traces(mode="lines+markers")
    return fig


def plot_bar_chart(
    df_input: Optional[pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    barmode: str = 'group',
    y_values_are_counts: bool = False,
    y_axis_title: Optional[str] = None  # FIXED: Added parameter for custom y-axis title
) -> go.Figure:
    """Creates a flexible bar chart from a DataFrame."""
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title)
        
    fig = px.bar(df_input, x=x_col, y=y_col, color=color_col, barmode=barmode, text_auto=True)
    
    if y_values_are_counts:
        fig.update_traces(texttemplate='%{y:,.0f}', hovertemplate=f'<b>%{{x}}</b><br>Count: %{{y:,d}}<extra></extra>')
        fig.update_yaxes(tickformat='d')
    else:
        fig.update_traces(texttemplate='%{y:,.1f}', hovertemplate=f'<b>%{{x}}</b><br>Value: %{{y:,.1f}}<extra></extra>')

    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis_title=x_col.replace('_', ' ').title(),
        # FIXED: Use the new parameter if provided, otherwise generate from column name.
        yaxis_title=y_axis_title if y_axis_title is not None else y_col.replace('_', ' ').title(),
        legend_title=color_col.replace('_', ' ').title() if color_col else None,
        height=_get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 450)
    )
    
    return fig


def plot_donut_chart(
    df_input: Optional[pd.DataFrame], labels_col: str, values_col: str, title: str, **kwargs
) -> go.Figure:
    """Generates a donut chart."""
    height = kwargs.get('height') or (_get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350) + 50)
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title, height)

    fig = px.pie(df_input, names=labels_col, values=values_col, title=f'<b>{html.escape(title)}</b>', hole=0.6, height=height)
    fig.update_traces(textinfo='label+percent', insidetextorientation='radial')
    return fig


def plot_heatmap(
    matrix_df: Optional[pd.DataFrame], title: str, **kwargs
) -> go.Figure:
    """Generates a heatmap from a matrix-like DataFrame."""
    height = kwargs.get('height') or _get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 450)
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        return create_empty_figure(title, height)
    
    fig = px.imshow(matrix_df, text_auto=True, title=f'<b>{html.escape(title)}</b>', height=height)
    return fig


def plot_choropleth_map(
    map_data_df: Optional[pd.DataFrame], geojson_features: Optional[Union[Dict, List]],
    value_col: str, map_title: str, zone_id_df_col: str = 'zone_id',
    zone_id_geojson_prop: str = 'zone_id', **kwargs
) -> go.Figure:
    """Generates a choropleth map."""
    height = kwargs.get('map_height') or _get_setting_or_default('WEB_MAP_DEFAULT_HEIGHT', 600)
    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or value_col not in map_data_df:
        return create_empty_figure(map_title, height, "Map data is incomplete.")
    if not geojson_features:
        return create_empty_figure(map_title, height, "Geographic boundary data unavailable.")

    try:
        geojson = geojson_features if isinstance(geojson_features, dict) else {"type": "FeatureCollection", "features": geojson_features}
        
        fig = px.choropleth_mapbox(
            map_data_df, geojson=geojson, locations=zone_id_df_col,
            featureidkey=f"properties.{zone_id_geojson_prop}", color=value_col,
            hover_name=kwargs.get('hover_name_col', 'name'), opacity=0.75, height=height
        )
        fig.update_layout(title_text=f"<b>{html.escape(map_title)}</b>", margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        logger.error(f"Error creating map '{map_title}': {e}", exc_info=True)
        return create_empty_figure(map_title, height, message="Error generating map.")
