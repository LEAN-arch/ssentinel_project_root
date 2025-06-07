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
    logging.basicConfig(level=logging.INFO)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"CRITICAL IMPORT ERROR in plots.py: {e}. Using fallback settings/colors.")
    
    class FallbackPlotSettings:
        THEME_FONT_FAMILY = 'sans-serif'; COLOR_TEXT_DARK = "#333333"; COLOR_BACKGROUND_CONTENT = "#FFFFFF";
        COLOR_BACKGROUND_PAGE = "#F0F2F6"; COLOR_ACTION_PRIMARY = "#007BFF"; COLOR_TEXT_HEADINGS_MAIN = "#111111";
        COLOR_BACKGROUND_CONTENT_TRANSPARENT="rgba(255,255,255,0.8)"; COLOR_BORDER_LIGHT="#E0E0E0"; COLOR_BORDER_MEDIUM="#BDBDBD";
        MAPBOX_STYLE_WEB = "carto-positron"; WEB_PLOT_DEFAULT_HEIGHT = 400; WEB_PLOT_COMPACT_HEIGHT = 350;
        WEB_MAP_DEFAULT_HEIGHT = 600; COLOR_TEXT_MUTED = '#757575'; COLOR_RISK_LOW = '#2ECC71';
        COLOR_RISK_MODERATE = '#FFA500'; COLOR_ACCENT_BRIGHT = '#FFC107'; COLOR_ACTION_SECONDARY = '#6C757D';
        COLOR_RISK_HIGH = '#FF0000'; COLOR_BACKGROUND_WHITE = '#FFFFFF';
    settings = FallbackPlotSettings()
    def get_theme_color(n, c="general", f=None): return f or "#CCCCCC"

logger = logging.getLogger(__name__)


# FIXED: Renamed the function to match its usage throughout the file, resolving the NameError.
def _get_setting_or_default(attr: str, default: Any) -> Any:
    return getattr(settings, attr, default)


# --- Global Setup for Plotly and Mapbox ---
MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = False
_mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
if _mapbox_token and len(_mapbox_token) > 20:
    try:
        px.set_mapbox_access_token(_mapbox_token)
        MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = True
        logger.info("Plotly: MAPBOX_ACCESS_TOKEN env var configured successfully.")
    except Exception as e:
        logger.error(f"Plotly: Error setting Mapbox token: {e}")
else:
    logger.warning("Plotly: MAPBOX_ACCESS_TOKEN not set. Maps will use open-source styles.")


def set_sentinel_plotly_theme():
    """Configures and applies a custom Plotly theme for the application."""
    theme_font = _get_setting_or_default('THEME_FONT_FAMILY', 'sans-serif')
    default_colorway = px.colors.qualitative.Plotly
    sentinel_colorway = [
        _get_setting_or_default('COLOR_ACTION_PRIMARY', default_colorway[0]),
        _get_setting_or_default('COLOR_RISK_LOW', default_colorway[1]),
        _get_setting_or_default('COLOR_RISK_MODERATE', default_colorway[2]),
        _get_setting_or_default('COLOR_ACCENT_BRIGHT', default_colorway[3]),
        _get_setting_or_default('COLOR_ACTION_SECONDARY', default_colorway[4]),
    ]
    
    layout_template = go.Layout(
        font=dict(family=theme_font, size=11, color=_get_setting_or_default('COLOR_TEXT_DARK', "#333333")),
        paper_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT', "#FFFFFF"),
        plot_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_PAGE', "#F0F2F6"),
        colorway=sentinel_colorway,
        xaxis=dict(gridcolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0")),
        yaxis=dict(gridcolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0")),
        # FIXED: The `color` property is now correctly nested inside the `font` dictionary for the title.
        title=dict(
            font=dict(
                family=theme_font, 
                size=16, 
                color=_get_setting_or_default('COLOR_TEXT_HEADINGS_MAIN', "#111111")
            ),
            x=0.05, 
            xanchor='left'
        ),
        legend=dict(bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT_TRANSPARENT', "rgba(255,255,255,0.8)"), borderwidth=0.5, orientation='h', y=1.02, x=1, xanchor='right'),
        margin=dict(l=60, r=20, t=80, b=60)
    )

    mapbox_style = _get_setting_or_default('MAPBOX_STYLE_WEB', "carto-positron")
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG and "mapbox" in mapbox_style.lower():
        mapbox_style = "carto-positron"
        logger.info(f"Plotly Theme: No Mapbox token; defaulting map style to '{mapbox_style}'.")
    
    layout_template.mapbox = dict(
        style=mapbox_style,
        center=dict(lat=_get_setting_or_default('MAP_DEFAULT_CENTER_LAT', 0), lon=_get_setting_or_default('MAP_DEFAULT_CENTER_LON', 0)),
        zoom=_get_setting_or_default('MAP_DEFAULT_ZOOM_LEVEL', 1)
    )
    
    pio.templates["sentinel_theme"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_theme"
    logger.info("Custom Plotly theme 'sentinel_theme' applied as default.")

try:
    set_sentinel_plotly_theme()
except Exception as e:
    logger.error(f"Failed to set custom Plotly theme: {e}", exc_info=True)


def create_empty_figure(chart_title: str, height: Optional[int] = None, message_text: str = "No data available.") -> go.Figure:
    """Creates a blank figure with a message, used as a placeholder."""
    final_height = height or _get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 400)
    fig = go.Figure()
    fig.update_layout(
        title_text=f'<b>{html.escape(chart_title)}</b>', height=final_height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=html.escape(message_text), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=12, font_color=_get_setting_or_default('COLOR_TEXT_MUTED', '#757575'))]
    )
    return fig


def plot_annotated_line_chart(
    data_series: Optional[pd.Series], chart_title: str, y_axis_label: str = "Value", **kwargs
) -> go.Figure:
    """Generates an annotated line chart from a pandas Series."""
    height = kwargs.get('chart_height', _get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350))
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title, height=height)

    series = data_series.copy()
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index, errors='coerce')
    series.dropna(inplace=True)
    if series.empty: return create_empty_figure(chart_title, height=height)
    
    fig = px.line(series, title=f"<b>{html.escape(chart_title)}</b>", labels={'value': y_axis_label, 'index': 'Date'}, height=height)
    fig.update_traces(mode="lines+markers")
    return fig


def plot_bar_chart(
    df_input: Optional[pd.DataFrame], x_col: str, y_col: str, title: str,
    color_col: Optional[str] = None, y_values_are_counts: bool = False,
    y_axis_title: Optional[str] = None, **kwargs
) -> go.Figure:
    """
    Creates a flexible bar chart. Accepts **kwargs to pass 'orientation', 'color_discrete_map', etc.
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title)
        
    try:
        # FIXED: Pass the entire kwargs dictionary to plotly express to handle 'orientation', etc.
        fig = px.bar(df_input, x=x_col, y=y_col, color=color_col, text_auto=True, **kwargs)
        
        is_horizontal = kwargs.get('orientation') == 'h'
        
        text_format = ',.0f' if y_values_are_counts else ',.1f'
        hover_data_format = ':,d' if y_values_are_counts else ':,.1f'

        if is_horizontal:
            text_template = f'%{{x:{text_format}}}'
            hovertemplate = f'<b>%{{y}}</b><br>{x_col.replace("_", " ")}: %{{x{hover_data_format}}}<extra></extra>'
        else:
            text_template = f'%{{y:{text_format}}}'
            hovertemplate = f'<b>%{{x}}</b><br>{y_col.replace("_", " ")}: %{{y{hover_data_format}}}<extra></extra>'

        fig.update_traces(texttemplate=text_template, hovertemplate=hovertemplate)
        
        fig.update_layout(
            title_text=f'<b>{html.escape(title)}</b>',
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_axis_title if y_axis_title is not None else y_col.replace('_', ' ').title(),
            legend_title=color_col.replace('_', ' ').title() if color_col else None
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
        return create_empty_figure(title, message_text=f"Chart Error: {e}")


def plot_donut_chart(df_input: Optional[pd.DataFrame], **kwargs) -> go.Figure:
    """Generates a donut chart."""
    title = kwargs.get('title', "Donut Chart")
    height = kwargs.get('height', _get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350) + 50)
    labels_col = kwargs.get('labels_col_name', 'label')
    values_col = kwargs.get('values_col_name', 'value')
    if not (isinstance(df_input, pd.DataFrame) and not df_input.empty and labels_col in df_input and values_col in df_input):
        return create_empty_figure(title, height=height)

    fig = px.pie(df_input, names=labels_col, values=values_col, title=f'<b>{html.escape(title)}</b>', hole=0.6, height=height)
    fig.update_traces(textinfo='label+percent', insidetextorientation='radial')
    return fig


def plot_heatmap(matrix_df: Optional[pd.DataFrame], **kwargs) -> go.Figure:
    """Generates a heatmap from a matrix-like DataFrame."""
    title = kwargs.get('chart_title', 'Heatmap')
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        return create_empty_figure(title)
    
    fig = px.imshow(matrix_df, text_auto=True, title=f'<b>{html.escape(title)}</b>')
    return fig


def plot_choropleth_map(map_data_df: Optional[pd.DataFrame], **kwargs) -> go.Figure:
    """Generates a choropleth map."""
    map_title = kwargs.get('map_title', "Map")
    height = kwargs.get('map_height', _get_setting_or_default('WEB_MAP_DEFAULT_HEIGHT', 600))
    value_col = kwargs.get('value_col_name')
    geojson_features = kwargs.get('geojson_features')
    zone_id_df_col = kwargs.get('zone_id_df_col', 'zone_id')
    
    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or not value_col or not geojson_features:
        return create_empty_figure(map_title, height=height, message_text="Map data or boundaries are missing.")

    try:
        geojson = geojson_features if isinstance(geojson_features, dict) else {"type": "FeatureCollection", "features": geojson_features}
        fig = px.choropleth_mapbox(
            map_data_df, geojson=geojson, locations=zone_id_df_col,
            featureidkey=f"properties.{kwargs.get('zone_id_geojson_prop', 'zone_id')}",
            color=value_col,
            hover_name=kwargs.get('hover_name_col', 'name'),
            height=height
        )
        fig.update_layout(title_text=f"<b>{html.escape(map_title)}</b>", margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        logger.error(f"Error creating map '{map_title}': {e}", exc_info=True)
        return create_empty_figure(map_title, height, message="Error generating map.")
        fig.update_layout(title_text=f"<b>{html.escape(map_title)}</b>", margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        logger.error(f"Error creating map '{map_title}': {e}", exc_info=True)
        return create_empty_figure(map_title, height, message="Error generating map.")
