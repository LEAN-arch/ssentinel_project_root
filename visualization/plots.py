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
    # Fallback logic remains unchanged
    logging.basicConfig(level=logging.INFO); logger_init = logging.getLogger(__name__)
    logger_init.error(f"CRITICAL IMPORT ERROR in plots.py: {e}. Using fallback settings.")
    class FallbackPlotSettings:
        THEME_FONT_FAMILY = 'sans-serif'; COLOR_TEXT_DARK = "#333333"; COLOR_BACKGROUND_CONTENT = "#FFFFFF";
        COLOR_BACKGROUND_PAGE = "#F0F2F6"; COLOR_ACTION_PRIMARY = "#007BFF"; COLOR_TEXT_HEADINGS_MAIN = "#111111";
        MAPBOX_STYLE_WEB = "carto-positron"; WEB_PLOT_DEFAULT_HEIGHT = 400; WEB_PLOT_COMPACT_HEIGHT = 350;
    settings = FallbackPlotSettings()
    def get_theme_color(n, c="general", f=None): return f or "#757575"

logger = logging.getLogger(__name__)

def _get_setting_or_default(attr: str, default: Any) -> Any: return getattr(settings, attr, default)

# --- Global Setup for Plotly and Mapbox (unchanged) ---
MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = bool(os.getenv("MAPBOX_ACCESS_TOKEN"))
if MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG: px.set_mapbox_access_token(os.getenv("MAPBOX_ACCESS_TOKEN"))

def set_sentinel_plotly_theme():
    """Configures and applies a custom Plotly theme for the application."""
    theme_font = _get_setting_or_default('THEME_FONT_FAMILY', 'sans-serif')
    colorway = [get_theme_color(i, "general") for i in range(8)]
    
    layout_template = go.Layout(
        font=dict(family=theme_font, size=11, color=_get_setting_or_default('COLOR_TEXT_DARK', "#333")),
        paper_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT', "#FFF"),
        plot_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_PAGE', "#F0F2F6"),
        colorway=colorway,
        # FIXED: The `color` property must be inside the `font` dictionary for the title.
        title=dict(
            font=dict(
                family=theme_font, 
                size=16, 
                color=_get_setting_or_default('COLOR_TEXT_HEADINGS_MAIN', "#111111")
            ),
            x=0.05, 
            xanchor='left'
        ),
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
        margin=dict(l=60, r=20, t=80, b=60)
    )

    mapbox_style = _get_setting_or_default('MAPBOX_STYLE_WEB', "carto-positron")
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG and "mapbox" in mapbox_style:
        mapbox_style = "carto-positron"
    layout_template.mapbox = dict(style=mapbox_style)
    
    pio.templates["sentinel_theme"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_theme"
    logger.info("Custom Plotly theme 'sentinel_theme' applied as default.")

# Apply the theme when the module is loaded.
try:
    set_sentinel_plotly_theme()
except Exception as e:
    logger.error(f"Failed to set custom Plotly theme: {e}", exc_info=True)


def create_empty_figure(chart_title: str, height: Optional[int] = None, message_text: str = "No data available.") -> go.Figure:
    final_height = height or _get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 400)
    fig = go.Figure()
    fig.update_layout(
        title_text=f'<b>{html.escape(chart_title)}</b>', height=final_height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=html.escape(message_text), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=12)]
    )
    return fig


def plot_bar_chart(
    df_input: Optional[pd.DataFrame], x_col: str, y_col: str, title: str,
    color_col: Optional[str] = None, y_values_are_counts: bool = False,
    y_axis_title: Optional[str] = None, **kwargs
) -> go.Figure:
    """
    Creates a flexible bar chart from a DataFrame.
    FIXED: Accepts **kwargs to pass 'orientation', 'color_discrete_map', etc., to px.bar.
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title)
        
    fig = px.bar(df_input, x=x_col, y=y_col, color=color_col, text_auto=True, **kwargs)
    
    if y_values_are_counts:
        fig.update_traces(texttemplate='%{y:,.0f}', hovertemplate=f'<b>%{{y}}</b><br>Count: %{{x:,d}}<extra></extra>' if kwargs.get('orientation') == 'h' else f'<b>%{{x}}</b><br>Count: %{{y:,d}}<extra></extra>')
        if kwargs.get('orientation') == 'h': fig.update_xaxes(tickformat='d')
        else: fig.update_yaxes(tickformat='d')
    
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title if y_axis_title is not None else y_col.replace('_', ' ').title(),
        legend_title=color_col.replace('_', ' ').title() if color_col else None,
        height=_get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 450)
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
