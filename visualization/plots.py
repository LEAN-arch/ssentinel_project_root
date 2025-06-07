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

# --- Robust Imports & Settings ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
    from .ui_elements import get_theme_color
except ImportError as e:
    # This fallback ensures the module can be imported even if dependencies are missing,
    # which is crucial for startup and error reporting.
    logging.basicConfig(level=logging.INFO)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"PLOTS IMPORT ERROR: {e}. Using fallback settings. Plots may not be styled correctly.")
    class FallbackPlotSettings:
        THEME_FONT_FAMILY = 'sans-serif'; COLOR_TEXT_DARK = "#333"; COLOR_BACKGROUND_CONTENT = "#FFF"
        COLOR_BACKGROUND_PAGE = "#F0F2F6"; COLOR_ACTION_PRIMARY = "#007BFF"; COLOR_BORDER_LIGHT = "#EEE"
        COLOR_BORDER_MEDIUM = "#CCC"; COLOR_TEXT_HEADINGS_MAIN = "#111"; COLOR_TEXT_MUTED = '#777'
        MAPBOX_STYLE_WEB = "carto-positron"; MAP_DEFAULT_CENTER_LAT = 0.0; MAP_DEFAULT_CENTER_LON = 0.0
        MAP_DEFAULT_ZOOM_LEVEL = 1; WEB_PLOT_DEFAULT_HEIGHT = 400; WEB_PLOT_COMPACT_HEIGHT = 350
        LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {}
    settings = FallbackPlotSettings()
    def get_theme_color(name_or_idx, category="general", fallback_color=None): return fallback_color or "#777"
    def convert_to_numeric(data, default_value=np.nan, target_type=None): return pd.to_numeric(data, errors='coerce').fillna(default_value)

logger = logging.getLogger(__name__)

# --- Mapbox Configuration ---
MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = False
if os.getenv("MAPBOX_ACCESS_TOKEN"):
    px.set_mapbox_access_token(os.getenv("MAPBOX_ACCESS_TOKEN"))
    MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = True
    logger.info("Plotly: Mapbox token found and configured.")
else:
    logger.warning("Plotly: MAPBOX_ACCESS_TOKEN env var not set. Maps may fall back to open styles.")


def set_sentinel_plotly_theme():
    """Applies a custom Plotly theme for consistent styling across all charts."""
    theme_font = getattr(settings, 'THEME_FONT_FAMILY', 'sans-serif')
    colorway = [get_theme_color(i, "general") for i in range(8)]
    
    # Intelligent map style selection
    mapbox_style = getattr(settings, 'MAPBOX_STYLE_WEB', 'carto-positron')
    open_map_styles = {"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner"}
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG and mapbox_style not in open_map_styles:
        logger.warning(f"Map style '{mapbox_style}' requires a token but none is set. Falling back to 'carto-positron'.")
        mapbox_style = 'carto-positron'

    layout_template = go.Layout(
        font=dict(family=theme_font, size=12, color=getattr(settings, 'COLOR_TEXT_DARK', '#333')),
        paper_bgcolor=getattr(settings, 'COLOR_BACKGROUND_CONTENT', '#FFF'),
        plot_bgcolor=getattr(settings, 'COLOR_BACKGROUND_PAGE', '#F0F2F6'),
        colorway=colorway,
        title=dict(font_size=16, x=0.05, xanchor='left', font_color=getattr(settings, 'COLOR_TEXT_HEADINGS_MAIN', '#111')),
        xaxis=dict(gridcolor=getattr(settings, 'COLOR_BORDER_LIGHT', '#EEE'), zeroline=False, automargin=True, title_standoff=10),
        yaxis=dict(gridcolor=getattr(settings, 'COLOR_BORDER_LIGHT', '#EEE'), zeroline=False, automargin=True, title_standoff=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=10),
        margin=dict(l=60, r=30, t=70, b=50),
        mapbox=dict(
            style=mapbox_style,
            center=dict(lat=getattr(settings, 'MAP_DEFAULT_CENTER_LAT', 0), lon=getattr(settings, 'MAP_DEFAULT_CENTER_LON', 0)),
            zoom=getattr(settings, 'MAP_DEFAULT_ZOOM_LEVEL', 1)
        )
    )
    pio.templates["sentinel_theme"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_theme"
    logger.info("Custom Plotly theme 'sentinel_theme' configured and applied.")

# --- Initialize Theme on Import ---
try:
    set_sentinel_plotly_theme()
except Exception as e:
    logger.error(f"Failed to set custom Plotly theme: {e}", exc_info=True)


def create_empty_figure(
    chart_title: str,
    height: Optional[int] = None,
    message: str = "No data available for the current selection."
) -> go.Figure:
    """Creates a standardized empty chart figure with a message."""
    final_height = height or getattr(settings, 'WEB_PLOT_COMPACT_HEIGHT', 350)
    fig = go.Figure()
    fig.update_layout(
        title_text=html.escape(chart_title),
        height=final_height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=html.escape(message),
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=getattr(settings, 'COLOR_TEXT_MUTED', '#777'))
        )]
    )
    return fig


def plot_annotated_line_chart(
    data_series: Optional[pd.Series],
    chart_title: str,
    y_axis_label: str,
    y_values_are_counts: bool = False,
    height: Optional[int] = None,
    line_color: Optional[str] = None
) -> go.Figure:
    """
    Generates a clean, readable line chart with annotations.

    Args:
        data_series: Time-indexed pandas Series.
        chart_title: The title of the chart.
        y_axis_label: The label for the Y-axis.
        y_values_are_counts: If True, forces integer ticks on the y-axis.
        height: Optional height of the chart in pixels.
        line_color: Optional hex code for the line color.
    """
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title, height)

    fig = go.Figure()
    trace_name = html.escape(y_axis_label)
    hover_format = 'd' if y_values_are_counts else ',.1f'

    fig.add_trace(go.Scatter(
        x=data_series.index,
        y=data_series.values,
        mode='lines+markers',
        name=trace_name,
        line=dict(color=(line_color or get_theme_color(0)), width=2.5),
        marker=dict(size=6),
        hovertemplate=f'<b>Date</b>: %{{x|%Y-%m-%d}}<br><b>{trace_name}</b>: %{{y:{hover_format}}}<extra></extra>'
    ))

    fig.update_layout(
        title_text=html.escape(chart_title),
        yaxis_title=html.escape(y_axis_label),
        height=(height or getattr(settings, 'WEB_PLOT_COMPACT_HEIGHT', 350))
    )

    # **CRITICAL IMPROVEMENT**: Enforce integer ticks for count-based data
    if y_values_are_counts:
        max_val = data_series.max()
        if pd.notna(max_val) and max_val > 0:
            # Generate integer ticks up to the max value
            tick_step = max(1, int(np.ceil(max_val / 5))) # Aim for about 5 ticks
            dtick = np.ceil(tick_step) if max_val > 10 else 1
            fig.update_yaxes(tickmode='linear', tick0=0, dtick=dtick, rangemode='tozero')
        else:
            fig.update_yaxes(tickmode='linear', tick0=0, dtick=1, rangemode='tozero')
            
    return fig


def plot_bar_chart(
    df_input: Optional[pd.DataFrame],
    x_col_name: str,
    y_col_name: str,
    chart_title: str,
    y_values_are_counts_flag: bool = False,
    color_col_name: Optional[str] = None,
    height: Optional[int] = None
) -> go.Figure:
    """Generates a standardized bar chart."""
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or x_col_name not in df_input.columns or y_col_name not in df_input.columns:
        return create_empty_figure(chart_title, height)
        
    try:
        fig = px.bar(
            df_input,
            x=x_col_name,
            y=y_col_name,
            color=color_col_name,
            text_auto=True,
            title=html.escape(chart_title),
            height=(height or getattr(settings, 'WEB_PLOT_COMPACT_HEIGHT', 350))
        )
        
        hover_template = f'<b>%{{x}}</b><br><b>{html.escape(y_col_name.replace("_", " ").title())}</b>: %{{y}}<extra></extra>'
        text_template = '%{y:,d}' if y_values_are_counts_flag else '%{y:,.1f}'
        
        fig.update_traces(texttemplate=text_template, hovertemplate=hover_template)
        fig.update_layout(
            xaxis_title=html.escape(x_col_name.replace('_', ' ').title()),
            yaxis_title=html.escape(y_col_name.replace('_', ' ').title()),
            legend_title=html.escape(color_col_name.replace('_', ' ').title()) if color_col_name else None
        )
        
        if y_values_are_counts_flag:
            fig.update_yaxes(tickformat='d', rangemode='tozero')

        return fig
    except Exception as e:
        logger.error(f"Error creating bar chart '{chart_title}': {e}", exc_info=True)
        return create_empty_figure(chart_title, height, message="Error generating chart.")


def plot_donut_chart(
    df_input: Optional[pd.DataFrame],
    labels_col: str,
    values_col: str,
    chart_title: str,
    height: Optional[int] = None,
    color_map: Optional[Dict[str, str]] = None
) -> go.Figure:
    """Generates a styled donut chart."""
    final_height = height or getattr(settings, 'WEB_PLOT_COMPACT_HEIGHT', 350)
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or labels_col not in df_input.columns or values_col not in df_input.columns:
        return create_empty_figure(chart_title, final_height, "Missing data for donut chart.")

    df_plot = df_input.copy()
    try:
        df_plot[values_col] = convert_to_numeric(df_plot[values_col], default_value=0)
        df_plot = df_plot[df_plot[values_col] > 1e-6].sort_values(by=values_col, ascending=False)
    except Exception as e:
        logger.error(f"Error preparing data for donut chart '{chart_title}': {e}", exc_info=True)
        return create_empty_figure(chart_title, final_height, "Error preparing data.")

    if df_plot.empty:
        return create_empty_figure(chart_title, final_height)
        
    fig = px.pie(
        df_plot,
        names=labels_col,
        values=values_col,
        title=html.escape(chart_title),
        hole=0.5,
        height=final_height,
        color_discrete_map=color_map
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percent: %{percent}<extra></extra>')
    fig.update_layout(showlegend=False, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def plot_heatmap(
    matrix_df: Optional[pd.DataFrame],
    chart_title: str,
    height: Optional[int] = None,
    color_scale: str = 'Viridis'
) -> go.Figure:
    """Generates a styled heatmap."""
    final_height = height or getattr(settings, 'WEB_PLOT_DEFAULT_HEIGHT', 400)
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        return create_empty_figure(chart_title, final_height)

    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale=color_scale,
        hoverongaps=False
    ))
    fig.update_layout(
        title_text=html.escape(chart_title),
        height=final_height,
        xaxis_nticks=len(matrix_df.columns),
        yaxis_nticks=len(matrix_df.index)
    )
    return fig


def plot_choropleth_map(
    map_data_df: Optional[pd.DataFrame],
    geojson_features: Optional[Dict[str, Any]],
    value_col_name: str,
    map_title: str,
    zone_id_geojson_prop: str = 'zone_id',
    zone_id_df_col: str = 'zone_id',
    color_scale_name: str = 'Viridis',
    height: Optional[int] = None
) -> go.Figure:
    """Generates a standardized choropleth map."""
    final_height = height or getattr(settings, 'WEB_MAP_DEFAULT_HEIGHT', 600)
    
    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or not geojson_features:
        return create_empty_figure(map_title, final_height, "Geographic or value data unavailable.")

    if value_col_name not in map_data_df.columns or zone_id_df_col not in map_data_df.columns:
         return create_empty_figure(map_title, final_height, f"Required columns missing for map.")

    try:
        fig = px.choropleth_mapbox(
            map_data_df,
            geojson=geojson_features,
            locations=zone_id_df_col,
            featureidkey=f"properties.{zone_id_geojson_prop}",
            color=value_col_name,
            color_continuous_scale=color_scale_name,
            opacity=0.7,
            title=html.escape(map_title),
            height=final_height
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        logger.error(f"Error creating map '{map_title}': {e}", exc_info=True)
        return create_empty_figure(map_title, final_height, message="Error generating map.")
