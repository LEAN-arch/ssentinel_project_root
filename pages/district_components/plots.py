# sentinel_project_root/pages/district_components/plots.py
# Plotting functions specifically for the District Dashboard.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import logging
from typing import Optional, Dict, Any, List, Union
import html

try:
    from config import settings
    from visualization.ui_elements import get_theme_color 
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"CRITICAL IMPORT ERROR in district_components/plots.py: {e}. Using fallback settings.")
    class FallbackPlotSettings:
        COLOR_TEXT_DARK = "#333333"; COLOR_BACKGROUND_CONTENT = "#FFFFFF"; COLOR_BACKGROUND_PAGE = "#F0F2F6";
        COLOR_ACTION_PRIMARY = "#007BFF"; COLOR_TEXT_HEADINGS_MAIN = "#111111"; COLOR_TEXT_MUTED = '#757575';
        WEB_PLOT_DEFAULT_HEIGHT = 400; MAPBOX_STYLE_WEB = "carto-positron"; WEB_MAP_DEFAULT_HEIGHT = 600;
        MAP_DEFAULT_CENTER_LAT = 0.0; MAP_DEFAULT_CENTER_LON = 0.0; MAP_DEFAULT_ZOOM_LEVEL = 1;
    settings = FallbackPlotSettings()
    def get_theme_color(name_or_idx, category="general", fallback_color_hex=None): return fallback_color_hex or "#757575"

logger = logging.getLogger(__name__)

def _get_setting(attr: str, default: Any) -> Any:
    return getattr(settings, attr, default)

def create_empty_figure(title: str, height: Optional[int] = None, message: str = "No data available to display.") -> go.Figure:
    """Creates a blank Plotly figure with a message."""
    final_height = height or _get_setting('WEB_PLOT_DEFAULT_HEIGHT', 400)
    fig = go.Figure()
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>', height=final_height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=html.escape(message), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=14, font_color=_get_setting('COLOR_TEXT_MUTED', '#757575'))]
    )
    return fig

def plot_choropleth_map(
    map_data: Optional[pd.DataFrame],
    geojson: Optional[Dict[str, Any]],
    value_col: str,
    title: str,
    location_id_col: str,
    geojson_prop_key: str,
    height: Optional[int] = None
) -> go.Figure:
    """Creates a choropleth map for district-level visualization."""
    final_height = height or _get_setting('WEB_MAP_DEFAULT_HEIGHT', 600)
    if not isinstance(map_data, pd.DataFrame) or map_data.empty or not geojson:
        return create_empty_figure(title, final_height)
        
    fig = px.choropleth_mapbox(
        map_data,
        geojson=geojson,
        locations=location_id_col,
        featureidkey=f"properties.{geojson_prop_key}",
        color=value_col,
        color_continuous_scale="Viridis",
        mapbox_style=_get_setting('MAPBOX_STYLE_WEB', 'carto-positron'),
        zoom=_get_setting('MAP_DEFAULT_ZOOM_LEVEL', 4),
        center={"lat": _get_setting('MAP_DEFAULT_CENTER_LAT', 0), "lon": _get_setting('MAP_DEFAULT_CENTER_LON', 0)},
        opacity=0.6,
        hover_name=location_id_col,
        hover_data={value_col: ':.2f'}
    )
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        margin={"r":0, "t":50, "l":0, "b":0},
        height=final_height
    )
    return fig

def plot_bar_chart(
    df_input: Optional[pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    barmode: str = 'stack',
    y_values_are_counts: bool = True,
    height: Optional[int] = None
) -> go.Figure:
    """Creates a stacked or grouped bar chart for district comparisons."""
    final_height = height or _get_setting('WEB_PLOT_DEFAULT_HEIGHT', 400)
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title, final_height)
        
    fig = px.bar(
        df_input,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode=barmode,
        text_auto=True,
        height=final_height
    )
    
    if y_values_are_counts:
        fig.update_traces(texttemplate='%{y:,d}', hovertemplate='%{x}<br>Count: %{y:,d}<extra></extra>')
        fig.update_yaxes(tickformat='d')
    else:
        fig.update_traces(texttemplate='%{y:,.1f}', hovertemplate='%{x}<br>Value: %{y:,.1f}<extra></extra>')

    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        legend_title=color_col.replace('_', ' ').title() if color_col else None
    )
    return fig
