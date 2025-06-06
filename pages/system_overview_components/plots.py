# sentinel_project_root/pages/system_overview_components/plots.py
# Plotting functions specifically for the National/System Overview Dashboard.

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
    logger_init.error(f"CRITICAL IMPORT ERROR in system_overview_components/plots.py: {e}. Using fallback settings.")
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

def plot_indicator(
    value: float,
    title: str,
    reference_value: Optional[float] = None,
    mode: str = "number+delta"
) -> go.Figure:
    """Creates a single KPI indicator figure."""
    fig = go.Figure(go.Indicator(
        mode=mode,
        value=value,
        title={"text": title},
        delta={'reference': reference_value} if reference_value is not None else None
    ))
    fig.update_layout(height=250)
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
    """Creates a choropleth map for national-level visualization."""
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
        zoom=_get_setting('MAP_DEFAULT_ZOOM_LEVEL', 3),
        center={"lat": _get_setting('MAP_DEFAULT_CENTER_LAT', 0), "lon": _get_setting('MAP_DEFAULT_CENTER_LON', 0)},
        opacity=0.6,
        hover_name=location_id_col
    )
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        margin={"r":0, "t":50, "l":0, "b":0},
        height=final_height
    )
    return fig
