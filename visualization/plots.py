# sentinel_project_root/visualization/plots.py
# Standardized, robust plotting functions for Sentinel Health Co-Pilot dashboards.

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import logging
import os
import html
from typing import Optional, List, Dict, Any

try:
    from config import settings
except ImportError:
    # Fallback for standalone execution or testing
    class FallbackSettings:
        THEME_FONT_FAMILY = 'sans-serif'
        COLOR_TEXT_DARK = "#333333"
        COLOR_BACKGROUND_CONTENT = "#FFFFFF"
        COLOR_TEXT_HEADINGS_MAIN = "#111111"
        MAPBOX_STYLE_WEB = "carto-positron"
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

# --- Theme and Mapbox Setup ---
def _get_setting(attr: str, default: Any) -> Any:
    """Safely gets a setting attribute."""
    return getattr(settings, attr, default)

def set_sentinel_plotly_theme():
    """Configures and applies a custom Plotly theme for the application."""
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
    mapbox_style = _get_setting('MAPBOX_STYLE_WEB', 'carto-positron')
    
    # If a premium Mapbox style is requested but no token is available, default to an open style.
    if "mapbox" in mapbox_style.lower() and not mapbox_token:
        logger.warning(f"Mapbox style '{mapbox_style}' requested but no MAPBOX_ACCESS_TOKEN found. Defaulting to 'carto-positron'.")
        mapbox_style = 'carto-positron'
    elif mapbox_token:
        px.set_mapbox_access_token(mapbox_token)

    layout_template = go.Layout(
        font=dict(family=_get_setting('THEME_FONT_FAMILY', 'sans-serif'), size=12, color=_get_setting('COLOR_TEXT_DARK', '#333')),
        title=dict(font=dict(size=18, color=_get_setting('COLOR_TEXT_HEADINGS_MAIN', '#111')), x=0.05, xanchor='left'),
        paper_bgcolor=_get_setting('COLOR_BACKGROUND_CONTENT', '#FFF'),
        plot_bgcolor=_get_setting('COLOR_BACKGROUND_CONTENT', '#FFF'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        mapbox_style=mapbox_style
    )
    pio.templates["sentinel_theme"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_theme"
    logger.info("Custom Plotly theme 'sentinel_theme' applied as default.")

try:
    set_sentinel_plotly_theme()
except Exception as e:
    logger.error(f"Failed to set custom Plotly theme: {e}", exc_info=True)

def create_empty_figure(title: str, message: str = "No data available.") -> go.Figure:
    """Creates a blank figure with a text message, used as a placeholder."""
    fig = go.Figure()
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14}}]
    )
    return fig
