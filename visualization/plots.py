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
    from .ui_elements import get_theme_color
except ImportError:
    # Fallback for standalone execution or testing
    class FallbackSettings:
        MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
        MAPBOX_STYLE_WEB = "carto-positron"
        WEB_MAP_DEFAULT_HEIGHT = 600
        THEME_FONT_FAMILY = 'sans-serif'
        COLOR_TEXT_DARK = "#333333"
        COLOR_BACKGROUND_CONTENT = "#FFFFFF"
        COLOR_TEXT_HEADINGS_MAIN = "#111111"
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

# --- Theme and Mapbox Setup ---
def _get_setting(attr: str, default: Any) -> Any:
    """Safely gets a setting attribute."""
    return getattr(settings, attr, default)

def set_sentinel_plotly_theme():
    """Configures and applies a custom Plotly theme for the application."""
    if os.getenv("MAPBOX_ACCESS_TOKEN"):
        px.set_mapbox_access_token(os.getenv("MAPBOX_ACCESS_TOKEN"))
        
    layout_template = go.Layout(
        font=dict(family=_get_setting('THEME_FONT_FAMILY', 'sans-serif'), size=12, color=_get_setting('COLOR_TEXT_DARK', '#333')),
        title=dict(font=dict(size=18, color=_get_setting('COLOR_TEXT_HEADINGS_MAIN', '#111')), x=0.05, xanchor='left'),
        paper_bgcolor=_get_setting('COLOR_BACKGROUND_CONTENT', '#FFF'),
        plot_bgcolor=_get_setting('COLOR_BACKGROUND_CONTENT', '#FFF'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        mapbox_style=_get_setting('MAPBOX_STYLE_WEB', 'carto-positron')
    )
    pio.templates["sentinel_theme"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_theme"
    logger.info("Custom Plotly theme 'sentinel_theme' applied as default.")

try:
    set_sentinel_plotly_theme()
except Exception as e:
    logger.error(f"Failed to set custom Plotly theme: {e}", exc_info=True)


# --- Base Plotting Utilities ---
def _create_base_figure(df: Optional[pd.DataFrame], title: str, required_cols: Optional[List[str]] = None) -> Optional[go.Figure]:
    """Handles common validation. Returns a figure on success, or an empty placeholder on failure."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_empty_figure(title, message="No data available.")
    if required_cols and any(col not in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return create_empty_figure(title, message=f"Data is missing required columns: {', '.join(missing)}")
    return None

def create_empty_figure(title: str, message: str = "No data available.") -> go.Figure:
    """Creates a blank figure with a text message, used as a placeholder."""
    fig = go.Figure()
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14}}]
    )
    return fig

# --- Standardized Chart Functions ---

def plot_bar_chart(df_input: pd.DataFrame, x_col: str, y_col: str, title: str, **kwargs) -> go.Figure:
    """Creates a standardized bar chart."""
    required = [x_col, y_col] + ([kwargs.get('color')] if kwargs.get('color') else [])
    base_fig = _create_base_figure(df_input, title, required)
    if base_fig: return base_fig
    
    try:
        fig = px.bar(df_input, x=x_col, y=y_col, title=f'<b>{html.escape(title)}</b>', text_auto=True, **kwargs)
        fig.update_traces(textposition='outside')
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
        return create_empty_figure(title, message=f"Chart Error: {e}")

def plot_donut_chart(df_input: pd.DataFrame, labels_col: str, values_col: str, title: str) -> go.Figure:
    base_fig = _create_base_figure(df_input, title, [labels_col, values_col])
    if base_fig: return base_fig
    fig = go.Figure(data=[go.Pie(labels=df_input[labels_col], values=df_input[values_col], hole=.5, textinfo='label+percent')])
    fig.update_layout(title_text=f'<b>{html.escape(title)}</b>')
    return fig

def plot_annotated_line_chart(series_input: pd.Series, title: str, y_axis_title: str) -> go.Figure:
    if not isinstance(series_input, pd.Series) or series_input.empty: return create_empty_figure(title)
    fig = px.line(series_input, title=f'<b>{html.escape(title)}</b>', markers=True)
    fig.update_layout(yaxis_title=y_axis_title, xaxis_title="Date/Time", showlegend=False)
    if len(series_input) > 1:
        max_val, min_val = series_input.max(), series_input.min()
        max_idx, min_idx = series_input.idxmax(), series_input.idxmin()
        fig.add_annotation(x=max_idx, y=max_val, text=f"Max: {max_val:.1f}", showarrow=True, arrowhead=2)
        fig.add_annotation(x=min_idx, y=min_val, text=f"Min: {min_val:.1f}", showarrow=True, arrowhead=2)
    return fig

def plot_heatmap(matrix_df: Optional[pd.DataFrame], title: str, **kwargs) -> go.Figure:
    base_fig = _create_base_figure(matrix_df, title)
    if base_fig: return base_fig
    try:
        fig = px.imshow(matrix_df, text_auto=kwargs.get('text_auto', True), aspect=kwargs.get('aspect', "auto"), **kwargs)
        fig.update_layout(title_text=f'<b>{html.escape(title)}</b>')
        return fig
    except Exception as e:
        logger.error(f"Failed to create heatmap '{title}': {e}", exc_info=True)
        return create_empty_figure(title, message=f"Chart Error: {e}")

# FIX: Restored the plot_choropleth_map function to resolve the ImportError.
def plot_choropleth_map(
    map_data_df: pd.DataFrame, geojson: Dict, locations_col: str,
    color_col: str, title: str, **kwargs
) -> go.Figure:
    """Creates a standardized choropleth map."""
    required = [locations_col, color_col]
    base_fig = _create_base_figure(map_data_df, title, required)
    if base_fig: return base_fig

    try:
        fig = px.choropleth_mapbox(
            map_data_df,
            geojson=geojson,
            locations=locations_col,
            color=color_col,
            featureidkey=kwargs.get('featureidkey', f"properties.{locations_col}"),
            hover_name=kwargs.get('hover_name_col', locations_col),
            mapbox_style=_get_setting('MAPBOX_STYLE_WEB', 'carto-positron'),
            zoom=kwargs.get('zoom', 9),
            center=kwargs.get('center', {"lat": 4.8, "lon": 7.0}), # Example center
            **kwargs
        )
        fig.update_layout(
            title_text=f'<b>{html.escape(title)}</b>',
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating map '{title}': {e}", exc_info=True)
        return create_empty_figure(title, message="Error generating map.")
