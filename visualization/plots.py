# sentinel_project_root/visualization/plots.py
# Standardized, robust plotting functions for Sentinel Health Co-Pilot dashboards.

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import logging
import html
from typing import Optional, List, Dict, Any

try:
    from config import settings
    from .ui_elements import get_theme_color
except ImportError:
    # Fallback for standalone execution or testing
    class FallbackSettings:
        THEME_FONT_FAMILY = 'sans-serif'
        COLOR_TEXT_DARK = "#333333"
        COLOR_BACKGROUND_CONTENT = "#FFFFFF"
        COLOR_TEXT_HEADINGS_MAIN = "#111111"
    settings = FallbackSettings()
    def get_theme_color(key: str) -> str: return {"risk_high": "red"}.get(key, "grey")

logger = logging.getLogger(__name__)

# --- Theme Setup ---
def _get_setting(attr: str, default: Any) -> Any:
    """Safely gets a setting attribute."""
    return getattr(settings, attr, default)

def set_sentinel_plotly_theme():
    """Configures and applies a custom Plotly theme for the application."""
    layout_template = go.Layout(
        font=dict(family=_get_setting('THEME_FONT_FAMILY', 'sans-serif'), size=12, color=_get_setting('COLOR_TEXT_DARK', '#333')),
        title=dict(font=dict(size=18, color=_get_setting('COLOR_TEXT_HEADINGS_MAIN', '#111')), x=0.05, xanchor='left'),
        paper_bgcolor=_get_setting('COLOR_BACKGROUND_CONTENT', '#FFF'),
        plot_bgcolor=_get_setting('COLOR_BACKGROUND_CONTENT', '#FFF'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=80, b=40)
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
    """
    Handles common validation. Returns a figure on success, or an empty placeholder on failure.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_empty_figure(title, message="No data available.")
    
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Plot '{title}': Missing required columns: {missing_cols}")
            return create_empty_figure(title, message=f"Data is missing required columns: {', '.join(missing_cols)}")
        
    return None # Indicates validation passed

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

def plot_bar_chart(
    df_input: pd.DataFrame, x_col: str, y_col: str, title: str,
    color_col: Optional[str] = None, orientation: str = 'v',
    x_axis_title: Optional[str] = None, y_axis_title: Optional[str] = None, **kwargs
) -> go.Figure:
    """Creates a standardized bar chart with robust axis title handling."""
    required = [x_col, y_col] + ([color_col] if color_col else [])
    base_fig = _create_base_figure(df_input, title, required)
    if base_fig: return base_fig
    
    try:
        labels = {
            x_col: x_axis_title or x_col.replace('_', ' ').title(),
            y_col: y_axis_title or y_col.replace('_', ' ').title(),
        }
        if color_col: labels[color_col] = color_col.replace('_', ' ').title()

        fig = px.bar(df_input, x=x_col, y=y_col, color=color_col, orientation=orientation, labels=labels, text_auto=True, **kwargs)
        fig.update_layout(title_text=f'<b>{html.escape(title)}</b>')
        fig.update_traces(textposition='outside')
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
        return create_empty_figure(title, message=f"Chart Error: {e}")

def plot_donut_chart(df_input: pd.DataFrame, labels_col: str, values_col: str, title: str) -> go.Figure:
    """Creates a standardized donut chart."""
    base_fig = _create_base_figure(df_input, title, [labels_col, values_col])
    if base_fig: return base_fig

    fig = go.Figure(data=[go.Pie(labels=df_input[labels_col], values=df_input[values_col], hole=.5, textinfo='label+percent', insidetextorientation='radial')])
    fig.update_layout(title_text=f'<b>{html.escape(title)}</b>')
    return fig

def plot_annotated_line_chart(series_input: pd.Series, title: str, y_axis_title: str) -> go.Figure:
    """Creates a line chart with annotations for min/max values."""
    if not isinstance(series_input, pd.Series) or series_input.empty:
        return create_empty_figure(title)

    fig = px.line(series_input, title=f'<b>{html.escape(title)}</b>', markers=True)
    fig.update_layout(yaxis_title=y_axis_title, xaxis_title="Date/Time", showlegend=False)
    
    if len(series_input) > 1:
        max_val, min_val = series_input.max(), series_input.min()
        max_idx, min_idx = series_input.idxmax(), series_input.idxmin()
        fig.add_annotation(x=max_idx, y=max_val, text=f"Max: {max_val:.1f}", showarrow=True, arrowhead=2, ax=0, ay=-40)
        fig.add_annotation(x=min_idx, y=min_val, text=f"Min: {min_val:.1f}", showarrow=True, arrowhead=2, ax=0, ay=40)
    return fig

# FIX: Restored the plot_heatmap function to resolve the ImportError.
def plot_heatmap(matrix_df: Optional[pd.DataFrame], title: str, **kwargs) -> go.Figure:
    """Generates a heatmap from a matrix-like DataFrame."""
    # The base figure check for heatmaps doesn't need to check columns, as the whole DF is used.
    base_fig = _create_base_figure(matrix_df, title, required_cols=None)
    if base_fig: return base_fig

    try:
        fig = px.imshow(
            matrix_df,
            text_auto=kwargs.get('text_auto', True),
            aspect=kwargs.get('aspect', "auto"),
            **kwargs
        )
        fig.update_layout(title_text=f'<b>{html.escape(title)}</b>')
        return fig
    except Exception as e:
        logger.error(f"Failed to create heatmap '{title}': {e}", exc_info=True)
        return create_empty_figure(title, message=f"Chart Error: {e}")
