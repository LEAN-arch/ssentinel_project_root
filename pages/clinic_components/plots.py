# sentinel_project_root/pages/clinic_components/plots.py
# Plotting functions specifically for the Clinic Dashboard.

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
    from data_processing.helpers import convert_to_numeric 
    from visualization.ui_elements import get_theme_color 
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"CRITICAL IMPORT ERROR in clinic_components/plots.py: {e}. Using fallback settings.")
    class FallbackPlotSettings:
        COLOR_TEXT_DARK = "#333333"; COLOR_BACKGROUND_CONTENT = "#FFFFFF"; COLOR_BACKGROUND_PAGE = "#F0F2F6";
        COLOR_ACTION_PRIMARY = "#007BFF"; COLOR_TEXT_HEADINGS_MAIN = "#111111"; COLOR_TEXT_MUTED = '#757575';
        WEB_PLOT_DEFAULT_HEIGHT = 400; WEB_PLOT_COMPACT_HEIGHT = 350; MAPBOX_STYLE_WEB = "carto-positron";
        MAP_DEFAULT_CENTER_LAT = 0.0; MAP_DEFAULT_CENTER_LON = 0.0; MAP_DEFAULT_ZOOM_LEVEL = 1;
    settings = FallbackPlotSettings()
    def get_theme_color(name_or_idx, category="general", fallback_color_hex=None): return fallback_color_hex or "#757575"
    def convert_to_numeric(data, default=np.nan, target_type=None): return pd.to_numeric(data, errors='coerce').fillna(default)

logger = logging.getLogger(__name__)

def _get_setting(attr: str, default: Any) -> Any:
    return getattr(settings, attr, default)

def create_empty_figure(title: str, height: Optional[int] = None, message: str = "No data to display.") -> go.Figure:
    """Creates a blank Plotly figure with a message."""
    final_height = height or _get_setting('WEB_PLOT_DEFAULT_HEIGHT', 400)
    fig = go.Figure()
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>', height=final_height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=html.escape(message), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=14, font_color=_get_setting('COLOR_TEXT_MUTED', '#757575'))]
    )
    return fig

def plot_annotated_line_chart(
    data_series: Optional[pd.Series],
    chart_title: str,
    y_axis_label: str,
    y_values_are_counts: bool = False,
    target_ref_line_val: Optional[float] = None
) -> go.Figure:
    """Creates an annotated line chart."""
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title)
    
    df = data_series.reset_index()
    df.columns = ['x', 'y']
    df['x'] = pd.to_datetime(df['x'])
    
    fig = px.line(df, x='x', y='y', title=None, markers=True)
    
    y_format = 'd' if y_values_are_counts else '.1f'
    fig.update_traces(hovertemplate=f'<b>%{{x|%b %d, %Y}}</b><br>{html.escape(y_axis_label)}: %{{y:{y_format}}}<extra></extra>')
    
    if target_ref_line_val is not None:
        fig.add_hline(y=target_ref_line_val, line_dash="dash", line_color="red", annotation_text="Target")

    fig.update_layout(
        title_text=f'<b>{html.escape(chart_title)}</b>',
        xaxis_title=None,
        yaxis_title=html.escape(y_axis_label),
        height=_get_setting('WEB_PLOT_COMPACT_HEIGHT', 350)
    )
    
    if y_values_are_counts:
        fig.update_yaxes(tickformat='d')
        
    return fig

def plot_bar_chart(
    df_input: Optional[pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    barmode: str = 'group',
    y_values_are_counts: bool = False
) -> go.Figure:
    """Creates a flexible bar chart from a DataFrame."""
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title)
        
    fig = px.bar(df_input, x=x_col, y=y_col, color=color_col, barmode=barmode, text_auto=True)
    
    text_template = '%{text:,d}' if y_values_are_counts else '%{text:,.1f}'
    hover_template = f'<b>%{{x}}</b><br>Count: %{{y:,d}}<extra></extra>' if y_values_are_counts else f'<b>%{{x}}</b><br>Value: %{{y:,.1f}}<extra></extra>'
    
    fig.update_traces(texttemplate=text_template, hovertemplate=hover_template)
    fig.update_layout(
        title_text=f'<b>{html.escape(title)}</b>',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        legend_title=color_col.replace('_', ' ').title() if color_col else None,
        height=_get_setting('WEB_PLOT_DEFAULT_HEIGHT', 450)
    )
    
    if y_values_are_counts:
        fig.update_yaxes(tickformat='d')

    return fig
