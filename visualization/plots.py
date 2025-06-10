# sentinel_project_root/visualization/plots.py
# SME PLATINUM STANDARD - CENTRALIZED & DEFINITIVE PLOTTING FACTORY (V9 - FINAL)

import logging
from typing import Any, Dict, Optional

import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from config import settings

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a hex color string to an rgba string for Plotly compatibility."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6: return 'rgba(0,0,0,0.1)'
    try:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    except ValueError:
        return 'rgba(0,0,0,0.1)'

# --- Theme Setup ---
def set_plotly_theme():
    """Sets the custom Sentinel theme as the default for all Plotly charts."""
    base_layout = {
        'font': {'family': "sans-serif", 'size': 12, 'color': settings.COLOR_TEXT_PRIMARY},
        'title': {'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': settings.COLOR_TEXT_HEADINGS}},
        'paper_bgcolor': settings.COLOR_BACKGROUND_CONTENT,
        'plot_bgcolor': settings.COLOR_BACKGROUND_CONTENT,
        'margin': dict(l=60, r=40, t=60, b=60),
        'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font={'size': 10}),
        'xaxis': {'showgrid': False, 'zeroline': False},
        'yaxis': {'gridcolor': '#e9ecef', 'zeroline': False},
    }
    sentinel_template = go.layout.Template(layout=base_layout)
    sentinel_template.layout.colorway = settings.PLOTLY_COLORWAY
    pio.templates['sentinel'] = sentinel_template
    pio.templates.default = 'sentinel'
    logger.debug("Custom 'sentinel' Plotly theme applied.")

# --- Factory Functions for Charts ---
def create_empty_figure(title: str, message: str = "No data available.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title_text=f"<b>{html.escape(title)}</b>",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": html.escape(message), "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14, "color": settings.COLOR_TEXT_MUTED}}]
    )
    return fig

def plot_bar_chart(
    df: pd.DataFrame, x_col: str, y_col: str, title: str,
    x_title: Optional[str] = None, y_title: Optional[str] = None, **px_kwargs: Any
) -> go.Figure:
    """Creates a themed bar chart with correct axis labeling."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_empty_figure(title)
        
    try:
        # SME FIX: The `labels` dictionary is the correct way to set axis titles in Plotly Express.
        # The invalid `x_title` and `y_title` arguments are removed from `**px_kwargs`.
        axis_labels = {
            x_col: x_title or x_col.replace('_', ' ').title(),
            y_col: y_title or y_col.replace('_', ' ').title()
        }
        
        y_is_int = pd.api.types.is_integer_dtype(df[y_col]) or (df[y_col].dropna() % 1 == 0).all()
        
        fig = px.bar(
            df, x=x_col, y=y_col, title=f"<b>{html.escape(title)}</b>",
            text_auto=True,
            labels=axis_labels, # Use the corrected labels argument
            **px_kwargs
        )
        
        fig.update_traces(
            texttemplate='%{y:,.0f}' if y_is_int else '%{y:,.2f}',
            textposition='outside'
        )
        
        if y_is_int:
            fig.update_yaxes(tickformat='d')
            
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
        return create_empty_figure(title, "Error generating chart.")


def plot_donut_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title)
    try:
        fig = px.pie(df, names=label_col, values=value_col, title=f"<b>{html.escape(title)}</b>", hole=0.5)
        fig.update_traces(textinfo='percent+label', textposition='inside', insidetextorientation='radial', marker_line_width=2, marker_line_color=settings.COLOR_BACKGROUND_CONTENT)
        fig.update_layout(legend_title_text=label_col.replace("_", " ").title()); return fig
    except Exception as e: logger.error(f"Failed to create donut chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")

def plot_line_chart(series: pd.Series, title: str, y_title: str) -> go.Figure:
    if not isinstance(series, pd.Series) or series.empty: return create_empty_figure(title)
    try:
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode='lines+markers', line=dict(color=settings.COLOR_PRIMARY, width=3)))
        fig.update_layout(title_text=f"<b>{html.escape(title)}</b>", yaxis_title=y_title, xaxis_title="Date"); return fig
    except Exception as e: logger.error(f"Failed to create line chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")

def plot_forecast_chart(forecast_df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty: return create_empty_figure(title, "No forecast data available.")
    fig = go.Figure()
    rgba_fill = _hex_to_rgba(settings.COLOR_ACCENT, 0.3)
    fig.add_trace(go.Scatter(x=forecast_df['forecast_date'].tolist() + forecast_df['forecast_date'].tolist()[::-1], y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1], fill='toself', fillcolor=rgba_fill, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='Uncertainty'))
    fig.add_trace(go.Scatter(x=forecast_df['forecast_date'], y=forecast_df['predicted_value'], mode='lines', line=dict(color=settings.COLOR_PRIMARY, width=3), name='Forecast'))
    fig.update_layout(title_text=f"<b>{html.escape(title)}</b>", yaxis_title=y_title, xaxis_title="Date", showlegend=True)
    return fig
