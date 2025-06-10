# sentinel_project_root/visualization/plots.py
# SME PLATINUM STANDARD - CENTRALIZED & DEFINITIVE PLOTTING FACTORY (V12 - FINAL)

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
    """Creates a themed bar chart with correct axis labeling and non-negative count axis."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_empty_figure(title)
        
    try:
        axis_labels = {
            x_col: x_title or x_col.replace('_', ' ').title(),
            y_col: y_title or y_col.replace('_', ' ').title()
        }
        
        orientation = px_kwargs.get('orientation', 'v')
        value_col = x_col if orientation == 'h' else y_col
        
        is_int = pd.api.types.is_integer_dtype(df[value_col]) or (df[value_col].dropna() % 1 == 0).all()
        
        fig = px.bar(
            df, x=x_col, y=y_col, title=f"<b>{html.escape(title)}</b>",
            text_auto=True,
            labels=axis_labels,
            **px_kwargs
        )
        
        text_template = '%{x:,.0f}' if orientation == 'h' else '%{y:,.0f}'
        if not is_int:
            text_template = '%{x:,.2f}' if orientation == 'h' else '%{y:,.2f}'
            
        fig.update_traces(texttemplate=text_template, textposition='outside')
        
        if is_int:
            if orientation == 'h':
                fig.update_xaxes(tickformat='d', range=[0, None])
            else:
                fig.update_yaxes(tickformat='d', range=[0, None])
            
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
        return create_empty_figure(title, "Error generating chart.")

def plot_donut_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title)
    try:
        fig = px.pie(df, names=label_col, values=value_col, title=f"<b>{html.escape(title)}</b>", hole=0.5)
        fig.update_traces(textinfo='percent+label', textposition='inside', insidetextorientation='radial', marker_line_width=2, marker_line_color=settings.COLOR_BACKGROUND_CONTENT, hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>')
        fig.update_layout(legend_title_text=label_col.replace("_", " ").title()); return fig
    except Exception as e: logger.error(f"Failed to create donut chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")

def plot_line_chart(series: pd.Series, title: str, y_title: str) -> go.Figure:
    """Creates a themed line chart with rich annotations for min/max values."""
    if not isinstance(series, pd.Series) or series.empty: return create_empty_figure(title)
    try:
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode='lines+markers', line=dict(color=settings.COLOR_PRIMARY, width=3), hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{html.escape(y_title)}: %{{y:,.2f}}<extra></extra>'))
        fig.update_layout(title_text=f"<b>{html.escape(title)}</b>", yaxis_title=y_title, xaxis_title="Date")
        
        if len(series.dropna()) > 1:
            max_val, min_val = series.max(), series.min()
            max_idx, min_idx = series.idxmax(), series.idxmin()
            anno_font = dict(color="white", size=10, family="sans-serif")
            fig.add_annotation(x=max_idx, y=max_val, text=f"Max<br>{max_val:,.1f}", showarrow=True, bgcolor=_hex_to_rgba(settings.COLOR_RISK_HIGH, 0.8), font=anno_font, borderpad=2, yshift=15, arrowhead=2)
            fig.add_annotation(x=min_idx, y=min_val, text=f"Min<br>{min_val:,.1f}", showarrow=True, bgcolor=_hex_to_rgba(settings.COLOR_RISK_LOW, 0.8), font=anno_font, borderpad=2, yshift=-15, arrowhead=2)
        
        return fig
    except Exception as e: logger.error(f"Failed to create line chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")

def plot_choropleth_map(df: pd.DataFrame, geojson: Dict, value_col: str, title: str, zone_id_col: str='zone_id', **px_kwargs: Any) -> go.Figure:
    """Creates a themed choropleth map for geospatial analysis."""
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title, "No geographic data.")
    try:
        fig = px.choropleth_mapbox(df, geojson=geojson, locations=zone_id_col, featureidkey="properties.zone_id", color=value_col, mapbox_style=settings.MAPBOX_STYLE, zoom=settings.MAP_DEFAULT_ZOOM, center={"lat": settings.MAP_DEFAULT_CENTER[0], "lon": settings.MAP_DEFAULT_CENTER[1]}, opacity=0.75, title=f"<b>{html.escape(title)}</b>", **px_kwargs)
        fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0}, mapbox_accesstoken=settings.MAPBOX_TOKEN); return fig
    except Exception as e: logger.error(f"Failed to create choropleth map '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating map.")

def plot_heatmap(df: pd.DataFrame, title: str, **px_kwargs: Any) -> go.Figure:
    """Creates a themed heatmap."""
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title)
    try:
        fig = px.imshow(df, text_auto=True, aspect="auto", title=f"<b>{html.escape(title)}</b>", **px_kwargs); return fig
    except Exception as e: logger.error(f"Failed to create heatmap '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")

def plot_forecast_chart(forecast_df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    """Plots a generic forecast with styled uncertainty intervals."""
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty: return create_empty_figure(title, "No forecast data available.")
    fig = go.Figure()
    rgba_fill = _hex_to_rgba(settings.COLOR_ACCENT, 0.3)
    fig.add_trace(go.Scatter(x=forecast_df['forecast_date'].tolist() + forecast_df['forecast_date'].tolist()[::-1], y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1], fill='toself', fillcolor=rgba_fill, line=dict(color='rgba(255,255,255,0)'), hoverinfo="none", name='Uncertainty', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df['forecast_date'], y=forecast_df['predicted_value'], mode='lines', line=dict(color=settings.COLOR_PRIMARY, width=3), name='Forecast', hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Forecast: %{{y:,.1f}}<extra></extra>"))
    fig.update_layout(title_text=f"<b>{html.escape(title)}</b>", yaxis_title=y_title, xaxis_title="Date", showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig
