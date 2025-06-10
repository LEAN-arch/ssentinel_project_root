# sentinel_project_root/visualization/plots.py
# SME PLATINUM STANDARD - CENTRALIZED PLOTTING FACTORY (V6 - DEFINITIVE COLOR FIX)

import logging
from typing import Any, Dict, Optional

import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from config import settings

logger = logging.getLogger(__name__)

# SME FIX: Add a helper function to convert hex to a valid rgba string for Plotly
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a hex color string to an rgba string."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'


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


def create_empty_figure(title: str, message: str = "No data available.") -> go.Figure:
    # ... [This function is correct and remains unchanged] ...
    fig = go.Figure()
    fig.update_layout(title_text=f"<b>{html.escape(title)}</b>", xaxis={"visible": False}, yaxis={"visible": False}, annotations=[{"text": html.escape(message), "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14, "color": settings.COLOR_TEXT_MUTED}}])
    return fig

# ... [plot_bar_chart, plot_donut_chart, plot_line_chart, plot_choropleth_map, plot_heatmap are correct and remain unchanged] ...
def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_title: Optional[str]=None, y_title: Optional[str]=None, **px_kwargs: Any) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title)
    try:
        y_is_int = pd.api.types.is_integer_dtype(df[y_col]) or (df[y_col].dropna() % 1 == 0).all()
        fig = px.bar(df, x=x_col, y=y_col, title=f"<b>{html.escape(title)}</b>", labels={x_col: x_title or x_col.replace('_', ' ').title(), y_col: y_title or y_col.replace('_', ' ').title()}, text_auto=True, **px_kwargs)
        fig.update_traces(texttemplate='%{y:,.0f}' if y_is_int else '%{y:,.2f}', textposition='outside', marker_line_width=1.5, marker_line_color=settings.COLOR_BACKGROUND_CONTENT, hovertemplate=f"<b>%{{x}}</b><br>{y_col.replace('_', ' ').title()}: %{{y:,.2f}}<extra></extra>")
        if y_is_int: fig.update_yaxes(tickformat='d')
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")
def plot_donut_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title)
    try:
        fig = px.pie(df, names=label_col, values=value_col, title=f"<b>{html.escape(title)}</b>", hole=0.5)
        fig.update_traces(textinfo='percent+label', textposition='inside', insidetextorientation='radial', hoverinfo='label+percent+value', marker_line_width=2, marker_line_color=settings.COLOR_BACKGROUND_CONTENT)
        fig.update_layout(legend_title_text=label_col.replace("_", " ").title()); return fig
    except Exception as e:
        logger.error(f"Failed to create donut chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")
def plot_line_chart(series: pd.Series, title: str, y_title: str, add_annotations: bool=True) -> go.Figure:
    if not isinstance(series, pd.Series) or series.empty: return create_empty_figure(title)
    try:
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode='lines+markers', line=dict(color=settings.COLOR_PRIMARY, width=3), marker=dict(size=6), hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{html.escape(y_title)}: %{{y:,.2f}}<extra></extra>'))
        fig.update_layout(title_text=f"<b>{html.escape(title)}</b>", yaxis_title=y_title, xaxis_title="Date")
        if add_annotations and len(series.dropna()) > 1:
            max_val, min_val, max_idx, min_idx = series.max(), series.min(), series.idxmax(), series.idxmin()
            anno_font = dict(color="white", size=10)
            fig.add_annotation(x=max_idx, y=max_val, text=f"Max<br>{max_val:,.1f}", showarrow=True, bgcolor=settings.COLOR_RISK_HIGH, font=anno_font, borderpad=2, yshift=15)
            fig.add_annotation(x=min_idx, y=min_val, text=f"Min<br>{min_val:,.1f}", showarrow=True, bgcolor=settings.COLOR_RISK_LOW, font=anno_font, borderpad=2, yshift=-15)
        return fig
    except Exception as e:
        logger.error(f"Failed to create line chart '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")
def plot_choropleth_map(df: pd.DataFrame, geojson: Dict, value_col: str, title: str, zone_id_col: str='zone_id', **px_kwargs: Any) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title, "No geographic data.")
    try:
        fig = px.choropleth_mapbox(df, geojson=geojson, locations=zone_id_col, featureidkey="properties.zone_id", color=value_col, mapbox_style=settings.MAPBOX_STYLE, zoom=settings.MAP_DEFAULT_ZOOM, center={"lat": settings.MAP_DEFAULT_CENTER[0], "lon": settings.MAP_DEFAULT_CENTER[1]}, opacity=0.75, title=f"<b>{html.escape(title)}</b>", **px_kwargs)
        fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0}, mapbox_accesstoken=settings.MAPBOX_TOKEN); return fig
    except Exception as e:
        logger.error(f"Failed to create choropleth map '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating map.")
def plot_heatmap(df: pd.DataFrame, title: str, **px_kwargs: Any) -> go.Figure:
    if not isinstance(df, pd.DataFrame) or df.empty: return create_empty_figure(title)
    try:
        fig = px.imshow(df, text_auto=True, aspect="auto", title=f"<b>{html.escape(title)}</b>", **px_kwargs); return fig
    except Exception as e:
        logger.error(f"Failed to create heatmap '{title}': {e}", exc_info=True); return create_empty_figure(title, "Error generating chart.")


def plot_forecast_chart(forecast_df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    """Plots a Prophet forecast with uncertainty intervals."""
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        return create_empty_figure(title, "No forecast data available.")
        
    fig = go.Figure()
    
    # SME FIX: Use the helper function to generate a valid rgba string for the fill color.
    rgba_fill_color = _hex_to_rgba(settings.COLOR_ACCENT, 0.3)
    
    fig.add_trace(go.Scatter(
        x=forecast_df['forecast_date'].tolist() + forecast_df['forecast_date'].tolist()[::-1],
        y=forecast_df['consumption_upper_bound'].tolist() + forecast_df['consumption_lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor=rgba_fill_color,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Uncertainty Interval'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['forecast_date'],
        y=forecast_df['predicted_daily_consumption'],
        mode='lines',
        line=dict(color=settings.COLOR_PRIMARY, width=3),
        name='Forecast'
    ))
    fig.update_layout(
        title_text=f"<b>{html.escape(title)}</b>",
        yaxis_title=y_title,
        xaxis_title="Date",
        showlegend=True
    )
    return fig
