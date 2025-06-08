# sentinel_project_root/visualization/plots.py
"""
Contains a centralized, theme-aware factory for creating standardized plots
for the Sentinel application, ensuring a consistent look and feel.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Any, Dict
import html

try:
    from .themes import sentinel_theme
    from .ui_elements import get_theme_color
except ImportError:
    logging.warning("Could not import Sentinel theme. Using fallback plotting styles.")
    sentinel_theme = "plotly_white"
    def get_theme_color(key: str, category="general", fallback_color_hex="#636EFA"):
        if category == "categorical_sequence": return px.colors.qualitative.Plotly
        return {"risk_high": "#EF553B", "risk_moderate": "#FFA15A", "risk_low": "#00CC96", "primary": "#007BFF"}.get(key, fallback_color_hex)

logger = logging.getLogger(__name__)

class ChartFactory:
    """A factory class for creating standardized, theme-consistent Plotly charts."""
    def __init__(self, theme: Optional[str] = None):
        self.theme = theme or "plotly_white"
        px.defaults.template = self.theme
        self._base_layout = {'title_x': 0.5, 'margin': dict(l=60, r=40, t=80, b=60), 'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 'xaxis': {'zeroline': False}, 'yaxis': {'zeroline': False}}

    def _apply_layout(self, fig: go.Figure, **kwargs) -> go.Figure:
        layout_updates = self._base_layout.copy(); layout_updates.update(kwargs)
        fig.update_layout(**layout_updates)
        return fig

    def create_empty_figure(self, title: str, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, annotations=[{"text": html.escape(message), "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return self._apply_layout(fig, title_text=f"<b>{html.escape(title)}</b>", showlegend=False)

    def create_bar_chart(self, df: pd.DataFrame, x: str, y: str, title: str, y_values_are_counts: bool, **px_kwargs) -> go.Figure:
        color, orientation = px_kwargs.get('color'), px_kwargs.get('orientation', 'v')
        required_cols = [x, y]; [required_cols.append(c) for c in [color] if c]
        if not all(col in df.columns for col in required_cols): return self.create_empty_figure(title, "Missing data columns.")
        try:
            y_title, x_title = px_kwargs.pop('y_axis_title', y.replace('_', ' ').title()), px_kwargs.pop('x_axis_title', x.replace('_', ' ').title())
            labels = {x: x_title, y: y_title}
            fig = px.bar(df, x=x, y=y, title=f"<b>{html.escape(title)}</b>", labels=labels, text_auto=True, **px_kwargs)
            value_col = x if orientation == 'h' else y
            if df[value_col].notna().any():
                max_val = df.groupby(y if orientation == 'h' else x)[value_col].sum().max() if px_kwargs.get('barmode') == 'stack' else df[value_col].max()
                if orientation == 'h': fig.update_xaxes(range=[0, max_val * 1.15])
                else: fig.update_yaxes(range=[0, max_val * 1.15])
            text_template = ',.0f' if y_values_are_counts else ',.2f'
            hover_template = f'<b>%{{y if orientation == "h" else x}}</b><br>{html.escape(x_title if orientation == "h" else y_title)}: %{{x if orientation == "h" else y}}:{text_template}<extra></extra>'
            if color: hover_template = f'<b>%{{y if orientation == "h" else x}}</b><br>{html.escape(color.replace("_", " ").title())}: %{{fullData.name}}<br>{html.escape(value_col.replace("_", " ").title())}: %{{x if orientation == "h" else y}}:{text_template}<extra></extra>'
            fig.update_traces(textposition='outside', cliponaxis=False, hovertemplate=hover_template, texttemplate=f'%{{x if orientation == "h" else y}}:{text_template}')
            if y_values_are_counts: fig.update_xaxes(tickformat='d') if orientation == 'h' else fig.update_yaxes(tickformat='d')
            return self._apply_layout(fig, legend_title_text=color.replace("_", " ").title() if color else "")
        except Exception as e:
            logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True); return self.create_empty_figure(title, "Error generating chart")
            
    def create_donut_chart(self, df: pd.DataFrame, labels: str, values: str, title: str) -> go.Figure:
        if not all(c in df.columns for c in [labels, values]): return self.create_empty_figure(title, "Missing data columns.")
        try:
            fig = px.pie(df, names=labels, values=values, title=f"<b>{html.escape(title)}</b>", hole=0.4, color_discrete_sequence=get_theme_color(None, category='categorical_sequence'))
            fig.update_traces(textinfo='percent', hoverinfo='label+percent+value')
            return self._apply_layout(fig, showlegend=True, legend_title_text=labels.replace("_", " ").title())
        except Exception as e:
            logger.error(f"Failed to create donut chart '{title}': {e}", exc_info=True); return self.create_empty_figure(title, "Error generating chart")

    def create_annotated_line_chart(self, series: pd.Series, title: str, y_title: str) -> go.Figure:
        try:
            fig = px.line(x=series.index, y=series.values, title=f"<b>{html.escape(title)}</b>", markers=True)
            fig.update_traces(line=dict(color=get_theme_color('primary')), hovertemplate=f'<b>%{{x}}</b><br>{html.escape(y_title)}: %{{y:,.2f}}<extra></extra>')
            valid_series = series.dropna()
            if len(valid_series) > 1:
                max_val, min_val = valid_series.max(), valid_series.min()
                max_idx, min_idx = valid_series.idxmax(), valid_series.idxmin()
                annotation_font = dict(color="white", size=10)
                if max_idx == min_idx:
                    fig.add_annotation(x=max_idx, y=max_val, text=f"Value: {max_val:,.1f}", showarrow=True, arrowhead=2, yshift=10)
                else:
                    fig.add_annotation(x=max_idx, y=max_val, text=f"Max: {max_val:,.1f}", showarrow=True, arrowhead=1, bgcolor="rgba(239, 85, 59, 0.8)", bordercolor=get_theme_color("risk_high"), font=annotation_font, borderpad=4)
                    fig.add_annotation(x=min_idx, y=min_val, text=f"Min: {min_val:,.1f}", showarrow=True, arrowhead=1, bgcolor="rgba(0, 204, 150, 0.8)", bordercolor=get_theme_color("risk_low"), font=annotation_font, borderpad=4, yshift=-10)
            return self._apply_layout(fig, yaxis_title=y_title, xaxis_title="Date/Time", showlegend=False)
        except Exception as e:
            logger.error(f"Failed to create annotated line chart '{title}': {e}", exc_info=True); return self.create_empty_figure(title, "Error generating chart")

    def create_choropleth_map(self, df: pd.DataFrame, geojson: Dict, locations: str, color: str, title: str, **px_kwargs) -> go.Figure:
        if not all(c in df.columns for c in [locations, color]): return self.create_empty_figure(title, "Missing data for map.")
        try:
            fig = px.choropleth_mapbox(df, geojson=geojson, locations=locations, color=color,
                                       mapbox_style=_get_setting('MAPBOX_STYLE_WEB', "carto-positron"),
                                       zoom=_get_setting('MAP_DEFAULT_ZOOM_LEVEL', 8),
                                       center={"lat": _get_setting('MAP_DEFAULT_CENTER_LAT', 0), "lon": _get_setting('MAP_DEFAULT_CENTER_LON', 0)},
                                       opacity=0.7, title=f"<b>{html.escape(title)}</b>", **px_kwargs)
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            return fig
        except Exception as e:
            logger.error(f"Failed to create choropleth map '{title}': {e}", exc_info=True); return self.create_empty_figure(title, "Error generating map")

_factory = ChartFactory(theme=sentinel_theme)
def create_empty_figure(title: str, message: str = "No data available") -> go.Figure: return _factory.create_empty_figure(title, message)
def plot_bar_chart(df_input: pd.DataFrame, x_col: str, y_col: str, title: str, y_values_are_counts: bool = False, **kwargs: Any) -> go.Figure:
    if not isinstance(df_input, pd.DataFrame) or df_input.empty: return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_bar_chart(df_input, x_col, y_col, title, y_values_are_counts=y_values_are_counts, **kwargs)
def plot_donut_chart(df_input: pd.DataFrame, labels_col: str, values_col: str, title: str) -> go.Figure:
    if not isinstance(df_input, pd.DataFrame) or df_input.empty: return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_donut_chart(df_input, labels_col, values_col, title)
def plot_annotated_line_chart(series_input: pd.Series, title: str, y_axis_title: str) -> go.Figure:
    if not isinstance(series_input, pd.Series) or series_input.empty: return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_annotated_line_chart(series_input, title, y_axis_title)
def plot_choropleth_map(df_input: pd.DataFrame, geojson: Dict, locations: str, color: str, title: str, **kwargs: Any) -> go.Figure:
    if not isinstance(df_input, pd.DataFrame) or df_input.empty: return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_choropleth_map(df_input, geojson, locations, color, title, **kwargs)
