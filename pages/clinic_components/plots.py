# sentinel_project_root/pages/clinic_components/plots.py
# A centralized, theme-aware factory for creating standardized plots for Sentinel Dashboards.

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from typing import Optional, Dict, Any, List
import html

# --- Sentinel System Imports ---
try:
    from config import settings
    from visualization.ui_elements import get_theme_color
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"CRITICAL IMPORT ERROR in plots.py: {e}. Plotting will use fallback settings.", exc_info=True)
    class FallbackPlotSettings:
        COLOR_TEXT_DARK = "#333333"; COLOR_BACKGROUND_CONTENT = "#FFFFFF"; COLOR_GRID = "#EAEAEA";
        COLOR_TEXT_HEADINGS_MAIN = "#111111"; COLOR_TEXT_MUTED = '#757575';
        WEB_PLOT_DEFAULT_HEIGHT = 400; WEB_PLOT_COMPACT_HEIGHT = 350;
    settings = FallbackPlotSettings()
    def get_theme_color(name_or_idx, category="general", fallback_color_hex=None):
        if category == "categorical_sequence": return px.colors.qualitative.Plotly
        return fallback_color_hex or "#007BFF"

logger = logging.getLogger(__name__)


class ChartFactory:
    """
    A factory class for creating standardized, theme-consistent Plotly charts.
    Encapsulates all styling logic to ensure a uniform look and feel.
    """
    def __init__(self):
        """Initializes the factory with a base layout configuration."""
        self._base_layout = {
            'paper_bgcolor': getattr(settings, 'COLOR_BACKGROUND_CONTENT', '#FFFFFF'),
            'plot_bgcolor': getattr(settings, 'COLOR_BACKGROUND_CONTENT', '#FFFFFF'),
            'font': {'family': 'sans-serif', 'size': 12, 'color': getattr(settings, 'COLOR_TEXT_DARK', '#333333')},
            'title': {'font': {'size': 16, 'color': getattr(settings, 'COLOR_TEXT_HEADINGS_MAIN', '#111111')}, 'x': 0.05, 'xanchor': 'left'},
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1, 'traceorder': 'normal'},
            'margin': dict(l=60, r=30, t=80, b=50),
            'hovermode': 'x unified',
            'xaxis': {'showgrid': False, 'showline': True, 'linecolor': getattr(settings, 'COLOR_GRID', '#EAEAEA')},
            'yaxis': {'gridcolor': getattr(settings, 'COLOR_GRID', '#EAEAEA')},
        }

    def _apply_layout(self, fig: go.Figure, title: str, height: int, overrides: Optional[Dict] = None) -> go.Figure:
        """Applies the base layout and any specific overrides to a figure."""
        layout = self._base_layout.copy()
        layout['height'] = height
        layout['title']['text'] = f"<b>{html.escape(title)}</b>"
        if overrides: layout.update(overrides)
        fig.update_layout(**layout)
        return fig

    def create_empty_figure(self, title: str, message: str = "No data available for this period.") -> go.Figure:
        """Creates a themed, blank Plotly figure with a message."""
        fig = go.Figure()
        fig.add_annotation(text=html.escape(message), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=get_theme_color('muted')))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return self._apply_layout(fig, title, getattr(settings, 'WEB_PLOT_COMPACT_HEIGHT', 350), {'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    def plot_annotated_line_chart(self, data_series: pd.Series, chart_title: str, y_axis_label: str, target_ref_line_val: Optional[float], y_values_are_counts: bool, date_format_hover: str) -> go.Figure:
        """Creates a themed, annotated line chart from a pandas Series."""
        df = data_series.reset_index(); df.columns = ['x', 'y']
        fig = go.Figure(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', line=dict(color=get_theme_color('primary')), marker=dict(size=6)))
        
        y_format = 'd' if y_values_are_counts else '.1f'
        fig.update_traces(hovertemplate=f'<b>%{{x|{date_format_hover}}}</b><br>{html.escape(y_axis_label)}: %{{y:{y_format}}}<extra></extra>')

        if target_ref_line_val is not None:
            fig.add_hline(y=target_ref_line_val, line_dash="dash", line_color=get_theme_color('danger'), annotation_text="Target", annotation_position="top right")

        fig = self._apply_layout(fig, chart_title, getattr(settings, 'WEB_PLOT_COMPACT_HEIGHT', 350))
        fig.update_layout(yaxis_title=html.escape(y_axis_label), xaxis_title=None)
        if y_values_are_counts: fig.update_yaxes(tickformat='d')
        return fig

    def plot_bar_chart(self, df_input: pd.DataFrame, x_col_name: str, y_col_name: str, chart_title: str, color_col_name: Optional[str], bar_mode_style: str, y_values_are_counts_flag: bool, x_axis_label_text: Optional[str], y_axis_label_text: Optional[str]) -> go.Figure:
        """Creates a themed, flexible bar chart from a DataFrame."""
        fig = px.bar(df_input, x=x_col_name, y=y_col_name, color=color_col_name, barmode=bar_mode_style, color_discrete_sequence=get_theme_color(None, category="categorical_sequence"))
        
        text_template = ',.0f' if y_values_are_counts_flag else ',.1f'
        fig.update_traces(texttemplate=f'%{{y:{text_template}}}', textposition='outside', cliponaxis=False)
        
        fig = self._apply_layout(fig, chart_title, getattr(settings, 'WEB_PLOT_DEFAULT_HEIGHT', 400))
        fig.update_layout(
            uniformtext_minsize=8, uniformtext_mode='hide',
            xaxis_title=x_axis_label_text or x_col_name.replace('_', ' ').title(),
            yaxis_title=y_axis_label_text or y_col_name.replace('_', ' ').title(),
            legend_title_text=color_col_name.replace('_', ' ').title() if color_col_name else None
        )
        if y_values_are_counts_flag: fig.update_yaxes(tickformat='d')
        return fig

# --- Public Factory Functions ---
# These functions provide a simple, clean, and developer-friendly interface.

def create_empty_figure(title: str, message: str = "No data to display.") -> go.Figure:
    """Public factory function to create a themed empty figure."""
    return ChartFactory().create_empty_figure(title, message)

def plot_annotated_line_chart(data_series: Optional[pd.Series], chart_title: str, y_axis_label: str, target_ref_line_val: Optional[float] = None, y_values_are_counts: bool = False, date_format_hover: str = '%b %d, %Y') -> go.Figure:
    """Creates a themed line chart with annotations and a consistent style."""
    factory = ChartFactory()
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return factory.create_empty_figure(chart_title)
    return factory.plot_annotated_line_chart(data_series, chart_title, y_axis_label, target_ref_line_val, y_values_are_counts, date_format_hover)

def plot_bar_chart(df_input: Optional[pd.DataFrame], x_col_name: str, y_col_name: str, chart_title: str, color_col_name: Optional[str] = None, bar_mode_style: str = 'group', y_values_are_counts_flag: bool = False, x_axis_label_text: Optional[str] = None, y_axis_label_text: Optional[str] = None) -> go.Figure:
    """Creates a themed bar chart with a consistent style."""
    factory = ChartFactory()
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return factory.create_empty_figure(chart_title)
    return factory.plot_bar_chart(df_input, x_col_name, y_col_name, chart_title, color_col_name, bar_mode_style, y_values_are_counts_flag, x_axis_label_text, y_axis_label_text)
