# sentinel_project_root/visualization/plots.py
"""
Contains a centralized, theme-aware factory for creating standardized plots
for the Sentinel application, ensuring a consistent look and feel.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Any
import html

try:
    from .themes import sentinel_theme
    from .ui_elements import get_theme_color
except ImportError:
    # Fallback for standalone execution or import errors
    logging.warning("Could not import Sentinel theme. Using fallback plotting styles.")
    sentinel_theme = "plotly_white"
    def get_theme_color(key: str, category="general", fallback_color_hex="#636EFA"):
        if category == "categorical_sequence":
            return px.colors.qualitative.Plotly
        return {"risk_high": "#EF553B", "risk_moderate": "#FFA15A", "risk_low": "#00CC96", "primary": "#007BFF"}.get(key, fallback_color_hex)

logger = logging.getLogger(__name__)

class ChartFactory:
    """
    A factory class for creating standardized, theme-consistent Plotly charts.
    Encapsulates all styling logic to ensure a uniform look and feel.
    """
    def __init__(self, theme: Optional[str] = None):
        """Initializes the factory with a base layout configuration."""
        self.theme = theme or "plotly_white"
        px.defaults.template = self.theme
        
        self._base_layout = {
            'title_x': 0.5,
            'margin': dict(l=60, r=40, t=80, b=60),
            'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            'xaxis': {'zeroline': False},
            'yaxis': {'zeroline': False},
        }

    def _apply_layout(self, fig: go.Figure, **kwargs) -> go.Figure:
        """Applies the base layout and any specific overrides to a figure."""
        layout_updates = self._base_layout.copy()
        layout_updates.update(kwargs)
        fig.update_layout(**layout_updates)
        return fig

    def create_empty_figure(self, title: str, message: str) -> go.Figure:
        """Creates a themed, blank Plotly figure with a message."""
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{"text": html.escape(message), "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
        )
        return self._apply_layout(fig, title_text=f"<b>{html.escape(title)}</b>", showlegend=False)

    def create_bar_chart(self, df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str], barmode: str, orientation: str, x_axis_title: Optional[str], y_axis_title: Optional[str]) -> go.Figure:
        """Creates a standardized bar chart from a DataFrame."""
        required_cols = [x, y]
        if color: required_cols.append(color)

        if not all(col in df.columns for col in required_cols):
            missing = sorted([col for col in required_cols if col not in df.columns])
            logger.error(f"Bar chart '{title}' missing columns: {missing}")
            return self.create_empty_figure(title, "Missing required data columns.")
        
        try:
            y_title = y_axis_title or y.replace('_', ' ').title()
            labels = {x: x_axis_title or x.replace('_', ' ').title(), y: y_title}
            
            fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, orientation=orientation,
                         title=f"<b>{html.escape(title)}</b>", labels=labels, text_auto=True,
                         color_discrete_sequence=get_theme_color(None, category='categorical_sequence'))
            
            # Add padding to y-axis to prevent text labels from being clipped
            if df[y].notna().any():
                if barmode == 'stack': max_y = df.groupby(x)[y].sum().max()
                else: max_y = df[y].max()
                fig.update_yaxes(range=[0, max_y * 1.15])
            
            hover_template = f'<b>%{{x}}</b><br>{html.escape(y_title)}: %{{y:,.2f}}<extra></extra>'
            if color:
                hover_template = f'<b>%{{x}}</b><br>{html.escape(color.replace("_", " ").title())}: %{{fullData.name}}<br>{html.escape(y_title)}: %{{y:,.2f}}<extra></extra>'

            fig.update_traces(textposition='outside', cliponaxis=False, hovertemplate=hover_template)
            return self._apply_layout(fig, legend_title_text=color.replace("_", " ").title() if color else "")
        except Exception as e:
            logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
            return self.create_empty_figure(title, "Error generating chart")

    def create_donut_chart(self, df: pd.DataFrame, labels: str, values: str, title: str) -> go.Figure:
        """Creates a standardized donut chart."""
        if not all(c in df.columns for c in [labels, values]):
            return self.create_empty_figure(title, "Missing required data columns.")
        try:
            fig = px.pie(df, names=labels, values=values, title=f"<b>{html.escape(title)}</b>", hole=0.4,
                         color_discrete_sequence=get_theme_color(None, category='categorical_sequence'))
            fig.update_traces(textinfo='percent', hoverinfo='label+percent+value') # Labels in legend, details on hover
            return self._apply_layout(fig, showlegend=True, legend_title_text=labels.replace("_", " ").title())
        except Exception as e:
            logger.error(f"Failed to create donut chart '{title}': {e}", exc_info=True)
            return self.create_empty_figure(title, "Error generating chart")

    def create_annotated_line_chart(self, series: pd.Series, title: str, y_title: str) -> go.Figure:
        """Creates a line chart with annotations for min/max values."""
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
            logger.error(f"Failed to create annotated line chart '{title}': {e}", exc_info=True)
            return self.create_empty_figure(title, "Error generating chart")

# --- Singleton Instance and Public API ---
_factory = ChartFactory(theme=sentinel_theme)

def create_empty_figure(title: str, message: str = "No data available") -> go.Figure:
    """Public factory function to create a themed empty figure with a title."""
    return _factory.create_empty_figure(title, message)

def plot_bar_chart(df_input: pd.DataFrame, x_col: str, y_col: str, title: str, color: Optional[str] = None, barmode: str = 'relative', orientation: str = 'v', x_axis_title: Optional[str] = None, y_axis_title: Optional[str] = None) -> go.Figure:
    """
    Creates a standardized, theme-aware bar chart.

    Args:
        df_input (pd.DataFrame): The DataFrame to plot.
        x_col (str): The column for the x-axis.
        y_col (str): The column for the y-axis.
        title (str): The chart title.
        color (Optional[str]): The column to use for coloring the bars. Defaults to None.
        barmode (str): One of 'group', 'stack', or 'relative'. Defaults to 'relative'.
        orientation (str): 'v' for vertical, 'h' for horizontal. Defaults to 'v'.
        x_axis_title (Optional[str]): A custom title for the x-axis. Defaults to a formatted column name.
        y_axis_title (Optional[str]): A custom title for the y-axis. Defaults to a formatted column name.
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_bar_chart(df_input, x_col, y_col, title, color=color, barmode=barmode, orientation=orientation, x_axis_title=x_axis_title, y_axis_title=y_axis_title)

def plot_donut_chart(df_input: pd.DataFrame, labels_col: str, values_col: str, title: str) -> go.Figure:
    """
    Creates a standardized, theme-aware donut chart.

    Args:
        df_input (pd.DataFrame): The DataFrame to plot.
        labels_col (str): The column containing the chart labels.
        values_col (str): The column containing the chart values.
        title (str): The chart title.
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_donut_chart(df_input, labels_col, values_col, title)

def plot_annotated_line_chart(series_input: pd.Series, title: str, y_axis_title: str) -> go.Figure:
    """
    Creates a standardized, theme-aware line chart with annotations for min/max values.

    Args:
        series_input (pd.Series): A pandas Series with a DatetimeIndex and numeric values.
        title (str): The chart title.
        y_axis_title (str): The label for the Y-axis.
    """
    if not isinstance(series_input, pd.Series) or series_input.empty:
        return create_empty_figure(title, f"No data for '{title}'")
    return _factory.create_annotated_line_chart(series_input, title, y_axis_title)
