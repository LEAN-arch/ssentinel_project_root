# sentinel_project_root/visualization/plots.py
"""
Contains standardized plotting functions for the Sentinel application.
This ensures a consistent look and feel across all dashboards.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Dict, Any

try:
    from .themes import sentinel_theme
    from .ui_elements import get_theme_color
except ImportError:
    # Fallback for standalone execution or import errors
    def get_theme_color(key: str) -> str:
        return {"risk_high": "red", "risk_moderate": "orange", "risk_low": "green"}.get(key, "grey")
    sentinel_theme = None

logger = logging.getLogger(__name__)
px.defaults.template = sentinel_theme

def create_empty_figure(message: str) -> go.Figure:
    """Creates a blank Plotly figure with a text message in the center."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
    )
    return fig

# --- FIX APPLIED HERE ---
# The function signature is updated to explicitly handle axis titles.
def plot_bar_chart(
    df_input: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    orientation: str = 'v',
    x_axis_title: Optional[str] = None,
    y_axis_title: Optional[str] = None,
    **kwargs: Any
) -> go.Figure:
    """
    Creates a standardized bar chart, now robustly handling custom axis titles.

    Args:
        df_input: The DataFrame to plot.
        x_col: The column for the x-axis.
        y_col: The column for the y-axis.
        title: The chart title.
        color_col: Optional column to color the bars.
        orientation: 'v' for vertical, 'h' for horizontal.
        x_axis_title: Optional custom title for the x-axis.
        y_axis_title: Optional custom title for the y-axis.
        **kwargs: Additional keyword arguments to pass to px.bar (e.g., barmode, color_discrete_map).
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(f"No data available for '{title}'")

    required_cols = [x_col, y_col]
    if color_col:
        required_cols.append(color_col)

    if not all(col in df_input.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_input.columns]
        logger.error(f"Failed to create bar chart '{title}': missing columns {missing}")
        return create_empty_figure(f"Data for '{title}' is missing required columns.")
    
    try:
        # The idiomatic way to set axis labels in px.bar is via the `labels` dictionary.
        labels = {}
        if x_axis_title:
            labels[x_col] = x_axis_title
        if y_axis_title:
            labels[y_col] = y_axis_title
            
        fig = px.bar(
            df_input,
            x=x_col,
            y=y_col,
            color=color_col,
            orientation=orientation,
            title=title,
            labels=labels, # Pass the constructed labels dictionary
            text_auto=True,
            **kwargs
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            title_x=0.5, # Center the title
            legend_title_text=color_col.replace("_", " ").title() if color_col else ""
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to create bar chart '{title}': {e}", exc_info=True)
        return create_empty_figure(f"Error generating chart: '{title}'")


def plot_donut_chart(df_input: pd.DataFrame, labels_col: str, values_col: str, title: str) -> go.Figure:
    """Creates a standardized donut chart."""
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(f"No data for '{title}'")
    if not all(c in df_input.columns for c in [labels_col, values_col]):
        return create_empty_figure(f"Missing columns for '{title}'")

    fig = go.Figure(data=[go.Pie(
        labels=df_input[labels_col],
        values=df_input[values_col],
        hole=.4,
        textinfo='percent+label'
    )])
    fig.update_layout(title_text=title, title_x=0.5)
    return fig

def plot_annotated_line_chart(series_input: pd.Series, title: str, y_axis_title: str) -> go.Figure:
    """Creates a line chart with annotations for min/max values."""
    if not isinstance(series_input, pd.Series) or series_input.empty:
        return create_empty_figure(f"No data for '{title}'")

    fig = px.line(x=series_input.index, y=series_input.values, title=title, markers=True)
    fig.update_layout(title_x=0.5, yaxis_title=y_axis_title, xaxis_title="Date/Time")
    
    if len(series_input) > 1:
        max_val, min_val = series_input.max(), series_input.min()
        max_idx, min_idx = series_input.idxmax(), series_input.idxmin()
        fig.add_annotation(x=max_idx, y=max_val, text=f"Max: {max_val:.1f}", showarrow=True, arrowhead=1)
        fig.add_annotation(x=min_idx, y=min_val, text=f"Min: {min_val:.1f}", showarrow=True, arrowhead=1)
        
    return fig
