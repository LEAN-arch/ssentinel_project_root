# ssentinel_project_root/visualization/plots.py
"""
Contains standardized plotting functions for the Sentinel application.
This ensures a consistent look and feel across all dashboards by using a central theme.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Dict, Any

try:
    # Centralized theme object is imported from its own module.
    from .themes import sentinel_theme
except ImportError:
    # This fallback allows the module to be used in isolation for testing
    # or if the themes.py file is unavailable.
    sentinel_theme = None
    logging.warning("Could not import sentinel_theme. Plots will use default Plotly theme.")

logger = logging.getLogger(__name__)

# --- THEME APPLICATION ---
# This single line applies the consistent theme to all Plotly Express charts
# created after this module has been imported.
px.defaults.template = sentinel_theme


def create_empty_figure(message: str) -> go.Figure:
    """Creates a blank Plotly figure with a text message in the center."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]
    )
    if sentinel_theme:
        fig.update_layout(template=sentinel_theme)
    return fig


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
    Creates a standardized bar chart with robust error handling and axis labeling.
    """
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(f"No data available for '{title}'")

    required_cols = [x_col, y_col]
    if color_col:
        required_cols.append(color_col)

    if not all(col in df_input.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_input.columns]
        logger.error(f"Failed to create bar chart '{title}': missing columns {missing}")
        return create_empty_figure(f"Data for '{title}' is missing required columns: {missing}")
    
    try:
        # The idiomatic way to set axis labels in Plotly Express is via the `labels` dictionary.
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
            labels=labels,
            text_auto=True,
            **kwargs
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            title_x=0.5, # Center the title
            legend_title_text=str(color_col).replace("_", " ").title() if color_col else ""
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

    try:
        fig = go.Figure(data=[go.Pie(
            labels=df_input[labels_col],
            values=df_input[values_col],
            hole=.4,
            textinfo='percent+label'
        )])
        fig.update_layout(title_text=title, title_x=0.5, showlegend=True)
        return fig
    except Exception as e:
        logger.error(f"Failed to create donut chart '{title}': {e}", exc_info=True)
        return create_empty_figure(f"Error generating chart: '{title}'")


def plot_annotated_line_chart(series_input: pd.Series, title: str, y_axis_title: str) -> go.Figure:
    """Creates a standardized line chart with annotations for min/max values."""
    if not isinstance(series_input, pd.Series) or series_input.empty:
        return create_empty_figure(f"No data for '{title}'")

    try:
        fig = px.line(x=series_input.index, y=series_input.values, title=title, markers=True)
        fig.update_layout(title_x=0.5, yaxis_title=y_axis_title, xaxis_title="Date/Time")
        
        # Add annotations only if there is more than one data point to compare
        if len(series_input.dropna()) > 1:
            max_val, min_val = series_input.max(), series_input.min()
            max_idx, min_idx = series_input.idxmax(), series_input.idxmin()
            
            # Add max annotation
            fig.add_annotation(x=max_idx, y=max_val, text=f"Max: {max_val:,.1f}", showarrow=True, arrowhead=1)
            # Add min annotation, ensuring it doesn't overlap with max if they are the same point
            if max_idx != min_idx or max_val != min_val:
                fig.add_annotation(x=min_idx, y=min_val, text=f"Min: {min_val:,.1f}", showarrow=True, arrowhead=1, yshift=-10)
            
        return fig
    except Exception as e:
        logger.error(f"Failed to create line chart '{title}': {e}", exc_info=True)
        return create_empty_figure(f"Error generating chart: '{title}'")
