# sentinel_project_root/visualization/plots.py
# Plotting functions for Sentinel Health Co-Pilot Web Dashboards using Plotly.

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio # For theme interaction
import os # For MAPBOX_TOKEN_SET_FLAG
from typing import Optional, List, Dict, Any, Union

from config import settings # Use new settings module
from .ui_elements import get_theme_color # Import from local ui_elements

logger = logging.getLogger(__name__)

# --- Mapbox Token Handling (Mirrors logic from original ui_visualization_helpers) ---
MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = False
try:
    _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV = os.getenv("MAPBOX_ACCESS_TOKEN")
    if _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV and \
       _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV.strip() and \
       len(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV) > 20: # Basic check for token-like string
        
        # Check if it's a "public" token like pk.xxxx or a "secret" sk.xxxx
        # For client-side Plotly.js maps, only public tokens (pk.) are safe.
        # If it's a secret token, we should avoid setting it directly in client-side code.
        # Plotly Express handles this by itself if the token is set as an env var.
        # For direct Mapbox GL JS style URLs, it might be different.
        # For now, assume if it's set, it's usable by Plotly Express.
        px.set_mapbox_access_token(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV) # Set for Plotly Express
        MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = True
        logger.info("Plotly: MAPBOX_ACCESS_TOKEN environment variable found and set for px.")
    else:
        logger.warning("Plotly: MAPBOX_ACCESS_TOKEN environment variable not found or seems invalid. Maps requiring a token might not render correctly or will use open styles.")
except Exception as e_mapbox_token:
    logger.error(f"Plotly: Error encountered while trying to set Mapbox token for px: {e_mapbox_token}")


# --- Plotly Theme Setup ---
def set_sentinel_plotly_theme(): # Renamed from set_sentinel_plotly_theme_web
    """Sets a custom Plotly theme ('sentinel_web_theme') as the default."""
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
    
    # Define a colorway using the new get_theme_color utility for consistency
    sentinel_colorway_list = [
        settings.COLOR_ACTION_PRIMARY,
        settings.COLOR_RISK_LOW,        # Often used for "good" series
        settings.COLOR_RISK_MODERATE,   # Often used for "warning" series
        settings.COLOR_ACCENT_BRIGHT,   # A distinct accent
        settings.COLOR_ACTION_SECONDARY,
        get_theme_color(5, color_category="general", fallback_color_hex="#00ACC1"), # Teal-like
        get_theme_color(6, color_category="general", fallback_color_hex="#5E35B1"), # Deep Purple
        get_theme_color(7, color_category="general", fallback_color_hex="#FF7043")  # Coral/Orange
    ]

    layout_config = go.Layout(
        font=dict(family=theme_font_family, size=11, color=settings.COLOR_TEXT_DARK),
        paper_bgcolor=settings.COLOR_BACKGROUND_CONTENT, # Background of the chart figure
        plot_bgcolor=settings.COLOR_BACKGROUND_PAGE,    # Background of the plotting area
        colorway=sentinel_colorway_list,
        xaxis=dict(
            gridcolor=settings.COLOR_BORDER_LIGHT,
            linecolor=settings.COLOR_BORDER_MEDIUM,
            zerolinecolor=settings.COLOR_BORDER_MEDIUM,
            zerolinewidth=1,
            title_font_size=12,
            tickfont_size=10,
            automargin=True
        ),
        yaxis=dict(
            gridcolor=settings.COLOR_BORDER_LIGHT,
            linecolor=settings.COLOR_BORDER_MEDIUM,
            zerolinecolor=settings.COLOR_BORDER_MEDIUM,
            zerolinewidth=1,
            title_font_size=12,
            tickfont_size=10,
            automargin=True
        ),
        title=dict(
            font=dict(family=theme_font_family, size=16, color=settings.COLOR_TEXT_HEADINGS_MAIN), # Slightly larger title
            x=0.05, xanchor='left', # Align title left
            y=0.95, yanchor='top',
            pad=dict(t=20, b=10) # Padding around title
        ),
        legend=dict(
            bgcolor=settings.COLOR_BACKGROUND_CONTENT_TRANSPARENT if hasattr(settings, 'COLOR_BACKGROUND_CONTENT_TRANSPARENT') else 'rgba(255,255,255,0.85)', # Slightly transparent legend
            bordercolor=settings.COLOR_BORDER_LIGHT,
            borderwidth=0.5,
            orientation='h', # Horizontal legend at top
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font_size=10
        ),
        margin=dict(l=60, r=20, t=80, b=60) # Adjusted margins for title/legend
    )

    # Determine effective Mapbox style based on token availability
    # Styles like 'streets-v11', 'satellite-v9' often require a token.
    # 'open-street-map', 'carto-positron', 'carto-darkmatter' are usually token-free.
    if MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        effective_mapbox_style_for_theme = settings.MAPBOX_STYLE_WEB # Use configured style if token is set
    else:
        # Fallback to a known open style if token is not set, regardless of MAPBOX_STYLE_WEB config
        open_styles = ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]
        if settings.MAPBOX_STYLE_WEB.lower() in open_styles:
            effective_mapbox_style_for_theme = settings.MAPBOX_STYLE_WEB
        else:
            effective_mapbox_style_for_theme = "carto-positron" # Default open fallback
            logger.info(f"Plotly Theme: Mapbox token not set, defaulting map style to '{effective_mapbox_style_for_theme}' instead of '{settings.MAPBOX_STYLE_WEB}'.")

    layout_config.mapbox = dict(
        style=effective_mapbox_style_for_theme,
        center=dict(lat=settings.MAP_DEFAULT_CENTER_LAT, lon=settings.MAP_DEFAULT_CENTER_LON),
        zoom=settings.MAP_DEFAULT_ZOOM_LEVEL # Use renamed config
    )

    pio.templates["sentinel_web_theme_custom"] = go.layout.Template(layout=layout_config)
    pio.templates.default = "plotly+sentinel_web_theme_custom" # Combine with plotly base for other defaults
    logger.info("Custom Plotly theme 'sentinel_web_theme_custom' set as default.")

# Call theme setup when module is imported
set_sentinel_plotly_theme()


# --- Plotting Functions ---

def create_empty_figure( # Renamed from _create_empty_plot_figure
    chart_title: str,
    height: Optional[int] = None,
    message_text: str = "No data available to display for the current selection." # More generic message
) -> go.Figure:
    """Creates a styled empty Plotly figure with a message."""
    final_height = height or settings.WEB_PLOT_DEFAULT_HEIGHT
    
    fig = go.Figure()
    fig.update_layout(
        title_text=f"{chart_title}", # Keep title simple, message in annotation
        height=final_height,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[dict(
            text=message_text,
            xref="paper", yref="paper", # Relative to plotting area
            x=0.5, y=0.5, # Centered
            showarrow=False,
            font=dict(size=12, color=settings.COLOR_TEXT_MUTED)
        )],
        paper_bgcolor=settings.COLOR_BACKGROUND_CONTENT, # Match theme
        plot_bgcolor=settings.COLOR_BACKGROUND_PAGE     # Match theme
    )
    return fig

def plot_annotated_line_chart( # Renamed
    data_series: Optional[pd.Series],
    chart_title: str,
    y_axis_label: str = "Value",
    line_color_hex: Optional[str] = None, # Renamed for clarity
    target_ref_line_val: Optional[float] = None, # Renamed
    target_ref_label_text: Optional[str] = None, # Renamed
    show_confidence_interval: bool = False, # Renamed
    lower_ci_series: Optional[pd.Series] = None, # Renamed
    upper_ci_series: Optional[pd.Series] = None, # Renamed
    chart_height: Optional[int] = None,
    show_anomalies_flag: bool = False, # Renamed
    anomaly_iqr_factor: float = 1.5, # Renamed
    date_format_hover: str = "%Y-%m-%d", # Renamed
    y_values_are_counts: bool = False # Renamed
) -> go.Figure:
    """
    Generates an annotated line chart with optional confidence intervals, target lines, and anomaly detection.
    """
    final_height = chart_height or settings.WEB_PLOT_COMPACT_HEIGHT

    if not isinstance(data_series, pd.Series) or data_series.empty:
        logger.debug(f"Line chart '{chart_title}': Input data_series is empty or invalid.")
        return create_empty_figure(chart_title, final_height, "No data for line chart.")

    # Ensure series index is datetime for time series plotting
    try:
        series_for_plot = data_series.copy()
        if not pd.api.types.is_datetime64_any_dtype(series_for_plot.index):
            series_for_plot.index = pd.to_datetime(series_for_plot.index, errors='coerce')
        series_for_plot.dropna(inplace=True) # Drop entries where index conversion failed
        series_for_plot = convert_to_numeric(series_for_plot, default_value=np.nan).dropna() # Ensure values are numeric and drop NaNs
    except Exception as e_conv:
        logger.warning(f"Line chart '{chart_title}': Error processing input series index/values: {e_conv}")
        return create_empty_figure(chart_title, final_height, "Invalid data format for line chart.")

    if series_for_plot.empty:
        logger.debug(f"Line chart '{chart_title}': Series became empty after cleaning.")
        return create_empty_figure(chart_title, final_height, "No valid data points after cleaning.")

    fig = go.Figure()
    actual_line_color = line_color_hex or get_theme_color(0) # Default to first theme color

    y_hover_format = 'd' if y_values_are_counts else ',.1f' # Integer for counts, float for others
    
    fig.add_trace(go.Scatter(
        x=series_for_plot.index,
        y=series_for_plot.values,
        mode="lines+markers",
        name=y_axis_label if y_axis_label else (series_for_plot.name or "Value"), # Use series name if available
        line=dict(color=actual_line_color, width=2),
        marker=dict(size=5, symbol='circle'),
        customdata=series_for_plot.values, # For hover
        hovertemplate=f'<b>Date</b>: %{{x|{date_format_hover}}}<br><b>{y_axis_label}</b>: %{{y:{y_hover_format}}}<extra></extra>'
    ))

    # Confidence Interval
    if show_confidence_interval and isinstance(lower_ci_series, pd.Series) and isinstance(upper_ci_series, pd.Series) and \
       not lower_ci_series.empty and not upper_ci_series.empty:
        # Align indices and clean CI series
        common_idx = series_for_plot.index.intersection(lower_ci_series.index).intersection(upper_ci_series.index)
        if not common_idx.empty:
            l_ci_aligned = convert_to_numeric(lower_ci_series.reindex(common_idx), np.nan).dropna()
            u_ci_aligned = convert_to_numeric(upper_ci_series.reindex(common_idx), np.nan).dropna()
            
            # Further align based on non-NaN values in both CIs
            final_ci_idx = l_ci_aligned.index.intersection(u_ci_aligned.index)
            if not final_ci_idx.empty and (u_ci_aligned.loc[final_ci_idx] >= l_ci_aligned.loc[final_ci_idx]).all():
                fill_color_rgba = px.colors.hex_to_rgb(actual_line_color)
                fill_color_str = f"rgba({fill_color_rgba[0]},{fill_color_rgba[1]},{fill_color_rgba[2]},0.15)" # Transparent version
                fig.add_trace(go.Scatter(
                    x=list(final_ci_idx) + list(final_ci_idx[::-1]), # x, then x reversed
                    y=list(u_ci_aligned.loc[final_ci_idx].values) + list(l_ci_aligned.loc[final_ci_idx].values[::-1]), # upper, then lower reversed
                    fill="toself",
                    fillcolor=fill_color_str,
                    line=dict(width=0), # No border line for fill
                    name="Conf. Interval",
                    hoverinfo="skip" # Don't show hover for the fill itself
                ))
            else:
                logger.debug(f"Line chart '{chart_title}': CI data invalid (lower > upper or empty after alignment).")
        else:
            logger.debug(f"Line chart '{chart_title}': CI series indices do not align with main data series.")


    # Target Reference Line
    if target_ref_line_val is not None:
        ref_label = target_ref_label_text or f"Target: {target_ref_line_val:,.2f}"
        fig.add_hline(
            y=target_ref_line_val,
            line_dash="dash",
            line_color=settings.COLOR_RISK_MODERATE, # Consistent color for targets
            line_width=1.2,
            annotation_text=ref_label,
            annotation_position="bottom right",
            annotation_font_size=9,
            annotation_font_color=settings.COLOR_TEXT_MUTED
        )

    # Anomaly Detection (Simple IQR based)
    if show_anomalies_flag and len(series_for_plot.dropna()) > 7 and series_for_plot.nunique() > 2: # Min data points for meaningful IQR
        q1 = series_for_plot.quantile(0.25)
        q3 = series_for_plot.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 1e-6: # Ensure IQR is positive and non-zero
            upper_bound = q3 + anomaly_iqr_factor * iqr
            lower_bound = q1 - anomaly_iqr_factor * iqr
            anomalies_series = series_for_plot[(series_for_plot < lower_bound) | (series_for_plot > upper_bound)]
            if not anomalies_series.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies_series.index,
                    y=anomalies_series.values,
                    mode='markers',
                    marker=dict(color=settings.COLOR_RISK_HIGH, size=8, symbol='circle-open-dot', line=dict(width=1.5)),
                    name='Anomaly',
                    customdata=anomalies_series.values,
                    hovertemplate=f'<b>Anomaly Date</b>: %{{x|{date_format_hover}}}<br><b>Value</b>: %{{y:{y_hover_format}}}<extra></extra>'
                ))
        else:
            logger.debug(f"Line chart '{chart_title}': IQR is zero or NaN, skipping anomaly detection.")


    x_axis_title_text = series_for_plot.index.name if series_for_plot.index.name else "Date/Time"
    y_axis_config = dict(title_text=y_axis_label, rangemode='tozero' if y_values_are_counts and series_for_plot.min() >= 0 else 'normal')
    if y_values_are_counts:
        y_axis_config['tickformat'] = 'd' # Integer format for y-axis if counts

    fig.update_layout(
        title_text=chart_title,
        xaxis_title=x_axis_title_text,
        yaxis=y_axis_config,
        height=final_height,
        hovermode="x unified", # Shows tooltips for all traces at a given x-value
        legend=dict(traceorder='normal') # Ensure main line appears above CI fill in legend
    )
    return fig


def plot_bar_chart( # Renamed
    df_input: Optional[pd.DataFrame],
    x_col_name: str, # Renamed
    y_col_name: str, # Renamed
    chart_title: str,
    color_col_name: Optional[str] = None, # Renamed
    bar_mode_style: str = 'group', # Renamed: 'group', 'stack', 'relative'
    orientation_bar: str = 'v', # Renamed: 'v' or 'h'
    y_axis_label_text: Optional[str] = None, # Renamed
    x_axis_label_text: Optional[str] = None, # Renamed
    chart_height: Optional[int] = None,
    show_text_on_bars: Union[bool, str] = True, # Renamed, True/'auto' or format string
    sort_by_col: Optional[str] = None, # Renamed
    sort_ascending_flag: bool = True, # Renamed
    text_format_str: Optional[str] = None, # Renamed
    y_values_are_counts_flag: bool = False, # Renamed
    custom_color_map: Optional[Dict[str, str]] = None # Renamed, maps color_col values to hex colors
) -> go.Figure:
    """
    Generates a flexible bar chart using Plotly Express.
    """
    final_height = chart_height or settings.WEB_PLOT_DEFAULT_HEIGHT

    if not isinstance(df_input, pd.DataFrame) or df_input.empty or \
       x_col_name not in df_input.columns or y_col_name not in df_input.columns:
        logger.debug(f"Bar chart '{chart_title}': Input DataFrame invalid or missing key columns x='{x_col_name}', y='{y_col_name}'.")
        return create_empty_figure(chart_title, final_height)

    df_for_plot = df_input.copy()
    
    # Prepare columns
    df_for_plot[x_col_name] = df_for_plot[x_col_name].astype(str) # Ensure x-axis is categorical
    df_for_plot[y_col_name] = convert_to_numeric(df_for_plot[y_col_name], default_value=0.0) # Y-axis must be numeric
    
    if y_values_are_counts_flag:
        df_for_plot[y_col_name] = df_for_plot[y_col_name].round().astype('Int64') # Use nullable int for counts

    df_for_plot.dropna(subset=[x_col_name, y_col_name], inplace=True) # Drop if essential values are NaN after conversion
    
    if df_for_plot.empty:
        logger.debug(f"Bar chart '{chart_title}': DataFrame empty after cleaning/type conversion.")
        return create_empty_figure(chart_title, final_height, f"No valid data for x='{x_col_name}', y='{y_col_name}'.")

    # Sorting
    if sort_by_col and sort_by_col in df_for_plot.columns:
        try:
            df_for_plot.sort_values(by=sort_by_col, ascending=sort_ascending_flag, inplace=True, na_position='last')
        except Exception as e_sort:
            logger.warning(f"Bar chart '{chart_title}': Sorting by '{sort_by_col}' failed: {e_sort}. Proceeding without sort.")
    
    # Determine text display format and hover format
    effective_text_val_format = text_format_str or ('.0f' if y_values_are_counts_flag else '.1f')
    y_hover_val_format = 'd' if y_values_are_counts_flag else effective_text_val_format

    # Axis labels
    final_y_axis_label = y_axis_label_text or y_col_name.replace('_', ' ').title()
    final_x_axis_label = x_axis_label_text or x_col_name.replace('_', ' ').title()
    legend_title_text = color_col_name.replace('_', ' ').title() if color_col_name and color_col_name in df_for_plot.columns else None

    # Resolve color map for consistency
    resolved_color_map = custom_color_map
    if not resolved_color_map and color_col_name and color_col_name in df_for_plot.columns and settings.LEGACY_DISEASE_COLORS_WEB:
        # Try to use disease colors if applicable
        is_disease_theme_applicable = any(str(val) in settings.LEGACY_DISEASE_COLORS_WEB for val in df_for_plot[color_col_name].dropna().unique())
        if is_disease_theme_applicable:
            resolved_color_map = {
                str(val): get_theme_color(str(val), color_category="disease", fallback_color_hex=get_theme_color(abs(hash(str(val))) % 8, color_category="general"))
                for val in df_for_plot[color_col_name].dropna().unique()
            }
    
    try:
        fig = px.bar(
            df_for_plot,
            x=x_col_name if orientation_bar == 'v' else y_col_name, # X/Y swap for horizontal
            y=y_col_name if orientation_bar == 'v' else x_col_name,
            title=chart_title,
            color=color_col_name if color_col_name in df_for_plot.columns else None,
            barmode=bar_mode_style,
            orientation=orientation_bar,
            height=final_height,
            labels={ # px uses these for axis titles and legend title
                y_col_name: final_y_axis_label,
                x_col_name: final_x_axis_label,
                color_col_name if color_col_name else "_": legend_title_text or '' # Dummy key if no color_col
            },
            text_auto=show_text_on_bars, # True, False, or a format string
            color_discrete_map=resolved_color_map
        )
    except Exception as e_px_bar:
        logger.error(f"Bar chart '{chart_title}': Plotly Express error: {e_px_bar}", exc_info=True)
        return create_empty_figure(chart_title, final_height, "Error during chart generation.")

    # Update traces for hover and text formatting
    text_template_on_bar = f'%{{y:{effective_text_val_format}}}' if orientation_bar == 'v' else f'%{{x:{effective_text_val_format}}}'
    if isinstance(show_text_on_bars, str) and show_text_on_bars != 'auto': # If specific format string provided for text_auto
        text_template_on_bar = show_text_on_bars
    
    hover_template_str = ""
    custom_data_for_hover = []

    if orientation_bar == 'v':
        hover_template_str = f'<b>{final_x_axis_label}</b>: %{{x}}<br><b>{final_y_axis_label}</b>: %{{y:{y_hover_val_format}}}'
    else: # Horizontal
        hover_template_str = f'<b>{final_y_axis_label}</b>: %{{y}}<br><b>{final_x_axis_label}</b>: %{{x:{y_hover_val_format}}}'

    if color_col_name and color_col_name in df_for_plot.columns and legend_title_text:
        hover_template_str += f'<br><b>{legend_title_text}</b>: %{{customdata[0]}}'
        custom_data_for_hover.append(color_col_name)
    
    hover_template_str += '<extra></extra>' # Remove trace info

    fig.update_traces(
        marker_line_width=0.8, marker_line_color='rgba(0,0,0,0.3)',
        textfont_size=9, textangle=0, # textangle only if text_auto is on
        textposition='outside' if orientation_bar == 'h' else ('auto' if bar_mode_style != 'stack' else 'inside'),
        cliponaxis=False, # Allows text to go slightly outside plot area if needed
        texttemplate=text_template_on_bar if show_text_on_bars else None,
        hovertemplate=hover_template_str,
        customdata=df_for_plot[custom_data_for_hover] if custom_data_for_hover else None
    )

    # Update layout for axes and legend
    y_axis_layout_config = {'title_text': final_y_axis_label}
    x_axis_layout_config = {'title_text': final_x_axis_label}
    
    value_axis_config = y_axis_layout_config if orientation_bar == 'v' else x_axis_layout_config
    category_axis_config = x_axis_layout_config if orientation_bar == 'v' else y_axis_layout_config

    if y_values_are_counts_flag:
        value_axis_config['tickformat'] = 'd' # Integer ticks for counts
        value_axis_config['rangemode'] = 'tozero' # Ensure axis starts at 0 for counts

    # Category ordering based on sort
    category_col_for_order = x_col_name if orientation_bar == 'v' else y_col_name
    if sort_by_col == category_col_for_order : # If sorted by the category axis itself
        category_axis_config['categoryorder'] = 'array'
        category_axis_config['categoryarray'] = df_for_plot[category_col_for_order].tolist()
    elif orientation_bar == 'h' and (not sort_by_col or sort_by_col == y_col_name): # Default sort for horizontal bars
         category_axis_config['categoryorder'] = 'total ascending' if sort_ascending_flag else 'total descending'
    elif orientation_bar == 'v' and sort_by_col and sort_by_col != x_col_name: # Sorted by value or color_col
        # Let Plotly handle default category order or rely on input df_for_plot order
        pass


    fig.update_layout(
        yaxis=y_axis_layout_config,
        xaxis=x_axis_layout_config,
        uniformtext_minsize=7, uniformtext_mode='hide', # For text on bars
        legend_title_text=legend_title_text
    )
    return fig


def plot_donut_chart( # Renamed
    df_input: Optional[pd.DataFrame],
    labels_col_name: str, # Renamed
    values_col_name: str, # Renamed
    chart_title: str,
    chart_height: Optional[int] = None,
    custom_color_map: Optional[Dict[str, str]] = None, # Renamed
    pull_slice_amount: float = 0.03, # Renamed
    center_annotation: Optional[str] = None, # Renamed
    values_are_counts: bool = True # Renamed
) -> go.Figure:
    """Generates a donut chart using Plotly."""
    final_height = chart_height or (settings.WEB_PLOT_COMPACT_HEIGHT + 50) # Donuts often need bit more height for legend

    if not isinstance(df_input, pd.DataFrame) or df_input.empty or \
       labels_col_name not in df_input.columns or values_col_name not in df_input.columns:
        logger.debug(f"Donut chart '{chart_title}': Input DataFrame invalid or missing key columns.")
        return create_empty_figure(chart_title, final_height)

    df_for_plot = df_input.copy()
    df_for_plot[values_col_name] = convert_to_numeric(df_for_plot[values_col_name], default_value=0.0)
    if values_are_counts:
        df_for_plot[values_col_name] = df_for_plot[values_col_name].round().astype('Int64')
    
    df_for_plot = df_for_plot[df_for_plot[values_col_name] > 0] # Only plot positive values
    if df_for_plot.empty:
        logger.debug(f"Donut chart '{chart_title}': No positive data values to plot.")
        return create_empty_figure(chart_title, final_height, "No positive data for donut chart.")

    # Sort by values (descending) for consistent pull and color assignment
    df_for_plot.sort_values(by=values_col_name, ascending=False, inplace=True)
    df_for_plot[labels_col_name] = df_for_plot[labels_col_name].astype(str)

    # Color mapping
    plot_colors_list = None
    if custom_color_map:
        plot_colors_list = [custom_color_map.get(str(label), get_theme_color(i, color_category="general")) for i, label in enumerate(df_for_plot[labels_col_name])]
    elif settings.LEGACY_DISEASE_COLORS_WEB and any(str(lbl) in settings.LEGACY_DISEASE_COLORS_WEB for lbl in df_for_plot[labels_col_name]):
        plot_colors_list = [get_theme_color(str(lbl), color_category="disease", fallback_color_hex=get_theme_color(i, color_category="general")) for i, lbl in enumerate(df_for_plot[labels_col_name])]
    else: # Default theme colorway
        plot_colors_list = [get_theme_color(i, color_category="general") for i in range(len(df_for_plot[labels_col_name]))]


    hover_value_format = 'd' if values_are_counts else '.2f'
    
    fig = go.Figure(data=[go.Pie(
        labels=df_for_plot[labels_col_name],
        values=df_for_plot[values_col_name],
        hole=0.58, # Donut hole size
        pull=[pull_slice_amount if i < min(3, len(df_for_plot)) else 0 for i in range(len(df_for_plot))], # Pull top slices
        textinfo='label+percent', # Show label and percentage on slices
        insidetextorientation='radial', # Orient text
        hoverinfo='label+value+percent', # Info on hover
        hovertemplate=f'<b>%{{label}}</b><br>Value: %{{value:{hover_value_format}}}<br>Percent: %{{percent}}<extra></extra>',
        marker=dict(colors=plot_colors_list, line=dict(color=settings.COLOR_BACKGROUND_WHITE, width=1.8)),
        sort=False # Already sorted df_for_plot
    )])
    
    annotations_list = []
    if center_annotation:
        annotations_list.append(dict(
            text=str(center_annotation), 
            x=0.5, y=0.5, font_size=14, 
            showarrow=False, font_color=settings.COLOR_TEXT_DARK
        ))

    fig.update_layout(
        title_text=chart_title,
        height=final_height,
        showlegend=True, # Enable legend
        legend=dict(
            orientation="v", # Vertical legend
            yanchor="middle", y=0.5, # Center vertically
            xanchor="right", x=1.18, # Position to the right
            font_size=9,
            traceorder="normal" # Match order of slices
        ),
        annotations=annotations_list if annotations_list else None,
        margin=dict(l=20, r=120, t=60, b=20) # Adjust right margin for legend
    )
    return fig


def plot_heatmap( # Renamed
    matrix_df_input: Optional[pd.DataFrame],
    chart_title: str,
    chart_height: Optional[int] = None,
    color_scale_name: str = 'RdBu_r', # Renamed
    z_midpoint_val: Optional[float] = 0.0, # Renamed for clarity, 0 is good for correlation-like data
    show_cell_text: bool = True, # Renamed
    text_format: str = '.2f', # Renamed, e.g. for correlation values
    show_colorbar: bool = True # Renamed
) -> go.Figure:
    """Generates a heatmap from a matrix DataFrame."""
    final_height = chart_height or (settings.WEB_PLOT_DEFAULT_HEIGHT + 80) # Heatmaps can be taller

    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty:
        logger.debug(f"Heatmap '{chart_title}': Input matrix_df_input is empty or invalid.")
        return create_empty_figure(chart_title, final_height)

    df_numeric_matrix = matrix_df_input.copy().apply(pd.to_numeric, errors='coerce')
    if df_numeric_matrix.isnull().all().all(): # Check if all values became NaN
        logger.debug(f"Heatmap '{chart_title}': All data non-numeric after coercion.")
        return create_empty_figure(chart_title, final_height, "All heatmap data is non-numeric or missing.")

    z_values_for_plot = df_numeric_matrix.values
    text_values_on_cells = np.vectorize(lambda x: f"{x:{text_format}}" if pd.notna(x) else '')(z_values_for_plot) if show_cell_text else None
    
    # Determine zmid: if data is all positive or all negative, zmid=0 might not be good.
    # If data spans positive and negative, zmid=0 is usually desired for diverging scales.
    final_zmid_value = z_midpoint_val
    z_flat_no_nan = z_values_for_plot[~np.isnan(z_values_for_plot)]
    if len(z_flat_no_nan) > 0:
        all_positive = np.all(z_flat_no_nan >= 0)
        all_negative = np.all(z_flat_no_nan <= 0)
        if all_positive or all_negative: # If data doesn't cross zero, zmid=0 is less useful
            final_zmid_value = None # Let Plotly auto-determine midpoint
    else: # All NaNs
        final_zmid_value = None


    fig = go.Figure(data=[go.Heatmap(
        z=z_values_for_plot,
        x=df_numeric_matrix.columns.astype(str).tolist(),
        y=df_numeric_matrix.index.astype(str).tolist(),
        colorscale=color_scale_name,
        zmid=final_zmid_value, # Midpoint of colorscale
        text=text_values_on_cells,
        texttemplate="%{text}" if show_cell_text and text_values_on_cells is not None else '',
        hoverongaps=False, # Don't show hover for NaN gaps
        xgap=1, ygap=1, # Small gaps between cells
        colorbar=dict(
            thickness=15, len=0.9, tickfont=dict(size=9),
            title=dict(text="Value", side="right", font=dict(size=10)), # Colorbar title
            outlinewidth=0.5, outlinecolor=settings.COLOR_BORDER_MEDIUM
        ) if show_colorbar else None
    )])
    
    # Adjust x-axis tick angle if labels are long or numerous
    x_tick_angle_val = 0
    if len(df_numeric_matrix.columns) > 8 or max(len(str(c)) for c in df_numeric_matrix.columns) > 10:
        x_tick_angle_val = -45

    fig.update_layout(
        title_text=chart_title,
        height=final_height,
        xaxis=dict(showgrid=False, tickangle=x_tick_angle_val, side='bottom'), # Ticks at bottom
        yaxis=dict(showgrid=False, autorange='reversed'), # Standard for heatmaps
        plot_bgcolor=settings.COLOR_BACKGROUND_WHITE # Clean white background for heatmap plot area
    )
    return fig


def plot_choropleth_map( # Renamed
    map_data_df: Optional[pd.DataFrame], # DataFrame with zone_id and value_col
    geojson_features: Optional[List[Dict[str,Any]]], # List of GeoJSON features
    value_col_name: str,
    map_title: str,
    zone_id_geojson_prop: str = 'zone_id', # Property in GeoJSON features to match df zone_id
    zone_id_df_col: str = 'zone_id',       # Column in df to match GeoJSON property
    color_scale_name: str = 'Viridis',
    hover_name_col: Optional[str] = None, # Column for hover name (e.g., 'name')
    hover_data_cols: Optional[List[str]] = None, # Additional columns for hover info
    map_height: Optional[int] = None,
    center_lat_val: Optional[float] = None, # Renamed
    center_lon_val: Optional[float] = None, # Renamed
    zoom_level_val: Optional[int] = None,   # Renamed
    mapbox_style_override: Optional[str] = None # Renamed
) -> go.Figure:
    """
    Generates a choropleth map using Plotly Express.
    Uses a list of GeoJSON features instead of a file path.
    """
    final_map_height = map_height or settings.WEB_MAP_DEFAULT_HEIGHT
    module_log_prefix = "ChoroplethMap"

    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or \
       value_col_name not in map_data_df.columns or \
       zone_id_df_col not in map_data_df.columns:
        logger.warning(f"({module_log_prefix}) '{map_title}': DataFrame for map is invalid or missing key columns (value: '{value_col_name}', id: '{zone_id_df_col}').")
        return create_empty_figure(map_title, final_map_height, "Map data is incomplete.")

    if not geojson_features or not isinstance(geojson_features, list) or len(geojson_features) == 0:
        logger.warning(f"({module_log_prefix}) '{map_title}': GeoJSON features list is empty or invalid.")
        return create_empty_figure(map_title, final_map_height, "Geographic boundary data (GeoJSON) unavailable.")

    df_for_map_plot = map_data_df.copy()
    # Ensure value column is numeric and id column is string for matching
    df_for_map_plot[value_col_name] = convert_to_numeric(df_for_map_plot[value_col_name], default_value=np.nan)
    df_for_map_plot[zone_id_df_col] = df_for_map_plot[zone_id_df_col].astype(str).str.strip()
    
    # Drop rows where the value or ID is NaN/missing, as they can't be plotted
    df_for_map_plot.dropna(subset=[value_col_name, zone_id_df_col], inplace=True)
    
    if df_for_map_plot.empty:
        logger.info(f"({module_log_prefix}) '{map_title}': DataFrame empty after cleaning for map plotting.")
        return create_empty_figure(map_title, final_map_height, f"No valid data for metric '{value_col_name}' to display on map.")

    # Construct a valid GeoJSON FeatureCollection dictionary for Plotly Express
    geojson_feature_collection = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # Determine hover name: prefer specified, then 'name' col, then zone_id_df_col
    actual_hover_name_col = zone_id_df_col # Default
    if hover_name_col and hover_name_col in df_for_map_plot.columns:
        actual_hover_name_col = hover_name_col
    elif 'name' in df_for_map_plot.columns: # Common column for display name
        actual_hover_name_col = 'name'
        
    # Prepare hover_data: dictionary format for px.choropleth_mapbox
    final_hover_data_dict = {}
    if hover_data_cols:
        for h_col in hover_data_cols:
            if h_col in df_for_map_plot.columns:
                final_hover_data_dict[h_col] = True # Show column with default formatting
    # Always include the value column in hover if not already specified
    if value_col_name not in final_hover_data_dict:
         final_hover_data_dict[value_col_name] = True


    # Resolve map style based on token and override
    effective_map_style = mapbox_style_override
    if not effective_map_style: # If no override, use theme's default
        try:
            active_theme_map_style = pio.templates[pio.templates.default].layout.mapbox.style
            effective_map_style = active_theme_map_style
        except (KeyError, AttributeError):
            logger.warning(f"({module_log_prefix}) Could not get mapbox style from current Plotly theme. Defaulting to 'carto-positron'.")
            effective_map_style = "carto-positron" # Absolute fallback
            
    # If the chosen style might require a token, but token is not set, switch to an open style
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_styles_check = ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]
        if effective_map_style and effective_map_style.lower() not in open_styles_check:
            logger.warning(f"({module_log_prefix}) Map style '{effective_map_style}' might require a token, but token is not set. Switching to 'carto-positron'.")
            effective_map_style = "carto-positron"


    try:
        fig = px.choropleth_mapbox(
            df_for_map_plot,
            geojson=geojson_feature_collection, # Pass the constructed GeoJSON object
            locations=zone_id_df_col,         # Column in df_for_map_plot with IDs
            featureidkey=f"properties.{zone_id_geojson_prop}", # Path to ID in GeoJSON features
            color=value_col_name,             # Column in df_for_map_plot for color
            color_continuous_scale=color_scale_name,
            hover_name=actual_hover_name_col,
            hover_data=final_hover_data_dict,
            mapbox_style=effective_map_style,
            center=dict(lat=center_lat_val or settings.MAP_DEFAULT_CENTER_LAT,
                        lon=center_lon_val or settings.MAP_DEFAULT_CENTER_LON),
            zoom=zoom_level_val or settings.MAP_DEFAULT_ZOOM_LEVEL,
            opacity=0.75, # Slightly more opaque for better visibility
            height=final_map_height,
            title=map_title
        )
    except Exception as e_map:
        logger.error(f"({module_log_prefix}) Error creating choropleth map '{map_title}': {e_map}", exc_info=True)
        return create_empty_figure(map_title, final_map_height, "Error during map generation.")

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0), # Tight margins
        # Mapbox token is handled by px.set_mapbox_access_token() or if MAPBOX_ACCESS_TOKEN env var is set
    )
    # The `fitbounds` argument is not directly available in px.choropleth_mapbox.
    # Zoom and center are used instead. If using go.Choroplethmapbox, `fitbounds` could be used.
    # fig.update_geos(fitbounds="locations", visible=False) # This is for go.Choropleth, not mapbox variant

    logger.info(f"({module_log_prefix}) Choropleth map '{map_title}' created successfully.")
    return fig
