# sentinel_project_root/visualization/plots.py
# Plotting functions for Sentinel Health Co-Pilot Web Dashboards using Plotly.

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio 
import os 
from typing import Optional, List, Dict, Any, Union
import re 

try:
    from config import settings
    from .ui_elements import get_theme_color # Ensure this import path is correct
except ImportError as e:
    logging.basicConfig(level=logging.ERROR) # Basic logger if import fails early
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in plots.py: {e}. Ensure paths and dependencies are correct.")
    raise

logger = logging.getLogger(__name__)

# --- Mapbox Token Handling (Module Level) ---
MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = False
_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV = os.getenv("MAPBOX_ACCESS_TOKEN")

if _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV and _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV.strip() and len(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV) > 20:
    try:
        px.set_mapbox_access_token(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV)
        MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = True
        logger.info("Plotly: MAPBOX_ACCESS_TOKEN environment variable found and configured for Plotly Express.")
    except Exception as e_mapbox_token_setup:
        logger.error(f"Plotly: Error setting Mapbox token for Plotly Express: {e_mapbox_token_setup}", exc_info=True)
else:
    logger.warning("Plotly: MAPBOX_ACCESS_TOKEN not found or invalid. Maps needing a token may use open styles or fail.")


# --- Plotly Theme Setup ---
def _get_setting_or_default(attr_name: str, default_value: Any) -> Any:
    """Safely gets a setting attribute or returns a default."""
    return getattr(settings, attr_name, default_value)

def set_sentinel_plotly_theme():
    """Sets a custom Plotly theme ('sentinel_web_theme_custom') as the default."""
    theme_font_family = _get_setting_or_default('THEME_FONT_FAMILY', '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif')
    
    # Define colorway ensuring all colors are valid hex strings
    default_colorway = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    sentinel_colorway = [
        _get_setting_or_default('COLOR_ACTION_PRIMARY', default_colorway[0]),
        _get_setting_or_default('COLOR_RISK_LOW', default_colorway[1]),
        _get_setting_or_default('COLOR_RISK_MODERATE', default_colorway[2]),
        _get_setting_or_default('COLOR_ACCENT_BRIGHT', default_colorway[3]),
        _get_setting_or_default('COLOR_ACTION_SECONDARY', default_colorway[4]),
        get_theme_color(5, "general", default_colorway[5]), 
        get_theme_color(6, "general", default_colorway[6]), 
        get_theme_color(7, "general", default_colorway[7])
    ]

    layout_template = go.Layout(
        font=dict(family=theme_font_family, size=11, color=_get_setting_or_default('COLOR_TEXT_DARK', "#333333")),
        paper_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT', "#FFFFFF"),
        plot_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_PAGE', "#F0F2F6"),
        colorway=sentinel_colorway,
        xaxis=dict(
            gridcolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0"),
            linecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"),
            zerolinecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"), 
            zerolinewidth=1,
            title_font_size=12, tickfont_size=10, automargin=True
        ),
        yaxis=dict(
            gridcolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0"),
            linecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"),
            zerolinecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"), 
            zerolinewidth=1,
            title_font_size=12, tickfont_size=10, automargin=True
        ),
        title=dict(
            font=dict(family=theme_font_family, size=16, color=_get_setting_or_default('COLOR_TEXT_HEADINGS_MAIN', "#111111")),
            x=0.05, xanchor='left', y=0.95, yanchor='top', pad=dict(t=20, b=10)
        ),
        legend=dict(
            bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT_TRANSPARENT', "rgba(255,255,255,0.8)"),
            bordercolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0"), 
            borderwidth=0.5, orientation='h',
            yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=10
        ),
        margin=dict(l=60, r=20, t=80, b=60) # Default margins
    )

    # Mapbox style handling
    mapbox_style_setting = _get_setting_or_default('MAPBOX_STYLE_WEB', "carto-positron")
    effective_mapbox_style = mapbox_style_setting
    
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_map_styles = {"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"}
        if mapbox_style_setting.lower() not in open_map_styles:
            effective_mapbox_style = "carto-positron" # Fallback to a known open style
            logger.info(f"Plotly Theme: Mapbox token not set and style '{mapbox_style_setting}' requires one. Defaulting theme map style to '{effective_mapbox_style}'.")
    
    layout_template.mapbox = dict(
        style=effective_mapbox_style,
        center=dict(lat=_get_setting_or_default('MAP_DEFAULT_CENTER_LAT', 0), lon=_get_setting_or_default('MAP_DEFAULT_CENTER_LON', 0)),
        zoom=_get_setting_or_default('MAP_DEFAULT_ZOOM_LEVEL', 1)
    )

    pio.templates["sentinel_web_theme_custom"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_web_theme_custom" # Apply as default for all plots
    logger.info("Custom Plotly theme 'sentinel_web_theme_custom' configured and applied as default.")

# Apply theme once on module import
try:
    set_sentinel_plotly_theme()
except Exception as e_theme_setup:
    logger.error(f"Failed to set custom Plotly theme: {e_theme_setup}", exc_info=True)
    # Fallback to Plotly default if custom theme fails
    pio.templates.default = "plotly" 
    logger.warning("Fell back to default 'plotly' theme due to error in custom theme setup.")


# --- Plotting Functions ---

def create_empty_figure(
    chart_title: str,
    height: Optional[int] = None,
    message_text: str = "No data available to display for the current selection."
) -> go.Figure:
    """Creates a styled empty Plotly figure with a message."""
    final_height = height or _get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 400)
    fig = go.Figure()
    fig.update_layout(
        title_text=chart_title, 
        height=final_height,
        xaxis=dict(visible=False), # Use dict() for updating layout properties
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message_text, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=_get_setting_or_default('COLOR_TEXT_MUTED', '#757575'))
        )],
        # Ensure these use theme colors if available, otherwise hardcoded fallbacks
        paper_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT', '#FFFFFF'),
        plot_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_PAGE', '#F0F2F6')
    )
    return fig

def plot_annotated_line_chart(
    data_series: Optional[pd.Series], chart_title: str, y_axis_label: str = "Value",
    line_color_hex: Optional[str] = None, target_ref_line_val: Optional[float] = None,
    target_ref_label_text: Optional[str] = None, show_confidence_interval: bool = False,
    lower_ci_series: Optional[pd.Series] = None, upper_ci_series: Optional[pd.Series] = None,
    chart_height: Optional[int] = None, show_anomalies_flag: bool = False,
    anomaly_iqr_factor: float = 1.5, date_format_hover: str = "%Y-%m-%d",
    y_values_are_counts: bool = False
) -> go.Figure:
    final_height = chart_height or _get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350)
    
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title, final_height, "No data provided for line chart.")

    # Prepare data: ensure datetime index and numeric values
    series_plot = data_series.copy() # Work on a copy
    if not pd.api.types.is_datetime64_any_dtype(series_plot.index):
        series_plot.index = pd.to_datetime(series_plot.index, errors='coerce')
    series_plot = series_plot.dropna(axis=0, how='all') # Drop rows where index became NaT
    
    # Attempt numeric conversion, explicitly handling potential errors for the whole series
    try:
        series_plot = pd.to_numeric(series_plot, errors='coerce')
    except Exception as e_conv:
        logger.warning(f"Line chart '{chart_title}': Could not convert data series to numeric: {e_conv}. Some values may be NaN.")
        # Fallback to object type if conversion utterly fails, Plotly might handle it or show error
    
    series_plot.dropna(inplace=True) # Remove any NaNs produced by conversion or originally present

    if series_plot.empty: 
        return create_empty_figure(chart_title, final_height, "No valid data points after cleaning for line chart.")

    fig = go.Figure()
    actual_line_color = line_color_hex or get_theme_color(0, "general", fallback_color_hex="#1f77b4") # Default blue
    y_hover_format_str = 'd' if y_values_are_counts else ',.1f' # Integer for counts, 1 decimal for others
    
    fig.add_trace(go.Scatter(
        x=series_plot.index, y=series_plot.values, mode="lines+markers",
        name=y_axis_label if y_axis_label and y_axis_label.strip() else (series_plot.name or "Value"),
        line=dict(color=actual_line_color, width=2), marker=dict(size=5, symbol='circle'),
        hovertemplate=f'<b>Date</b>: %{{x|{date_format_hover}}}<br><b>{y_axis_label or "Value"}</b>: %{{y:{y_hover_format_str}}}<extra></extra>'
    ))

    # Confidence Interval
    if show_confidence_interval and isinstance(lower_ci_series, pd.Series) and isinstance(upper_ci_series, pd.Series) \
       and not lower_ci_series.empty and not upper_ci_series.empty:
        try:
            l_ci = lower_ci_series.copy(); l_ci.index = pd.to_datetime(l_ci.index, errors='coerce'); l_ci = pd.to_numeric(l_ci, errors='coerce')
            u_ci = upper_ci_series.copy(); u_ci.index = pd.to_datetime(u_ci.index, errors='coerce'); u_ci = pd.to_numeric(u_ci, errors='coerce')
            
            aligned_df = pd.concat([series_plot.rename("value"), l_ci.rename("lower"), u_ci.rename("upper")], axis=1).dropna()
            if not aligned_df.empty and (aligned_df["upper"] >= aligned_df["lower"]).all(): # Ensure upper > lower
                fill_rgba_tuple = px.colors.hex_to_rgb(actual_line_color)
                fill_color_str = f"rgba({fill_rgba_tuple[0]},{fill_rgba_tuple[1]},{fill_rgba_tuple[2]},0.15)" # Lighter fill
                fig.add_trace(go.Scatter(
                    x=list(aligned_df.index) + list(aligned_df.index[::-1]), # x, then x reversed
                    y=list(aligned_df["upper"]) + list(aligned_df["lower"][::-1]), # upper, then lower reversed
                    fill="toself", fillcolor=fill_color_str, line=dict(width=0), name="Confidence Interval", hoverinfo="skip"
                ))
            else: logger.warning(f"Line chart '{chart_title}': CI data invalid or alignment failed.")
        except Exception as e_ci:
            logger.warning(f"Line chart '{chart_title}': Error processing confidence interval: {e_ci}", exc_info=True)


    # Target Reference Line
    if target_ref_line_val is not None and pd.notna(target_ref_line_val):
        ref_label = target_ref_label_text or f"Target: {target_ref_line_val:,.2f}"
        fig.add_hline(
            y=target_ref_line_val, line_dash="dash", 
            line_color=get_theme_color("risk_moderate", fallback_color_hex="#FFA500"), line_width=1.2,
            annotation_text=ref_label, annotation_position="bottom right",
            annotation_font_size=9, annotation_font_color=get_theme_color("text_muted", fallback_color_hex="#757575")
        )

    # Anomaly Detection (Simple IQR based)
    if show_anomalies_flag and len(series_plot) > 7 and series_plot.nunique() > 2: # Min data points for meaningful IQR
        q1, q3 = series_plot.quantile(0.25), series_plot.quantile(0.75)
        iqr_value = q3 - q1
        if pd.notna(iqr_value) and iqr_value > 1e-6: # Check for valid IQR (not NaN and not extremely small)
            upper_bound, lower_bound = q3 + anomaly_iqr_factor * iqr_value, q1 - anomaly_iqr_factor * iqr_value
            anomalies_series = series_plot[(series_plot < lower_bound) | (series_plot > upper_bound)]
            if not anomalies_series.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies_series.index, y=anomalies_series.values, mode='markers',
                    marker=dict(color=get_theme_color("risk_high", fallback_color_hex="#FF0000"), size=8, 
                                symbol='circle-open-dot', line=dict(width=1.5)),
                    name='Anomaly', 
                    hovertemplate=f'<b>Anomaly</b>: %{{x|{date_format_hover}}}<br><b>Value</b>: %{{y:{y_hover_format_str}}}<extra></extra>'
                ))

    x_axis_title = series_plot.index.name if series_plot.index.name and series_plot.index.name.strip() else "Date/Time"
    yaxis_config = dict(title_text=y_axis_label, rangemode='tozero' if y_values_are_counts and series_plot.min() >= 0 else 'normal')
    if y_values_are_counts: yaxis_config['tickformat'] = 'd' # Integer format for counts

    fig.update_layout(
        title_text=chart_title, 
        xaxis_title=x_axis_title, 
        yaxis=yaxis_config, 
        height=final_height, 
        hovermode="x unified", # Shows all trace info for a given x
        legend=dict(traceorder='normal') # Ensure main line appears before CI/anomalies in legend
    )
    return fig


def plot_bar_chart(
    df_input: Optional[pd.DataFrame], x_col_name: str, y_col_name: str, chart_title: str,
    color_col_name: Optional[str] = None, bar_mode_style: str = 'group', orientation_bar: str = 'v',
    y_axis_label_text: Optional[str] = None, x_axis_label_text: Optional[str] = None,
    chart_height: Optional[int] = None, show_text_on_bars: Union[bool, str] = True,
    sort_by_col: Optional[str] = None, sort_ascending_flag: bool = True,
    text_format_str: Optional[str] = None, y_values_are_counts_flag: bool = False,
    custom_color_map: Optional[Dict[str, str]] = None
) -> go.Figure:
    final_height = chart_height or _get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 450)
    
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or \
       x_col_name not in df_input.columns or y_col_name not in df_input.columns:
        return create_empty_figure(chart_title, final_height, f"Missing required columns ('{x_col_name}', '{y_col_name}') or empty data.")

    df_plot = df_input.copy()
    # Ensure x-axis (categorical) is string for reliable plotting
    df_plot[x_col_name] = df_plot[x_col_name].astype(str).str.strip()
    # Convert y-axis (value) to numeric
    df_plot[y_col_name] = convert_to_numeric(
        df_plot[y_col_name], 
        default_value=0.0, # Default to 0 for aggregation, but will be dropped by dropna if it was originally NaN
        target_type=int if y_values_are_counts_flag else float
    )
    df_plot.dropna(subset=[x_col_name, y_col_name], inplace=True) # Drop rows where key columns are NaN
    
    if df_plot.empty: 
        return create_empty_figure(chart_title, final_height, f"No valid data for x='{x_col_name}', y='{y_col_name}' after cleaning.")

    # Sorting
    if sort_by_col and sort_by_col in df_plot.columns:
        try: 
            df_plot.sort_values(by=sort_by_col, ascending=sort_ascending_flag, inplace=True, na_position='last')
        except Exception as e_sort: 
            logger.warning(f"Bar chart '{chart_title}': Sorting by '{sort_by_col}' failed: {e_sort}.")
    
    # Determine text format for bar values
    effective_text_format = text_format_str
    if not effective_text_format:
        effective_text_format = '.0f' if y_values_are_counts_flag else '.1f' # Integer for counts, 1 decimal otherwise
    
    y_hover_format_bar_str = 'd' if y_values_are_counts_flag else effective_text_format # Consistent hover and text
    
    # Axis labels
    x_label = x_axis_label_text or x_col_name.replace('_', ' ').title()
    y_label = y_axis_label_text or y_col_name.replace('_', ' ').title()
    
    # Legend title
    legend_title_str = None
    if color_col_name and color_col_name in df_plot.columns:
        df_plot[color_col_name] = df_plot[color_col_name].astype(str) # Ensure color column is string for discrete mapping
        legend_title_str = color_col_name.replace('_', ' ').title()

    # Color mapping
    final_color_map = custom_color_map
    if not final_color_map and color_col_name and color_col_name in df_plot.columns:
        # Attempt to use legacy disease colors if applicable
        legacy_colors = _get_setting_or_default('LEGACY_DISEASE_COLORS_WEB', {})
        if any(str(val) in legacy_colors for val in df_plot[color_col_name].dropna().unique()):
            final_color_map = { 
                str(val): get_theme_color(str(val), "disease", get_theme_color(abs(hash(str(val))) % 8, "general"))
                for val in df_plot[color_col_name].dropna().unique() 
            }
    
    try:
        fig = px.bar(
            df_plot, 
            x=x_col_name if orientation_bar == 'v' else y_col_name,
            y=y_col_name if orientation_bar == 'v' else x_col_name, 
            title=chart_title,
            color=color_col_name if color_col_name in df_plot.columns else None,
            barmode=bar_mode_style, 
            orientation=orientation_bar, 
            height=final_height,
            labels={y_col_name: y_label, x_col_name: x_label, (color_col_name or "_"): legend_title_str or ''}, # Use underscore if no color col
            text_auto=show_text_on_bars if isinstance(show_text_on_bars, bool) else False, # px.bar text_auto takes bool
            color_discrete_map=final_color_map
        )
    except Exception as e_px_bar:
        logger.error(f"Error generating bar chart '{chart_title}' with Plotly Express: {e_px_bar}", exc_info=True)
        return create_empty_figure(chart_title, final_height, f"Error during bar chart generation: {e_px_bar}")

    # Refine text and hovertemplate
    text_template_str = f'%{{y:{effective_text_format}}}' if orientation_bar == 'v' else f'%{{x:{effective_text_format}}}'
    if isinstance(show_text_on_bars, str) and show_text_on_bars.lower() != 'auto': # Allow custom text template
        text_template_str = show_text_on_bars
    
    hover_x_var, hover_y_var = ('x', 'y') if orientation_bar == 'v' else ('y', 'x')
    x_axis_hover_label = x_label if orientation_bar == "v" else y_label
    y_axis_hover_label = y_label if orientation_bar == "v" else x_label
    
    current_hover_template = f'<b>{x_axis_hover_label}</b>: %{{{hover_x_var}}}<br>' + \
                             f'<b>{y_axis_hover_label}</b>: %{{{hover_y_var}:{y_hover_format_bar_str}}}'
    
    custom_data_for_hover = []
    if color_col_name and color_col_name in df_plot.columns and legend_title_str:
        current_hover_template += f'<br><b>{legend_title_str}</b>: %{{customdata[0]}}'
        custom_data_for_hover.append(color_col_name)
    current_hover_template += '<extra></extra>' # Remove trace info

    fig.update_traces(
        marker_line_width=0.8, marker_line_color='rgba(0,0,0,0.3)', 
        textfont_size=9,
        textposition='outside' if orientation_bar == 'h' else ('auto' if bar_mode_style != 'stack' else 'inside'),
        cliponaxis=False, 
        texttemplate=text_template_str if show_text_on_bars else None, # Apply text only if show_text_on_bars is True
        hovertemplate=current_hover_template,
        customdata=df_plot[custom_data_for_hover] if custom_data_for_hover else None
    )

    # Axis configurations
    yaxis_final_config = {'title_text': y_label}
    xaxis_final_config = {'title_text': x_label}
    value_axis_config_ref = yaxis_final_config if orientation_bar == 'v' else xaxis_final_config
    category_axis_config_ref = xaxis_final_config if orientation_bar == 'v' else yaxis_final_config

    if y_values_are_counts_flag:
        value_axis_config_ref.update({'tickformat': 'd', 'rangemode': 'tozero'})
    
    # Category ordering based on sort
    category_col_for_order = x_col_name if orientation_bar == 'v' else y_col_name
    if sort_by_col == category_col_for_order: # If sorted by the category axis itself
        category_axis_config_ref.update({'categoryorder': 'array', 'categoryarray': df_plot[category_col_for_order].tolist()})
    elif orientation_bar == 'h' and (not sort_by_col or sort_by_col == y_col_name): # Sort horizontal bars by value
         category_axis_config_ref['categoryorder'] = 'total ascending' if sort_ascending_flag else 'total descending'
    
    fig.update_layout(
        yaxis=yaxis_final_config, xaxis=xaxis_final_config, 
        uniformtext_minsize=7, uniformtext_mode='hide', # Standard text handling
        legend_title_text=legend_title_str
    )
    return fig


def plot_donut_chart(
    df_input: Optional[pd.DataFrame], labels_col_name: str, values_col_name: str, chart_title: str,
    chart_height: Optional[int] = None, custom_color_map: Optional[Dict[str, str]] = None,
    pull_slice_amount: float = 0.03, center_annotation_text: Optional[str] = None, # Renamed for clarity
    values_are_counts: bool = True
) -> go.Figure:
    final_height = chart_height or (_get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350) + 50) # Donuts often need a bit more height
    
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or \
       labels_col_name not in df_input.columns or values_col_name not in df_input.columns:
        return create_empty_figure(chart_title, final_height, "Missing data or columns for donut chart.")

    df_plot = df_input.copy()
    # Convert values, ensuring positive values for pie/donut charts
    df_plot[values_col_name] = convert_to_numeric(
        df_plot[values_col_name], 
        default_value=0.0, # Default to 0, will be filtered out
        target_type=int if values_are_counts else float
    )
    df_plot = df_plot[df_plot[values_col_name] > 1e-6].sort_values(by=values_col_name, ascending=False) # Filter out zero/negative, small epsilon for float
    df_plot[labels_col_name] = df_plot[labels_col_name].astype(str).str.strip() # Ensure labels are clean strings
    
    if df_plot.empty: 
        return create_empty_figure(chart_title, final_height, "No positive data available for donut chart.")

    # Color resolution
    plot_colors = None
    unique_labels = df_plot[labels_col_name].unique()
    if custom_color_map:
        plot_colors = [custom_color_map.get(str(lbl), get_theme_color(i % 8, "general")) for i, lbl in enumerate(unique_labels)]
    else:
        legacy_disease_colors = _get_setting_or_default('LEGACY_DISEASE_COLORS_WEB', {})
        if any(str(lbl) in legacy_disease_colors for lbl in unique_labels):
            plot_colors = [get_theme_color(str(lbl), "disease", get_theme_color(i % 8, "general")) for i, lbl in enumerate(unique_labels)]
        else: # Fallback to general theme colorway
            plot_colors = [get_theme_color(i % 8, "general") for i in range(len(unique_labels))] # Cycle through 8 general colors
    
    hover_value_format_str = 'd' if values_are_counts else '.2f' # Integer for counts, 2 decimals for other values
    
    # Pull largest slices for emphasis
    num_slices_to_pull = min(3, len(df_plot)) # Pull up to 3 largest slices
    pull_values = [pull_slice_amount if i < num_slices_to_pull else 0 for i in range(len(df_plot))]

    fig = go.Figure(data=[go.Pie(
        labels=df_plot[labels_col_name], 
        values=df_plot[values_col_name], 
        hole=0.58, # Standard donut hole size
        pull=pull_values,
        textinfo='label+percent', # Show label and percentage on slices
        insidetextorientation='radial', 
        hoverinfo='label+value+percent',
        hovertemplate=f'<b>%{{label}}</b><br>Value: %{{value:{hover_value_format_str}}}<br>Percent: %{{percent}}<extra></extra>',
        marker=dict(colors=plot_colors, line=dict(color=get_theme_color("white", fallback_color_hex="#FFFFFF"), width=1.8)),
        sort=False # Already sorted by value
    )])
    
    fig_annotations = []
    if center_annotation_text and center_annotation_text.strip():
        fig_annotations.append(dict(
            text=str(center_annotation_text), 
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font_color=get_theme_color("text_dark", fallback_color_hex="#333333")
        ))
        
    fig.update_layout(
        title_text=chart_title, 
        height=final_height, 
        showlegend=True, # Consider if legend is needed with textinfo on slices
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.18, font_size=9, traceorder="normal"),
        annotations=fig_annotations if fig_annotations else None,
        margin=dict(l=20, r=120, t=60, b=20) # Adjust right margin for legend
    )
    return fig


def plot_heatmap(
    matrix_df_input: Optional[pd.DataFrame], chart_title: str, chart_height: Optional[int] = None,
    color_scale_name: str = 'RdBu_r', # Red-Blue diverging, reversed (red high, blue low)
    z_midpoint_val: Optional[float] = 0.0, # Useful for diverging scales centered at 0
    show_cell_text: bool = True, text_format: str = '.2f', show_colorbar: bool = True
) -> go.Figure:
    final_height = chart_height or (_get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 450) + 80) # Heatmaps can be tall
    
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty:
        return create_empty_figure(chart_title, final_height, "No data provided for heatmap.")

    # Ensure data is numeric, coercing errors
    df_matrix = matrix_df_input.copy().apply(pd.to_numeric, errors='coerce')
    
    if df_matrix.isnull().all().all(): # Check if all values became NaN after coercion
        return create_empty_figure(chart_title, final_height, "All heatmap data is non-numeric or missing.")

    z_values_for_plot = df_matrix.values
    # Prepare text for cells if requested
    cell_text_values = None
    if show_cell_text:
        # Vectorized formatting of cell text, handles NaNs by showing empty string
        cell_text_values = np.vectorize(lambda x: f"{x:{text_format}}" if pd.notna(x) else '')(z_values_for_plot)
    
    # Determine zmid: if all data is positive or all negative, auto-calculate midpoint (None)
    # Otherwise, use the provided z_midpoint_val (often 0 for diverging scales)
    z_mid_final = z_midpoint_val
    valid_z_flat = z_values_for_plot[~np.isnan(z_values_for_plot)] # Flatten and remove NaNs
    if len(valid_z_flat) > 0:
        if (valid_z_flat >= 0).all() or (valid_z_flat <= 0).all(): # All positive or all negative
            z_mid_final = None 
    elif len(valid_z_flat) == 0: # All values were NaN
        z_mid_final = None

    fig = go.Figure(data=[go.Heatmap(
        z=z_values_for_plot, 
        x=df_matrix.columns.astype(str).tolist(), 
        y=df_matrix.index.astype(str).tolist(),
        colorscale=color_scale_name, 
        zmid=z_mid_final, 
        text=cell_text_values,
        texttemplate="%{text}" if show_cell_text and cell_text_values is not None else '', # Display text in cells
        hoverongaps=False, # Don't show hover for NaN cells
        xgap=1, ygap=1, # Small gaps between cells
        colorbar=dict(
            thickness=15, len=0.9, 
            tickfont=dict(size=9),
            title=dict(text="Value", side="right", font=dict(size=10)),
            outlinewidth=0.5, 
            outlinecolor=get_theme_color("border_medium", fallback_color_hex="#BDBDBD")
        ) if show_colorbar else None
    )])
    
    # Adjust x-axis tick angle for readability if many columns or long labels
    x_tick_angle_final = 0
    if len(df_matrix.columns) > 8 or max(len(str(c)) for c in df_matrix.columns) > 10:
        x_tick_angle_final = -45
        
    fig.update_layout(
        title_text=chart_title, 
        height=final_height,
        xaxis=dict(showgrid=False, tickangle=x_tick_angle_final, side='bottom'), # Ticks at bottom for heatmaps often clearer
        yaxis=dict(showgrid=False, autorange='reversed'), # Typically Y is reversed for matrix-like display
        plot_bgcolor=get_theme_color("white", fallback_color_hex="#FFFFFF") # Clean background for heatmap
    )
    return fig


def plot_choropleth_map(
    map_data_df: Optional[pd.DataFrame], 
    geojson_features: Optional[Union[Dict[str,Any], List[Dict[str,Any]]]], # Accept raw dict or list of features
    value_col_name: str, 
    map_title: str, 
    zone_id_geojson_prop: str = 'zone_id', # Property in GeoJSON feature.properties that matches zone_id_df_col
    zone_id_df_col: str = 'zone_id',       # Column in map_data_df for location matching
    color_scale_name: str = 'Viridis',
    hover_name_col: Optional[str] = None, # Column for main hover label
    hover_data_cols: Optional[List[str]] = None, # Additional columns for hover data
    map_height: Optional[int] = None, 
    center_lat_val: Optional[float] = None,
    center_lon_val: Optional[float] = None, 
    zoom_level_val: Optional[int] = None,
    mapbox_style_override: Optional[str] = None # Allow overriding theme's map style
) -> go.Figure:
    final_height = map_height or _get_setting_or_default('WEB_MAP_DEFAULT_HEIGHT', 600)
    log_prefix = "ChoroplethMap"

    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or \
       value_col_name not in map_data_df.columns or zone_id_df_col not in map_data_df.columns:
        logger.warning(f"({log_prefix}) '{map_title}': Map DataFrame invalid or missing key columns ('{value_col_name}', '{zone_id_df_col}').")
        return create_empty_figure(map_title, final_height, "Map data is incomplete.")
    
    if not geojson_features:
        logger.warning(f"({log_prefix}) '{map_title}': GeoJSON features data is missing.")
        return create_empty_figure(map_title, final_height, "Geographic boundary data unavailable.")
    
    # Ensure geojson_features is a FeatureCollection dict if it's a list
    geojson_for_plotly = geojson_features
    if isinstance(geojson_features, list): # If passed as a list of features
        geojson_for_plotly = {"type": "FeatureCollection", "features": geojson_features}
    elif not (isinstance(geojson_features, dict) and geojson_features.get("type") == "FeatureCollection"):
        logger.warning(f"({log_prefix}) '{map_title}': GeoJSON features not in expected FeatureCollection format or list of features.")
        return create_empty_figure(map_title, final_height, "Invalid geographic boundary data format.")


    df_map = map_data_df.copy()
    # Ensure value column is numeric and zone ID column is string for matching
    df_map[value_col_name] = convert_to_numeric(df_map[value_col_name], default_value=np.nan)
    df_map[zone_id_df_col] = df_map[zone_id_df_col].astype(str).str.strip()
    df_map.dropna(subset=[value_col_name, zone_id_df_col], inplace=True) # Critical columns must be valid
    
    if df_map.empty: 
        return create_empty_figure(map_title, final_height, f"No valid data for '{value_col_name}' to display on map after cleaning.")

    # Determine hover name: use provided, fallback to 'name' column, then zone_id_df_col
    hover_name_final = zone_id_df_col # Default fallback
    if hover_name_col and hover_name_col in df_map.columns:
        hover_name_final = hover_name_col
    elif 'name' in df_map.columns: # Common column name for display
        hover_name_final = 'name'
    
    # Configure hover_data: ensure value_col_name is included if not explicitly in hover_data_cols
    hover_data_config_dict: Dict[str, bool] = {}
    if hover_data_cols:
        for h_col in hover_data_cols:
            if h_col in df_map.columns: 
                hover_data_config_dict[h_col] = True
    if value_col_name not in hover_data_config_dict: # Always show the colored value in hover
        hover_data_config_dict[value_col_name] = True 

    # Determine Mapbox style
    final_mapbox_style = mapbox_style_override
    if not final_mapbox_style: # If no override, use theme default
        try: 
            final_mapbox_style = pio.templates[pio.templates.default].layout.mapbox.style
        except AttributeError: # If theme or mapbox style not set in theme
            final_mapbox_style = "carto-positron" # Absolute fallback
            logger.warning(f"({log_prefix}) Default map style not found in theme. Using 'carto-positron'.")
            
    # If the chosen style requires a token and token is not set, switch to an open style
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_map_styles_set = {"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"}
        if final_mapbox_style and final_mapbox_style.lower() not in open_map_styles_set:
            logger.warning(f"({log_prefix}) Mapbox token not available, but style '{final_mapbox_style}' may require it. Switching to 'carto-positron'.")
            final_mapbox_style = "carto-positron"

    try:
        fig = px.choropleth_mapbox(
            df_map, 
            geojson=geojson_for_plotly, 
            locations=zone_id_df_col, # Column in df_map to match with GeoJSON feature IDs
            featureidkey=f"properties.{zone_id_geojson_prop}", # Path in GeoJSON to the ID
            color=value_col_name, # Column for color intensity
            color_continuous_scale=color_scale_name,
            hover_name=hover_name_final,
            hover_data=hover_data_config_dict if hover_data_config_dict else None, # Pass None if empty
            mapbox_style=final_mapbox_style,
            center=dict(
                lat=center_lat_val if center_lat_val is not None else _get_setting_or_default('MAP_DEFAULT_CENTER_LAT', 0),
                lon=center_lon_val if center_lon_val is not None else _get_setting_or_default('MAP_DEFAULT_CENTER_LON', 0)
            ),
            zoom=zoom_level_val if zoom_level_val is not None else _get_setting_or_default('MAP_DEFAULT_ZOOM_LEVEL', 1),
            opacity=0.75, 
            height=final_height, 
            title=map_title
        )
    except Exception as e_map_create:
        logger.error(f"({log_prefix}) Error creating choropleth map '{map_title}': {e_map_create}", exc_info=True)
        return create_empty_figure(map_title, final_height, f"Error during map generation: {e_map_create}")

    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0)) # Tight margins for maps
    logger.info(f"({log_prefix}) Choropleth map '{map_title}' created successfully.")
    return fig
