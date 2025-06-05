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
import re # For string manipulations

from config import settings
from .ui_elements import get_theme_color # Import from local ui_elements

logger = logging.getLogger(__name__)

# --- Mapbox Token Handling ---
MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = False # Initialize module-level flag
try:
    _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV = os.getenv("MAPBOX_ACCESS_TOKEN")
    if _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV and \
       _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV.strip() and \
       len(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV) > 20: # Basic check
        
        # Plotly Express handles token internally if set in env var or via px.set_mapbox_access_token
        # We set it explicitly for px to ensure it's recognized by Plotly Express.
        px.set_mapbox_access_token(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV)
        MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = True
        logger.info("Plotly: MAPBOX_ACCESS_TOKEN environment variable found and configured for Plotly Express.")
    else:
        logger.warning("Plotly: MAPBOX_ACCESS_TOKEN not found or invalid. Maps needing a token may use open styles or fail.")
except Exception as e_mapbox_token_setup:
    logger.error(f"Plotly: Error setting Mapbox token for Plotly Express: {e_mapbox_token_setup}", exc_info=True)


# --- Plotly Theme Setup ---
def set_sentinel_plotly_theme():
    """Sets a custom Plotly theme ('sentinel_web_theme_custom') as the default."""
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
    
    sentinel_colorway = [
        settings.COLOR_ACTION_PRIMARY, settings.COLOR_RISK_LOW, settings.COLOR_RISK_MODERATE,
        settings.COLOR_ACCENT_BRIGHT, settings.COLOR_ACTION_SECONDARY,
        get_theme_color(5, color_category="general", fallback_color_hex="#00ACC1"), # Teal
        get_theme_color(6, color_category="general", fallback_color_hex="#5E35B1"), # Deep Purple
        get_theme_color(7, color_category="general", fallback_color_hex="#FF7043")  # Coral
    ]

    layout_settings = go.Layout(
        font=dict(family=theme_font_family, size=11, color=settings.COLOR_TEXT_DARK),
        paper_bgcolor=settings.COLOR_BACKGROUND_CONTENT,
        plot_bgcolor=settings.COLOR_BACKGROUND_PAGE,
        colorway=sentinel_colorway,
        xaxis=dict(gridcolor=settings.COLOR_BORDER_LIGHT, linecolor=settings.COLOR_BORDER_MEDIUM,
                   zerolinecolor=settings.COLOR_BORDER_MEDIUM, zerolinewidth=1,
                   title_font_size=12, tickfont_size=10, automargin=True),
        yaxis=dict(gridcolor=settings.COLOR_BORDER_LIGHT, linecolor=settings.COLOR_BORDER_MEDIUM,
                   zerolinecolor=settings.COLOR_BORDER_MEDIUM, zerolinewidth=1,
                   title_font_size=12, tickfont_size=10, automargin=True),
        title=dict(font=dict(family=theme_font_family, size=16, color=settings.COLOR_TEXT_HEADINGS_MAIN),
                   x=0.05, xanchor='left', y=0.95, yanchor='top', pad=dict(t=20, b=10)),
        legend=dict(bgcolor=settings.COLOR_BACKGROUND_CONTENT_TRANSPARENT,
                    bordercolor=settings.COLOR_BORDER_LIGHT, borderwidth=0.5, orientation='h',
                    yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=10),
        margin=dict(l=60, r=20, t=80, b=60)
    )

    # Determine effective Mapbox style for the theme
    effective_mapbox_style = settings.MAPBOX_STYLE_WEB
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_styles = ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]
        if settings.MAPBOX_STYLE_WEB.lower() not in open_styles:
            effective_mapbox_style = "carto-positron" # Default open fallback
            logger.info(f"Plotly Theme: Mapbox token not set, theme map style defaulting to '{effective_mapbox_style}'.")
    
    layout_settings.mapbox = dict(
        style=effective_mapbox_style,
        center=dict(lat=settings.MAP_DEFAULT_CENTER_LAT, lon=settings.MAP_DEFAULT_CENTER_LON),
        zoom=settings.MAP_DEFAULT_ZOOM_LEVEL
    )

    pio.templates["sentinel_web_theme_custom"] = go.layout.Template(layout=layout_settings)
    pio.templates.default = "plotly+sentinel_web_theme_custom"
    logger.info("Custom Plotly theme 'sentinel_web_theme_custom' applied.")

set_sentinel_plotly_theme() # Apply theme on module import


# --- Plotting Functions ---

def create_empty_figure(
    chart_title: str,
    height: Optional[int] = None,
    message_text: str = "No data available to display for the current selection."
) -> go.Figure:
    """Creates a styled empty Plotly figure with a message."""
    final_height = height or settings.WEB_PLOT_DEFAULT_HEIGHT
    fig = go.Figure()
    fig.update_layout(
        title_text=chart_title, height=final_height,
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[dict(text=message_text, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=12, color=settings.COLOR_TEXT_MUTED))],
        paper_bgcolor=settings.COLOR_BACKGROUND_CONTENT, # Ensure empty plot matches theme
        plot_bgcolor=settings.COLOR_BACKGROUND_PAGE
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
    final_height = chart_height or settings.WEB_PLOT_COMPACT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title, final_height, "No data for line chart.")

    series_plot = data_series.copy()
    if not pd.api.types.is_datetime64_any_dtype(series_plot.index):
        series_plot.index = pd.to_datetime(series_plot.index, errors='coerce')
    series_plot = series_plot.dropna(axis=0, how='all') # Drop rows where index is NaT
    series_plot = convert_to_numeric(series_plot, default_value=np.nan).dropna()

    if series_plot.empty: return create_empty_figure(chart_title, final_height, "No valid data points after cleaning.")

    fig = go.Figure()
    actual_line_color = line_color_hex or get_theme_color(0, "general")
    y_hover_fmt = 'd' if y_values_are_counts else ',.1f'
    
    fig.add_trace(go.Scatter(
        x=series_plot.index, y=series_plot.values, mode="lines+markers",
        name=y_axis_label if y_axis_label else (series_plot.name or "Value"),
        line=dict(color=actual_line_color, width=2), marker=dict(size=5, symbol='circle'),
        hovertemplate=f'<b>Date</b>: %{{x|{date_format_hover}}}<br><b>{y_axis_label}</b>: %{{y:{y_hover_fmt}}}<extra></extra>'
    ))

    if show_confidence_interval and isinstance(lower_ci_series, pd.Series) and isinstance(upper_ci_series, pd.Series):
        # Ensure CI series have datetime index and numeric values, align with main series
        l_ci = lower_ci_series.copy(); l_ci.index = pd.to_datetime(l_ci.index, errors='coerce'); l_ci = convert_to_numeric(l_ci, np.nan)
        u_ci = upper_ci_series.copy(); u_ci.index = pd.to_datetime(u_ci.index, errors='coerce'); u_ci = convert_to_numeric(u_ci, np.nan)
        
        # Align all three series by common valid datetime index
        aligned_df = pd.concat([series_plot.rename("value"), l_ci.rename("lower"), u_ci.rename("upper")], axis=1).dropna()
        if not aligned_df.empty and (aligned_df["upper"] >= aligned_df["lower"]).all():
            fill_rgba = px.colors.hex_to_rgb(actual_line_color)
            fill_color = f"rgba({fill_rgba[0]},{fill_rgba[1]},{fill_rgba[2]},0.15)"
            fig.add_trace(go.Scatter(
                x=list(aligned_df.index) + list(aligned_df.index[::-1]),
                y=list(aligned_df["upper"]) + list(aligned_df["lower"][::-1]),
                fill="toself", fillcolor=fill_color, line=dict(width=0), name="Conf. Interval", hoverinfo="skip"
            ))

    if target_ref_line_val is not None:
        ref_label = target_ref_label_text or f"Target: {target_ref_line_val:,.2f}"
        fig.add_hline(y=target_ref_line_val, line_dash="dash", line_color=get_theme_color("risk_moderate"), line_width=1.2,
                      annotation_text=ref_label, annotation_position="bottom right",
                      annotation_font_size=9, annotation_font_color=get_theme_color("text_muted"))

    if show_anomalies_flag and len(series_plot.dropna()) > 7 and series_plot.nunique() > 2:
        q1, q3 = series_plot.quantile(0.25), series_plot.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 1e-6:
            upper_b, lower_b = q3 + anomaly_iqr_factor * iqr, q1 - anomaly_iqr_factor * iqr
            anomalies = series_plot[(series_plot < lower_b) | (series_plot > upper_b)]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values, mode='markers',
                    marker=dict(color=get_theme_color("risk_high"), size=8, symbol='circle-open-dot', line=dict(width=1.5)),
                    name='Anomaly', hovertemplate=f'<b>Anomaly</b>: %{{x|{date_format_hover}}}<br><b>Value</b>: %{{y:{y_hover_fmt}}}<extra></extra>'
                ))

    x_title = series_plot.index.name if series_plot.index.name else "Date/Time"
    y_axis_cfg = dict(title_text=y_axis_label, rangemode='tozero' if y_values_are_counts and series_plot.min() >= 0 else 'normal')
    if y_values_are_counts: y_axis_cfg['tickformat'] = 'd'

    fig.update_layout(title_text=chart_title, xaxis_title=x_title, yaxis=y_axis_cfg, height=final_height, hovermode="x unified", legend=dict(traceorder='normal'))
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
    final_height = chart_height or settings.WEB_PLOT_DEFAULT_HEIGHT
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or x_col_name not in df_input.columns or y_col_name not in df_input.columns:
        return create_empty_figure(chart_title, final_height)

    df_plot = df_input.copy()
    df_plot[x_col_name] = df_plot[x_col_name].astype(str)
    df_plot[y_col_name] = convert_to_numeric(df_plot[y_col_name], default_value=0.0, target_type=int if y_values_are_counts_flag else float)
    df_plot.dropna(subset=[x_col_name, y_col_name], inplace=True)
    if df_plot.empty: return create_empty_figure(chart_title, final_height, f"No valid data for x='{x_col_name}', y='{y_col_name}'.")

    if sort_by_col and sort_by_col in df_plot.columns:
        try: df_plot.sort_values(by=sort_by_col, ascending=sort_ascending_flag, inplace=True, na_position='last')
        except: logger.warning(f"Bar chart '{chart_title}': Sorting by '{sort_by_col}' failed.")
    
    text_fmt = text_format_str if text_format_str else ('.0f' if y_values_are_counts_flag else '.1f')
    y_hover_fmt_bar = 'd' if y_values_are_counts_flag else text_fmt
    x_label = x_axis_label_text or x_col_name.replace('_', ' ').title()
    y_label = y_axis_label_text or y_col_name.replace('_', ' ').title()
    legend_title = color_col_name.replace('_', ' ').title() if color_col_name and color_col_name in df_plot.columns else None

    resolved_color_map = custom_color_map
    if not resolved_color_map and color_col_name and color_col_name in df_plot.columns and settings.LEGACY_DISEASE_COLORS_WEB:
        if any(str(val) in settings.LEGACY_DISEASE_COLORS_WEB for val in df_plot[color_col_name].dropna().unique()):
            resolved_color_map = { str(val): get_theme_color(str(val), "disease", get_theme_color(abs(hash(str(val))) % 8, "general"))
                                   for val in df_plot[color_col_name].dropna().unique() }
    
    try:
        fig = px.bar(df_plot, x=x_col_name if orientation_bar == 'v' else y_col_name,
                     y=y_col_name if orientation_bar == 'v' else x_col_name, title=chart_title,
                     color=color_col_name if color_col_name in df_plot.columns else None,
                     barmode=bar_mode_style, orientation=orientation_bar, height=final_height,
                     labels={y_col_name: y_label, x_col_name: x_label, (color_col_name or "_"): legend_title or ''},
                     text_auto=show_text_on_bars, color_discrete_map=resolved_color_map)
    except Exception as e_px: return create_empty_figure(chart_title, final_height, f"Error generating bar chart: {e_px}")

    text_template = f'%{{y:{text_fmt}}}' if orientation_bar == 'v' else f'%{{x:{text_fmt}}}'
    if isinstance(show_text_on_bars, str) and show_text_on_bars != 'auto': text_template = show_text_on_bars
    
    hover_x, hover_y = ('x', 'y') if orientation_bar == 'v' else ('y', 'x')
    hover_template = f'<b>{x_label if orientation_bar == "v" else y_label}</b>: %{{{hover_x}}}<br><b>{y_label if orientation_bar == "v" else x_label}</b>: %{{{hover_y}:{y_hover_fmt_bar}}}'
    custom_data_hover = []
    if color_col_name and color_col_name in df_plot.columns and legend_title:
        hover_template += f'<br><b>{legend_title}</b>: %{{customdata[0]}}'
        custom_data_hover.append(color_col_name)
    hover_template += '<extra></extra>'

    fig.update_traces(marker_line_width=0.8, marker_line_color='rgba(0,0,0,0.3)', textfont_size=9,
                      textposition='outside' if orientation_bar == 'h' else ('auto' if bar_mode_style != 'stack' else 'inside'),
                      cliponaxis=False, texttemplate=text_template if show_text_on_bars else None,
                      hovertemplate=hover_template, customdata=df_plot[custom_data_hover] if custom_data_hover else None)

    y_axis_cfg = {'title_text': y_label}; x_axis_cfg = {'title_text': x_label}
    val_axis_cfg = y_axis_cfg if orientation_bar == 'v' else x_axis_cfg
    cat_axis_cfg = x_axis_cfg if orientation_bar == 'v' else y_axis_cfg
    if y_values_are_counts_flag: val_axis_cfg.update({'tickformat': 'd', 'rangemode': 'tozero'})
    
    cat_col_order = x_col_name if orientation_bar == 'v' else y_col_name
    if sort_by_col == cat_col_order:
        cat_axis_cfg.update({'categoryorder': 'array', 'categoryarray': df_plot[cat_col_order].tolist()})
    elif orientation_bar == 'h' and (not sort_by_col or sort_by_col == y_col_name):
         cat_axis_cfg['categoryorder'] = 'total ascending' if sort_ascending_flag else 'total descending'
    fig.update_layout(yaxis=y_axis_cfg, xaxis=x_axis_cfg, uniformtext_minsize=7, uniformtext_mode='hide', legend_title_text=legend_title)
    return fig


def plot_donut_chart(
    df_input: Optional[pd.DataFrame], labels_col_name: str, values_col_name: str, chart_title: str,
    chart_height: Optional[int] = None, custom_color_map: Optional[Dict[str, str]] = None,
    pull_slice_amount: float = 0.03, center_annotation: Optional[str] = None, values_are_counts: bool = True
) -> go.Figure:
    final_height = chart_height or (settings.WEB_PLOT_COMPACT_HEIGHT + 50)
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or labels_col_name not in df_input.columns or values_col_name not in df_input.columns:
        return create_empty_figure(chart_title, final_height)

    df_plot = df_input.copy()
    df_plot[values_col_name] = convert_to_numeric(df_plot[values_col_name], 0.0, target_type=int if values_are_counts else float)
    df_plot = df_plot[df_plot[values_col_name] > 0].sort_values(by=values_col_name, ascending=False)
    df_plot[labels_col_name] = df_plot[labels_col_name].astype(str)
    if df_plot.empty: return create_empty_figure(chart_title, final_height, "No positive data for donut chart.")

    colors = None
    if custom_color_map: colors = [custom_color_map.get(str(lbl), get_theme_color(i, "general")) for i, lbl in enumerate(df_plot[labels_col_name])]
    elif settings.LEGACY_DISEASE_COLORS_WEB and any(str(lbl) in settings.LEGACY_DISEASE_COLORS_WEB for lbl in df_plot[labels_col_name]):
        colors = [get_theme_color(str(lbl), "disease", get_theme_color(i, "general")) for i, lbl in enumerate(df_plot[labels_col_name])]
    else: colors = [get_theme_color(i, "general") for i in range(len(df_plot[labels_col_name]))]
    
    hover_val_fmt = 'd' if values_are_counts else '.2f'
    fig = go.Figure(data=[go.Pie(
        labels=df_plot[labels_col_name], values=df_plot[values_col_name], hole=0.58,
        pull=[pull_slice_amount if i < min(3, len(df_plot)) else 0 for i in range(len(df_plot))],
        textinfo='label+percent', insidetextorientation='radial', hoverinfo='label+value+percent',
        hovertemplate=f'<b>%{{label}}</b><br>Value: %{{value:{hover_val_fmt}}}<br>Percent: %{{percent}}<extra></extra>',
        marker=dict(colors=colors, line=dict(color=get_theme_color("white"), width=1.8)), sort=False
    )])
    
    annotations = [dict(text=str(center_annotation), x=0.5, y=0.5, font_size=14, showarrow=False, font_color=get_theme_color("text_dark"))] if center_annotation else None
    fig.update_layout(title_text=chart_title, height=final_height, showlegend=True,
                      legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.18, font_size=9, traceorder="normal"),
                      annotations=annotations, margin=dict(l=20, r=120, t=60, b=20))
    return fig


def plot_heatmap(
    matrix_df_input: Optional[pd.DataFrame], chart_title: str, chart_height: Optional[int] = None,
    color_scale_name: str = 'RdBu_r', z_midpoint_val: Optional[float] = 0.0,
    show_cell_text: bool = True, text_format: str = '.2f', show_colorbar: bool = True
) -> go.Figure:
    final_height = chart_height or (settings.WEB_PLOT_DEFAULT_HEIGHT + 80)
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty:
        return create_empty_figure(chart_title, final_height)

    df_matrix = matrix_df_input.copy().apply(pd.to_numeric, errors='coerce')
    if df_matrix.isnull().all().all():
        return create_empty_figure(chart_title, final_height, "All heatmap data is non-numeric or missing.")

    z_vals = df_matrix.values
    cell_text = np.vectorize(lambda x: f"{x:{text_format}}" if pd.notna(x) else '')(z_vals) if show_cell_text else None
    
    zmid_final = z_midpoint_val
    z_flat_valid = z_vals[~np.isnan(z_vals)]
    if len(z_flat_valid) > 0 and (np.all(z_flat_valid >= 0) or np.all(z_flat_valid <= 0)):
        zmid_final = None # Auto-midpoint if data doesn't cross zero
    elif len(z_flat_valid) == 0: zmid_final = None # All NaNs

    fig = go.Figure(data=[go.Heatmap(
        z=z_vals, x=df_matrix.columns.astype(str).tolist(), y=df_matrix.index.astype(str).tolist(),
        colorscale=color_scale_name, zmid=zmid_final, text=cell_text,
        texttemplate="%{text}" if show_cell_text and cell_text is not None else '',
        hoverongaps=False, xgap=1, ygap=1,
        colorbar=dict(thickness=15, len=0.9, tickfont=dict(size=9),
                      title=dict(text="Value", side="right", font=dict(size=10)),
                      outlinewidth=0.5, outlinecolor=get_theme_color("border_medium")) if show_colorbar else None
    )])
    
    x_tick_angle = -45 if len(df_matrix.columns) > 8 or max(len(str(c)) for c in df_matrix.columns) > 10 else 0
    fig.update_layout(title_text=chart_title, height=final_height,
                      xaxis=dict(showgrid=False, tickangle=x_tick_angle, side='bottom'),
                      yaxis=dict(showgrid=False, autorange='reversed'), plot_bgcolor=get_theme_color("white"))
    return fig


def plot_choropleth_map(
    map_data_df: Optional[pd.DataFrame], geojson_features: Optional[List[Dict[str,Any]]],
    value_col_name: str, map_title: str, zone_id_geojson_prop: str = 'zone_id',
    zone_id_df_col: str = 'zone_id', color_scale_name: str = 'Viridis',
    hover_name_col: Optional[str] = None, hover_data_cols: Optional[List[str]] = None,
    map_height: Optional[int] = None, center_lat_val: Optional[float] = None,
    center_lon_val: Optional[float] = None, zoom_level_val: Optional[int] = None,
    mapbox_style_override: Optional[str] = None
) -> go.Figure:
    final_height = map_height or settings.WEB_MAP_DEFAULT_HEIGHT
    log_prefix = "ChoroplethMap"

    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or \
       value_col_name not in map_data_df.columns or zone_id_df_col not in map_data_df.columns:
        logger.warning(f"({log_prefix}) '{map_title}': Map DataFrame invalid or missing key columns.")
        return create_empty_figure(map_title, final_height, "Map data is incomplete.")
    if not geojson_features or not isinstance(geojson_features, list) or len(geojson_features) == 0:
        logger.warning(f"({log_prefix}) '{map_title}': GeoJSON features list empty/invalid.")
        return create_empty_figure(map_title, final_height, "Geographic boundary data unavailable.")

    df_map = map_data_df.copy()
    df_map[value_col_name] = convert_to_numeric(df_map[value_col_name], np.nan)
    df_map[zone_id_df_col] = df_map[zone_id_df_col].astype(str).str.strip()
    df_map.dropna(subset=[value_col_name, zone_id_df_col], inplace=True)
    if df_map.empty: return create_empty_figure(map_title, final_height, f"No valid data for '{value_col_name}' to display on map.")

    geojson_fc = {"type": "FeatureCollection", "features": geojson_features}
    hover_name = hover_name_col if hover_name_col and hover_name_col in df_map.columns else \
                 ('name' if 'name' in df_map.columns else zone_id_df_col)
    
    hover_data_cfg: Dict[str, bool] = {}
    if hover_data_cols:
        for h_col in hover_data_cols:
            if h_col in df_map.columns: hover_data_cfg[h_col] = True
    if value_col_name not in hover_data_cfg: hover_data_cfg[value_col_name] = True # Ensure value is in hover

    map_style = mapbox_style_override
    if not map_style:
        try: map_style = pio.templates[pio.templates.default].layout.mapbox.style
        except: map_style = "carto-positron"; logger.warning(f"({log_prefix}) Using fallback map style 'carto-positron'.")
            
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_styles = ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]
        if map_style and map_style.lower() not in open_styles:
            logger.warning(f"({log_prefix}) Map style '{map_style}' might need token. Switching to 'carto-positron'.")
            map_style = "carto-positron"

    try:
        fig = px.choropleth_mapbox(
            df_map, geojson=geojson_fc, locations=zone_id_df_col,
            featureidkey=f"properties.{zone_id_geojson_prop}", color=value_col_name,
            color_continuous_scale=color_scale_name, hover_name=hover_name,
            hover_data=hover_data_cfg, mapbox_style=map_style,
            center=dict(lat=center_lat_val or settings.MAP_DEFAULT_CENTER_LAT,
                        lon=center_lon_val or settings.MAP_DEFAULT_CENTER_LON),
            zoom=zoom_level_val or settings.MAP_DEFAULT_ZOOM_LEVEL,
            opacity=0.75, height=final_height, title=map_title
        )
    except Exception as e:
        logger.error(f"({log_prefix}) Error creating choropleth map '{map_title}': {e}", exc_info=True)
        return create_empty_figure(map_title, final_height, "Error during map generation.")

    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    logger.info(f"({log_prefix}) Choropleth map '{map_title}' created.")
    return fig
