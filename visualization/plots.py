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
import html 
import re 

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric 
    from .ui_elements import get_theme_color 
except ImportError as e:
    logging.basicConfig(level=logging.INFO) 
    logger_init = logging.getLogger(__name__) 
    logger_init.error(f"CRITICAL IMPORT ERROR in plots.py: {e}. Using fallback settings/colors. Ensure config.py, helpers.py and ui_elements.py are correct.")
    class FallbackPlotSettings:
        THEME_FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
        COLOR_TEXT_DARK = "#333333"; COLOR_BACKGROUND_CONTENT = "#FFFFFF"; COLOR_BACKGROUND_PAGE = "#F0F2F6";
        COLOR_ACTION_PRIMARY = "#007BFF"; COLOR_RISK_LOW = "#2ECC71"; COLOR_RISK_MODERATE = "#FFA500";
        COLOR_ACCENT_BRIGHT = "#FFC107"; COLOR_ACTION_SECONDARY = "#6C757D";
        COLOR_BORDER_LIGHT = "#E0E0E0"; COLOR_BORDER_MEDIUM = "#BDBDBD"; COLOR_TEXT_HEADINGS_MAIN = "#111111";
        COLOR_BACKGROUND_CONTENT_TRANSPARENT = "rgba(255,255,255,0.8)"; COLOR_TEXT_MUTED = '#757575';
        COLOR_RISK_HIGH = "#FF0000"; COLOR_BACKGROUND_WHITE = "#FFFFFF";
        LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {}
        MAPBOX_STYLE_WEB = "carto-positron"; MAP_DEFAULT_CENTER_LAT = 0.0; MAP_DEFAULT_CENTER_LON = 0.0; MAP_DEFAULT_ZOOM_LEVEL = 1;
        WEB_PLOT_DEFAULT_HEIGHT = 400; WEB_PLOT_COMPACT_HEIGHT = 350; WEB_MAP_DEFAULT_HEIGHT = 600;
    settings = FallbackPlotSettings()
    if 'get_theme_color' not in globals():
        def get_theme_color(name_or_idx, category="general", fallback_color_hex=None): # type: ignore
            return fallback_color_hex or getattr(settings, 'COLOR_TEXT_MUTED', "#CCCCCC") 
    if 'convert_to_numeric' not in globals():
        def convert_to_numeric(data_input, default_value=np.nan, target_type=None): # type: ignore
            return pd.to_numeric(data_input, errors='coerce').fillna(default_value)

    logger_init.warning("plots.py: Using fallback settings and/or basic helper functions due to an import error. Plot styles/functionality may differ.")

logger = logging.getLogger(__name__)

def _get_setting_or_default(attr_name: str, default_value: Any) -> Any:
    """Safely gets a setting attribute or returns a default."""
    return getattr(settings, attr_name, default_value)

MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = False
_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV = os.getenv("MAPBOX_ACCESS_TOKEN")
if _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV and _SENTINEL_MAPBOX_ACCESS_TOKEN_ENV.strip() and len(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV) > 20:
    try:
        px.set_mapbox_access_token(_SENTINEL_MAPBOX_ACCESS_TOKEN_ENV)
        MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG = True
        logger.info("Plotly: MAPBOX_ACCESS_TOKEN env var found and configured for Plotly Express.")
    except Exception as e_mapbox_token_setup:
        logger.error(f"Plotly: Error setting Mapbox token: {e_mapbox_token_setup}", exc_info=True)
else:
    logger.warning("Plotly: MAPBOX_ACCESS_TOKEN not found/invalid. Maps needing token may use open styles/fail.")

def set_sentinel_plotly_theme():
    theme_font_family = _get_setting_or_default('THEME_FONT_FAMILY', '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif')
    default_colorway = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    sentinel_colorway = [
        _get_setting_or_default('COLOR_ACTION_PRIMARY', default_colorway[0]),
        _get_setting_or_default('COLOR_RISK_LOW', default_colorway[1]),
        _get_setting_or_default('COLOR_RISK_MODERATE', default_colorway[2]),
        _get_setting_or_default('COLOR_ACCENT_BRIGHT', default_colorway[3]),
        _get_setting_or_default('COLOR_ACTION_SECONDARY', default_colorway[4]),
        get_theme_color(5, "general", fallback_color_hex=default_colorway[5]), 
        get_theme_color(6, "general", fallback_color_hex=default_colorway[6]), 
        get_theme_color(7, "general", fallback_color_hex=default_colorway[7])
    ]
    layout_template = go.Layout(
        font=dict(family=theme_font_family, size=11, color=_get_setting_or_default('COLOR_TEXT_DARK', "#333333")),
        paper_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT', "#FFFFFF"),
        plot_bgcolor=_get_setting_or_default('COLOR_BACKGROUND_PAGE', "#F0F2F6"),
        colorway=sentinel_colorway,
        xaxis=dict(gridcolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0"), linecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"), zerolinecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"), zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True),
        yaxis=dict(gridcolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0"), linecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"), zerolinecolor=_get_setting_or_default('COLOR_BORDER_MEDIUM', "#BDBDBD"), zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True),
        title=dict(font=dict(family=theme_font_family, size=16, color=_get_setting_or_default('COLOR_TEXT_HEADINGS_MAIN', "#111111")), x=0.05, xanchor='left', y=0.95, yanchor='top', pad=dict(t=20, b=10)),
        legend=dict(bgcolor=_get_setting_or_default('COLOR_BACKGROUND_CONTENT_TRANSPARENT', "rgba(255,255,255,0.8)"), bordercolor=_get_setting_or_default('COLOR_BORDER_LIGHT', "#E0E0E0"), borderwidth=0.5, orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=10),
        margin=dict(l=60, r=20, t=80, b=60) 
    )
    mapbox_style_setting = _get_setting_or_default('MAPBOX_STYLE_WEB', "carto-positron")
    effective_mapbox_style = mapbox_style_setting
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_map_styles = {"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"}
        if mapbox_style_setting.lower() not in open_map_styles:
            effective_mapbox_style = "carto-positron" 
            logger.info(f"Plotly Theme: Mapbox token not set and style '{mapbox_style_setting}' requires one. Defaulting theme map style to '{effective_mapbox_style}'.")
    layout_template.mapbox = dict(style=effective_mapbox_style, center=dict(lat=_get_setting_or_default('MAP_DEFAULT_CENTER_LAT', 0), lon=_get_setting_or_default('MAP_DEFAULT_CENTER_LON', 0)), zoom=_get_setting_or_default('MAP_DEFAULT_ZOOM_LEVEL', 1))
    pio.templates["sentinel_web_theme_custom"] = go.layout.Template(layout=layout_template)
    pio.templates.default = "plotly+sentinel_web_theme_custom" 
    logger.info("Custom Plotly theme 'sentinel_web_theme_custom' configured and applied as default.")
try:
    set_sentinel_plotly_theme()
except Exception as e_theme_setup:
    logger.error(f"Failed to set custom Plotly theme: {e_theme_setup}", exc_info=True)
    pio.templates.default = "plotly" 
    logger.warning("Fell back to default 'plotly' theme due to error in custom theme setup.")

def create_empty_figure(chart_title: str, height: Optional[int] = None, message_text: str = "No data available to display for the current selection.") -> go.Figure:
    final_height = height or _get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 400)
    fig = go.Figure()
    fig.update_layout(
        title_text=html.escape(chart_title), height=final_height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=html.escape(message_text), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=12, color=_get_setting_or_default('COLOR_TEXT_MUTED', '#757575')))],
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
    log_prefix_plot = f"PlotLine/{html.escape(chart_title[:30])}" 

    if not isinstance(data_series, pd.Series) or data_series.empty:
        return create_empty_figure(chart_title, final_height, "No data provided for this trend line.")

    series_plot = data_series.copy() 
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(series_plot.index):
            series_plot.index = pd.to_datetime(series_plot.index, errors='coerce')
        
        series_plot = series_plot[pd.notna(series_plot.index)] 
        if series_plot.empty:
            return create_empty_figure(chart_title, final_height, "Invalid date/time values in data.")
        
        if y_values_are_counts:
            series_plot = convert_to_numeric(series_plot, default_value=0, target_type=int)
        else:
            series_plot = convert_to_numeric(series_plot, default_value=np.nan, target_type=float)
            series_plot.dropna(inplace=True) 
        
    except Exception as e_prep:
        logger.error(f"({log_prefix_plot}) CRITICAL ERROR during data series preparation: {e_prep}", exc_info=True)
        return create_empty_figure(chart_title, final_height, "Critical error preparing data for line chart.")

    if series_plot.empty: 
        return create_empty_figure(chart_title, final_height, "No valid data points for this trend after cleaning.")

    fig = go.Figure()
    actual_line_color = line_color_hex or get_theme_color(0, "general", fallback_color_hex=_get_setting_or_default('COLOR_ACTION_PRIMARY', "#1E88E5"))
    y_hover_format_str = 'd' if y_values_are_counts else ',.1f' 
    trace_name = html.escape(y_axis_label if y_axis_label and y_axis_label.strip() else (str(series_plot.name) if series_plot.name else "Value"))
    
    fig.add_trace(go.Scatter(
        x=series_plot.index, y=series_plot.values, mode="lines+markers",
        name=trace_name,
        line=dict(color=actual_line_color, width=2), 
        marker=dict(size=5, symbol='circle'),
        hovertemplate=f'<b>Date</b>: %{{x|{date_format_hover}}}<br><b>{trace_name}</b>: %{{y:{y_hover_format_str}}}<extra></extra>'
    ))
    
    yaxis_final_config = dict(title_text=html.escape(y_axis_label))
    if y_values_are_counts: 
        yaxis_final_config['tickformat'] = 'd' 
    
    fig.update_layout(title_text=html.escape(chart_title), yaxis=yaxis_final_config, height=final_height)
    return fig

def plot_bar_chart(
    df_input: Optional[pd.DataFrame], x_col_name: str, y_col_name: str, chart_title: str,
    color_col_name: Optional[str] = None, bar_mode_style: str = 'group',
    y_values_are_counts_flag: bool = False, # CORRECTED: Added the missing parameter
    **kwargs 
) -> go.Figure:
    log_prefix_plot = f"PlotBar/{html.escape(chart_title[:30])}"
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return create_empty_figure(chart_title)
    
    df_plot = df_input.copy()
    
    try:
        fig = px.bar(df_plot, x=x_col_name, y=y_col_name, color=color_col_name, barmode=bar_mode_style, text_auto=True)
        
        # CORRECTED: Use the flag to apply integer-specific formatting for counts.
        if y_values_are_counts_flag:
            fig.update_traces(texttemplate='%{y:,d}', hovertemplate=f'<b>%{{x}}</b><br>Count: %{{y:,d}}<extra></extra>')
            fig.update_yaxes(tickformat='d')
        else:
            fig.update_traces(texttemplate='%{y:,.1f}', hovertemplate=f'<b>%{{x}}</b><br>Value: %{{y:,.1f}}<extra></extra>')

        fig.update_layout(
            title_text=f'<b>{html.escape(chart_title)}</b>',
            xaxis_title=x_col_name.replace('_', ' ').title(),
            yaxis_title=y_col_name.replace('_', ' ').title(),
            legend_title=color_col_name.replace('_', ' ').title() if color_col_name else None
        )
        return fig
    except Exception as e:
        logger.error(f"({log_prefix_plot}) Error creating bar chart: {e}", exc_info=True)
        return create_empty_figure(chart_title, message="Error generating chart.")

def plot_donut_chart(
    df_input: Optional[pd.DataFrame], labels_col_name: str, values_col_name: str, chart_title: str,
    chart_height: Optional[int] = None, custom_color_map: Optional[Dict[str, str]] = None,
    pull_slice_amount: float = 0.03, center_annotation_text: Optional[str] = None, 
    values_are_counts: bool = True
) -> go.Figure:
    final_height = chart_height or (_get_setting_or_default('WEB_PLOT_COMPACT_HEIGHT', 350) + 50)
    log_prefix_plot = f"PlotDonut/{html.escape(chart_title[:30])}"
    if not isinstance(df_input, pd.DataFrame) or df_input.empty or labels_col_name not in df_input.columns or values_col_name not in df_input.columns:
        return create_empty_figure(chart_title, final_height, "Missing data for donut chart.")
    df_plot = df_input.copy()
    try:
        df_plot[values_col_name] = convert_to_numeric(df_plot[values_col_name], default_value=0, target_type=int if values_are_counts else float)
        df_plot = df_plot[df_plot[values_col_name] > 1e-6].sort_values(by=values_col_name, ascending=False) 
        df_plot[labels_col_name] = df_plot[labels_col_name].astype(str).str.strip() 
    except Exception as e_prep_donut:
        logger.error(f"({log_prefix_plot}) Error preparing data: {e_prep_donut}", exc_info=True)
        return create_empty_figure(chart_title, final_height, "Error preparing data for donut chart.")
    if df_plot.empty: 
        return create_empty_figure(chart_title, final_height, "No positive data available for donut chart.")

    plot_colors_list = None; unique_labels_list = df_plot[labels_col_name].unique()
    if custom_color_map: plot_colors_list = [custom_color_map.get(str(lbl), get_theme_color(i % 8, "general")) for i, lbl in enumerate(unique_labels_list)]
    else:
        legacy_colors_map_donut = _get_setting_or_default('LEGACY_DISEASE_COLORS_WEB', {})
        if any(str(lbl) in legacy_colors_map_donut for lbl in unique_labels_list): plot_colors_list = [get_theme_color(str(lbl), "disease", get_theme_color(i % 8, "general")) for i, lbl in enumerate(unique_labels_list)]
        else: plot_colors_list = [get_theme_color(i % 8, "general") for i in range(len(unique_labels_list))]
    
    hover_value_fmt = 'd' if values_are_counts else '.2f'
    num_slices_to_pull_val = min(3, len(df_plot)) 
    pull_values_list = [pull_slice_amount if i < num_slices_to_pull_val else 0 for i in range(len(df_plot))]
    fig = go.Figure(data=[go.Pie(labels=df_plot[labels_col_name], values=df_plot[values_col_name], hole=0.58, pull=pull_values_list, textinfo='label+percent', insidetextorientation='radial', hoverinfo='label+value+percent', hovertemplate=f'<b>%{{label}}</b><br>Value: %{{value:{hover_value_fmt}}}<br>Percent: %{{percent}}<extra></extra>', marker=dict(colors=plot_colors_list, line=dict(color=get_theme_color("white", fallback_color_hex="#FFFFFF"), width=1.8)), sort=False)])
    fig_annotations_list = []
    if center_annotation_text and center_annotation_text.strip():
        fig_annotations_list.append(dict(text=html.escape(str(center_annotation_text)), x=0.5, y=0.5, font_size=14, showarrow=False, font_color=get_theme_color("text_dark", fallback_color_hex="#333333")))
    fig.update_layout(title_text=html.escape(chart_title), height=final_height, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.18, font_size=9, traceorder="normal"), annotations=fig_annotations_list if fig_annotations_list else None, margin=dict(l=20, r=120, t=60, b=20))
    return fig

def plot_heatmap(
    matrix_df_input: Optional[pd.DataFrame], chart_title: str, chart_height: Optional[int] = None,
    color_scale_name: str = 'RdBu_r', z_midpoint_val: Optional[float] = 0.0,
    show_cell_text: bool = True, text_format: str = '.2f', show_colorbar: bool = True
) -> go.Figure:
    final_height = chart_height or (_get_setting_or_default('WEB_PLOT_DEFAULT_HEIGHT', 450) + 80)
    log_prefix_plot = f"PlotHeatmap/{html.escape(chart_title[:30])}"
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty:
        return create_empty_figure(chart_title, final_height)
    df_matrix = matrix_df_input.copy().apply(pd.to_numeric, errors='coerce')
    if df_matrix.isnull().all().all():
        return create_empty_figure(chart_title, final_height, "All heatmap data is non-numeric or missing.")
    z_values = df_matrix.values; cell_text_content = None
    if show_cell_text: cell_text_content = np.vectorize(lambda x: f"{x:{text_format}}" if pd.notna(x) else '')(z_values)
    zmid_final_val = z_midpoint_val; valid_z_flat_vals = z_values[~np.isnan(z_values)]
    if len(valid_z_flat_vals) > 0 and ((valid_z_flat_vals >= 0).all() or (valid_z_flat_vals <= 0).all()): zmid_final_val = None 
    elif len(valid_z_flat_vals) == 0: zmid_final_val = None
    fig = go.Figure(data=[go.Heatmap(z=z_values, x=df_matrix.columns.astype(str).tolist(), y=df_matrix.index.astype(str).tolist(), colorscale=color_scale_name, zmid=zmid_final_val, text=cell_text_content, texttemplate="%{text}" if show_cell_text and cell_text_content is not None else '', hoverongaps=False, xgap=1, ygap=1, colorbar=dict(thickness=15, len=0.9, tickfont=dict(size=9), title=dict(text="Value", side="right", font=dict(size=10)), outlinewidth=0.5, outlinecolor=get_theme_color("border_medium", fallback_color_hex="#BDBDBD")) if show_colorbar else None)])
    x_tick_angle = -45 if len(df_matrix.columns) > 8 or max(len(str(c)) for c in df_matrix.columns) > 10 else 0
    fig.update_layout(title_text=html.escape(chart_title), height=final_height, xaxis=dict(showgrid=False, tickangle=x_tick_angle, side='bottom'), yaxis=dict(showgrid=False, autorange='reversed'), plot_bgcolor=get_theme_color("white", fallback_color_hex="#FFFFFF"))
    return fig

def plot_choropleth_map(
    map_data_df: Optional[pd.DataFrame], geojson_features: Optional[Union[Dict[str,Any], List[Dict[str,Any]]]],
    value_col_name: str, map_title: str, zone_id_geojson_prop: str = 'zone_id',
    zone_id_df_col: str = 'zone_id', color_scale_name: str = 'Viridis',
    hover_name_col: Optional[str] = None, hover_data_cols: Optional[List[str]] = None,
    map_height: Optional[int] = None, center_lat_val: Optional[float] = None,
    center_lon_val: Optional[float] = None, zoom_level_val: Optional[int] = None,
    mapbox_style_override: Optional[str] = None
) -> go.Figure:
    final_height = map_height or _get_setting_or_default('WEB_MAP_DEFAULT_HEIGHT', 600)
    log_prefix = f"ChoroplethMap/{html.escape(map_title[:30])}"
    if not isinstance(map_data_df, pd.DataFrame) or map_data_df.empty or value_col_name not in map_data_df.columns or zone_id_df_col not in map_data_df.columns:
        return create_empty_figure(map_title, final_height, "Map data is incomplete.")
    if not geojson_features:
        return create_empty_figure(map_title, final_height, "Geographic boundary data unavailable.")
    geojson_plotly_input = geojson_features
    if isinstance(geojson_features, list): geojson_plotly_input = {"type": "FeatureCollection", "features": geojson_features}
    elif not (isinstance(geojson_features, dict) and geojson_features.get("type") == "FeatureCollection"):
        return create_empty_figure(map_title, final_height, "Invalid geographic boundary data format.")
    df_map_plot = map_data_df.copy()
    try:
        df_map_plot[value_col_name] = convert_to_numeric(df_map_plot[value_col_name], default_value=np.nan) 
        df_map_plot[zone_id_df_col] = df_map_plot[zone_id_df_col].astype(str).str.strip()
        df_map_plot.dropna(subset=[value_col_name, zone_id_df_col], inplace=True)
    except Exception as e_prep_map:
        logger.error(f"({log_prefix}) Error preparing map data: {e_prep_map}", exc_info=True)
        return create_empty_figure(map_title, final_height, "Error preparing data for map.")
    if df_map_plot.empty: return create_empty_figure(map_title, final_height, f"No valid data for '{html.escape(value_col_name)}' to display on map after cleaning.")
    
    hover_name_final_val = zone_id_df_col 
    if hover_name_col and hover_name_col in df_map_plot.columns: hover_name_final_val = hover_name_col
    elif 'name' in df_map_plot.columns: hover_name_final_val = 'name'
    
    hover_data_cfg_dict: Dict[str, Union[bool, str]] = {} 
    if hover_data_cols:
        for h_col in hover_data_cols:
            if h_col in df_map_plot.columns: hover_data_cfg_dict[h_col] = True 
    if value_col_name not in hover_data_cfg_dict: hover_data_cfg_dict[value_col_name] = True 
    
    final_mapbox_style_val = mapbox_style_override
    if not final_mapbox_style_val:
        try: final_mapbox_style_val = pio.templates[pio.templates.default].layout.mapbox.style
        except AttributeError: final_mapbox_style_val = "carto-positron"; logger.warning(f"({log_prefix}) Default map style not in theme. Using 'carto-positron'.")
    if not MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG:
        open_styles_set = {"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"}
        if final_mapbox_style_val and final_mapbox_style_val.lower() not in open_styles_set:
            logger.warning(f"({log_prefix}) Mapbox token not available, style '{final_mapbox_style_val}' may require it. Switching to 'carto-positron'.")
            final_mapbox_style_val = "carto-positron"
    try:
        fig = px.choropleth_mapbox(df_map_plot, geojson=geojson_plotly_input, locations=zone_id_df_col, featureidkey=f"properties.{zone_id_geojson_prop}", color=value_col_name, color_continuous_scale=color_scale_name, hover_name=hover_name_final_val, hover_data=hover_data_cfg_dict if hover_data_cfg_dict else None, mapbox_style=final_mapbox_style_val, center=dict(lat=center_lat_val if center_lat_val is not None else _get_setting_or_default('MAP_DEFAULT_CENTER_LAT', 0), lon=center_lon_val if center_lon_val is not None else _get_setting_or_default('MAP_DEFAULT_CENTER_LON', 0)), zoom=zoom_level_val if zoom_level_val is not None else _get_setting_or_default('MAP_DEFAULT_ZOOM_LEVEL', 1), opacity=0.75, height=final_height, title=None)
    except Exception as e_map_px:
        logger.error(f"({log_prefix}) Error creating px.choropleth_mapbox: {e_map_px}", exc_info=True)
        return create_empty_figure(map_title, final_height, f"Map generation error: {e_map_px}")
    fig.update_layout(title_text=html.escape(map_title), margin=dict(l=0, r=0, t=50, b=0))
    logger.info(f"({log_prefix}) Choropleth map '{map_title}' created.")
    return fig
