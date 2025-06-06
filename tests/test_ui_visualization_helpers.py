# sentinel_project_root/tests/test_ui_visualization_helpers.py
# Pytest tests for UI and Plotting helpers in visualization module for Sentinel.

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from unittest.mock import patch, MagicMock # Added MagicMock for st.container
import html
import json
import os # For MAPBOX_TOKEN_SET_FLAG access from plots module
import hashlib # Added for hash test

from visualization.plots import (
    set_sentinel_plotly_theme, create_empty_figure, plot_choropleth_map,
    plot_annotated_line_chart, plot_bar_chart, plot_donut_chart, plot_heatmap,
    MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG as SUT_MAPBOX_TOKEN_FLAG # Import the flag from the SUT
)
from visualization.ui_elements import (
    get_theme_color, render_kpi_card, render_traffic_light_indicator,
    display_custom_styled_kpi_box
)
from config import settings

# Apply Sentinel theme once for all tests in this module
@pytest.fixture(scope="module", autouse=True)
def apply_sentinel_theme_for_viz_tests():
    set_sentinel_plotly_theme()

# --- Tests for Core Theming and Color Utilities ---
def test_get_theme_color_utility():
    assert get_theme_color("risk_high") == settings.COLOR_RISK_HIGH
    assert get_theme_color("action_primary") == settings.COLOR_ACTION_PRIMARY
    fallback_hex = "#ABC123"
    assert get_theme_color("unknown_color", "unknown_cat", fallback_hex) == fallback_hex
    
    legacy_colors = settings.LEGACY_DISEASE_COLORS_WEB
    if legacy_colors and "TB" in legacy_colors:
        assert get_theme_color("TB", "disease") == legacy_colors["TB"]
    else: # Test fallback for disease if not in legacy map
        assert isinstance(get_theme_color("TB", "disease", "#BADA55"), str)

    # Test general colorway (index-based access, uses predefined list in ui_elements)
    assert isinstance(get_theme_color(0, "general"), str)
    assert isinstance(get_theme_color(100, "general"), str) # Test modulo behavior

# --- Tests for HTML Component Renderers ---
@patch('visualization.ui_elements.st') # Patch st where it's used in ui_elements
def test_render_kpi_card_html_structure(mock_st_ui):
    mock_st_ui.markdown = MagicMock() # Mock the markdown function specifically
    mock_st_ui.container = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())) # Mock container as context manager

    title, value, icon, status, units, help_txt = "Active Cases", "120", "🦠", "MODERATE_CONCERN", "cases", "Total active cases."
    render_kpi_card(title=title, value_str=value, icon=icon, status_level=status, units=units, help_text=help_txt, container_border=False)
    
    mock_st_ui.markdown.assert_called_once()
    html_out, _ = mock_st_ui.markdown.call_args # KWARGS in _
    output_html = html_out[0]
    
    assert 'class="kpi-card status-moderate-concern"' in output_html
    assert f'<h3 class="kpi-title">{html.escape(title)}</h3>' in output_html
    assert f'<p class="kpi-value">{html.escape(value)} <span class=\'kpi-units\'>{html.escape(units)}</span></p>' in output_html
    assert f'title="{html.escape(help_txt)}"' in output_html
    assert html.escape(icon) in output_html

    mock_st_ui.markdown.reset_mock()
    render_kpi_card(title="Risk Change", value_str="-2.5", delta_value="-0.5%", delta_is_positive=False, status_level="GOOD_PERFORMANCE", units="pts")
    html_out_delta, _ = mock_st_ui.markdown.call_args
    output_delta_html = html_out_delta[0]
    assert 'class="kpi-card status-good-performance"' in output_delta_html
    assert f'<p class="kpi-delta negative">{html.escape("-0.5%")}</p>' in output_delta_html

@patch('visualization.ui_elements.st')
def test_render_traffic_light_indicator_html(mock_st_ui_traffic):
    mock_st_ui_traffic.markdown = MagicMock()
    mock_st_ui_traffic.container = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

    msg, status, details = "Power Status", "HIGH_RISK", "Backup generator active."
    render_traffic_light_indicator(message=msg, status_level=status, details_text=details)
    
    mock_st_ui_traffic.markdown.assert_called_once()
    html_out_tl, _ = mock_st_ui_traffic.markdown.call_args
    output_html_tl = html_out_tl[0]
    
    assert 'class="traffic-light-dot status-high-risk"' in output_html_tl
    assert f'<span class="traffic-light-message">{html.escape(msg)}</span>' in output_html_tl
    assert f'<span class="traffic-light-details">{html.escape(details)}</span>' in output_html_tl

@patch('visualization.ui_elements.st.markdown')
def test_display_custom_styled_kpi_box_html(mock_st_markdown_custom_kpi):
    label, value, sub_text, color = "Total Population", 12345, "Estimated", settings.COLOR_RISK_MODERATE
    display_custom_styled_kpi_box(label=label, value=value, sub_text=sub_text, highlight_edge_color=color)
    
    mock_st_markdown_custom_kpi.assert_called_once()
    html_out_custom, _ = mock_st_markdown_custom_kpi.call_args
    output_html_custom = html_out_custom[0]

    assert 'class="custom-markdown-kpi-box highlight-amber-edge"' in output_html_custom # amber for moderate
    assert f'<div class="custom-kpi-label-top-condition">{html.escape(label)}</div>' in output_html_custom
    assert f'<div class="custom-kpi-value-large">{html.escape("12,345")}</div>' in output_html_custom # Comma formatted
    assert f'<div class="custom-kpi-subtext-small">{html.escape(sub_text)}</div>' in output_html_custom


# --- Tests for Plotting Functionality ---
def test_create_empty_figure_properties():
    title, height, msg = "Empty Plot", 350, "No data found."
    fig = create_empty_figure(chart_title=title, height=height, message_text=msg)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == title
    assert fig.layout.height == height
    assert not fig.layout.xaxis.visible and not fig.layout.yaxis.visible
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == msg

def test_plot_annotated_line_chart(sample_series_data_plotting_fixture: pd.Series):
    title = "Sample Line Chart"
    fig = plot_annotated_line_chart(sample_series_data_plotting_fixture, chart_title=title)
    assert isinstance(fig, go.Figure) and fig.layout.title.text == title
    assert len(fig.data) >= 1 and fig.data[0].type == 'scatter' and \
           ('lines' in fig.data[0].mode.lower() if fig.data[0].mode else False)

    fig_empty = plot_annotated_line_chart(pd.Series(dtype=float), "Empty Line")
    assert "Empty Line" in fig_empty.layout.title.text and len(fig_empty.layout.annotations) > 0

def test_plot_bar_chart(sample_bar_df_plotting_fixture: pd.DataFrame):
    title = "Sample Bar Chart"
    fig = plot_bar_chart(sample_bar_df_plotting_fixture, x_col_name='category', y_col_name='value', chart_title=title, color_col_name='group')
    assert isinstance(fig, go.Figure) and fig.layout.title.text == title
    assert len(fig.data) > 0 and fig.data[0].type == 'bar'

    fig_empty = plot_bar_chart(pd.DataFrame(columns=['x','y']), x_col_name='x', y_col_name='y', chart_title="Empty Bar")
    assert "Empty Bar" in fig_empty.layout.title.text and len(fig_empty.layout.annotations) > 0

def test_plot_donut_chart(sample_donut_df_plotting_fixture: pd.DataFrame):
    title = "Sample Donut Chart"
    fig = plot_donut_chart(sample_donut_df_plotting_fixture, labels_col_name='label', values_col_name='count', chart_title=title)
    assert isinstance(fig, go.Figure) and fig.layout.title.text == title
    assert len(fig.data) == 1 and fig.data[0].type == 'pie' and fig.data[0].hole is not None and fig.data[0].hole > 0.4

    fig_empty = plot_donut_chart(pd.DataFrame(columns=['l','v']), labels_col_name='l', values_col_name='v', chart_title="Empty Donut")
    assert "Empty Donut" in fig_empty.layout.title.text and len(fig_empty.layout.annotations) > 0

def test_plot_heatmap(sample_heatmap_df_plotting_fixture: pd.DataFrame):
    if sample_heatmap_df_plotting_fixture.empty: pytest.skip("Sample heatmap DF empty.")
    title = "Sample Heatmap"
    fig = plot_heatmap(sample_heatmap_df_plotting_fixture, chart_title=title)
    assert isinstance(fig, go.Figure) and fig.layout.title.text == title
    assert len(fig.data) == 1 and fig.data[0].type == 'heatmap'

    fig_empty = plot_heatmap(pd.DataFrame(), chart_title="Empty Heatmap")
    assert "Empty Heatmap" in fig_empty.layout.title.text and len(fig_empty.layout.annotations) > 0

def test_plot_choropleth_map(sample_map_data_df_plotting_fixture: pd.DataFrame, sample_zone_data_df_main_fixture: pd.DataFrame):
    title = "Zonal Map"
    geojson_features: Optional[List[Dict[str, Any]]] = None
    if 'geometry_obj' in sample_zone_data_df_main_fixture.columns and sample_zone_data_df_main_fixture['geometry_obj'].notna().any():
        geojson_features = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}} 
                            for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r['geometry_obj']) and isinstance(r['geometry_obj'], dict)]
    if not geojson_features or not isinstance(sample_map_data_df_plotting_fixture, pd.DataFrame) or \
       sample_map_data_df_plotting_fixture.empty or 'avg_risk_score' not in sample_map_data_df_plotting_fixture.columns or \
       'zone_id' not in sample_map_data_df_plotting_fixture.columns:
        pytest.skip("Sample data not configured for choropleth map test.")

    fig = plot_choropleth_map(map_data_df=sample_map_data_df_plotting_fixture, geojson_features=geojson_features,
                              value_col_name='avg_risk_score', map_title=title, zone_id_df_col='zone_id', zone_id_geojson_prop='zone_id')
    assert isinstance(fig, go.Figure) and fig.layout.title.text == title
    assert fig.layout.mapbox.style is not None
    assert len(fig.data) >= 1 and fig.data[0].type == 'choroplethmapbox'

    fig_empty_data = plot_choropleth_map(pd.DataFrame(columns=['zone_id', 'val']), geojson_features, 'val', "Empty Data Map", 'zone_id')
    assert "Empty Data Map" in fig_empty_data.layout.title.text and len(fig_empty_data.layout.annotations) > 0
    
    fig_empty_geojson = plot_choropleth_map(sample_map_data_df_plotting_fixture, [], 'avg_risk_score', "Empty GeoJSON Map", 'zone_id')
    assert "Empty GeoJSON Map" in fig_empty_geojson.layout.title.text and len(fig_empty_geojson.layout.annotations) > 0

@patch('visualization.plots.MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG', False)
def test_map_no_token_override_fallback(sample_map_data_df_plotting_fixture: pd.DataFrame, sample_zone_data_df_main_fixture: pd.DataFrame):
    geojson_features: Optional[List[Dict[str, Any]]] = None
    if 'geometry_obj' in sample_zone_data_df_main_fixture.columns and sample_zone_data_df_main_fixture['geometry_obj'].notna().any():
        geojson_features = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}} 
                            for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r['geometry_obj']) and isinstance(r['geometry_obj'], dict)]
    if not geojson_features or not isinstance(sample_map_data_df_plotting_fixture, pd.DataFrame) or sample_map_data_df_plotting_fixture.empty or \
       'avg_risk_score' not in sample_map_data_df_plotting_fixture.columns or 'zone_id' not in sample_map_data_df_plotting_fixture.columns:
        pytest.skip("Sample data not configured for no-token map override style test.")

    fig = plot_choropleth_map(map_data_df=sample_map_data_df_plotting_fixture, geojson_features=geojson_features,
                              value_col_name='avg_risk_score', map_title="No Token Fallback", zone_id_df_col='zone_id',
                              mapbox_style_override="mapbox://styles/mapbox/streets-v11") # Token-requiring style
    assert fig.layout.mapbox.style.lower() in ["carto-positron", "open-street-map"]

@patch('visualization.plots.MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG', False)
def test_map_no_token_theme_fallback(sample_map_data_df_plotting_fixture: pd.DataFrame, sample_zone_data_df_main_fixture: pd.DataFrame, monkeypatch):
    geojson_features: Optional[List[Dict[str, Any]]] = None
    if 'geometry_obj' in sample_zone_data_df_main_fixture.columns and sample_zone_data_df_main_fixture['geometry_obj'].notna().any():
        geojson_features = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}} 
                            for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r['geometry_obj']) and isinstance(r['geometry_obj'], dict)]
    if not geojson_features or not isinstance(sample_map_data_df_plotting_fixture, pd.DataFrame) or sample_map_data_df_plotting_fixture.empty or \
       'avg_risk_score' not in sample_map_data_df_plotting_fixture.columns or 'zone_id' not in sample_map_data_df_plotting_fixture.columns:
        pytest.skip("Sample data not configured for no-token theme map style test.")

    monkeypatch.setattr(settings, 'MAPBOX_STYLE_WEB', "mapbox://styles/mapbox/satellite-v9") # Token-requiring
    set_sentinel_plotly_theme() # Re-apply with mocked flag and new setting
    
    fig = plot_choropleth_map(map_data_df=sample_map_data_df_plotting_fixture, geojson_features=geojson_features,
                              value_col_name='avg_risk_score', map_title="No Token Theme Fallback", zone_id_df_col='zone_id',
                              mapbox_style_override=None) # Use theme default
    
    theme_map_style = pio.templates[pio.templates.default].layout.mapbox.style
    assert theme_map_style.lower() in ["carto-positron", "open-street-map"]
    assert fig.layout.mapbox.style == theme_map_style
