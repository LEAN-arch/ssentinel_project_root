# sentinel_project_root/tests/test_ui_visualization_helpers.py
# Pytest tests for UI and Plotting helpers in visualization module for Sentinel.
# Path reflects old structure, tests adapted for new module structure.

import pytest
import pandas as pd
# GeoPandas import removed. Map tests will use DataFrames and GeoJSON features list.
import plotly.graph_objects as go
import plotly.io as pio # For accessing Plotly theme properties
from unittest.mock import patch # For mocking st.markdown
import html # For checking escaped HTML strings
import json # For GeoJSON feature list

# Import functions and constants from the new 'visualization' module
from visualization.plots import (
    set_sentinel_plotly_theme,
    create_empty_figure,
    plot_choropleth_map, # This now takes GeoJSON features list
    plot_annotated_line_chart,
    plot_bar_chart,
    plot_donut_chart,
    plot_heatmap
)
from visualization.ui_elements import (
    get_theme_color,
    render_kpi_card,
    render_traffic_light_indicator
)
# Import MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG from the SUT (plots.py) to check its state
from visualization.plots import MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG as SUT_MAPBOX_TOKEN_FLAG_IN_PLOTS

from config import settings # Use new settings module

# Fixtures (e.g., sample_series_data_plotting_fixture, sample_map_data_df_plotting_fixture, etc.)
# are sourced from conftest.py. sample_map_data_df_plotting_fixture now returns a DataFrame.

# Apply the Sentinel web theme once for all tests in this module.
# This ensures that get_theme_color and plot functions use the correct theme defaults.
@pytest.fixture(scope="module", autouse=True)
def apply_sentinel_theme_for_visualization_tests(): # Renamed fixture
    set_sentinel_plotly_theme()

# --- Tests for Core Theming and Color Utilities (visualization.ui_elements) ---
def test_get_theme_color_utility_specifics(): # Renamed test
    assert get_theme_color("risk_high") == settings.COLOR_RISK_HIGH, "Risk high color from settings mismatch."
    assert get_theme_color("action_primary") == settings.COLOR_ACTION_PRIMARY, "Action primary color from settings mismatch."
    assert get_theme_color("positive_delta") == settings.COLOR_POSITIVE_DELTA, "Positive delta color from settings mismatch."
    
    custom_fallback_hex_test = "#FEDCBA" # Unique fallback for this test
    assert get_theme_color("a_totally_unknown_color_name", color_category="unknown_cat", fallback_color_hex=custom_fallback_hex_test) == custom_fallback_hex_test, \
        "Fallback color was not used correctly for an unknown color name/category."
    
    # Test LEGACY_DISEASE_COLORS_WEB from settings
    legacy_disease_colors_test = getattr(settings, 'LEGACY_DISEASE_COLORS_WEB', {})
    if legacy_disease_colors_test and "TB" in legacy_disease_colors_test:
        assert get_theme_color("TB", color_category="disease") == legacy_disease_colors_test["TB"], \
            "Disease color for 'TB' (from legacy map in settings) mismatch."
    else:
        # If "TB" not in legacy colors, it should still return a string (either Plotly theme default or final fallback)
        assert isinstance(get_theme_color("TB", color_category="disease", fallback_color_hex="#BADA55"), str), \
            "Disease color for 'TB' (when not in specific legacy map) did not return a string type."

    # Test getting a color from the currently set default Plotly theme's colorway (if Plotly available to get_theme_color)
    # This part of get_theme_color was made a placeholder to avoid direct pio import in ui_elements.
    # If it were active, this test would be:
    # active_template_test = pio.templates.get(pio.templates.default)
    # if active_template_test and hasattr(active_template_test.layout, 'colorway') and active_template_test.layout.colorway:
    #     expected_colorway_first_test = active_template_test.layout.colorway[0]
    #     assert get_theme_color(0, color_category="general") == expected_colorway_first_test, \
    #            "General theme color (index 0 from Plotly colorway) mismatch."
    # else:
    #     pytest.fail("Could not access colorway from default Plotly template. Theme setup in plots.py might be an issue.")
    # For now, test that it falls back gracefully if Plotly colorway access isn't there.
    assert isinstance(get_theme_color(0, color_category="general", fallback_color_hex="#ABCDEF"), str), \
        "get_theme_color (index 0, general) did not return a string."


# --- Tests for HTML Component Renderers (visualization.ui_elements) ---
@patch('visualization.ui_elements.st.markdown') # Patch where st.markdown is called in ui_elements
def test_render_kpi_card_html_structure_check(mock_st_markdown_render_kpi): # Renamed
    kpi_title_test = "Active Cases Today"
    kpi_value_test = "125"
    kpi_icon_test = "🦠"
    kpi_status_test = "MODERATE_CONCERN" # Pythonic, maps to kebab-case CSS
    kpi_units_test = "cases"
    kpi_help_test = "Total active cases identified based on recent diagnostics."
    
    render_kpi_card(
        title=kpi_title_test, value_str=kpi_value_test, icon=kpi_icon_test, 
        status_level=kpi_status_test, units=kpi_units_test, help_text=kpi_help_test,
        container_border=False # Test without outer st.container
    )
    mock_st_markdown_render_kpi.assert_called_once()
    html_args_kpi_test, _ = mock_st_markdown_render_kpi.call_args
    output_html_kpi_test = html_args_kpi_test[0]
    
    assert 'class="kpi-card status-moderate-concern"' in output_html_kpi_test, "KPI card status CSS class incorrect in output."
    assert f'<h3 class="kpi-title">{html.escape(kpi_title_test)}</h3>' in output_html_kpi_test, "KPI title rendering incorrect in output."
    assert f'<p class="kpi-value">{html.escape(kpi_value_test)} <span class=\'kpi-units\'>{html.escape(kpi_units_test)}</span></p>' in output_html_kpi_test, \
        "KPI value and units rendering incorrect in output."
    assert f'title="{html.escape(kpi_help_test)}"' in output_html_kpi_test, "KPI help text (tooltip attribute) incorrect in output."
    assert html.escape(kpi_icon_test) in output_html_kpi_test, "KPI icon rendering incorrect in output."

    mock_st_markdown_render_kpi.reset_mock()
    render_kpi_card(title="Avg. Risk Score Change", value_str="-3.5", delta_value="-0.8%", delta_is_positive=False, status_level="GOOD_PERFORMANCE", units="points")
    html_args_delta_kpi, _ = mock_st_markdown_render_kpi.call_args
    output_html_delta_kpi_test = html_args_delta_kpi[0]
    assert 'class="kpi-card status-good-performance"' in output_html_delta_kpi_test, "Delta KPI status class incorrect in output."
    assert f'<p class="kpi-delta negative">{html.escape("-0.8%")}</p>' in output_html_delta_kpi_test, "Negative delta rendering incorrect in output."

@patch('visualization.ui_elements.st.markdown') # Patch in ui_elements
def test_render_traffic_light_indicator_html_structure_check(mock_st_markdown_render_traffic): # Renamed
    tl_message_test = "Facility Power Status"
    tl_status_test = "HIGH_RISK" # Pythonic, maps to CSS status-high-risk
    tl_details_test = "Main power offline. Backup generator active."

    render_traffic_light_indicator(message=tl_message_test, status_level=tl_status_test, details_text=tl_details_test)
    mock_st_markdown_render_traffic.assert_called_once()
    html_args_tl_test, _ = mock_st_markdown_render_traffic.call_args
    output_html_tl_test = html_args_tl_test[0]
    
    assert 'class="traffic-light-dot status-high-risk"' in output_html_tl_test, "Traffic light dot CSS class incorrect in output."
    assert f'<span class="traffic-light-message">{html.escape(tl_message_test)}</span>' in output_html_tl_test, "Traffic light message rendering incorrect."
    assert f'<span class="traffic-light-details">{html.escape(tl_details_test)}</span>' in output_html_tl_test, "Traffic light details rendering incorrect."


# --- Tests for Plotting Functionality (visualization.plots) ---
def test_create_empty_figure_has_correct_properties(): # Renamed test
    chart_title_empty_test = "Empty Dataset Visualization"
    height_empty_test = 380
    message_empty_test = "No data found for the selected criteria to plot this chart."
    fig_empty_test = create_empty_figure(chart_title=chart_title_empty_test, height=height_empty_test, message_text=message_empty_test)
    
    assert isinstance(fig_empty_test, go.Figure), "create_empty_figure did not return a Plotly Figure object."
    assert fig_empty_test.layout.title.text == chart_title_empty_test, "Empty plot title text incorrect." # Message is now an annotation
    assert fig_empty_test.layout.height == height_empty_test, "Empty plot height incorrect."
    assert not fig_empty_test.layout.xaxis.visible and not fig_empty_test.layout.yaxis.visible, "Axes should be invisible for an empty plot figure."
    assert len(fig_empty_test.layout.annotations) == 1 and fig_empty_test.layout.annotations[0].text == message_empty_test, \
        "Empty plot message annotation incorrect or missing."

def test_plot_annotated_line_chart_basic_functionality(sample_series_data_plotting_fixture: pd.Series): # Renamed
    chart_title_line_test = "Sample Time Series Data"
    fig_line_test = plot_annotated_line_chart(sample_series_data_plotting_fixture, chart_title_line_test)
    assert isinstance(fig_line_test, go.Figure), "plot_annotated_line_chart did not return a Plotly Figure."
    assert fig_line_test.layout.title.text == chart_title_line_test, "Line chart title incorrect."
    assert len(fig_line_test.data) >= 1 and fig_line_test.data[0].type == 'scatter' and \
           ('lines' in fig_line_test.data[0].mode.lower() if fig_line_test.data[0].mode else False), \
        "Line chart trace configuration (type or mode) incorrect."

    fig_line_empty_test = plot_annotated_line_chart(pd.Series(dtype=float), "Empty Series Data Line Chart") # Pass empty series
    assert "Empty Series Data Line Chart" in fig_line_empty_test.layout.title.text and \
           len(fig_line_empty_test.layout.annotations) > 0 and \
           "No data for line chart." in fig_line_empty_test.layout.annotations[0].text, \
        "Empty line chart did not display correct title/message annotation."


def test_plot_bar_chart_basic_functionality(sample_bar_df_plotting_fixture: pd.DataFrame): # Renamed
    chart_title_bar_test = "Sample Bar Chart Distribution"
    fig_bar_test = plot_bar_chart(
        sample_bar_df_plotting_fixture, x_col_name='category_label_plot', y_col_name='value_count_plot', 
        chart_title=chart_title_bar_test, color_col_name='grouping_col_plot' # Use fixture column names
    )
    assert isinstance(fig_bar_test, go.Figure), "plot_bar_chart did not return a Plotly Figure."
    assert fig_bar_test.layout.title.text == chart_title_bar_test, "Bar chart title incorrect."
    assert len(fig_bar_test.data) > 0 and fig_bar_test.data[0].type == 'bar', "Bar chart trace configuration incorrect."

    fig_bar_empty_test = plot_bar_chart(pd.DataFrame(columns=['x_ax','y_ax']), x_col_name='x_ax', y_col_name='y_ax', chart_title="Empty Bar Data Plot")
    assert "Empty Bar Data Plot" in fig_bar_empty_test.layout.title.text and \
           len(fig_bar_empty_test.layout.annotations) > 0 and \
           "No data available" in fig_bar_empty_test.layout.annotations[0].text.lower(), \
        "Empty bar chart did not display correct title/message."


def test_plot_donut_chart_basic_functionality(sample_donut_df_plotting_fixture: pd.DataFrame): # Renamed
    chart_title_donut_test = "Status Distribution (Donut)"
    fig_donut_test = plot_donut_chart(
        sample_donut_df_plotting_fixture, labels_col_name='risk_level_label_plot', values_col_name='case_counts_plot', chart_title=chart_title_donut_test
    )
    assert isinstance(fig_donut_test, go.Figure), "plot_donut_chart did not return a Plotly Figure."
    assert fig_donut_test.layout.title.text == chart_title_donut_test, "Donut chart title incorrect."
    assert len(fig_donut_test.data) == 1 and fig_donut_test.data[0].type == 'pie' and \
           fig_donut_test.data[0].hole is not None and fig_donut_test.data[0].hole > 0.4, \
        "Donut chart trace configuration (type or hole size) incorrect."

    fig_donut_empty_test = plot_donut_chart(pd.DataFrame(columns=['lab','val']), labels_col_name='lab', values_col_name='val', chart_title="Empty Donut Data Plot")
    assert "Empty Donut Data Plot" in fig_donut_empty_test.layout.title.text and \
           len(fig_donut_empty_test.layout.annotations) > 0 and \
           "No data available" in fig_donut_empty_test.layout.annotations[0].text.lower(), \
        "Empty donut chart did not display correct title/message."


def test_plot_heatmap_basic_functionality(sample_heatmap_df_plotting_fixture: pd.DataFrame): # Renamed
    if sample_heatmap_df_plotting_fixture.empty:
        pytest.skip("Sample heatmap DataFrame fixture is empty for this test.")
    chart_title_heatmap_test = "Sample Correlation Heatmap"
    fig_heatmap_test = plot_heatmap(sample_heatmap_df_plotting_fixture, chart_title=chart_title_heatmap_test)
    assert isinstance(fig_heatmap_test, go.Figure), "plot_heatmap did not return a Plotly Figure."
    assert fig_heatmap_test.layout.title.text == chart_title_heatmap_test, "Heatmap title incorrect."
    assert len(fig_heatmap_test.data) == 1 and fig_heatmap_test.data[0].type == 'heatmap', "Heatmap trace type incorrect."

    fig_heatmap_empty_test = plot_heatmap(pd.DataFrame(), chart_title="Empty Heatmap Data Plot")
    assert "Empty Heatmap Data Plot" in fig_heatmap_empty_test.layout.title.text and \
           len(fig_heatmap_empty_test.layout.annotations) > 0 and \
           ("No data available" in fig_heatmap_empty_test.layout.annotations[0].text.lower() or \
            "all data non-numeric" in fig_heatmap_empty_test.layout.annotations[0].text.lower()), \
           "Empty heatmap did not display correct title/message."


def test_plot_choropleth_map_basic_functionality(
    sample_map_data_df_plotting_fixture: pd.DataFrame, # This is now a DataFrame
    sample_zone_data_df_main_fixture: pd.DataFrame   # This contains 'geometry_obj'
): # Renamed
    chart_title_map_test = "Zonal Metric Map Visualization"
    
    # Prepare GeoJSON features list from the sample_zone_data_df_main_fixture
    geojson_features_for_map_test: Optional[List[Dict[str, Any]]] = None
    if 'geometry_obj' in sample_zone_data_df_main_fixture.columns and \
       sample_zone_data_df_main_fixture['geometry_obj'].notna().any():
        
        geojson_features_for_map_test = []
        for _, row_zone_geom in sample_zone_data_df_main_fixture.iterrows():
            if pd.notna(row_zone_geom['geometry_obj']) and isinstance(row_zone_geom['geometry_obj'], dict):
                feature = {
                    "type": "Feature",
                    "geometry": row_zone_geom['geometry_obj'],
                    "properties": {"zone_id": str(row_zone_geom.get('zone_id', 'N/A'))} # Ensure zone_id is string
                }
                geojson_features_for_map_test.append(feature)
    
    if not geojson_features_for_map_test:
        pytest.skip("Could not extract valid GeoJSON features from sample_zone_data_df_main_fixture for map test.")

    # Ensure the map_data_df has the necessary columns for plotting
    if not isinstance(sample_map_data_df_plotting_fixture, pd.DataFrame) or \
       sample_map_data_df_plotting_fixture.empty or \
       'avg_risk_score' not in sample_map_data_df_plotting_fixture.columns or \
       'zone_id' not in sample_map_data_df_plotting_fixture.columns:
        pytest.skip("Sample DataFrame for choropleth map test (sample_map_data_df_plotting_fixture) is not correctly configured or empty.")

    fig_map_test = plot_choropleth_map(
        map_data_df=sample_map_data_df_plotting_fixture,
        geojson_features=geojson_features_for_map_test, # Pass list of features
        value_col_name='avg_risk_score', 
        map_title=chart_title_map_test,
        zone_id_df_col='zone_id', # Column in DataFrame for joining
        zone_id_geojson_prop='zone_id' # Property in GeoJSON features for joining
    )
    assert isinstance(fig_map_test, go.Figure), "plot_choropleth_map did not return a Plotly Figure."
    assert fig_map_test.layout.title.text == chart_title_map_test, "Choropleth map title incorrect."
    assert fig_map_test.layout.mapbox.style is not None, "Mapbox style should be set by theme or default in plot function."
    assert len(fig_map_test.data) >= 1 and fig_map_test.data[0].type == 'choroplethmapbox', "Choropleth map trace type incorrect."

    # Test with empty DataFrame for map data
    fig_map_empty_data_test = plot_choropleth_map(
        map_data_df=pd.DataFrame(columns=['zone_id', 'value']), 
        geojson_features=geojson_features_for_map_test, 
        value_col_name='value', map_title="Empty Data Map", zone_id_df_col='zone_id'
    )
    assert "Empty Data Map" in fig_map_empty_data_test.layout.title.text and \
           len(fig_map_empty_data_test.layout.annotations) > 0 and \
           "Map data is incomplete" in fig_map_empty_data_test.layout.annotations[0].text, \
        "Empty map data did not display correct title/message."

    # Test with empty GeoJSON features
    fig_map_empty_geojson_test = plot_choropleth_map(
        map_data_df=sample_map_data_df_plotting_fixture,
        geojson_features=[], # Empty list
        value_col_name='avg_risk_score', map_title="Empty GeoJSON Map", zone_id_df_col='zone_id'
    )
    assert "Empty GeoJSON Map" in fig_map_empty_geojson_test.layout.title.text and \
           len(fig_map_empty_geojson_test.layout.annotations) > 0 and \
           "Geographic boundary data (GeoJSON) unavailable" in fig_map_empty_geojson_test.layout.annotations[0].text, \
        "Empty GeoJSON features did not display correct title/message."


# Test Mapbox token fallback behavior (using SUT_MAPBOX_TOKEN_FLAG_IN_PLOTS from visualization.plots)
# These tests rely on the SUT_MAPBOX_TOKEN_FLAG_IN_PLOTS being correctly set by visualization.plots
# If the token is actually set in the test environment, these tests might behave as if token is present.
# The patch is more reliable for forcing the "no token" scenario.

@patch('visualization.plots.MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG', False) # Mock the flag in visualization.plots
def test_plot_choropleth_map_no_token_falls_back_for_override_style(
    sample_map_data_df_plotting_fixture: pd.DataFrame,
    sample_zone_data_df_main_fixture: pd.DataFrame
):
    """Tests that if mapbox_style_override requires a token but token flag is False, it falls back to an open style."""
    geojson_features_map_no_token: Optional[List[Dict[str, Any]]] = None
    if 'geometry_obj' in sample_zone_data_df_main_fixture.columns and sample_zone_data_df_main_fixture['geometry_obj'].notna().any():
        geojson_features_map_no_token = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}} 
                                         for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r['geometry_obj'])]
    if not geojson_features_map_no_token or \
       not isinstance(sample_map_data_df_plotting_fixture, pd.DataFrame) or sample_map_data_df_plotting_fixture.empty or \
       'avg_risk_score' not in sample_map_data_df_plotting_fixture.columns or 'zone_id' not in sample_map_data_df_plotting_fixture.columns:
        pytest.skip("Sample data not configured for no-token map override style test.")

    fig_no_token_override_test = plot_choropleth_map(
        map_data_df=sample_map_data_df_plotting_fixture, geojson_features=geojson_features_map_no_token,
        value_col_name='avg_risk_score', map_title="No Token Fallback (Override Style)", zone_id_df_col='zone_id',
        mapbox_style_override="mapbox://styles/mapbox/streets-v11" # A style that typically requires a token
    )
    # Plotly Express itself often defaults to 'carto-positron' if token invalid for a private style
    # Our theme setter also aims for 'carto-positron' or 'open-street-map'.
    # The plot_choropleth_map function should also ensure a fallback if MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG is False.
    assert fig_no_token_override_test.layout.mapbox.style.lower() in ["carto-positron", "open-street-map"], \
           f"Map style did not fall back to an open style when token is False and override needs token. Got: {fig_no_token_override_test.layout.mapbox.style}"


@patch('visualization.plots.MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG', False) # Mock flag in SUT
def test_plot_choropleth_map_no_token_theme_uses_fallback_style(
    sample_map_data_df_plotting_fixture: pd.DataFrame,
    sample_zone_data_df_main_fixture: pd.DataFrame,
    monkeypatch # Pytest fixture for modifying attributes
):
    """Tests that if the theme's default map style (from settings) needs a token but token flag is False, the plot falls back."""
    geojson_features_map_theme_fallback: Optional[List[Dict[str, Any]]] = None
    if 'geometry_obj' in sample_zone_data_df_main_fixture.columns and sample_zone_data_df_main_fixture['geometry_obj'].notna().any():
        geojson_features_map_theme_fallback = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}} 
                                              for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r['geometry_obj'])]
    if not geojson_features_map_theme_fallback or \
       not isinstance(sample_map_data_df_plotting_fixture, pd.DataFrame) or sample_map_data_df_plotting_fixture.empty or \
       'avg_risk_score' not in sample_map_data_df_plotting_fixture.columns or 'zone_id' not in sample_map_data_df_plotting_fixture.columns:
        pytest.skip("Sample data not configured for no-token theme map style test.")

    # Temporarily set settings.MAPBOX_STYLE_WEB to a token-requiring style for this test
    monkeypatch.setattr(settings, 'MAPBOX_STYLE_WEB', "mapbox://styles/mapbox/satellite-v9")
    # Re-apply the theme because settings.MAPBOX_STYLE_WEB changed AND MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG is mocked to False for this test scope
    set_sentinel_plotly_theme() 
    
    fig_theme_fallback_test = plot_choropleth_map(
        map_data_df=sample_map_data_df_plotting_fixture, geojson_features=geojson_features_map_theme_fallback,
        value_col_name='avg_risk_score', map_title="No Token Fallback (Theme Style Test)", zone_id_df_col='zone_id',
        mapbox_style_override=None # CRITICAL: let it use the theme's default mapbox_style
    )
    # The theme itself (set by set_sentinel_plotly_theme) should have chosen an open style
    # because MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG is False within this test's context (due to patch).
    theme_map_style_in_effect = pio.templates[pio.templates.default].layout.mapbox.style
    assert theme_map_style_in_effect.lower() in ["carto-positron", "open-street-map"], \
        f"Theme's default map style should be an open one when token is False. Got: {theme_map_style_in_effect}"
    assert fig_theme_fallback_test.layout.mapbox.style == theme_map_style_in_effect, \
        "Plot map style did not match the theme's (fallback) map style when token is False."
