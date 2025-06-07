# sentinel_project_root/tests/test_ui_visualization_helpers.py
# Pytest tests for UI and Plotting helpers in visualization module for Sentinel.

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from unittest.mock import patch, MagicMock
import html
import json
import os
import hashlib

# --- Module Imports ---
from visualization.plots import (
    set_sentinel_plotly_theme, create_empty_figure, plot_choropleth_map,
    plot_annotated_line_chart, plot_bar_chart, plot_donut_chart, plot_heatmap
)
from visualization.ui_elements import (
    get_theme_color, render_kpi_card, render_traffic_light_indicator,
    display_custom_styled_kpi_box
)
from config import settings

# --- Fixtures for Plotting Tests ---

@pytest.fixture(scope="module", autouse=True)
def apply_sentinel_theme_for_viz_tests():
    """Apply the custom Sentinel Plotly theme once for all tests in this module."""
    set_sentinel_plotly_theme()

@pytest.fixture(scope="module")
def sample_series_data_plotting_fixture() -> pd.Series:
    """Provides a sample time series for plotting tests."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10, freq='D'))
    data = np.random.randint(50, 150, size=10)
    return pd.Series(data, index=dates, name="visits")

@pytest.fixture(scope="module")
def sample_bar_df_plotting_fixture() -> pd.DataFrame:
    """Provides a sample DataFrame for bar chart plotting tests."""
    data = {
        'category': ['Alpha', 'Beta', 'Gamma', 'Alpha', 'Beta', 'Gamma'],
        'value': [10, 25, 15, 12, 18, 22],
        'group': ['Group 1', 'Group 1', 'Group 1', 'Group 2', 'Group 2', 'Group 2']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def sample_donut_df_plotting_fixture() -> pd.DataFrame:
    """Provides a sample DataFrame for donut chart plotting tests."""
    data = {'label': ['Complete', 'Pending', 'Failed'], 'count': [150, 45, 12]}
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def sample_heatmap_df_plotting_fixture() -> pd.DataFrame:
    """Provides a sample DataFrame for heatmap plotting tests."""
    data = np.random.rand(5, 7)
    rows = [f'Zone {chr(65+i)}' for i in range(5)]
    cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    return pd.DataFrame(data, index=rows, columns=cols)

@pytest.fixture(scope="module")
def sample_map_data_df_plotting_fixture() -> pd.DataFrame:
    """Provides sample aggregated data for map plotting tests."""
    data = {
        'zone_id': [f'Zone{chr(65 + i)}' for i in range(5)],
        'avg_risk_score': [75.5, 42.1, 68.9, 81.0, 55.3],
        'total_cases': [12, 5, 9, 15, 7]
    }
    return pd.DataFrame(data)


# --- Tests for Core Theming and Color Utilities ---
def test_get_theme_color_utility():
    assert get_theme_color("risk_high") == settings.COLOR_RISK_HIGH
    assert get_theme_color("action_primary") == settings.COLOR_ACTION_PRIMARY
    assert get_theme_color(0, "general") == pio.templates[pio.templates.default].layout.colorway[0]

# --- Tests for HTML Component Renderers ---
@patch('visualization.ui_elements.st')
def test_render_kpi_card_html_structure(mock_st):
    """Tests that KPI cards are rendered with the correct HTML structure and classes."""
    mock_st.markdown = MagicMock()
    render_kpi_card(title="Active Cases", value_str="120", status_level="MODERATE_CONCERN")
    
    html_out, _ = mock_st.markdown.call_args
    assert 'class="kpi-card status-moderate-concern"' in html_out[0]
    assert '<h3 class="kpi-title">Active Cases</h3>' in html_out[0]
    assert '<p class="kpi-value">120</p>' in html_out[0]

@patch('visualization.ui_elements.st')
def test_render_traffic_light_indicator_html(mock_st):
    """Tests that traffic light indicators have the correct HTML classes."""
    mock_st.markdown = MagicMock()
    render_traffic_light_indicator(message="Power Status", status_level="HIGH_RISK")
    
    html_out, _ = mock_st.markdown.call_args
    assert 'class="traffic-light-dot status-high-risk"' in html_out[0]
    assert '<span class="traffic-light-message">Power Status</span>' in html_out[0]


# --- Tests for Plotting Functionality ---
def test_create_empty_figure_properties():
    """Verifies that empty figures are created with the correct message and layout."""
    fig = create_empty_figure(chart_title="Empty Plot", message_text="No data.")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Empty Plot"
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == "No data."

def test_plot_annotated_line_chart(sample_series_data_plotting_fixture: pd.Series):
    """Tests the creation of an annotated line chart."""
    fig = plot_annotated_line_chart(sample_series_data_plotting_fixture, chart_title="Line Chart")
    assert isinstance(fig, go.Figure) and len(fig.data) > 0 and fig.data[0].type == 'scatter'

def test_plot_bar_chart(sample_bar_df_plotting_fixture: pd.DataFrame):
    """Tests the creation of a bar chart."""
    fig = plot_bar_chart(sample_bar_df_plotting_fixture, 'category', 'value', "Bar Chart")
    assert isinstance(fig, go.Figure) and len(fig.data) > 0 and fig.data[0].type == 'bar'

def test_plot_donut_chart(sample_donut_df_plotting_fixture: pd.DataFrame):
    """Tests the creation of a donut chart."""
    fig = plot_donut_chart(sample_donut_df_plotting_fixture, 'label', 'count', "Donut Chart")
    assert isinstance(fig, go.Figure) and len(fig.data) > 0 and fig.data[0].type == 'pie' and fig.data[0].hole > 0

def test_plot_heatmap(sample_heatmap_df_plotting_fixture: pd.DataFrame):
    """Tests the creation of a heatmap."""
    if sample_heatmap_df_plotting_fixture.empty: pytest.skip("Sample heatmap DF is empty.")
    fig = plot_heatmap(sample_heatmap_df_plotting_fixture, "Heatmap")
    assert isinstance(fig, go.Figure) and len(fig.data) > 0 and fig.data[0].type == 'heatmap'

def test_plot_choropleth_map(sample_map_data_df_plotting_fixture: pd.DataFrame, sample_zone_data_df_main_fixture: pd.DataFrame):
    """Tests the creation of a choropleth map."""
    geojson_features = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}}
                        for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r.get('geometry_obj'))]
    if not geojson_features or sample_map_data_df_plotting_fixture.empty:
        pytest.skip("Sample data not configured for choropleth map test.")

    fig = plot_choropleth_map(
        map_data_df=sample_map_data_df_plotting_fixture,
        geojson_features=geojson_features,
        value_col_name='avg_risk_score',
        map_title="Zonal Risk Map",
        zone_id_df_col='zone_id'
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0 and fig.data[0].type == 'choroplethmapbox'

@patch('visualization.plots.MAPBOX_TOKEN_SET_IN_PLOTLY_FLAG', False)
def test_map_no_token_override_fallback(sample_map_data_df_plotting_fixture: pd.DataFrame, sample_zone_data_df_main_fixture: pd.DataFrame):
    """Tests that map styles requiring a token fall back to a free style when no token is set."""
    geojson_features = [{"type": "Feature", "geometry": r['geometry_obj'], "properties": {"zone_id": str(r.get('zone_id'))}}
                        for _, r in sample_zone_data_df_main_fixture.iterrows() if pd.notna(r.get('geometry_obj'))]
    if not geojson_features or sample_map_data_df_plotting_fixture.empty:
        pytest.skip("Sample data not configured for no-token map style test.")

    fig = plot_choropleth_map(
        map_data_df=sample_map_data_df_plotting_fixture,
        geojson_features=geojson_features,
        value_col_name='avg_risk_score',
        map_title="No Token Fallback",
        mapbox_style_override="mapbox://styles/mapbox/streets-v11"  # A token-requiring style
    )
    # The plot function should detect no token is set and fall back to a free style.
    assert fig.layout.mapbox.style.lower() in ["carto-positron", "open-street-map"]
