# sentinel_project_root/tests/test_visualization.py
# SME PLATINUM STANDARD - VISUALIZATION & UI TESTS

import html
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from visualization import (create_empty_figure, plot_bar_chart,
                           plot_donut_chart, plot_line_chart, render_kpi_card,
                           set_plotly_theme)

# Fixtures are sourced from conftest.py

@pytest.fixture(scope="module", autouse=True)
def apply_theme():
    """Apply the custom Plotly theme for all tests in this module."""
    set_plotly_theme()

# --- Plotting Tests ---
def test_create_empty_figure_properties():
    """Verifies that empty figures are created with the correct message and layout."""
    fig = create_empty_figure(title="Empty Test", message="No data here.")
    assert isinstance(fig, go.Figure)
    assert "Empty Test" in fig.layout.title.text
    assert fig.layout.annotations[0].text == "No data here."

def test_plot_line_chart_structure(health_records_df):
    """Tests that line charts are generated with the correct structure."""
    series = health_records_df.set_index('encounter_date')['age'].resample('M').mean()
    fig = plot_line_chart(series, title="Monthly Age Trend", y_title="Average Age")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0 and fig.data[0].type == 'scatter'
    assert "Monthly Age Trend" in fig.layout.title.text
    assert fig.layout.yaxis.title.text == "Average Age"

def test_plot_bar_chart_structure(enriched_health_records_df):
    """Tests that bar charts are generated with the correct structure."""
    df = enriched_health_records_df['gender'].value_counts().reset_index()
    fig = plot_bar_chart(df, x_col='gender', y_col='count', title="Gender Distribution")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0 and fig.data[0].type == 'bar'
    assert "Gender Distribution" in fig.layout.title.text

def test_plot_donut_chart_structure(enriched_health_records_df):
    """Tests that donut charts are generated with the correct structure."""
    df = enriched_health_records_df.drop_duplicates('patient_id')['diagnosis'].value_counts().nlargest(5).reset_index()
    fig = plot_donut_chart(df, label_col='diagnosis', value_col='count', title="Top 5 Diagnoses")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0 and fig.data[0].type == 'pie'
    assert fig.data[0].hole > 0.4 # Check that it's a donut, not a pie
    assert "Top 5 Diagnoses" in fig.layout.title.text

# --- UI Element Tests ---
@patch('visualization.ui_elements.st')
def test_render_kpi_card_html(mock_st):
    """Tests that KPI cards render with the correct HTML structure and classes."""
    mock_st.markdown = MagicMock()
    render_kpi_card(
        title="Test KPI", value=123.45, unit="tests",
        status_level="HIGH_CONCERN", help_text="A test tooltip."
    )
    
    html_out, kwargs = mock_st.markdown.call_args
    html_content = html_out[0]
    
    assert 'class="kpi-card status-high-concern"' in html_content
    assert f'title="{html.escape("A test tooltip.")}"' in html_content
    assert '<div class="kpi-title">Test KPI</div>' in html_content
    assert '<p class="kpi-value">123.45' in html_content
    assert '<span class="kpi-units">tests</span>' in html_content
    assert kwargs['unsafe_allow_html'] is True
