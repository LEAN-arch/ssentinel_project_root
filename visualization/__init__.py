# sentinel_project_root/visualization/__init__.py
"""
Initializes the visualization package, defining its public API.

This package provides a suite of high-level, theme-aware functions for
creating standardized Plotly charts and custom HTML/CSS UI components
for the Sentinel Streamlit application.
"""

# --- Core Plotting Functions from plots.py ---
from .plots import (
    set_plotly_theme,
    create_empty_figure,
    plot_bar_chart,
    plot_donut_chart,
    plot_line_chart,
    plot_choropleth_map,
    plot_heatmap,
    plot_forecast_chart,
)

# --- Custom UI Element Renderers from ui_elements.py ---
from .ui_elements import (
    get_theme_color,
    render_kpi_card,
    render_traffic_light_indicator,
    render_custom_kpi,
    load_and_inject_css,
)

# --- Define the canonical public API for the package ---
__all__ = [
    # plots.py
    "set_plotly_theme",
    "create_empty_figure",
    "plot_bar_chart",
    "plot_donut_chart",
    "plot_line_chart",
    "plot_choropleth_map",
    "plot_heatmap",
    "plot_forecast_chart",
    
    # ui_elements.py
    "get_theme_color",
    "render_kpi_card",
    "render_traffic_light_indicator",
    "render_custom_kpi",
    "load_and_inject_css",
]
