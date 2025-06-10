# sentinel_project_root/visualization/__init__.py
# SME PLATINUM STANDARD - ROBUST & EXPLICIT PACKAGE API (V5 - DEFINITIVE FIX)

"""
Initializes the visualization package, defining its public API.
This file explicitly exports all public-facing functions from its submodules,
providing a single, consistent import point for the rest of the application.
"""

# --- Core Plotting Functions from plots.py ---
from .plots import (
    set_plotly_theme,
    create_empty_figure,
    plot_bar_chart,
    plot_donut_chart,
    plot_line_chart,
    plot_forecast_chart,
)

# --- Custom UI Element Renderers from ui_elements.py ---
# SME FIX: All necessary UI components are now imported and exposed.
from .ui_elements import (
    load_and_inject_css,
    get_theme_color,
    render_kpi_card,
    render_traffic_light_indicator,
    render_custom_kpi,
)

# --- Define the canonical public API for the package ---
__all__ = [
    # from plots.py
    "set_plotly_theme",
    "create_empty_figure",
    "plot_bar_chart",
    "plot_donut_chart",
    "plot_line_chart",
    "plot_forecast_chart",
    
    # from ui_elements.py
    "load_and_inject_css",
    "get_theme_color",
    "render_kpi_card",
    "render_traffic_light_indicator",
    "render_custom_kpi",
]
