# sentinel_project_root/visualization/__init__.py
"""
Initializes the visualization package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.

This file defines the public API of the visualization package.
"""

# --- Import functions from submodules to expose them publicly ---

# From plots.py
from .plots import (
    create_empty_figure,
    plot_bar_chart,
    plot_donut_chart,
    plot_annotated_line_chart
)

# From ui_elements.py
from .ui_elements import (
    get_theme_color,
    render_kpi_card,
    render_traffic_light_indicator,
    display_custom_styled_kpi_box
)

# From themes.py (assuming it exists to define the plotly theme template)
try:
    from .themes import sentinel_theme
except ImportError:
    # Fallback if themes.py doesn't exist or has an issue
    sentinel_theme = "plotly_white"


# --- Define __all__ for explicit public API definition ---
# This tells tools and developers which names are part of the public API
# and controls 'from visualization import *' behavior.
__all__ = [
    # plots
    "create_empty_figure",
    "plot_bar_chart",
    "plot_donut_chart",
    "plot_annotated_line_chart",
    
    # ui_elements
    "get_theme_color",
    "render_kpi_card",
    "render_traffic_light_indicator",
    "display_custom_styled_kpi_box",
    
    # themes
    "sentinel_theme"
]
