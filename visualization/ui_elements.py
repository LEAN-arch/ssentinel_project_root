# sentinel_project_root/visualization/ui_elements.py
# Contains functions for rendering standardized UI components for Sentinel dashboards.
# This version uses native Streamlit components for robustness and maintainability.

import streamlit as st
from typing import Dict, Any, Optional

try:
    from config import settings
except ImportError:
    # Fallback for standalone execution or testing
    class FallbackSettings:
        THEME_COLORS = {
            "general": {"primary": "#007BFF", "secondary": "#6C757D"},
            "risk": {"high": "#FF4136", "moderate": "#FF851B", "low": "#2ECC40"}
        }
    settings = FallbackSettings()

# --- THEME COLOR MANAGEMENT ---

def get_theme_color(key: str, category: str = "general", fallback: Optional[str] = None) -> str:
    """
    Safely retrieves a color from the theme settings dictionary.
    
    Args:
        key: The color key (e.g., 'primary', 'high').
        category: The color category (e.g., 'general', 'risk').
        fallback: A default color to return if the requested color is not found.
        
    Returns:
        A hex color string.
    """
    try:
        theme_colors = getattr(settings, 'THEME_COLORS', {})
        return theme_colors.get(category, {}).get(key, fallback or "#CCCCCC")
    except Exception:
        return fallback or "#CCCCCC"

# --- STANDARD UI COMPONENTS ---

def render_kpi_card(
    title: str,
    value_str: str,
    icon: str,
    units: str = "",
    help_text: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal"
):
    """
    Renders a standardized Key Performance Indicator (KPI) card using st.metric.
    
    Args:
        title: The title of the KPI.
        value_str: The main value to display, pre-formatted as a string.
        icon: An emoji icon for the KPI.
        units: The units for the value (e.g., '%', 'ms').
        help_text: Optional tooltip text.
        delta: Optional string for the delta value (e.g., "+12%").
        delta_color: "normal", "inverse", or "off".
    """
    # Using st.metric is the standard, robust way to create KPIs in Streamlit.
    st.metric(
        label=f"{icon} {title}",
        value=f"{value_str}{units}",
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )

def render_traffic_light_indicator(message: str, status_level: str):
    """
    Renders a colored box with a message, like a traffic light, using native st components.
    
    Args:
        message: The text to display.
        status_level: The status level ('HIGH_RISK', 'MODERATE_CONCERN', 'ACCEPTABLE', 'NO_DATA').
    """
    # Mapping our custom status levels to Streamlit's native functions
    status_map = {
        "HIGH_RISK": "error",
        "MODERATE_CONCERN": "warning",
        "ACCEPTABLE": "success",
        "NO_DATA": "info"
    }
    
    # Get the appropriate Streamlit function (st.error, st.warning, etc.)
    render_func = getattr(st, status_map.get(status_level, "info"), st.info)
    
    # Call the function with the message, which is more maintainable than custom HTML.
    render_func(message)

def render_info_box(header: str, content: str, icon: str = "ℹ️"):
    """
    Renders a simple bordered box with a header and content using st.container.
    """
    with st.container(border=True):
        st.markdown(f"**{icon} {header}**")
        st.markdown(content)
