# ssentinel_project_root/visualization/ui_elements.py
"""
Contains functions for rendering standardized, theme-aware UI components.
This module leverages native Streamlit components for robustness and maintainability.
"""
import streamlit as st
from typing import Dict, Optional

try:
    from config import settings
except ImportError:
    # This fallback allows the module to be used in isolation for testing
    # or if the main settings file is unavailable.
    class FallbackSettings:
        THEME_COLORS = {
            "risk": {"high": "#FF4136", "moderate": "#FF851B", "low": "#2ECC40"},
            "general": {"primary": "#007BFF", "secondary": "#6C757D"}
        }
    settings = FallbackSettings()

# --- THEME COLOR MANAGEMENT ---

def get_theme_color(key: str, category: str = "risk", fallback: str = "#6C757D") -> str:
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
        # Safely access nested dictionaries
        theme_colors = getattr(settings, 'THEME_COLORS', {})
        return theme_colors.get(category, {}).get(key, fallback)
    except Exception:
        return fallback

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
        units: Optional units for the value (e.g., '%', 'days').
        help_text: Optional tooltip text.
        delta: Optional string for the delta value (e.g., "+12%").
        delta_color: "normal", "inverse", or "off".
    """
    # Using st.metric is the standard, robust way to create KPIs in Streamlit.
    st.metric(
        label=f"{icon} {title}",
        value=f"{value_str}{' ' + units if units else ''}",
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
    # This provides a layer of abstraction, making the calling code cleaner.
    status_map = {
        "HIGH_RISK": st.error,
        "MODERATE_CONCERN": st.warning,
        "ACCEPTABLE": st.success,
        "NO_DATA": st.info
    }
    
    # Get the appropriate Streamlit function (e.g., st.error), with st.info as a safe default
    render_func = status_map.get(status_level, st.info)
    
    # Call the function with the message
    render_func(message)

def render_info_box(header: str, content: str, icon: str = "ℹ️"):
    """
    Renders a simple bordered box with a header and content using st.container.
    """
    with st.container(border=True):
        st.markdown(f"**{icon} {header}**")
        st.markdown(content)

# --- Preserved for backward compatibility ---
# This function's logic is now superseded by the get_theme_color function above,
# but it's kept to avoid breaking any un-seen pages that might still be using it.
# A future refactor could deprecate this and update all call sites.
def get_color_by_risk(risk_level: str) -> str:
    """
    [DEPRECATED] Returns a color based on a risk level string.
    Use get_theme_color(risk_level) instead.
    """
    return get_theme_color(key=risk_level.lower(), category='risk')
