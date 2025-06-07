# sentinel_project_root/visualization/ui_elements.py
"""
Contains standardized UI component rendering functions for the Sentinel application.
This ensures a consistent look and feel across all dashboards and handles theming.
"""
import streamlit as st
import html
import logging
from typing import Optional, Any, Union, List
import pandas as pd

# --- Robust Settings Import with Fallback ---
try:
    from config import settings
except ImportError:
    logging.basicConfig(level=logging.INFO)
    _logger_init_fallback = logging.getLogger(__name__)
    _logger_init_fallback.error("CRITICAL IMPORT ERROR in ui_elements.py. Using fallback settings.")

    class FallbackSettings:
        COLOR_RISK_HIGH = "#D32F2F"; COLOR_RISK_MODERATE = "#FFA000"; COLOR_RISK_LOW = "#388E3C";
        COLOR_ACTION_PRIMARY = "#1976D2"; PLOTLY_COLORWAY = ["#1976D2", "#388E3C", "#FFA000", "#D32F2F"]
        COLOR_POSITIVE_DELTA = "#388E3C"; COLOR_NEGATIVE_DELTA = "#D32F2F"; COLOR_TEXT_MUTED = "#6c757d"
    settings = FallbackSettings()

logger = logging.getLogger(__name__)


def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets a setting attribute or returns a default."""
    return getattr(settings, attr_name, default_value)


def get_theme_color(
    name_or_index: Any,
    category: str = "general",
    fallback_color: str = "#6c757d"
) -> Union[str, List[str]]:
    """
    Retrieves a theme color or color sequence from settings.

    Args:
        name_or_index (Any): The semantic name of the color (e.g., 'risk_high') or an integer index for a categorical color.
        category (str): The category of color. Use 'categorical_sequence' to get the full list for plots.
        fallback_color (str): The color to return if no match is found.

    Returns:
        Union[str, List[str]]: The hex color code or a list of hex color codes.
    """
    if category == "categorical_sequence":
        return _get_setting('PLOTLY_COLORWAY', [fallback_color])

    if isinstance(name_or_index, str):
        key = name_or_index.lower().strip().replace('_', '-')
        # A single map for all semantic color lookups for consistency and maintainability.
        direct_map = {
            "risk-high": "COLOR_RISK_HIGH", "high-concern": "COLOR_RISK_HIGH", "high-risk": "COLOR_RISK_HIGH",
            "risk-moderate": "COLOR_RISK_MODERATE", "moderate-concern": "COLOR_RISK_MODERATE",
            "risk-low": "COLOR_RISK_LOW", "acceptable": "COLOR_RISK_LOW", "good-performance": "COLOR_RISK_LOW",
            "primary": "COLOR_ACTION_PRIMARY",
            "positive": "COLOR_POSITIVE_DELTA",
            "negative": "COLOR_NEGATIVE_DELTA", "danger": "COLOR_RISK_HIGH",
            "muted": "COLOR_TEXT_MUTED", "neutral": "COLOR_TEXT_MUTED"
        }
        setting_name = direct_map.get(key)
        if setting_name:
            return _get_setting(setting_name, fallback_color)
    
    return fallback_color


def render_kpi_card(
    title: str, value_str: str, icon: str = "●", status_level: str = "NEUTRAL",
    delta_value: Optional[str] = None, delta_is_positive: Optional[bool] = None,
    help_text: Optional[str] = None, units: Optional[str] = None
) -> None:
    """
    Renders a custom HTML KPI card in Streamlit.

    Args:
        title (str): The main title of the KPI.
        value_str (str): The primary value to display, pre-formatted as a string.
        icon (str): An emoji to display next to the title. Defaults to "●".
        status_level (str): A semantic status level (e.g., 'GOOD_PERFORMANCE', 'HIGH_CONCERN')
                            which maps to a CSS class for styling. Defaults to "NEUTRAL".
        delta_value (Optional[str]): A string representing the change in value. Defaults to None.
        delta_is_positive (Optional[bool]): Determines the color of the delta. True for green, False for red.
                                           Required if delta_value is provided. Defaults to None.
        help_text (Optional[str]): Text to display in a tooltip on hover. Defaults to None.
        units (Optional[str]): Units to display next to the primary value. Defaults to None.
    """
    status_class = f"status-{str(status_level).lower().replace('_', '-')}"
    
    delta_html = ""
    if delta_value is not None and delta_is_positive is not None:
        delta_class = "positive" if delta_is_positive else "negative"
        delta_html = f'<p class="kpi-delta {delta_class}">{html.escape(str(delta_value))}</p>'
    
    tooltip = f'title="{html.escape(str(help_text))}"' if help_text else ''
    units_html = f" <span class='kpi-units'>{html.escape(str(units))}</span>" if units else ""

    kpi_html = f"""
    <div class="kpi-card {status_class}" {tooltip}>
        <div class="kpi-header">
            <span class="kpi-icon">{html.escape(icon)}</span>
            <h3 class="kpi-title">{html.escape(title)}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{html.escape(value_str)}{units_html}</p>
            {delta_html}
        </div>
    </div>
    """
    try:
        st.markdown(kpi_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering KPI Card '{title}': {e}", exc_info=True)


def render_traffic_light_indicator(
    message: str, status_level: str, details_text: Optional[str] = None
) -> None:
    """
    Renders a custom HTML traffic light status indicator.

    Args:
        message (str): The primary message to display.
        status_level (str): A semantic status level (e.g., 'ACCEPTABLE', 'HIGH_RISK') that maps to a CSS class.
        details_text (Optional[str]): Smaller, secondary text to display. Defaults to None.
    """
    status_class = f"status-{str(status_level).lower().replace('_', '-')}"
    details_html = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text else ""

    traffic_html = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {status_class}"></span>
        <span class="traffic-light-message">{html.escape(message)}</span>
        {details_html}
    </div>
    """
    try:
        st.markdown(traffic_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering Traffic Light '{message}': {e}", exc_info=True)


def display_custom_styled_kpi_box(
    label: str, value: Union[str, int, float, None],
    sub_text: Optional[str] = None, highlight_status: Optional[str] = None
) -> None:
    """
    Renders a custom-styled KPI box with an optional colored edge based on status.

    Args:
        label (str): The label displayed at the top of the box.
        value (Union[str, int, float, None]): The primary value. Will be formatted automatically.
        sub_text (Optional[str]): Smaller text displayed below the value. Defaults to None.
        highlight_status (Optional[str]): A semantic status (e.g., 'HIGH_RISK') to color the left edge.
                                          Defaults to None (no colored edge).
    """
    value_display = "N/A"
    if pd.notna(value):
        if isinstance(value, float) and not value.is_integer():
            value_display = f"{value:,.1f}"
        elif isinstance(value, (int, float)):
            value_display = f"{int(value):,}"
        else:
            value_display = str(value)
    
    sub_text_html = f'<div class="custom-kpi-subtext">{html.escape(str(sub_text))}</div>' if sub_text else ""
    
    # Generate CSS class directly from the semantic status name for consistency.
    edge_class = f"highlight-edge-{str(highlight_status).lower().replace('_', '-')}" if highlight_status else ""

    box_html = f"""
    <div class="custom-kpi-box {edge_class}">
        <div class="custom-kpi-label">{html.escape(label)}</div>
        <div class="custom-kpi-value">{html.escape(value_display)}</div>
        {sub_text_html}
    </div>
    """
    try:
        st.markdown(box_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering Custom KPI Box '{label}': {e}", exc_info=True)
