import streamlit as st
import html
import logging
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

# --- Robust Settings Import with Fallback ---
try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    # FIXED: Use the correct `__name__` magic variable.
    _logger_init_fallback = logging.getLogger(__name__)
    _logger_init_fallback.error(f"CRITICAL IMPORT ERROR in ui_elements.py: {e}. Using fallback settings.")

    class FallbackSettings:
        COLOR_RISK_HIGH = "#D32F2F"; COLOR_RISK_MODERATE = "#FFA000"; COLOR_RISK_LOW = "#388E3C";
        COLOR_ACTION_PRIMARY = "#1976D2"; LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {}
    settings = FallbackSettings()

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


def _get_setting_attr(attr_name: str, default_value: Any) -> Any:
    """Safely gets a setting attribute or returns a default."""
    return getattr(settings, attr_name, default_value)


def get_theme_color(
    color_name_or_index: Any,
    color_category: str = "general",
    fallback_color_hex: Optional[str] = None
) -> str:
    """Retrieves a theme color from settings by name or index."""
    direct_map = {
        "risk_high": 'COLOR_RISK_HIGH', "risk_moderate": 'COLOR_RISK_MODERATE', "risk_low": 'COLOR_RISK_LOW',
        "action_primary": 'COLOR_ACTION_PRIMARY', "positive_delta": 'COLOR_POSITIVE_DELTA', "negative_delta": 'COLOR_NEGATIVE_DELTA'
    }
    
    if isinstance(color_name_or_index, str):
        key = color_name_or_index.lower().strip()
        if key in direct_map:
            return _get_setting_attr(direct_map[key], fallback_color_hex or "#757575")
        
        if color_category == "disease":
            legacy_colors = _get_setting_attr('LEGACY_DISEASE_COLORS_WEB', {})
            return legacy_colors.get(color_name_or_index, fallback_color_hex or "#6c757d")
    
    if isinstance(color_name_or_index, int) and color_category == "general":
        colorway = _get_setting_attr('PLOTLY_COLORWAY', [_get_setting_attr('COLOR_ACTION_PRIMARY', '#1976D2')])
        return colorway[color_name_or_index % len(colorway)]

    return fallback_color_hex or "#6c757d"


def render_kpi_card(
    title: str, value_str: str, icon: str = "●", status_level: str = "NEUTRAL",
    delta_value: Optional[str] = None, delta_is_positive: Optional[bool] = None,
    help_text: Optional[str] = None, units: Optional[str] = None, **kwargs
) -> None:
    """Renders a custom HTML KPI card in Streamlit."""
    status_class = f"status-{str(status_level).lower().replace('_', '-')}"
    
    delta_html = ""
    if delta_value is not None:
        delta_class = "positive" if delta_is_positive else "negative"
        delta_html = f'<p class="kpi-delta {delta_class}">{html.escape(str(delta_value))}</p>'
    
    tooltip = f'title="{html.escape(str(help_text))}"' if help_text else ''
    units_html = f" <span class='kpi-units'>{html.escape(str(units))}</span>" if units else ""

    # FIXED: CSS class names match the stylesheet.
    kpi_html = f"""
    <div class="kpi-card {status_class}" {tooltip}>
        <div class="kpi-card-header">
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
        # FIXED: Wrap in container to isolate component layout.
        with st.container():
            st.markdown(kpi_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering KPI Card '{title}': {e}", exc_info=True)


def render_traffic_light_indicator(
    message: str, status_level: str, details_text: Optional[str] = None, **kwargs
) -> None:
    """Renders a custom HTML traffic light status indicator."""
    status_class = f"status-{str(status_level).lower().replace('_', '-')}"
    details_html = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text else ""

    # FIXED: CSS class names match the stylesheet.
    traffic_html = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {status_class}"></span>
        <span class="traffic-light-message">{html.escape(message)}</span>
        {details_html}
    </div>
    """
    try:
        with st.container():
            st.markdown(traffic_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering Traffic Light '{message}': {e}", exc_info=True)


def display_custom_styled_kpi_box(
    label: str, value: Union[str, int, float, None],
    sub_text: Optional[str] = None, highlight_edge_color: Optional[str] = None
) -> None:
    """Renders a custom-styled KPI box with a colored edge."""
    value_display = "N/A"
    if pd.notna(value):
        if isinstance(value, (int, float)):
            value_display = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.1f}" if isinstance(value, float) and not value.is_integer() else f"{int(value)}"
        else:
            value_display = str(value)
    
    sub_text_html = f'<div class="custom-kpi-subtext-small">{html.escape(str(sub_text))}</div>' if sub_text else ""
    
    edge_class = ""
    if highlight_edge_color:
        color_map = {
            _get_setting_attr('COLOR_RISK_HIGH', "#D32F2F").upper(): "highlight-red-edge",
            _get_setting_attr('COLOR_RISK_MODERATE', "#FFA000").upper(): "highlight-amber-edge",
            _get_setting_attr('COLOR_RISK_LOW', "#388E3C").upper(): "highlight-green-edge",
        }
        edge_class = color_map.get(highlight_edge_color.upper(), "")

    box_html = f"""
    <div class="custom-markdown-kpi-box {edge_class}">
        <div class="custom-kpi-label-top-condition">{html.escape(label)}</div>
        <div class="custom-kpi-value-large">{html.escape(value_display)}</div>
        {sub_text_html}
    </div>
    """
    try:
        with st.container():
            st.markdown(box_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error rendering Custom KPI Box '{label}': {e}", exc_info=True)
