# sentinel_project_root/visualization/ui_elements.py
# UI element rendering functions for Sentinel Health Co-Pilot Web Dashboards,
# primarily producing HTML/Markdown for Streamlit.

import streamlit as st # Only used for st.markdown and st.container
import html
import logging
from typing import Optional, Dict, Any, Union # Added Union

from config import settings

logger = logging.getLogger(__name__)

# --- Color Utility ---
def get_theme_color(
    color_name_or_index: Any,
    color_category: str = "general",
    fallback_color_hex: Optional[str] = None
) -> str:
    """
    Retrieves a color from the application's theme settings or a predefined list.
    This version avoids direct Plotly theme inspection to keep ui_elements independent.
    """
    # Direct mapping for named theme colors from settings.py
    direct_color_map: Dict[str, str] = {
        "risk_high": settings.COLOR_RISK_HIGH, "risk_moderate": settings.COLOR_RISK_MODERATE,
        "risk_low": settings.COLOR_RISK_LOW, "risk_neutral": settings.COLOR_RISK_NEUTRAL,
        "action_primary": settings.COLOR_ACTION_PRIMARY, "action_secondary": settings.COLOR_ACTION_SECONDARY,
        "positive_delta": settings.COLOR_POSITIVE_DELTA, "negative_delta": settings.COLOR_NEGATIVE_DELTA,
        "text_dark": settings.COLOR_TEXT_DARK, "headings_main": settings.COLOR_TEXT_HEADINGS_MAIN,
        "accent_bright": settings.COLOR_ACCENT_BRIGHT, "background_content": settings.COLOR_BACKGROUND_CONTENT,
        "background_page": settings.COLOR_BACKGROUND_PAGE, "border_light": settings.COLOR_BORDER_LIGHT,
        "border_medium": settings.COLOR_BORDER_MEDIUM, "white": settings.COLOR_BACKGROUND_WHITE,
        "subtle_background": settings.COLOR_BACKGROUND_SUBTLE,
        "link_default": settings.COLOR_TEXT_LINK_DEFAULT
    }

    if isinstance(color_name_or_index, str):
        color_name_lower = color_name_or_index.lower()
        if color_name_lower in direct_color_map:
            return direct_color_map[color_name_lower]
        if color_category == "disease" and settings.LEGACY_DISEASE_COLORS_WEB.get(color_name_or_index): # Check original case for disease map
            return settings.LEGACY_DISEASE_COLORS_WEB[color_name_or_index]

    # Fallback general colorway (matches the one defined in plots.py for consistency if Plotly isn't used)
    general_sentinel_colorway = [
        settings.COLOR_ACTION_PRIMARY, settings.COLOR_RISK_LOW, settings.COLOR_RISK_MODERATE,
        settings.COLOR_ACCENT_BRIGHT, settings.COLOR_ACTION_SECONDARY,
        "#00ACC1", "#5E35B1", "#FF7043" # Teal, Deep Purple, Coral
    ]
    if isinstance(color_name_or_index, int) and color_category == "general":
        try:
            return general_sentinel_colorway[color_name_or_index % len(general_sentinel_colorway)]
        except IndexError: # Should not happen with modulo
            pass # Fall through to other fallbacks

    if fallback_color_hex:
        return fallback_color_hex
    
    # Absolute fallback if nothing else matches: use a neutral text color
    logger.debug(f"Color '{color_name_or_index}' (category: '{color_category}') not found, using absolute fallback.")
    return settings.COLOR_TEXT_MUTED 


# --- HTML/Markdown Component Renderers ---

def render_kpi_card(
    title: str,
    value_str: str,
    icon: str = "●", 
    status_level: str = "NEUTRAL",
    delta_value: Optional[str] = None,
    delta_is_positive: Optional[bool] = None,
    help_text: Optional[str] = None,
    units: Optional[str] = None,
    container_border: bool = True # Default to True as per original use
) -> None:
    """Renders a KPI card using Streamlit Markdown and custom CSS."""
    css_status_class = f"status-{status_level.lower().replace('_', '-')}" if status_level else "status-neutral"
    
    delta_html = ""
    if delta_value is not None: # Check for None explicitly, as empty string might be valid delta_value
        delta_class = "neutral" # Default delta class
        if delta_is_positive is True: delta_class = "positive"
        elif delta_is_positive is False: delta_class = "negative"
        delta_html = f'<p class="kpi-delta {delta_class}">{html.escape(str(delta_value))}</p>'

    tooltip_attr = f'title="{html.escape(str(help_text))}"' if help_text else ''
    units_html = f" <span class='kpi-units'>{html.escape(str(units))}</span>" if units else ""

    # Ensured all dynamic parts are escaped.
    kpi_html = f"""
    <div class="kpi-card {css_status_class}" {tooltip_attr}>
        <div class="kpi-card-header">
            <span class="kpi-icon" role="img" aria-label="icon">{html.escape(icon)}</span>
            <h3 class="kpi-title">{html.escape(str(title))}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{html.escape(str(value_str))}{units_html}</p>
            {delta_html}
        </div>
    </div>
    """
    try:
        if container_border:
            with st.container(border=True): # Updated to use border=True if Streamlit version supports
                st.markdown(kpi_html, unsafe_allow_html=True)
        else:
            st.markdown(kpi_html, unsafe_allow_html=True)
    except Exception as e: # Catch if st.container(border=True) is not supported by current st version
        logger.warning(f"KPI Card: st.container(border=True) might not be supported or other st.markdown error: {e}. Falling back.")
        st.markdown(kpi_html, unsafe_allow_html=True) # Fallback to plain markdown


def render_traffic_light_indicator(
    message: str,
    status_level: str, 
    details_text: Optional[str] = None,
    container_border: bool = False
) -> None:
    """Renders a traffic light style indicator using Streamlit Markdown and custom CSS."""
    css_dot_class = f"status-{status_level.lower().replace('_', '-')}" if status_level else "status-neutral"
    details_html = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text else ""

    traffic_html = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {css_dot_class}" role="img" aria-label="{html.escape(status_level.replace('_', ' '))} status"></span>
        <span class="traffic-light-message">{html.escape(str(message))}</span>
        {details_html}
    </div>
    """
    try:
        if container_border:
            with st.container(border=True):
                st.markdown(traffic_html, unsafe_allow_html=True)
        else:
            st.markdown(traffic_html, unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Traffic Light: st.container(border=True) might not be supported or other st.markdown error: {e}. Falling back.")
        st.markdown(traffic_html, unsafe_allow_html=True)


def display_custom_styled_kpi_box(
    label: str,
    value: Union[str, int, float],
    sub_text: Optional[str] = None,
    highlight_edge_color: Optional[str] = None 
) -> None:
    """
    Renders a custom KPI box using Markdown, styled via `custom-markdown-kpi-box` CSS.
    """
    edge_class = ""
    if highlight_edge_color:
        # Map specific hex colors to CSS classes if defined in style_web_reports.css
        if highlight_edge_color.upper() == settings.COLOR_RISK_HIGH.upper(): edge_class = "highlight-red-edge"
        elif highlight_edge_color.upper() == settings.COLOR_RISK_MODERATE.upper(): edge_class = "highlight-amber-edge"
        elif highlight_edge_color.upper() == settings.COLOR_RISK_LOW.upper(): edge_class = "highlight-green-edge"
        # Add more mappings if other edge colors are defined in CSS
    
    # Format value: add comma for thousands if numeric and >= 1000
    value_display_str = str(value)
    if isinstance(value, (int, float)):
        if value >= 1000 or value <= -1000 : # Check magnitude for formatting
            value_display_str = f"{value:,.0f}"
        elif isinstance(value, float) and not value.is_integer():
            value_display_str = f"{value:.1f}" # One decimal for non-integer floats
        else: # Integer or whole float < 1000
            value_display_str = f"{int(value)}"


    sub_text_html_part = f'<div class="custom-kpi-subtext-small">{html.escape(str(sub_text))}</div>' if sub_text else ""

    box_html_content = f"""
    <div class="custom-markdown-kpi-box {edge_class}">
        <div class="custom-kpi-label-top-condition">{html.escape(str(label))}</div>
        <div class="custom-kpi-value-large">{html.escape(value_display_str)}</div>
        {sub_text_html_part}
    </div>
    """
    st.markdown(box_html_content, unsafe_allow_html=True)
