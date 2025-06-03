# sentinel_project_root/visualization/ui_elements.py
# UI element rendering functions for Sentinel Health Co-Pilot Web Dashboards,
# primarily producing HTML/Markdown for Streamlit.

import streamlit as st
import html
import logging
from typing import Optional, Dict, Any

from config import settings # Use new settings module

logger = logging.getLogger(__name__)

# --- Color Utility (Moved from plots.py as it's general UI theming) ---

def get_theme_color(
    color_name_or_index: Any,
    color_category: str = "general", # e.g., "general", "risk", "action", "disease", "delta"
    fallback_color_hex: Optional[str] = None
) -> str:
    """
    Retrieves a color from the application's theme settings or Plotly's default colorway.

    Args:
        color_name_or_index: Name of the color (e.g., "risk_high", "TB") or an integer index for Plotly colorway.
        color_category: Category of the color to fetch ("general", "risk", "action", "disease", "delta").
        fallback_color_hex: A hex color string to use if no other color can be resolved.

    Returns:
        A hex color string.
    """
    # Direct mapping for named theme colors from settings.py
    direct_color_map: Dict[str, str] = {
        "risk_high": settings.COLOR_RISK_HIGH,
        "risk_moderate": settings.COLOR_RISK_MODERATE,
        "risk_low": settings.COLOR_RISK_LOW,
        "risk_neutral": settings.COLOR_RISK_NEUTRAL,
        "action_primary": settings.COLOR_ACTION_PRIMARY,
        "action_secondary": settings.COLOR_ACTION_SECONDARY,
        "positive_delta": settings.COLOR_POSITIVE_DELTA,
        "negative_delta": settings.COLOR_NEGATIVE_DELTA,
        "text_dark": settings.COLOR_TEXT_DARK,
        "headings_main": settings.COLOR_TEXT_HEADINGS_MAIN,
        "accent_bright": settings.COLOR_ACCENT_BRIGHT,
        "background_content": settings.COLOR_BACKGROUND_CONTENT,
        "background_page": settings.COLOR_BACKGROUND_PAGE,
        "border_light": settings.COLOR_BORDER_LIGHT,
    }

    if isinstance(color_name_or_index, str) and color_name_or_index in direct_color_map:
        return direct_color_map[color_name_or_index]

    # Specific category handling
    if color_category == "disease" and isinstance(color_name_or_index, str) and \
       settings.LEGACY_DISEASE_COLORS_WEB.get(color_name_or_index):
        return settings.LEGACY_DISEASE_COLORS_WEB[color_name_or_index]
    
    # Fallback to Plotly's current theme colorway if general category or index-based
    # This part requires Plotly to be imported and theme set, which is usually done in plots.py
    # For now, to keep this module independent of plotly for just this function,
    # we'll use a predefined fallback or the user-supplied one.
    # If a more dynamic Plotly theme color is needed here, `pio` would need to be imported.
    
    # Placeholder for Plotly theme colorway access if it were here:
    # try:
    #     import plotly.io as pio
    #     active_template = pio.templates.get(pio.templates.default) # get for safety
    #     if active_template and hasattr(active_template.layout, 'colorway') and active_template.layout.colorway:
    #         colorway = active_template.layout.colorway
    #         idx = color_name_or_index if isinstance(color_name_or_index, int) else abs(hash(str(color_name_or_index)))
    #         return colorway[idx % len(colorway)]
    # except ImportError:
    #     logger.debug("Plotly not available for dynamic theme colorway in get_theme_color.")
    # except Exception as e:
    #      logger.warning(f"Could not get Plotly theme color for '{color_name_or_index}': {e}")

    # Final fallback logic
    if fallback_color_hex:
        return fallback_color_hex
    # Absolute fallback if nothing else matches
    return settings.COLOR_TEXT_DARK # A sensible default text color


# --- HTML/Markdown Component Renderers ---

def render_kpi_card( # Renamed from render_web_kpi_card
    title: str,
    value_str: str,
    icon: str = "●", # Default generic icon
    status_level: str = "NEUTRAL", # E.g., "HIGH_RISK", "ACCEPTABLE", "GOOD_PERFORMANCE"
    delta_value: Optional[str] = None, # Renamed from delta
    delta_is_positive: Optional[bool] = None,
    help_text: Optional[str] = None,
    units: Optional[str] = None, # Added units parameter
    container_border: bool = True # Option to render within st.container(border=True)
) -> None:
    """
    Renders a Key Performance Indicator (KPI) card using Streamlit Markdown.
    Relies on CSS classes defined in `style_web_reports.css`.
    """
    # Convert Pythonic status_level (e.g., HIGH_RISK) to kebab-case for CSS (e.g., status-high-risk)
    css_status_class = f"status-{status_level.lower().replace('_', '-')}" if status_level else "status-neutral"
    
    delta_html_part = ""
    if delta_value:
        delta_class = "positive" if delta_is_positive else ("negative" if delta_is_positive is False else "neutral")
        delta_html_part = f'<p class="kpi-delta {delta_class}">{html.escape(str(delta_value))}</p>'

    tooltip_attribute = f'title="{html.escape(str(help_text))}"' if help_text else ''
    units_html_part = f" <span class='kpi-units'>{html.escape(str(units))}</span>" if units else ""

    kpi_card_html_content = f"""
    <div class="kpi-card {css_status_class}" {tooltip_attribute}>
        <div class="kpi-card-header">
            <span class="kpi-icon" role="img" aria-label="icon">{html.escape(icon)}</span>
            <h3 class="kpi-title">{html.escape(str(title))}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{html.escape(str(value_str))}{units_html_part}</p>
            {delta_html_part}
        </div>
    </div>
    """
    # The styles are assumed to be loaded globally by app.py from style_web_reports.css
    # No <style> block here to avoid repetition and ensure global CSS is the source of truth.
    
    if container_border:
        with st.container(border=True):
            st.markdown(kpi_card_html_content, unsafe_allow_html=True)
    else:
        st.markdown(kpi_card_html_content, unsafe_allow_html=True)


def render_traffic_light_indicator( # Renamed from render_web_traffic_light_indicator
    message: str,
    status_level: str, # E.g., "HIGH_RISK", "MODERATE_CONCERN", "ACCEPTABLE"
    details_text: Optional[str] = None,
    container_border: bool = False # Option for consistency
) -> None:
    """
    Renders a traffic light style indicator using Streamlit Markdown.
    Relies on CSS classes defined in `style_web_reports.css`.
    """
    css_dot_status_class = f"status-{status_level.lower().replace('_', '-')}" if status_level else "status-neutral"
    
    details_html_part = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text else ""

    traffic_light_html_content = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {css_dot_status_class}" role="img" aria-label="{status_level.replace('_', ' ')} status"></span>
        <span class="traffic-light-message">{html.escape(str(message))}</span>
        {details_html_part}
    </div>
    """
    
    if container_border:
        with st.container(border=True):
            st.markdown(traffic_light_html_content, unsafe_allow_html=True)
    else:
        st.markdown(traffic_light_html_content, unsafe_allow_html=True)


def display_custom_styled_kpi_box( # Specific to Population Dashboard's markdown boxes
    label: str,
    value: Union[str, int, float],
    sub_text: Optional[str] = None,
    highlight_edge_color: Optional[str] = None # e.g., settings.COLOR_RISK_HIGH
) -> None:
    """
    Renders a custom KPI box using Markdown, similar to the style in the original Population Dashboard.
    Relies on CSS classes like `custom-markdown-kpi-box`.
    """
    edge_class = "highlight-red-edge" if highlight_edge_color == settings.COLOR_RISK_HIGH else \
                 ("highlight-amber-edge" if highlight_edge_color == settings.COLOR_RISK_MODERATE else "")
                 # Add more edge color classes if needed, or pass class directly

    value_display = f"{value:,.0f}" if isinstance(value, (int, float)) and value >= 1000 else str(value)
    
    sub_text_html = f'<div class="custom-kpi-subtext-small">{html.escape(sub_text)}</div>' if sub_text else ""

    html_content = f"""
    <div class="custom-markdown-kpi-box {edge_class}">
        <div class="custom-kpi-label-top-condition">{html.escape(label)}</div>
        <div class="custom-kpi-value-large">{html.escape(value_display)}</div>
        {sub_text_html}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

# Potential additional UI helper:
# def render_alert_banner(message: str, level: str = "info", icon: Optional[str] = None):
#     """ Renders a more prominent alert banner using st.info, st.warning, st.error """
#     if level == "info":
#         st.info(f"{icon or 'ℹ️'} {message}" if icon else message)
#     elif level == "warning":
#         st.warning(f"{icon or '⚠️'} {message}" if icon else message)
#     elif level == "error":
#         st.error(f"{icon or '🚨'} {message}" if icon else message)
#     else: # Default to info
#         st.info(message)
