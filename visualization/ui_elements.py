# sentinel_project_root/visualization/ui_elements.py
# UI element rendering functions for Sentinel Health Co-Pilot Web Dashboards,
# primarily producing HTML/Markdown for Streamlit.

import streamlit as st 
import html
import logging
from typing import Optional, Dict, Any, Union

try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR) # Basic logger if import fails early
    logger = logging.getLogger(__name__)
    logger.error(f"Critical import error in ui_elements.py: {e}. Ensure config.py is accessible.")
    # Define fallback settings attributes if config fails to import, to prevent NameErrors later
    class FallbackSettings:
        COLOR_RISK_HIGH = "#FF4B4B"
        COLOR_RISK_MODERATE = "#FFA500"
        COLOR_RISK_LOW = "#2ECC71"
        COLOR_RISK_NEUTRAL = "#757575"
        COLOR_ACTION_PRIMARY = "#007BFF"
        COLOR_ACTION_SECONDARY = "#6C757D"
        COLOR_POSITIVE_DELTA = "#28A745"
        COLOR_NEGATIVE_DELTA = "#DC3545"
        COLOR_TEXT_DARK = "#212529"
        COLOR_TEXT_HEADINGS_MAIN = "#343A40"
        COLOR_ACCENT_BRIGHT = "#FFC107"
        COLOR_BACKGROUND_CONTENT = "#FFFFFF"
        COLOR_BACKGROUND_PAGE = "#F8F9FA"
        COLOR_BORDER_LIGHT = "#DEE2E6"
        COLOR_BORDER_MEDIUM = "#CED4DA"
        COLOR_BACKGROUND_WHITE = "#FFFFFF"
        COLOR_BACKGROUND_SUBTLE = "#F1F3F5"
        COLOR_TEXT_LINK_DEFAULT = "#007BFF"
        COLOR_TEXT_MUTED = "#6C757D"
        LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {} # Empty dict as fallback
    settings = FallbackSettings()
    logger.warning("ui_elements.py: Using fallback settings due to import error.")


logger = logging.getLogger(__name__)

# --- Color Utility ---
# Helper to safely get attributes from settings
def _get_setting_attr(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

def get_theme_color(
    color_name_or_index: Any,
    color_category: str = "general",
    fallback_color_hex: Optional[str] = None # Specific fallback for this call
) -> str:
    """
    Retrieves a color from the application's theme settings or a predefined list.
    """
    # Direct mapping for named theme colors from settings.py
    direct_color_map: Dict[str, str] = {
        "risk_high": _get_setting_attr('COLOR_RISK_HIGH', "#FF4B4B"),
        "risk_moderate": _get_setting_attr('COLOR_RISK_MODERATE', "#FFA500"),
        "risk_low": _get_setting_attr('COLOR_RISK_LOW', "#2ECC71"),
        "risk_neutral": _get_setting_attr('COLOR_RISK_NEUTRAL', "#757575"),
        "action_primary": _get_setting_attr('COLOR_ACTION_PRIMARY', "#007BFF"),
        "action_secondary": _get_setting_attr('COLOR_ACTION_SECONDARY', "#6C757D"),
        "positive_delta": _get_setting_attr('COLOR_POSITIVE_DELTA', "#28A745"),
        "negative_delta": _get_setting_attr('COLOR_NEGATIVE_DELTA', "#DC3545"),
        "text_dark": _get_setting_attr('COLOR_TEXT_DARK', "#212529"),
        "headings_main": _get_setting_attr('COLOR_TEXT_HEADINGS_MAIN', "#343A40"),
        "accent_bright": _get_setting_attr('COLOR_ACCENT_BRIGHT', "#FFC107"),
        "background_content": _get_setting_attr('COLOR_BACKGROUND_CONTENT', "#FFFFFF"),
        "background_page": _get_setting_attr('COLOR_BACKGROUND_PAGE', "#F8F9FA"),
        "border_light": _get_setting_attr('COLOR_BORDER_LIGHT', "#DEE2E6"),
        "border_medium": _get_setting_attr('COLOR_BORDER_MEDIUM', "#CED4DA"),
        "white": _get_setting_attr('COLOR_BACKGROUND_WHITE', "#FFFFFF"),
        "subtle_background": _get_setting_attr('COLOR_BACKGROUND_SUBTLE', "#F1F3F5"),
        "link_default": _get_setting_attr('COLOR_TEXT_LINK_DEFAULT', "#007BFF"),
        "text_muted": _get_setting_attr('COLOR_TEXT_MUTED', "#6C757D") # Added muted text color to map
    }

    if isinstance(color_name_or_index, str):
        color_name_lower = color_name_or_index.lower().strip()
        if color_name_lower in direct_color_map:
            return direct_color_map[color_name_lower]
        
        # Check legacy disease colors (case-sensitive original key)
        legacy_disease_colors = _get_setting_attr('LEGACY_DISEASE_COLORS_WEB', {})
        if color_category == "disease" and isinstance(legacy_disease_colors, dict) and color_name_or_index in legacy_disease_colors:
            return legacy_disease_colors[color_name_or_index]

    # Fallback general colorway (ensure these also use _get_setting_attr for resilience)
    general_sentinel_colorway = [
        direct_color_map["action_primary"], direct_color_map["risk_low"], direct_color_map["risk_moderate"],
        direct_color_map["accent_bright"], direct_color_map["action_secondary"],
        "#00ACC1", "#5E35B1", "#FF7043" # Generic fallbacks if settings are missing
    ]
    if isinstance(color_name_or_index, int) and color_category == "general":
        if general_sentinel_colorway: # Ensure list is not empty
            return general_sentinel_colorway[color_name_or_index % len(general_sentinel_colorway)]
        # Fall through if colorway itself is empty (should not happen with defaults)

    if fallback_color_hex and isinstance(fallback_color_hex, str):
        return fallback_color_hex
    
    logger.debug(f"Color '{color_name_or_index}' (category: '{color_category}') not found. Using absolute fallback color_text_muted.")
    return direct_color_map["text_muted"] # Absolute fallback


# --- HTML/Markdown Component Renderers ---

def render_kpi_card(
    title: str,
    value_str: str, # Main value, expected as string (e.g., "N/A", "123", "10.5%")
    icon: str = "●", 
    status_level: str = "NEUTRAL", # E.g., "HIGH_RISK", "MODERATE_CONCERN", "ACCEPTABLE", "GOOD_PERFORMANCE", "NO_DATA"
    delta_value: Optional[str] = None, # E.g., "+5", "-2.1%"
    delta_is_positive: Optional[bool] = None, # True for good change, False for bad
    help_text: Optional[str] = None,
    units: Optional[str] = None, # E.g., "%", "days", "cases"
    container_border: bool = True # Controls if st.container(border=True) is used
) -> None:
    """Renders a KPI card using Streamlit Markdown and custom CSS. Assumes CSS classes are defined."""
    
    # Sanitize inputs and prepare CSS classes/HTML parts
    safe_title = html.escape(str(title))
    safe_value_str = html.escape(str(value_str))
    safe_icon = html.escape(icon)
    safe_status_level = html.escape(str(status_level).lower().replace('_', '-').strip()) if status_level else "neutral"
    css_status_class = f"status-{safe_status_level}"

    delta_html = ""
    if delta_value is not None and str(delta_value).strip(): # Check if delta_value is meaningful
        safe_delta_value = html.escape(str(delta_value))
        delta_color_class = "neutral" # Default class for delta
        if delta_is_positive is True: 
            delta_color_class = "positive"
        elif delta_is_positive is False: 
            delta_color_class = "negative"
        delta_html = f'<p class="kpi-delta {delta_color_class}">{safe_delta_value}</p>'

    tooltip_attr_str = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''
    units_html_str = f" <span class='kpi-units'>{html.escape(str(units))}</span>" if units and str(units).strip() else ""

    kpi_html_content = f"""
    <div class="kpi-card {css_status_class}" {tooltip_attr_str}>
        <div class="kpi-card-header">
            <span class="kpi-icon" role="img" aria-label="icon">{safe_icon}</span>
            <h3 class="kpi-title">{safe_title}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{safe_value_str}{units_html_str}</p>
            {delta_html}
        </div>
    </div>
    """
    try:
        # Use st.container only if a border is requested and supported.
        # Otherwise, just use st.markdown directly.
        if container_border:
            try: # Attempt to use border=True, introduced in later Streamlit versions
                with st.container(border=True):
                    st.markdown(kpi_html_content, unsafe_allow_html=True)
            except TypeError: # Fallback if 'border' kwarg is not supported
                logger.debug("KPI Card: st.container(border=True) not supported by this Streamlit version. Rendering without explicit container for border.")
                # For older versions, the border would need to be part of the kpi-card CSS
                st.markdown(kpi_html_content, unsafe_allow_html=True) 
        else:
            st.markdown(kpi_html_content, unsafe_allow_html=True)
    except Exception as e_render_kpi: 
        logger.error(f"KPI Card: Error during st.markdown for title '{safe_title}': {e_render_kpi}", exc_info=True)
        st.error(f"Error rendering KPI: {safe_title}") # Show a user-friendly error


def render_traffic_light_indicator(
    message: str,
    status_level: str, # E.g., "HIGH_RISK", "MODERATE_CONCERN", "ACCEPTABLE"
    details_text: Optional[str] = None,
    container_border: bool = False # Controls if st.container(border=True) is used
) -> None:
    """Renders a traffic light style indicator using Streamlit Markdown and custom CSS."""
    safe_message = html.escape(str(message))
    safe_status_level = html.escape(str(status_level).lower().replace('_', '-').strip()) if status_level else "neutral"
    css_dot_class = f"status-{safe_status_level}"
    
    details_html_str = ""
    if details_text and str(details_text).strip():
        details_html_str = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>'

    traffic_html_content = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {css_dot_class}" role="img" aria-label="{html.escape(str(status_level).replace('_', ' '))} status"></span>
        <span class="traffic-light-message">{safe_message}</span>
        {details_html_str}
    </div>
    """
    try:
        if container_border:
            try:
                with st.container(border=True):
                    st.markdown(traffic_html_content, unsafe_allow_html=True)
            except TypeError:
                logger.debug("Traffic Light: st.container(border=True) not supported. Rendering without explicit container.")
                st.markdown(traffic_html_content, unsafe_allow_html=True)
        else:
            st.markdown(traffic_html_content, unsafe_allow_html=True)
    except Exception as e_render_traffic:
        logger.error(f"Traffic Light: Error during st.markdown for message '{safe_message}': {e_render_traffic}", exc_info=True)
        st.error(f"Error rendering indicator: {safe_message}")


def display_custom_styled_kpi_box(
    label: str,
    value: Union[str, int, float], # The main value to display
    sub_text: Optional[str] = None, # Additional smaller text below the value
    highlight_edge_color: Optional[str] = None # Hex color string for left edge highlight
) -> None:
    """
    Renders a custom KPI box using Markdown, styled via `custom-markdown-kpi-box` CSS.
    The CSS should define classes like .highlight-red-edge based on hex colors.
    """
    safe_label = html.escape(str(label))
    
    # Format the main value for display
    value_display_str = str(value) # Default to string representation
    if isinstance(value, (int, float)):
        if pd.notna(value): # Ensure it's not NaN before formatting
            if abs(value) >= 1000: 
                value_display_str = f"{value:,.0f}" # Comma separated for thousands
            elif isinstance(value, float) and not value.is_integer():
                value_display_str = f"{value:.1f}" # One decimal for non-integer floats
            else: 
                value_display_str = f"{int(value)}" # Clean integer
        else:
            value_display_str = "N/A" # Display for NaN numeric values
    safe_value_display = html.escape(value_display_str)

    sub_text_html_part_str = ""
    if sub_text and str(sub_text).strip():
        sub_text_html_part_str = f'<div class="custom-kpi-subtext-small">{html.escape(str(sub_text))}</div>'
    
    # Determine edge highlight class
    edge_highlight_class = ""
    if highlight_edge_color and isinstance(highlight_edge_color, str):
        # This relies on specific CSS classes being defined e.g., .highlight-red-edge
        # The mapping should be robust using _get_setting_attr for color constants
        color_upper = highlight_edge_color.upper()
        if color_upper == _get_setting_attr('COLOR_RISK_HIGH', "").upper(): edge_highlight_class = "highlight-red-edge"
        elif color_upper == _get_setting_attr('COLOR_RISK_MODERATE', "").upper(): edge_highlight_class = "highlight-amber-edge"
        elif color_upper == _get_setting_attr('COLOR_RISK_LOW', "").upper(): edge_highlight_class = "highlight-green-edge"
        # Add more color-to-class mappings if needed, or consider a more generic CSS approach
        else: logger.debug(f"No specific CSS class mapping for highlight_edge_color: {highlight_edge_color}")

    box_html_content = f"""
    <div class="custom-markdown-kpi-box {edge_highlight_class}">
        <div class="custom-kpi-label-top-condition">{safe_label}</div>
        <div class="custom-kpi-value-large">{safe_value_display}</div>
        {sub_text_html_part_str}
    </div>
    """
    try:
        st.markdown(box_html_content, unsafe_allow_html=True)
    except Exception as e_render_custom_kpi:
        logger.error(f"Custom KPI Box: Error during st.markdown for label '{safe_label}': {e_render_custom_kpi}", exc_info=True)
        st.error(f"Error rendering custom KPI: {safe_label}")
