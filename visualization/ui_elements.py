# sentinel_project_root/visualization/ui_elements.py
# UI element rendering functions for Sentinel Health Co-Pilot Web Dashboards,
# primarily producing HTML/Markdown for Streamlit.

import streamlit as st 
import html
import logging
from typing import Optional, Dict, Any, Union
import pandas as pd # Imported for pd.notna, pd.isna

# Attempt to import settings, with a robust fallback mechanism
try:
    from config import settings
except ImportError as e:
    # This block provides fallback settings if the main config.py is missing or has import errors.
    # It allows ui_elements.py to be imported and potentially used with default styling.
    logging.basicConfig(level=logging.INFO) # Ensure logger is available
    logger_init = logging.getLogger(__name__) # Use a different name to avoid conflict
    logger_init.error(f"Critical import error for 'config.settings' in ui_elements.py: {e}. Using fallback settings.")
    
    class FallbackSettings:
        """Provides default values if the main settings module cannot be imported."""
        # Color definitions (using common web-safe colors as fallbacks)
        COLOR_RISK_HIGH = "#D32F2F"  # Red
        COLOR_RISK_MODERATE = "#FFA000" # Amber/Orange
        COLOR_RISK_LOW = "#388E3C"   # Green
        COLOR_RISK_NEUTRAL = "#757575" # Grey
        COLOR_ACTION_PRIMARY = "#1976D2" # Blue
        COLOR_ACTION_SECONDARY = "#607D8B" # Blue Grey
        COLOR_POSITIVE_DELTA = "#388E3C" # Green
        COLOR_NEGATIVE_DELTA = "#D32F2F" # Red
        COLOR_TEXT_DARK = "#212121"      # Very Dark Grey
        COLOR_TEXT_HEADINGS_MAIN = "#333333" # Dark Grey
        COLOR_TEXT_MUTED = "#757575"     # Medium Grey
        COLOR_TEXT_LINK_DEFAULT = "#1976D2" # Blue
        COLOR_ACCENT_BRIGHT = "#FFC107"  # Amber/Yellow
        COLOR_BACKGROUND_CONTENT = "#FFFFFF" # White
        COLOR_BACKGROUND_PAGE = "#ECEFF1"    # Light Grey
        COLOR_BACKGROUND_SUBTLE = "#F5F5F5"  # Very Light Grey
        COLOR_BACKGROUND_WHITE = "#FFFFFF"   # White
        COLOR_BORDER_LIGHT = "#E0E0E0"   # Light Grey Border
        COLOR_BORDER_MEDIUM = "#BDBDBD"  # Medium Grey Border
        LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {} # Empty dict
        # Add any other settings attributes used by this module with sensible defaults
        PROJECT_ROOT_DIR = "." # Current directory as fallback
        APP_FAVICON_PATH = "assets/favicon.ico" # Example
        APP_LAYOUT = "wide"

    settings = FallbackSettings() # Instantiate the fallback
    logger_init.warning("ui_elements.py: Using fallback settings due to an import error with 'config.settings'. Some visual styles may differ.")

logger = logging.getLogger(__name__)

# --- Helper for Safe Settings Access ---
def _get_setting_attr(attr_name: str, default_value: Any) -> Any:
    """Safely gets an attribute from the settings object, providing a default if not found."""
    return getattr(settings, attr_name, default_value)

# --- Color Utility ---
def get_theme_color(
    color_name_or_index: Any,
    color_category: str = "general",
    fallback_color_hex: Optional[str] = None # Specific fallback for this call
) -> str:
    """
    Retrieves a color from the application's theme settings or a predefined list.
    Ensures robust handling of missing settings or invalid inputs.
    """
    # Direct mapping for named theme colors from settings.py using the helper
    direct_color_map: Dict[str, str] = {
        "risk_high": _get_setting_attr('COLOR_RISK_HIGH', "#D32F2F"),
        "risk_moderate": _get_setting_attr('COLOR_RISK_MODERATE', "#FFA000"),
        "risk_low": _get_setting_attr('COLOR_RISK_LOW', "#388E3C"),
        "risk_neutral": _get_setting_attr('COLOR_RISK_NEUTRAL', "#757575"),
        "action_primary": _get_setting_attr('COLOR_ACTION_PRIMARY', "#1976D2"),
        "action_secondary": _get_setting_attr('COLOR_ACTION_SECONDARY', "#607D8B"),
        "positive_delta": _get_setting_attr('COLOR_POSITIVE_DELTA', "#388E3C"),
        "negative_delta": _get_setting_attr('COLOR_NEGATIVE_DELTA', "#D32F2F"),
        "text_dark": _get_setting_attr('COLOR_TEXT_DARK', "#212121"),
        "headings_main": _get_setting_attr('COLOR_TEXT_HEADINGS_MAIN', "#333333"),
        "accent_bright": _get_setting_attr('COLOR_ACCENT_BRIGHT', "#FFC107"),
        "background_content": _get_setting_attr('COLOR_BACKGROUND_CONTENT', "#FFFFFF"),
        "background_page": _get_setting_attr('COLOR_BACKGROUND_PAGE', "#ECEFF1"),
        "border_light": _get_setting_attr('COLOR_BORDER_LIGHT', "#E0E0E0"),
        "border_medium": _get_setting_attr('COLOR_BORDER_MEDIUM', "#BDBDBD"),
        "white": _get_setting_attr('COLOR_BACKGROUND_WHITE', "#FFFFFF"),
        "subtle_background": _get_setting_attr('COLOR_BACKGROUND_SUBTLE', "#F5F5F5"),
        "link_default": _get_setting_attr('COLOR_TEXT_LINK_DEFAULT', "#1976D2"),
        "text_muted": _get_setting_attr('COLOR_TEXT_MUTED', "#757575")
    }

    if isinstance(color_name_or_index, str):
        color_name_lower = color_name_or_index.lower().strip()
        if color_name_lower in direct_color_map:
            return direct_color_map[color_name_lower]
        
        legacy_disease_colors = _get_setting_attr('LEGACY_DISEASE_COLORS_WEB', {})
        if color_category == "disease" and isinstance(legacy_disease_colors, dict) and color_name_or_index in legacy_disease_colors:
            # Ensure the color value itself is a string
            color_val = legacy_disease_colors[color_name_or_index]
            return str(color_val) if color_val is not None else direct_color_map["text_muted"]


    # Fallback general colorway
    general_sentinel_colorway = [
        direct_color_map["action_primary"], direct_color_map["risk_low"], 
        direct_color_map["risk_moderate"], direct_color_map["accent_bright"], 
        direct_color_map["action_secondary"],
        "#00ACC1", "#5E35B1", "#FF7043" # Generic fallbacks if settings for first 5 are missing
    ]
    if isinstance(color_name_or_index, int) and color_category == "general":
        if general_sentinel_colorway: # Ensure list is not empty
            return general_sentinel_colorway[color_name_or_index % len(general_sentinel_colorway)]
        # Fall through if colorway itself is empty (should not happen with defaults)

    if fallback_color_hex and isinstance(fallback_color_hex, str) and fallback_color_hex.startswith("#"):
        return fallback_color_hex
    
    logger.debug(f"Color '{color_name_or_index}' (category: '{color_category}') not found. Using absolute fallback: text_muted.")
    return direct_color_map["text_muted"] 


# --- HTML/Markdown Component Renderers ---

def render_kpi_card(
    title: str,
    value_str: str, # Main value, expected as string (e.g., "N/A", "123", "10.5%")
    icon: str = "●", 
    status_level: str = "NEUTRAL", # E.g., "HIGH_RISK", "MODERATE_CONCERN", "ACCEPTABLE"
    delta_value: Optional[str] = None, # E.g., "+5", "-2.1%"
    delta_is_positive: Optional[bool] = None, # True for good change, False for bad
    help_text: Optional[str] = None,
    units: Optional[str] = None, # E.g., "%", "days", "cases"
    container_border: bool = True # If True, adds 'kpi-card-bordered' class for CSS styling
) -> None:
    """
    Renders a KPI card using Streamlit Markdown and custom CSS.
    The border is now controlled by adding a CSS class if container_border is True.
    """
    
    # Sanitize inputs and prepare CSS classes/HTML parts
    safe_title = html.escape(str(title))
    safe_value_str = html.escape(str(value_str)) # Value is already string
    safe_icon = html.escape(str(icon))
    
    # Sanitize status_level for CSS class
    safe_status_for_css = "neutral" # Default
    if status_level and isinstance(status_level, str):
        safe_status_for_css = html.escape(status_level.lower().replace('_', '-').strip())
    css_status_class = f"status-{safe_status_for_css}"

    delta_html_content = ""
    if delta_value is not None and str(delta_value).strip(): 
        safe_delta_val = html.escape(str(delta_value))
        delta_css_class = "neutral" 
        if delta_is_positive is True: delta_css_class = "positive"
        elif delta_is_positive is False: delta_css_class = "negative"
        delta_html_content = f'<p class="kpi-delta {delta_css_class}">{safe_delta_val}</p>'

    tooltip_html_attr = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''
    units_html_content = f" <span class='kpi-units'>{html.escape(str(units))}</span>" if units and str(units).strip() else ""
    
    border_class_str = " kpi-card-bordered" if container_border else ""

    kpi_html_final = f"""
    <div class="kpi-card {css_status_class}{border_class_str}" {tooltip_html_attr}>
        <div class="kpi-card-header">
            <span class="kpi-icon" role="img" aria-label="icon">{safe_icon}</span>
            <h3 class="kpi-title">{safe_title}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{safe_value_str}{units_html_content}</p>
            {delta_html_content}
        </div>
    </div>
    """
    try:
        st.markdown(kpi_html_final, unsafe_allow_html=True)
    except Exception as e_render_kpi: 
        logger.error(f"KPI Card: Error during st.markdown for title '{safe_title}': {e_render_kpi}", exc_info=True)
        # Attempt to display a simpler error message in the UI if possible
        try: st.error(f"Error rendering KPI: {safe_title}")
        except: pass # Silently fail if st.error itself fails (e.g., outside Streamlit context)


def render_traffic_light_indicator(
    message: str,
    status_level: str, # E.g., "HIGH_RISK", "MODERATE_CONCERN", "ACCEPTABLE"
    details_text: Optional[str] = None,
    container_border: bool = False # If True, adds 'traffic-light-bordered' class
) -> None:
    """Renders a traffic light style indicator using Streamlit Markdown and custom CSS."""
    safe_message = html.escape(str(message))
    safe_status_for_css = "neutral"
    if status_level and isinstance(status_level, str):
        safe_status_for_css = html.escape(status_level.lower().replace('_', '-').strip())
    css_dot_class = f"status-{safe_status_for_css}"
    
    details_html_content = ""
    if details_text and str(details_text).strip():
        details_html_content = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>'

    border_class_str = " traffic-light-bordered" if container_border else ""

    traffic_html_final = f"""
    <div class="traffic-light-indicator{border_class_str}">
        <span class="traffic-light-dot {css_dot_class}" role="img" aria-label="{html.escape(str(status_level).replace('_', ' '))} status"></span>
        <span class="traffic-light-message">{safe_message}</span>
        {details_html_content}
    </div>
    """
    try:
        st.markdown(traffic_html_final, unsafe_allow_html=True)
    except Exception as e_render_traffic:
        logger.error(f"Traffic Light: Error during st.markdown for message '{safe_message}': {e_render_traffic}", exc_info=True)
        try: st.error(f"Error rendering indicator: {safe_message}")
        except: pass


def display_custom_styled_kpi_box(
    label: str,
    value: Union[str, int, float], # The main value to display
    sub_text: Optional[str] = None, # Additional smaller text below the value
    highlight_edge_color: Optional[str] = None # Hex color string for left edge highlight (maps to CSS class)
) -> None:
    """
    Renders a custom KPI box using Markdown. Relies on CSS class `custom-markdown-kpi-box`
    and potentially `highlight-red-edge`, `highlight-amber-edge`, `highlight-green-edge`.
    """
    safe_label = html.escape(str(label))
    
    # Format the main value for display
    value_display_str = "N/A" # Default if value is NaN or unformattable
    if pd.notna(value): # Check if value is not NaN or None
        if isinstance(value, (int, float)):
            if abs(value) >= 1000: 
                value_display_str = f"{value:,.0f}" 
            elif isinstance(value, float) and not value.is_integer() and not np.isinf(value):
                value_display_str = f"{value:.1f}" 
            elif not np.isinf(value): # Integer or whole float
                value_display_str = f"{int(value)}"
            else: # Handle infinities
                value_display_str = "Inf" if value > 0 else "-Inf"
        else: # If not int/float but notna, convert to string
            value_display_str = str(value)
    safe_value_display = html.escape(value_display_str)

    sub_text_html_part = ""
    if sub_text and str(sub_text).strip():
        sub_text_html_part = f'<div class="custom-kpi-subtext-small">{html.escape(str(sub_text))}</div>'
    
    edge_highlight_css_class = ""
    if highlight_edge_color and isinstance(highlight_edge_color, str):
        color_upper_case = highlight_edge_color.upper()
        # Robustly get color settings with fallbacks
        if color_upper_case == _get_setting_attr('COLOR_RISK_HIGH', "#D32F2F").upper(): edge_highlight_css_class = "highlight-red-edge"
        elif color_upper_case == _get_setting_attr('COLOR_RISK_MODERATE', "#FFA000").upper(): edge_highlight_css_class = "highlight-amber-edge"
        elif color_upper_case == _get_setting_attr('COLOR_RISK_LOW', "#388E3C").upper(): edge_highlight_css_class = "highlight-green-edge"
        else: logger.debug(f"No specific CSS class mapping for highlight_edge_color: {highlight_edge_color}")

    box_html_final = f"""
    <div class="custom-markdown-kpi-box {edge_highlight_css_class}">
        <div class="custom-kpi-label-top-condition">{safe_label}</div>
        <div class="custom-kpi-value-large">{safe_value_display}</div>
        {sub_text_html_part}
    </div>
    """
    try:
        st.markdown(box_html_final, unsafe_allow_html=True)
    except Exception as e_render_custom_kpi:
        logger.error(f"Custom KPI Box: Error during st.markdown for label '{safe_label}': {e_render_custom_kpi}", exc_info=True)
        try: st.error(f"Error rendering custom KPI: {safe_label}")
        except: pass
