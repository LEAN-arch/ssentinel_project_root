:# sentinel_project_root/visualization/ui_elements.py
UI element rendering functions for Sentinel Health Co-Pilot Web Dashboards,
primarily producing HTML/Markdown for Streamlit.
import streamlit as st
import html
import logging
from typing import Optional, Dict, Any, Union
import pandas as pd # For pd.notna, pd.isna, which are robust NaN checks
--- Robust Settings Import with Fallback ---
SETTINGS_IMPORTED_SUCCESSFULLY = False
try:
from config import settings
SETTINGS_IMPORTED_SUCCESSFULLY = True
except ImportError as e_config_import:
logging.basicConfig(level=logging.INFO)
_logger_init_fallback = logging.getLogger(name)
_logger_init_fallback.error(
f"CRITICAL IMPORT ERROR: 'config.settings' could not be imported in ui_elements.py: {e_config_import}. "
"Using internal fallback settings. THIS IS NOT INTENDED FOR PRODUCTION."
)
class FallbackSettings:
    COLOR_RISK_HIGH = "#D32F2F"; COLOR_RISK_MODERATE = "#FFA000"; COLOR_RISK_LOW = "#388E3C";
    COLOR_RISK_NEUTRAL = "#757575"; COLOR_ACTION_PRIMARY = "#1976D2"; COLOR_ACTION_SECONDARY = "#607D8B";
    COLOR_POSITIVE_DELTA = "#388E3C"; COLOR_NEGATIVE_DELTA = "#D32F2F"; COLOR_TEXT_DARK = "#212121";
    COLOR_TEXT_HEADINGS_MAIN = "#333333"; COLOR_TEXT_MUTED = "#757575";
    COLOR_TEXT_LINK_DEFAULT = "#1976D2"; COLOR_ACCENT_BRIGHT = "#FFC107"; 
    COLOR_BACKGROUND_CONTENT = "#FFFFFF"; COLOR_BACKGROUND_PAGE = "#ECEFF1"; 
    COLOR_BACKGROUND_SUBTLE = "#F5F5F5"; COLOR_BACKGROUND_WHITE = "#FFFFFF";   
    COLOR_BORDER_LIGHT = "#E0E0E0"; COLOR_BORDER_MEDIUM = "#BDBDBD";  
    LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {} 
settings = FallbackSettings()
Use code with caution.
logger = logging.getLogger(name)
--- Safe Settings Access Helper ---
def _get_setting_attr(attr_name: str, default_value: Any) -> Any:
val = getattr(settings, attr_name, default_value)
if val is default_value and not hasattr(settings, attr_name) and default_value is not None:
logger.debug(f"Setting '{attr_name}' not found; using provided default: '{default_value}'.")
return val
--- Color Utility ---
def get_theme_color(
color_name_or_index: Any,
color_category: str = "general",
fallback_color_hex: Optional[str] = None
) -> str:
direct_color_map: Dict[str, str] = {
"risk_high": _get_setting_attr('COLOR_RISK_HIGH', "#E53935"), "risk_moderate": _get_setting_attr('COLOR_RISK_MODERATE', "#FFB300"),
"risk_low": _get_setting_attr('COLOR_RISK_LOW', "#43A047"), "risk_neutral": _get_setting_attr('COLOR_RISK_NEUTRAL', "#757575"),
"action_primary": _get_setting_attr('COLOR_ACTION_PRIMARY', "#1E88E5"), "action_secondary": _get_setting_attr('COLOR_ACTION_SECONDARY', "#78909C"),
"positive_delta": _get_setting_attr('COLOR_POSITIVE_DELTA', "#4CAF50"), "negative_delta": _get_setting_attr('COLOR_NEGATIVE_DELTA', "#F44336"),
"text_dark": _get_setting_attr('COLOR_TEXT_DARK', "#263238"), "headings_main": _get_setting_attr('COLOR_TEXT_HEADINGS_MAIN', "#37474F"),
"accent_bright": _get_setting_attr('COLOR_ACCENT_BRIGHT', "#FFCA28"), "background_content": _get_setting_attr('COLOR_BACKGROUND_CONTENT', "#FFFFFF"),
"background_page": _get_setting_attr('COLOR_BACKGROUND_PAGE', "#CFD8DC"), "border_light": _get_setting_attr('COLOR_BORDER_LIGHT', "#ECEFF1"),
"border_medium": _get_setting_attr('COLOR_BORDER_MEDIUM', "#B0BEC5"), "white": _get_setting_attr('COLOR_BACKGROUND_WHITE', "#FFFFFF"),
"subtle_background": _get_setting_attr('COLOR_BACKGROUND_SUBTLE', "#FAFAFA"), "link_default": _get_setting_attr('COLOR_TEXT_LINK_DEFAULT', "#0277BD"),
"text_muted": _get_setting_attr('COLOR_TEXT_MUTED', "#616161")
}
if isinstance(color_name_or_index, str):
    color_name_key = color_name_or_index.lower().strip()
    if color_name_key in direct_color_map:
        return direct_color_map[color_name_key]
    
    legacy_disease_colors_map = _get_setting_attr('LEGACY_DISEASE_COLORS_WEB', {})
    if color_category == "disease" and isinstance(legacy_disease_colors_map, dict) and color_name_or_index in legacy_disease_colors_map:
        return str(legacy_disease_colors_map[color_name_or_index])

general_theme_colorway = [
    direct_color_map["action_primary"], direct_color_map["risk_low"], direct_color_map["risk_moderate"],
    direct_color_map["accent_bright"], direct_color_map["action_secondary"],
    "#00BCD4", "#673AB7", "#FF5722"
]
if isinstance(color_name_or_index, int) and color_category == "general":
    return general_theme_colorway[color_name_or_index % len(general_theme_colorway)]

if fallback_color_hex: return fallback_color_hex
return direct_color_map["text_muted"]
Use code with caution.
--- HTML/Markdown Component Renderers ---
def render_kpi_card(
title: str, value_str: str, icon: str = "●", status_level: str = "NEUTRAL",
delta_value: Optional[str] = None, delta_is_positive: Optional[bool] = None,
help_text: Optional[str] = None, units: Optional[str] = None, container_border: bool = True
) -> None:
safe_title = html.escape(str(title).strip())
safe_value_str = html.escape(str(value_str).strip())
safe_icon = html.escape(str(icon).strip())
status_str_clean = str(status_level).lower().replace('_', '-').strip() if status_level else "neutral"
css_status_class = f"status-{html.escape(status_str_clean)}"

delta_html_block = ""
if delta_value is not None and str(delta_value).strip(): 
    safe_delta_str = html.escape(str(delta_value).strip())
    delta_color_css_class = "neutral" 
    if delta_is_positive is True: delta_color_css_class = "positive"
    elif delta_is_positive is False: delta_color_css_class = "negative"
    delta_html_block = f'<p class="kpi-delta {delta_color_css_class}">{safe_delta_str}</p>'

tooltip_html_attribute = f'title="{html.escape(str(help_text).strip())}"' if help_text and str(help_text).strip() else ''
units_html_block = f" <span class='kpi-units'>{html.escape(str(units).strip())}</span>" if units and str(units).strip() else ""

border_css_class_str = " kpi-card-bordered" if container_border else ""

# CORRECTED: Changed class names to match CSS (e.g., "kpi-card" instead of "sentinel-kpi-card")
kpi_html_content = f"""
<div class="kpi-card {css_status_class}{border_css_class_str}" {tooltip_html_attribute}>
    <div class="kpi-card-header">
        <span class="kpi-icon" role="img" aria-label="KPI icon">{safe_icon}</span>
        <h3 class="kpi-title">{safe_title}</h3>
    </div>
    <div class="kpi-body">
        <p class="kpi-value">{safe_value_str}{units_html_block}</p>
        {delta_html_block}
    </div>
</div>
"""
try:
    # CORRECTED: Wrap the markdown in a container to isolate it from parent layout elements.
    with st.container():
        st.markdown(kpi_html_content, unsafe_allow_html=True)
except Exception as e_render_kpi: 
    logger.error(f"KPI Card Render Error: Title '{safe_title}', Details: {e_render_kpi}", exc_info=True)
    try: st.error(f"Error rendering KPI: {safe_title}")
    except: pass
Use code with caution.
def render_traffic_light_indicator(
message: str, status_level: str, details_text: Optional[str] = None, container_border: bool = False
) -> None:
safe_message = html.escape(str(message).strip())
status_str_clean = str(status_level).lower().replace('_', '-').strip() if status_level else "neutral"
css_dot_class_name = f"status-{html.escape(status_str_clean)}"
details_html_block = ""
if details_text and str(details_text).strip():
    details_html_block = f'<span class="traffic-light-details">{html.escape(str(details_text).strip())}</span>'

border_css_class_str = " traffic-light-bordered" if container_border else ""
aria_label_status = html.escape(str(status_level).replace('_', ' ').strip())

# CORRECTED: Changed class names to match CSS
traffic_html_content = f"""
<div class="traffic-light-indicator{border_css_class_str}">
    <span class="traffic-light-dot {css_dot_class_name}" role="img" aria-label="{aria_label_status} status indicator"></span>
    <span class="traffic-light-message">{safe_message}</span>
    {details_html_block}
</div>
"""
try:
    with st.container():
        st.markdown(traffic_html_content, unsafe_allow_html=True)
except Exception as e_render_traffic:
    logger.error(f"Traffic Light Indicator Render Error: Message '{safe_message}', Details: {e_render_traffic}", exc_info=True)
    try: st.error(f"Error rendering indicator: {safe_message}")
    except: pass
Use code with caution.
def display_custom_styled_kpi_box(
label: str, value: Union[str, int, float, None],
sub_text: Optional[str] = None, highlight_edge_color: Optional[str] = None
) -> None:
safe_label = html.escape(str(label).strip())
value_display_formatted = "N/A"
if pd.notna(value):
    if isinstance(value, (int, float)):
        if np.isinf(value): value_display_formatted = "Inf" if value > 0 else "-Inf"
        elif abs(value) >= 1000: value_display_formatted = f"{value:,.0f}" 
        elif isinstance(value, float) and not value.is_integer(): value_display_formatted = f"{value:.1f}" 
        else: value_display_formatted = f"{int(value)}"
    else:
        value_display_formatted = str(value).strip()
safe_value_display = html.escape(value_display_formatted)

sub_text_html_block = ""
if sub_text and str(sub_text).strip():
    sub_text_html_block = f'<div class="custom-kpi-subtext-small">{html.escape(str(sub_text).strip())}</div>'

edge_highlight_css_class_name = ""
if highlight_edge_color and isinstance(highlight_edge_color, str) and highlight_edge_color.strip():
    color_upper = highlight_edge_color.strip().upper()
    color_class_mappings = {
        _get_setting_attr('COLOR_RISK_HIGH', "#D32F2F").upper(): "highlight-red-edge",
        _get_setting_attr('COLOR_RISK_MODERATE', "#FFA000").upper(): "highlight-amber-edge",
        _get_setting_attr('COLOR_RISK_LOW', "#388E3C").upper(): "highlight-green-edge",
        _get_setting_attr('COLOR_ACTION_PRIMARY', "#1976D2").upper(): "highlight-blue-edge",
    }
    edge_highlight_css_class_name = color_class_mappings.get(color_upper, "")
    if not edge_highlight_css_class_name:
        logger.debug(f"No specific CSS class mapping for highlight_edge_color: '{highlight_edge_color}'.")

box_html_final_content = f"""
<div class="custom-markdown-kpi-box {edge_highlight_css_class_name}">
    <div class="custom-kpi-label-top-condition">{safe_label}</div>
    <div class="custom-kpi-value-large">{safe_value_display}</div>
    {sub_text_html_block}
</div>
"""
try:
    with st.container():
        st.markdown(box_html_final_content, unsafe_allow_html=True)
except Exception as e_render_custom_box:
    logger.error(f"Custom Styled KPI Box Render Error: Label '{safe_label}', Details: {e_render_custom_box}", exc_info=True)
    try: st.error(f"Error rendering custom KPI: {safe_label}")
    except: pass
