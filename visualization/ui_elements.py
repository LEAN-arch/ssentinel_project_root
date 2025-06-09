# sentinel_project_root/visualization/ui_elements.py
# SME PLATINUM STANDARD - THEME-AWARE UI COMPONENTS

import html
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
import streamlit as st

from config import settings

logger = logging.getLogger(__name__)

@st.cache_resource
def load_and_inject_css(css_path: Union[str, Path]):
    """Loads a CSS file and injects it into the Streamlit application."""
    path = Path(css_path)
    if not path.is_file():
        logger.warning(f"CSS file not found at: {path}. UI may not be styled correctly.")
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.debug(f"Successfully loaded and injected CSS from {path}.")
    except Exception as e:
        logger.error(f"Error loading CSS from {path}: {e}", exc_info=True)


def get_theme_color(semantic_name: str, fallback: str = "#6c757d") -> str:
    """
    Retrieves a theme color from settings using a semantic name.
    e.g., 'risk_high', 'primary', 'positive_delta'.
    """
    attr_name = f"COLOR_{semantic_name.upper()}"
    return getattr(settings, attr_name, fallback)


def render_kpi_card(
    title: str,
    value: Any,
    unit: str = "",
    status_level: Optional[str] = None,
    delta: Optional[float] = None,
    delta_is_improvement: Optional[bool] = None,
    help_text: Optional[str] = None,
    icon: str = "💡"
) -> None:
    """
    Renders a rich, custom HTML KPI card in Streamlit.
    """
    # --- Value Formatting ---
    if pd.isna(value):
        value_str = "N/A"
    elif isinstance(value, float) and not value.is_integer():
        value_str = f"{value:,.2f}"
    elif isinstance(value, (int, float)):
        value_str = f"{int(value):,}"
    else:
        value_str = str(value)
        
    # --- Delta Formatting ---
    delta_html = ""
    if delta is not None and delta_is_improvement is not None:
        delta_class = "positive" if delta_is_improvement else "negative"
        arrow = "▲" if delta_is_improvement else "▼"
        delta_html = f'<p class="kpi-delta {delta_class}">{arrow} {delta:+.1f}%</p>'

    # --- CSS and Tooltip ---
    status_class = f"status-{status_level.lower().replace('_', '-')}" if status_level else ""
    tooltip_attr = f'title="{html.escape(help_text)}"' if help_text else ""
    unit_html = f'<span class="kpi-units">{html.escape(unit)}</span>' if unit else ""

    card_html = f"""
    <div class="kpi-card {status_class}" {tooltip_attr}>
        <div class="kpi-header">
            <span class="kpi-icon">{html.escape(icon)}</span>
            <div class="kpi-title">{html.escape(title)}</div>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{html.escape(value_str)}{unit_html}</p>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_traffic_light_indicator(
    message: str,
    status_level: str,
    details: Optional[str] = None
) -> None:
    """Renders a custom HTML traffic light status indicator."""
    status_class = f"status-{status_level.lower().replace('_', '-')}"
    details_html = f'<div class="traffic-light-details">{html.escape(details)}</div>' if details else ""

    indicator_html = f"""
    <div class="traffic-light-indicator">
        <div class="traffic-light-dot {status_class}"></div>
        <div class="traffic-light-message">{html.escape(message)}</div>
        {details_html}
    </div>
    """
    st.markdown(indicator_html, unsafe_allow_html=True)


def render_custom_kpi(
    label: str,
    value: Any,
    sub_text: Optional[str] = None,
    highlight_status: Optional[str] = None
) -> None:
    """Renders a custom-styled KPI box with an optional colored edge."""
    # --- Value Formatting ---
    if pd.isna(value):
        value_display = "N/A"
    elif isinstance(value, float) and not value.is_integer():
        value_display = f"{value:,.1f}"
    elif isinstance(value, (int, float)):
        value_display = f"{int(value):,}"
    else:
        value_display = str(value)
    
    sub_text_html = f'<div class="custom-kpi-subtext">{html.escape(sub_text)}</div>' if sub_text else ""
    edge_class = f"highlight-{highlight_status.lower().replace('_', '-')}-edge" if highlight_status else ""

    kpi_box_html = f"""
    <div class="custom-markdown-kpi-box {edge_class}">
        <div class="custom-kpi-label-top">{html.escape(label)}</div>
        <div class="custom-kpi-value-large">{html.escape(value_display)}</div>
        {sub_text_html}
    </div>
    """
    st.markdown(kpi_box_html, unsafe_allow_html=True)
