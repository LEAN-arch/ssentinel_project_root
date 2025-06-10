# sentinel_project_root/pages/05_System_Glossary.py
# SME PLATINUM STANDARD - COMPREHENSIVE GLOSSARY (V6 - FINAL)

import html
import logging

import streamlit as st
from config import settings

# --- Page Setup ---
st.set_page_config(page_title="Glossary", page_icon="📜", layout="wide")
logger = logging.getLogger(__name__)


# --- Helper Function ---
def display_term(term: str, definition: str, config_key: str = None):
    """Renders a single glossary term in a standardized format."""
    st.markdown(f"#### {html.escape(term)}")
    st.markdown(f"*{html.escape(definition)}*")
    
    if config_key:
        try:
            value = settings
            for key in config_key.split('.'):
                value = getattr(value, key)
            
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                value_str = f"{str(value)[:100]}..."
            else:
                value_str = str(value)
            st.caption(f"`settings.{config_key}` = `{html.escape(value_str)}`")
        except AttributeError:
            st.caption(f"`settings.{config_key}` (Not found in current config)")
    st.markdown("---")

# --- Glossary Data ---
GLOSSARY_TERMS = [
    # --- System Architecture & Core Concepts ---
    {"term": "Sentinel Health Co-Pilot", "definition": "An edge-first health intelligence and action support system designed for resource-limited environments, prioritizing offline functionality and actionable insights for frontline health workers (FHWs).", "category": "System"},
    {"term": "Personal Edge Device (PED)", "definition": "A ruggedized smartphone or similar device used by FHWs. It runs the Sentinel native application for offline-first task management, data capture, and AI-driven guidance.", "category": "System"},
    {"term": "Edge AI", "definition": "Artificial intelligence models optimized to run directly on local devices (like a PED) with minimal resources, enabling real-time decision support without internet connectivity.", "category": "System"},
    {"term": "KPI (Key Performance Indicator)", "definition": "A measurable value that demonstrates how effectively a health system objective is being achieved. Sentinel KPIs are designed to be inferential and decision-grade.", "category": "System"},

    # --- Analytics Terms ---
    {"term": "AI Risk Score", "definition": "A predictive score (0-100) indicating a patient's risk of adverse health outcomes, calculated from various health and contextual data points. Higher scores denote higher risk.", "category": "Analytics", "config_key": "ANALYTICS.risk_score_moderate_threshold"},
    {"term": "Follow-up Priority", "definition": "An AI-generated score (0-100) that ranks patients by the urgency of required follow-up, helping CHWs prioritize their visits.", "category": "Analytics", "config_key": "MODEL_WEIGHTS.risk_score_multiplier"},
    {"term": "Prophet Forecast", "definition": "An advanced time-series forecasting model by Meta used in Sentinel to predict trends (e.g., supply consumption, patient load), accounting for seasonality.", "category": "Analytics", "config_key": "ANALYTICS.prophet_forecast_days"},

    # --- Clinical & Operational Terms ---
    {"term": "TAT (Test Turnaround Time)", "definition": "The duration from when a lab sample is collected to when a conclusive result is available. A key metric for diagnostic efficiency.", "category": "Operations", "config_key": "KEY_TEST_TYPES"},
    {"term": "Zonal Analysis", "definition": "The aggregation and comparison of health metrics across different predefined geographical areas (zones) to identify hotspots, resource gaps, and performance variations.", "category": "Operations"},
    # SME FIX: The string literal is now correctly terminated.
    {"term": "SpO₂ (Oxygen Saturation)", "definition": "An estimate of the amount of oxygen in the blood, expressed as a percentage. A critical metric for respiratory health.", "category": "Operations", "config_key": "ANALYTICS.spo2_critical_threshold_pct"},
]

# --- Main Page Execution ---
def main():
    st.title("📜 System Glossary")
    st.markdown("Definitions for common terms, abbreviations, and concepts used throughout the Sentinel platform.")
    st.divider()

    search_query = st.text_input("Search Glossary", placeholder="e.g., Risk Score, TAT, Edge AI")

    query = search_query.lower()
    if query:
        filtered_terms = [
            t for t in GLOSSARY_TERMS 
            if query in t['term'].lower() or query in t['definition'].lower()
        ]
    else:
        filtered_terms = GLOSSARY_TERMS

    if not filtered_terms and query:
        st.warning(f"No results found for '{html.escape(query)}'.")

    categories = sorted(list(set(t['category'] for t in filtered_terms)), key=lambda x: (x != 'System', x != 'Analytics', x))
    for category in categories:
        st.header(category)
        for term_data in filtered_terms:
            if term_data['category'] == category:
                display_term(term_data['term'], term_data['definition'], term_data.get('config_key'))

    st.divider()
    st.caption(settings.APP_FOOTER_TEXT)

if __name__ == "__main__":
    main()
