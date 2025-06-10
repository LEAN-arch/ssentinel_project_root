# sentinel_project_root/pages/05_System_Glossary.py
# SME PLATINUM STANDARD - COMPREHENSIVE GLOSSARY (V8 - FINAL)

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
# This list is the single source of truth for all glossary terms.
GLOSSARY_TERMS = [
    # --- System Architecture & Core Concepts ---
    {"term": "Sentinel Health Co-Pilot", "definition": "An edge-first health intelligence and action support system designed for resource-limited environments, prioritizing offline functionality and actionable insights for frontline health workers (FHWs).", "category": "System"},
    {"term": "Personal Edge Device (PED)", "definition": "A ruggedized smartphone, wearable, or low-power device used by FHWs. It runs the Sentinel native application for offline-first task management, data capture, and AI-driven guidance.", "category": "System"},
    {"term": "Edge AI / TinyML", "definition": "Artificial intelligence models (e.g., TensorFlow Lite) optimized to execute directly on PEDs with minimal computational resources, enabling offline decision support, anomaly detection, and risk stratification at the point of care.", "category": "System"},
    {"term": "KPI (Key Performance Indicator)", "definition": "A measurable value that demonstrates how effectively a health system objective is being achieved. Sentinel KPIs are designed to be inferential and decision-grade.", "category": "System"},
    {"term": "Just-In-Time (JIT) Guidance", "definition": "A feature of the PED application that provides context-specific instructions, protocols, or media (e.g., a video on how to administer a test) to the FHW at the exact moment it is needed during their workflow.", "category": "System"},
    {"term": "Haptic Patterns", "definition": "Programmed sequences of vibrations on a PED used to convey information non-visually. This provides discreet alerts for critical events, task reminders, or confirmations without requiring the user to look at the screen.", "category": "System", "config_key": "HAPTIC_PATTERNS_PATH"},
    {"term": "Opportunistic Sync", "definition": "A data synchronization strategy where devices transfer data to higher tiers only when a viable, low-cost communication channel is available (e.g., Bluetooth, local Wi-Fi). This is vital for environments with intermittent or expensive internet access.", "category": "System"},

    # --- Analytics Terms ---
    {"term": "AI Risk Score", "definition": "A predictive score (0-100) indicating a patient's risk of adverse health outcomes, calculated from various health and contextual data points. Higher scores denote higher risk.", "category": "Analytics", "config_key": "ANALYTICS.risk_score_moderate_threshold"},
    {"term": "Follow-up Priority", "definition": "An AI-generated score (0-100) that ranks patients by the urgency of required follow-up, helping CHWs prioritize their visits.", "category": "Analytics", "config_key": "MODEL_WEIGHTS.risk_score_multiplier"},
    {"term": "Prophet Forecast", "definition": "An advanced time-series forecasting model by Meta used in Sentinel to predict trends (e.g., supply consumption, patient load), accounting for seasonality.", "category": "Analytics", "config_key": "ANALYTICS.prophet_forecast_days"},

    # --- Clinical & Operational Terms ---
    {"term": "Screening Cascade / Funnel", "definition": "A visualization that shows the progression of patients through the steps of a screening program (e.g., number of symptomatic patients -> number tested -> number positive -> number linked to care). It is a powerful tool for identifying bottlenecks where patients are being lost.", "category": "Clinical & Operational"},
    {"term": "Syndromic Surveillance", "definition": "The practice of monitoring patient-reported symptoms (e.g., fever, cough, rash) in a population to detect potential disease outbreaks or unusual health events earlier than traditional diagnostic methods would allow.", "category": "Clinical & Operational"},
    {"term": "Linkage to Care", "definition": "The process of ensuring a patient who has been diagnosed with a condition successfully follows up and begins the appropriate treatment or care regimen. This is a critical step in turning a diagnosis into a positive health outcome.", "category": "Clinical & Operational"},
    {"term": "TAT (Test Turnaround Time)", "definition": "The total time elapsed from sample collection to when a validated result is available. A key metric for diagnostic efficiency.", "category": "Clinical & Operational", "config_key": "KEY_TEST_TYPES"},
    {"term": "Zonal Analysis", "definition": "The aggregation and comparison of health metrics across different predefined geographical areas (zones) to identify hotspots, resource gaps, and performance variations.", "category": "Clinical & Operational"},
    {"term": "SpO₂ (Oxygen Saturation)", "definition": "An estimate of the amount of oxygen in the blood, expressed as a percentage. A critical metric for respiratory health.", "category": "Clinical & Operational", "config_key": "ANALYTICS.spo2_critical_threshold_pct"},
    {"term": "Neglected Tropical Diseases (NTDs)", "definition": "A diverse group of communicable diseases that prevail in tropical and subtropical conditions and primarily affect impoverished communities. Sentinel can be configured to track and support screening programs for NTDs prevalent in a specific region.", "category": "Clinical & Operational"},

    # --- Technical & Data Terms ---
    {"term": "Data Contract", "definition": "An implicit or explicit agreement between software components on the structure and type of data they will exchange. Errors like `KeyError` often indicate a broken data contract, where one component expects a data column that another component failed to provide.", "category": "Technical & Data"},
    {"term": "Pydantic", "definition": "A Python library used extensively in Sentinel for data validation and settings management. It ensures that all configurations and data structures conform to a predefined schema, preventing bugs and improving system robustness.", "category": "Technical & Data"},
    {"term": "API", "definition": "Application Programming Interface. A set of rules that allows different software applications to communicate and exchange data.", "category": "Technical & Data"},
    {"term": "CSV", "definition": "Comma-Separated Values. A simple text file format for tabular data, used for raw data import and simple reporting.", "category": "Technical & Data"},
    {"term": "GeoJSON", "definition": "An open standard JSON-based format for encoding geographic data structures (e.g., points, lines, polygons). Used for operational zone boundaries.", "category": "Technical & Data"},
    {"term": "JSON", "definition": "JavaScript Object Notation. A lightweight, human-readable data-interchange format used for configuration files and API data exchange.", "category": "Technical & Data"},
    {"term": "SQLite", "definition": "An embedded SQL database engine used for local data storage on PEDs and hubs due to its portability and lack of a separate server.", "category": "Technical & Data"},
]

# --- Main Page Execution ---
def main():
    st.title("📜 System Glossary")
    st.markdown("A comprehensive dictionary of terms, abbreviations, and concepts used throughout the Sentinel platform.")
    st.divider()

    search_query = st.text_input("Search Glossary", placeholder="e.g., Risk Score, Screening Cascade, Edge AI")

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

    # Define a logical sort order for the categories
    category_order = {"System": 0, "Analytics": 1, "Clinical & Operational": 2, "Technical & Data": 3}
    categories = sorted(list(set(t['category'] for t in filtered_terms)), key=lambda x: category_order.get(x, 99))
    
    for category in categories:
        st.header(category)
        for term_data in filtered_terms:
            if term_data['category'] == category:
                display_term(term_data['term'], term_data['definition'], term_data.get('config_key'))

    st.divider()
    st.caption(settings.APP_FOOTER_TEXT)

if __name__ == "__main__":
    main()
