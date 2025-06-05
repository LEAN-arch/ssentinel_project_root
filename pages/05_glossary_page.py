# sentinel_project_root/pages/glossary_page.py
# Glossary of Terms for the "Sentinel Health Co-Pilot" System.

import streamlit as st
import logging
from typing import Optional, Any, Union, Dict, List # Added Union, Dict, List
import html # For escaping
from pathlib import Path # Added for path manipulation

from config import settings

logger = logging.getLogger(__name__)

# --- Page Configuration (Call this early) ---
try:
    page_icon_value = "📜" # Default icon
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_FAVICON_PATH'):
        # Ensure APP_FAVICON_PATH is relative to PROJECT_ROOT_DIR
        favicon_path = Path(settings.PROJECT_ROOT_DIR) / settings.APP_FAVICON_PATH
        if favicon_path.is_file():
            page_icon_value = str(favicon_path)
        else:
            logger.warning(f"Favicon for Glossary page not found: {favicon_path}")

    page_layout_value = "wide" # Default layout
    if hasattr(settings, 'APP_LAYOUT'):
        page_layout_value = settings.APP_LAYOUT
        
    st.set_page_config(
        page_title=f"Glossary - {settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'App'}",
        page_icon=page_icon_value,
        layout=page_layout_value
    )
except Exception as e_page_config:
    logger.error(f"Error applying page configuration for Glossary page: {e_page_config}", exc_info=True)
    st.set_page_config(page_title="Glossary", page_icon="📜", layout="wide") # Fallback

# --- Sidebar ---
st.sidebar.markdown("---") # Visual separator
try:
    if hasattr(settings, 'PROJECT_ROOT_DIR') and hasattr(settings, 'APP_LOGO_SMALL_PATH'):
        project_root_path = Path(settings.PROJECT_ROOT_DIR)
        # APP_LOGO_SMALL_PATH should be relative to PROJECT_ROOT_DIR, e.g., "assets/logo.png"
        logo_path_sidebar = project_root_path / settings.APP_LOGO_SMALL_PATH

        if logo_path_sidebar.is_file():
            st.sidebar.image(str(logo_path_sidebar.resolve()), width=150) # Adjust width as needed
        else:
            logger.warning(f"Sidebar logo for Glossary page not found at resolved path: {logo_path_sidebar.resolve()}")
            st.sidebar.caption("Logo not found.")
    else:
        logger.warning("PROJECT_ROOT_DIR or APP_LOGO_SMALL_PATH missing in settings for Glossary sidebar logo.")
        st.sidebar.caption("Logo config missing.")
except Exception as e_logo_glossary: # Catch any other unexpected error during logo loading
    logger.error(f"Unexpected error displaying Glossary sidebar logo: {e_logo_glossary}", exc_info=True)
    st.sidebar.caption("Error loading logo.")
st.sidebar.markdown("---") # Visual separator
# You can add other sidebar elements for the glossary page here if needed, e.g., quick links to sections
# st.sidebar.header("Sections")
# st.sidebar.markdown("[System Concepts](#system-architecture-core-concepts)")
# st.sidebar.markdown("[Clinical & Operational](#clinical-epidemiological-operational-terms)")
# st.sidebar.markdown("[Technical & Data](#technical-data-platform-terms)")


# --- Main Page Content ---
st.title(f"📜 {settings.APP_NAME} - Glossary of Terms")
st.markdown(
    "Definitions for common terms, abbreviations, metrics, and system-specific concepts "
    "used throughout the Sentinel Health Co-Pilot platform."
)
st.divider()

def _display_glossary_term(
    term_name: str,
    definition_text: str,
    related_config_variable_name: Optional[str] = None
) -> None:
    """ Displays a glossary term and its definition in a standardized format. """
    st.markdown(f"#### {html.escape(term_name)}") # Escape term name
    st.markdown(f"*{html.escape(definition_text)}*") # Escape definition
    
    if related_config_variable_name:
        config_value_display: Any = None # Use Any for flexibility
        try:
            config_value_display = getattr(settings, related_config_variable_name, None)
        except AttributeError: 
            # This should ideally not be reached if getattr's default is used,
            # but good for robustness if settings object is complex.
            logger.warning(f"AttributeError accessing settings.{related_config_variable_name}")
            pass 

        display_val_str = f"(config '{html.escape(related_config_variable_name)}' not found or value is None)"
        if config_value_display is not None:
            try:
                if isinstance(config_value_display, list):
                    # Show first few elements for lists to avoid long displays
                    preview_list = [str(item) for item in config_value_display[:min(5, len(config_value_display))]]
                    display_val_str = ", ".join(preview_list)
                    if len(config_value_display) > 5: display_val_str += ", ..."
                elif isinstance(config_value_display, dict):
                    # Show first few items for dicts
                    dict_items_preview = list(config_value_display.items())[:min(2, len(config_value_display))]
                    preview_dict_str_parts = [f"{repr(k)}: {repr(v)}" for k, v in dict_items_preview]
                    display_val_str = "{ " + ", ".join(preview_dict_str_parts) + " }"
                    if len(config_value_display) > 2: 
                        display_val_str = display_val_str[:-1] + ", ...}" # Add ellipsis inside brace
                else:
                    display_val_str = str(config_value_display)
                
                # Truncate very long strings (like paths) for display
                if len(display_val_str) > 150: 
                    display_val_str = display_val_str[:147] + "..."
            except Exception as e_format_config: # Catch errors during formatting
                logger.error(f"Error formatting config value for {related_config_variable_name}: {e_format_config}")
                display_val_str = "(Error formatting value)"
        
        st.caption(f"*(Related config: `settings.{html.escape(related_config_variable_name)}` = `{html.escape(display_val_str)}`)*")
    st.markdown("---")


# --- Section I: Sentinel Health Co-Pilot System Concepts ---
# Using markdown anchors for potential sidebar linking
st.header("🌐 System Architecture & Core Concepts", anchor="system-architecture-core-concepts")
_display_glossary_term(
    "Sentinel Health Co-Pilot", 
    "An edge-first health intelligence and action support system for resource-limited, high-risk LMIC environments, prioritizing offline functionality, actionable insights for frontline health workers (FHWs), and resilient data flow."
)
_display_glossary_term(
    "Personal Edge Device (PED)", 
    "A ruggedized smartphone, wearable, or low-power SoC used by FHWs. Runs native apps with on-device Edge AI for real-time monitoring, alerts, task management, and JIT guidance, primarily offline."
)
_display_glossary_term(
    "Edge AI / TinyML", 
    "AI models (e.g., TFLite) optimized for PEDs or local Supervisor Hubs, enabling offline decision support, anomaly detection, and risk stratification with minimal resources."
)
_display_glossary_term(
    "Supervisor Hub (Tier 1)", 
    "Optional intermediary device for CHW Supervisors. Locally aggregates PED data, allows localized team oversight, and batches data for Facility Nodes.",
    "HUB_SQLITE_DB_NAME" # Ensure this exists in settings
)
_display_glossary_term(
    "Facility Node (Tier 2)", 
    "Local server/PC at a clinic. Aggregates data from Hubs/PEDs, performs local analytics, interfaces with EMRs, generates reports, and stages data for wider sync.",
    "FACILITY_NODE_DB_TYPE" # Ensure this exists in settings
)
_display_glossary_term(
    "Regional/Cloud Node (Tier 3)", 
    "Optional centralized infrastructure for population-level analytics, AI model training, and national reporting. Receives data from Facility Nodes."
)
# Check if ALERT_SPO2_CRITICAL_LOW_PCT exists in settings before using it in f-string
spo2_critical_low_setting = settings.ALERT_SPO2_CRITICAL_LOW_PCT if hasattr(settings, 'ALERT_SPO2_CRITICAL_LOW_PCT') else 'N/A'
_display_glossary_term(
    "Lean Data Inputs", 
    f"Core principle: collect minimal viable data with maximum predictive power for Edge AI and FHW actionability in LMIC settings (e.g., age group, chronic condition flag, SpO₂ < {spo2_critical_low_setting}%, observed fatigue, key symptoms).",
    "ALERT_SPO2_CRITICAL_LOW_PCT"
)
_display_glossary_term(
    "Action Code / Suggested Action Code", 
    "System-internal code (e.g., 'ACTION_SPO2_MANAGE_URGENT') generated by alerts, AI, or tasks. Maps to PED UI elements (pictograms, guidance) via configuration files.",
    "PICTOGRAM_MAP_JSON_PATH" # Ensure this exists
)
_display_glossary_term(
    "Opportunistic Sync", 
    "Data synchronization strategy where devices transfer data to higher tiers only when a viable, low-cost, stable communication channel is available (e.g., Bluetooth, local Wi-Fi, QR).",
    "EDGE_DATA_SYNC_PROTOCOLS_SUPPORTED" # Ensure this exists
)

# --- Section II: Clinical, Epidemiological & Operational Terms ---
st.header("🩺 Clinical, Epidemiological & Operational Terms", anchor="clinical-epidemiological-operational-terms")
risk_score_high_setting = settings.RISK_SCORE_HIGH_THRESHOLD if hasattr(settings, 'RISK_SCORE_HIGH_THRESHOLD') else 'N/A'
_display_glossary_term(
    "AI Risk Score (Patient/Worker)", 
    f"Simulated algorithmic score (0-100) predicting health risk or adverse outcome likelihood. High Risk ≥ {risk_score_high_setting}.", 
    "RISK_SCORE_HIGH_THRESHOLD"
)
fatigue_high_setting = settings.FATIGUE_INDEX_HIGH_THRESHOLD if hasattr(settings, 'FATIGUE_INDEX_HIGH_THRESHOLD') else 'N/A'
_display_glossary_term(
    "AI Follow-up Priority Score / Task Priority Score", 
    f"Simulated score (0-100) from AI/rules to prioritize patient follow-ups or tasks. High priority e.g., ≥ {fatigue_high_setting}.", 
    "FATIGUE_INDEX_HIGH_THRESHOLD"
)
heat_risk_setting = settings.ALERT_AMBIENT_HEAT_INDEX_RISK_C if hasattr(settings, 'ALERT_AMBIENT_HEAT_INDEX_RISK_C') else 'N/A'
heat_danger_setting = settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C if hasattr(settings, 'ALERT_AMBIENT_HEAT_INDEX_DANGER_C') else 'N/A'
_display_glossary_term(
    "Ambient Heat Index (°C)", 
    f"Perceived heat combining humidity and air temperature. Sentinel alerts: Risk at {heat_risk_setting}°C, Danger at {heat_danger_setting}°C.", 
    "ALERT_AMBIENT_HEAT_INDEX_DANGER_C"
)
key_condition_example = (settings.KEY_CONDITIONS_FOR_ACTION[0] 
                         if hasattr(settings, 'KEY_CONDITIONS_FOR_ACTION') and settings.KEY_CONDITIONS_FOR_ACTION 
                         else 'TB')
_display_glossary_term(
    "Condition (Key Actionable)", 
    f"Specific health conditions prioritized by Sentinel for monitoring, alerts, and response protocols. Defined in `settings.KEY_CONDITIONS_FOR_ACTION` (e.g., '{key_condition_example}').", 
    "KEY_CONDITIONS_FOR_ACTION"
)
_display_glossary_term(
    "Encounter (CHW/Clinic)", 
    "Any patient interaction with the health system or CHW documented in Sentinel (home visits, clinic consults, alert responses, follow-ups, remote check-ins)."
)
facility_coverage_low_setting = (settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT 
                                 if hasattr(settings, 'DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT') else 'N/A')
_display_glossary_term(
    "Facility Coverage Score (Zonal)", 
    f"District-level metric (0-100%) of health facility access/capacity relative to zone population. Low coverage (e.g., < {facility_coverage_low_setting}%) may trigger DHO review.", 
    "DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT"
)
fatigue_moderate_setting = settings.FATIGUE_INDEX_MODERATE_THRESHOLD if hasattr(settings, 'FATIGUE_INDEX_MODERATE_THRESHOLD') else 'N/A'
# fatigue_high_setting already defined above
_display_glossary_term(
    "Fatigue Index Score (Worker)", 
    f"Simulated score (0-100) of FHW fatigue, derived by Edge AI from HRV, activity, or self-reports. Alerts: Moderate ≥ {fatigue_moderate_setting}, High ≥ {fatigue_high_setting}.", 
    "FATIGUE_INDEX_HIGH_THRESHOLD"
)
hrv_low_setting = settings.STRESS_HRV_LOW_THRESHOLD_MS if hasattr(settings, 'STRESS_HRV_LOW_THRESHOLD_MS') else 'N/A'
_display_glossary_term(
    "HRV (Heart Rate Variability)", 
    f"Variation in time between heartbeats (ms, e.g., RMSSD). Low HRV (e.g., < {hrv_low_setting}ms) can indicate increased physiological stress/fatigue.",
    "STRESS_HRV_LOW_THRESHOLD_MS"
)
spo2_warn_low_setting = settings.ALERT_SPO2_WARNING_LOW_PCT if hasattr(settings, 'ALERT_SPO2_WARNING_LOW_PCT') else 'N/A'
# spo2_critical_low_setting already defined above
_display_glossary_term(
    "SpO₂ (Peripheral Capillary Oxygen Saturation)", 
    f"Estimate of blood oxygen saturation (%). Critical Low: < {spo2_critical_low_setting}%. Warning Low: < {spo2_warn_low_setting}%.", 
    "ALERT_SPO2_CRITICAL_LOW_PCT"
)
tat_target_setting = settings.TARGET_TEST_TURNAROUND_DAYS if hasattr(settings, 'TARGET_TEST_TURNAROUND_DAYS') else 'N/A'
_display_glossary_term(
    "TAT (Test Turnaround Time)", 
    f"Time from sample collection/registration to result availability. General critical test target ~{tat_target_setting} days. Specifics in `settings.KEY_TEST_TYPES_FOR_ANALYSIS`.", 
    "TARGET_TEST_TURNAROUND_DAYS"
)

# --- Section III: Technical & Data Format Terms ---
st.header("💻 Technical, Data & Platform Terms", anchor="technical-data-platform-terms")
_display_glossary_term(
    "API (Application Programming Interface)", 
    "Rules/protocols for software communication. Used in Sentinel for data sync between tiers (e.g., Facility to Regional) or external system integration (e.g., DHIS2, EMRs).",
    "FHIR_SERVER_ENDPOINT_LOCAL" # Ensure this exists
)
_display_glossary_term(
    "CSV (Comma-Separated Values)", 
    "Simple text file format for tabular data. Used for raw data import (e.g., health records) and simple reporting.",
    "HEALTH_RECORDS_CSV_PATH" # Ensure this exists
)
_display_glossary_term(
    "FHIR (Fast Healthcare Interoperability Resources)", 
    "Pronounced 'Fire'. HL7® international standard for exchanging EHRs using 'Resources' and an API. Sentinel aims for FHIR support at Tiers 2 & 3.",
    "FHIR_SERVER_ENDPOINT_LOCAL" # Ensure this exists
)
default_crs_setting = settings.DEFAULT_CRS_STANDARD if hasattr(settings, 'DEFAULT_CRS_STANDARD') else 'EPSG:4326'
_display_glossary_term(
    "GeoJSON", 
    f"Open standard JSON-based format for encoding geographic data (points, lines, polygons) and attributes. Used for operational zone boundaries. Default CRS: {default_crs_setting}.",
    "ZONE_GEOMETRIES_GEOJSON_FILE_PATH" # Ensure this exists
)
_display_glossary_term(
    "JSON (JavaScript Object Notation)", 
    "Lightweight, human-readable data-interchange format. Used for configuration files (e.g., escalation protocols) and API data exchange.",
    "ESCALATION_PROTOCOLS_JSON_PATH" # Ensure this exists
)
_display_glossary_term(
    "Pictogram", 
    "Simple iconic image representing a concept, action, or object. Used in Sentinel PED UIs for clarity, especially for low-literacy users or multilingual contexts. Mapped via configuration.",
    "PICTOGRAM_MAP_JSON_PATH" # Ensure this exists
)
qr_packet_size_setting = settings.QR_PACKET_MAX_SIZE_BYTES if hasattr(settings, 'QR_PACKET_MAX_SIZE_BYTES') else 'N/A'
_display_glossary_term(
    "QR Code Packet", 
    f"Method for offline data transfer by encoding data into QR codes. Useful for PED-to-Hub/PED-to-PED exchange. Max single packet size: {qr_packet_size_setting} bytes.",
    "QR_PACKET_MAX_SIZE_BYTES"
)
_display_glossary_term(
    "SQLite", 
    "Embedded SQL database engine. Used in Sentinel for local data storage on PEDs and Supervisor Hubs due to portability and no need for a separate server.",
    "PED_SQLITE_DB_NAME" # Ensure this exists
)
_display_glossary_term(
    "TFLite (TensorFlow Lite)", 
    "Open-source deep learning framework for running TensorFlow models on mobile, embedded, and IoT devices. Key for Edge AI on Sentinel PEDs.",
    "EDGE_MODEL_VITALS_DETERIORATION" # Ensure this exists
)

st.divider()
footer_text = settings.APP_FOOTER_TEXT if hasattr(settings, 'APP_FOOTER_TEXT') else "Sentinel Health Co-Pilot."
st.caption(footer_text)

app_name_log = settings.APP_NAME if hasattr(settings, 'APP_NAME') else 'App'
app_version_log = settings.APP_VERSION if hasattr(settings, 'APP_VERSION') else 'N/A'
logger.info(f"Glossary page for {app_name_log} (v{app_version_log}) loaded.")
