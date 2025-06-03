# sentinel_project_root/pages/glossary_page.py
# Glossary of Terms for the "Sentinel Health Co-Pilot" System.
# Renamed from 5_glossary.py

import streamlit as st
import logging # For page-specific logging if needed
from typing import Optional # For type hinting in helper function

from config import settings # Use new settings module

# --- Page Specific Logger ---
logger = logging.getLogger(__name__)

# --- Page Configuration (Handled by Streamlit based on filename & app.py) ---
# st.set_page_config is called once in the main app.py.
# This page will inherit global settings like layout.
st.title(f"📜 {settings.APP_NAME} - Glossary of Terms")
st.markdown(
    "This page provides definitions for common terms, abbreviations, metrics, and system-specific concepts "
    "used throughout the Sentinel Health Co-Pilot platform, its dashboards, and documentation."
)
st.divider()

# --- Helper function for consistent term formatting and display ---
def _display_glossary_term( # Renamed for clarity within this module
    term_name: str, # Renamed for clarity
    definition_text: str, # Renamed
    related_config_variable_name: Optional[str] = None # Renamed
) -> None:
    """ Displays a glossary term and its definition in a standardized format. """
    st.markdown(f"#### {term_name}")
    st.markdown(f"*{definition_text}*") # Italicize definition for emphasis
    
    if related_config_variable_name:
        # Attempt to display the actual value from settings if the variable name is valid
        try:
            config_value_to_display = getattr(settings, related_config_variable_name, None)
            if config_value_to_display is not None:
                # For list values, join them for better display rather than raw list print
                if isinstance(config_value_to_display, list):
                    display_val_str = ", ".join(map(str, config_value_to_display[:5])) # Show first 5 elements
                    if len(config_value_to_display) > 5:
                        display_val_str += "..."
                elif isinstance(config_value_to_display, dict):
                     display_val_str = f"{ {k:v for i,(k,v) in enumerate(config_value_to_display.items()) if i < 2} }" # Show first 2 items
                     if len(config_value_to_display) > 2: display_val_str = display_val_str[:-1] + ", ...}"

                else:
                    display_val_str = str(config_value_to_display)
                
                # Limit display length for very long strings (like paths)
                if len(display_val_str) > 150:
                    display_val_str = display_val_str[:147] + "..."

                st.caption(f"*(Related config: `settings.{related_config_variable_name}` = `{display_val_str}`)*")
            else: # Variable name not found in settings (should ideally not happen if validated)
                st.caption(f"*(Related configuration in `config/settings.py`: `{related_config_variable_name}`)*")
        except AttributeError: # Should not happen if getattr used with default
             st.caption(f"*(Related configuration in `config/settings.py`: `{related_config_variable_name}`)*")
        except Exception as e_cfg_disp: # Catch any other error during display formatting
            logger.warning(f"Could not display config value for {related_config_variable_name}: {e_cfg_disp}")
            st.caption(f"*(Related configuration in `config/settings.py`: `{related_config_variable_name}` - error displaying value)*")
            
    st.markdown("---") # Visual separator after each term definition


# --- Section I: Sentinel Health Co-Pilot System Concepts ---
st.header("🌐 System Architecture & Core Concepts")
_display_glossary_term(
    term_name="Sentinel Health Co-Pilot", 
    definition_text=(
        "An edge-first health intelligence and action support system. It's designed for resource-limited, "
        "high-risk LMIC environments, prioritizing offline functionality, actionable insights for frontline "
        "health workers (FHWs), and resilient data flow."
    )
)
_display_glossary_term(
    term_name="Personal Edge Device (PED)", 
    definition_text=(
        "A ruggedized smartphone, wearable sensor array, or low-power System-on-Chip (SoC) device utilized by FHWs. "
        "It runs native applications featuring on-device Edge AI for real-time physiological monitoring, "
        "environmental sensing, alert generation, prioritized task management, and Just-In-Time (JIT) guidance protocols, "
        "primarily designed for offline operation."
    )
)
_display_glossary_term(
    term_name="Edge AI / TinyML", 
    definition_text=(
        "Artificial Intelligence (AI) models, often leveraging frameworks like TensorFlow Lite (TFLite) or "
        "specialized TinyML libraries, optimized to execute directly on PEDs or local Supervisor Hubs. "
        "These models operate with minimal computational resources, enabling offline decision support, "
        "anomaly detection, and risk stratification at the point of care."
    )
)
_display_glossary_term(
    term_name="Supervisor Hub (Tier 1)", 
    definition_text=(
        "An optional intermediary device (e.g., tablet, rugged phone) used by a CHW Supervisor or team leader. "
        "It locally aggregates data from team members' PEDs via short-range communication (e.g., Bluetooth, Wi-Fi Direct), "
        "allowing for localized team oversight, simple dashboard views, and batched data transfer to higher tiers (Facility Nodes)."
    ),
    related_config_variable_name="HUB_SQLITE_DB_NAME"
)
_display_glossary_term(
    term_name="Facility Node (Tier 2)", 
    definition_text=(
        "A local server, PC, or robust computing device situated at a clinic or health post. It aggregates data from "
        "Supervisor Hubs or directly from PEDs, can perform more complex local analytics, potentially interfaces with local "
        "EMRs, generates facility-level reports (like the Clinic Console), and serves as a staging point for wider data sync."
    ),
    related_config_variable_name="FACILITY_NODE_DB_TYPE"
)
_display_glossary_term(
    term_name="Regional/Cloud Node (Tier 3)", 
    definition_text=(
        "Optional centralized infrastructure (on-premise regional server or cloud platform) for population-level analytics, "
        "epidemiological surveillance, advanced AI model training, and national-level health reporting. Receives data from Facility Nodes."
    )
)
_display_glossary_term(
    term_name="Lean Data Inputs", 
    definition_text=(
        f"A core design principle focused on collecting only the minimum viable data points with maximum predictive power "
        f"for Edge AI models and direct actionability by FHWs, tailored for constrained LMIC settings. Examples: age group, "
        f"chronic condition flag, SpO₂ < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%, observed fatigue flag, key symptoms."
    ),
    related_config_variable_name="ALERT_SPO2_CRITICAL_LOW_PCT"
)
_display_glossary_term(
    term_name="Action Code / Suggested Action Code", 
    definition_text=(
        "A system-internal code (e.g., 'ACTION_SPO2_MANAGE_URGENT', 'TASK_VISIT_VITALS_URGENT') generated by an alert, "
        "AI model, or task. On a PED, this maps (via `pictogram_map.json` or `escalation_protocols.json`) to display "
        "pictograms, JIT guidance, automated communication, or protocol steps."
    ),
    related_config_variable_name="PICTOGRAM_MAP_JSON_PATH" # Example of related config
)
_display_glossary_term(
    term_name="Opportunistic Sync", 
    definition_text=(
        "Data synchronization strategy where Sentinel devices (PEDs, Hubs, Nodes) transfer data to higher tiers only when a "
        "viable, low-cost, stable communication channel is available (e.g., Bluetooth, local Wi-Fi, brief cellular, SD card/QR). "
        "Vital for intermittent connectivity."
    ),
    related_config_variable_name="EDGE_DATA_SYNC_PROTOCOLS_SUPPORTED"
)

# --- Section II: Clinical, Epidemiological & Operational Terms ---
st.header("🩺 Clinical, Epidemiological & Operational Terms")
_display_glossary_term(
    term_name="AI Risk Score (Patient/Worker)", 
    definition_text=(
        f"A simulated algorithmic score (typically 0-100) predicting an individual's general health risk or likelihood of "
        f"adverse outcomes, derived from vitals, symptoms, demographics, and context. High Risk ≥ {settings.RISK_SCORE_HIGH_THRESHOLD}."
    ), 
    related_config_variable_name="RISK_SCORE_HIGH_THRESHOLD"
)
_display_glossary_term(
    term_name="AI Follow-up Priority Score / Task Priority Score", 
    definition_text=(
        f"A simulated score (0-100) from AI/rules to prioritize patient follow-ups or tasks. High priority scores "
        f"(e.g., ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}) indicate more urgent attention needed."
    ), 
    related_config_variable_name="FATIGUE_INDEX_HIGH_THRESHOLD" # Re-using as general high priority threshold example
)
_display_glossary_term(
    term_name="Ambient Heat Index (°C)", 
    definition_text=(
        f"A measure of perceived heat when relative humidity is combined with air temperature. Sentinel uses this for heat "
        f"stress alerts (Risk at {settings.ALERT_AMBIENT_HEAT_INDEX_RISK_C}°C, Danger at {settings.ALERT_AMBIENT_HEAT_INDEX_DANGER_C}°C)."
    ), 
    related_config_variable_name="ALERT_AMBIENT_HEAT_INDEX_DANGER_C"
)
_display_glossary_term(
    term_name="Condition (Key Actionable)", 
    definition_text=(
        f"Specific health conditions prioritized by Sentinel for monitoring, alerts, and response protocols. "
        f"Defined in `settings.KEY_CONDITIONS_FOR_ACTION` (e.g., '{settings.KEY_CONDITIONS_FOR_ACTION[0]}')."
    ), 
    related_config_variable_name="KEY_CONDITIONS_FOR_ACTION"
)
_display_glossary_term(
    term_name="Encounter (CHW/Clinic)", 
    definition_text=(
        "Any interaction a patient has with the health system or a CHW documented within Sentinel, including home visits, "
        "clinic consultations, alert responses, scheduled follow-ups, or remote check-ins."
    )
)
_display_glossary_term(
    term_name="Facility Coverage Score (Zonal)", 
    definition_text=(
        f"A district-level metric (0-100%) reflecting adequacy of health facility access and capacity relative to zone population. "
        f"Low coverage (e.g., < {settings.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%) may trigger DHO review."
    ), 
    related_config_variable_name="DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT"
)
_display_glossary_term(
    term_name="Fatigue Index Score (Worker)", 
    definition_text=(
        f"A simulated score (0-100) indicating a FHW's fatigue level, derived by Edge AI on their PED from HRV, "
        f"activity, or self-reports. Alert levels: Moderate ≥ {settings.FATIGUE_INDEX_MODERATE_THRESHOLD}, High ≥ {settings.FATIGUE_INDEX_HIGH_THRESHOLD}."
    ), 
    related_config_variable_name="FATIGUE_INDEX_HIGH_THRESHOLD"
)
_display_glossary_term(
    term_name="HRV (Heart Rate Variability)", 
    definition_text=(
        f"Variation in time interval between consecutive heartbeats, measured in ms (e.g., RMSSD). Low HRV "
        f"(e.g., < {settings.STRESS_HRV_LOW_THRESHOLD_MS}ms) can indicate increased physiological stress or fatigue."
    ),
    related_config_variable_name="STRESS_HRV_LOW_THRESHOLD_MS"
)
_display_glossary_term(
    term_name="SpO₂ (Peripheral Capillary Oxygen Saturation)", 
    definition_text=(
        f"Estimate of oxygen in blood hemoglobin, as a percentage (%). Measured non-invasively. "
        f"Critical Low SpO₂ threshold: < {settings.ALERT_SPO2_CRITICAL_LOW_PCT}%. Warning Low SpO₂: < {settings.ALERT_SPO2_WARNING_LOW_PCT}%."
    ), 
    related_config_variable_name="ALERT_SPO2_CRITICAL_LOW_PCT"
)
_display_glossary_term(
    term_name="TAT (Test Turnaround Time)", 
    definition_text=(
        f"Total time from sample collection/registration to result availability. Target TATs vary; general critical test target "
        f"~{settings.TARGET_TEST_TURNAROUND_DAYS} days. Specific targets in `settings.KEY_TEST_TYPES_FOR_ANALYSIS`."
    ), 
    related_config_variable_name="TARGET_TEST_TURNAROUND_DAYS"
)


# --- Section III: Technical & Data Format Terms ---
st.header("💻 Technical, Data & Platform Terms")
_display_glossary_term(
    term_name="API (Application Programming Interface)", 
    definition_text=(
        "A defined set of rules and protocols allowing software applications/components to communicate and exchange data. "
        "Used in Sentinel for data sync between tiers (e.g., Facility to Regional Node) or integration with external systems (e.g., DHIS2, EMRs)."
    ),
    related_config_variable_name="FHIR_SERVER_ENDPOINT_LOCAL" # Example related API endpoint
)
_display_glossary_term(
    term_name="CSV (Comma-Separated Values)", 
    definition_text=(
        "A simple text file format where data values in a table are stored as plain text, with values separated by commas "
        "and rows by new lines. Used for raw data import (e.g., health records) and simple reporting."
    ),
    related_config_variable_name="HEALTH_RECORDS_CSV_PATH"
)
_display_glossary_term(
    term_name="FHIR (Fast Healthcare Interoperability Resources)", 
    definition_text=(
        "Pronounced 'Fire'. An HL7® international standard for exchanging electronic health records (EHR) using 'Resources' "
        "and an API. Sentinel aims for FHIR support at Tiers 2 & 3 for interoperability."
    ),
    related_config_variable_name="FHIR_SERVER_ENDPOINT_LOCAL" # From original config, ensure it's in new settings
)
_display_glossary_term(
    term_name="GeoJSON", 
    definition_text=(
        f"An open standard JSON-based format for encoding geographic data structures (points, lines, polygons) and their attributes. "
        f"Used in Sentinel for operational zone boundaries. Default CRS: {settings.DEFAULT_CRS_STANDARD}."
    ),
    related_config_variable_name="ZONE_GEOMETRIES_GEOJSON_FILE_PATH"
)
# GeoPandas term removed as per instructions. If a replacement concept like "DataFrame with Geometry Objects" is used, define that.
# For now, GeoJSON is the primary geospatial data format term.
_display_glossary_term(
    term_name="JSON (JavaScript Object Notation)", 
    definition_text=(
        "A lightweight, human-readable data-interchange format. Used extensively in Sentinel for configuration files "
        "(e.g., escalation protocols, pictogram maps) and API data exchange."
    ),
    related_config_variable_name="ESCALATION_PROTOCOLS_JSON_PATH" # Example JSON config
)
_display_glossary_term(
    term_name="Pictogram", 
    definition_text=(
        "A simple, iconic image representing a concept, action, or object. Used in Sentinel PED UIs for clarity, "
        "especially for low-literacy users or multilingual contexts. Mapped via `pictogram_map.json`."
    ),
    related_config_variable_name="PICTOGRAM_MAP_JSON_PATH"
)
_display_glossary_term(
    term_name="QR Code Packet", 
    definition_text=(
        f"Method for offline data transfer by encoding data into QR codes, displayed on one device and scanned by another. "
        f"Useful for PED-to-Hub/PED-to-PED exchange. Max single packet size: {settings.QR_PACKET_MAX_SIZE_BYTES} bytes."
    ),
    related_config_variable_name="QR_PACKET_MAX_SIZE_BYTES"
)
_display_glossary_term(
    term_name="SQLite", 
    definition_text=(
        "An embedded SQL database engine implemented as a C-library. Used in Sentinel for local data storage on "
        "PEDs and Supervisor Hubs due to its portability and lack of need for a separate server process."
    ),
    related_config_variable_name="PED_SQLITE_DB_NAME"
)
_display_glossary_term(
    term_name="TFLite (TensorFlow Lite)", 
    definition_text=(
        "An open-source deep learning framework from Google for running TensorFlow models on mobile, embedded, and IoT devices. "
        "Enables on-device ML with low latency and small binary size. Key for Edge AI on Sentinel PEDs."
    ),
    related_config_variable_name="EDGE_MODEL_VITALS_DETERIORATION" # Example TFLite model filename
)

st.divider()
st.caption(settings.APP_FOOTER_TEXT)
logger.info(f"Glossary page for {settings.APP_NAME} (v{settings.APP_VERSION}) loaded successfully.")
