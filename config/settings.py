# sentinel_project_root/config/settings.py
# Centralized Configuration for "Sentinel Health Co-Pilot"

import os
import logging
from datetime import datetime

# --- Base Project Directory ---
# Assumes this file is in sentinel_project_root/config/settings.py
# PROJECT_ROOT_DIR will be sentinel_project_root
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# --- Logger for Path Validation ---
# Basic logger setup for this module, primary app logging should be in app.py
settings_logger = logging.getLogger(__name__)
# Avoid basicConfig here; let the main app or Streamlit handle global config.
# If this module is imported before global logging is set, warnings might not show in console
# unless explicitly handled by the importing module or if Streamlit's logger catches them.

# --- Path Validation Helper ---
def validate_path(path: str, description: str, is_dir: bool = False) -> str:
    """Validate file or directory path, log warning if missing."""
    if not os.path.exists(path):
        settings_logger.warning(f"{description} not found at expected path: {path}")
    elif is_dir and not os.path.isdir(path):
        settings_logger.warning(f"{description} is not a directory: {path}")
    elif not is_dir and not os.path.isfile(path):
        settings_logger.warning(f"{description} is not a file: {path}")
    return path

# --- I. Core System & Directory Configuration ---
APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "4.0.0-refactored" # Updated version
ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {datetime.now().year} {ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org" # Ensure this is a valid contact

# Directories
ASSETS_DIR = validate_path(os.path.join(PROJECT_ROOT_DIR, "assets"), "Assets directory", is_dir=True)
DATA_SOURCES_DIR = validate_path(os.path.join(PROJECT_ROOT_DIR, "data_sources"), "Data sources directory", is_dir=True)
# FACILITY_NODE_DATA_DIR = validate_path(os.path.join(PROJECT_ROOT_DIR, "facility_node_data"), "Facility Node data directory", is_dir=True) # Create if used
# LOCAL_DATA_DIR_PED_SIM = validate_path(os.path.join(PROJECT_ROOT_DIR, "local_data_ped_sim"), "PED simulation data directory", is_dir=True) # Create if used

# Key Asset Files
APP_LOGO_SMALL_PATH = validate_path(os.path.join(ASSETS_DIR, "sentinel_logo_small.png"), "Small app logo")
APP_LOGO_LARGE_PATH = validate_path(os.path.join(ASSETS_DIR, "sentinel_logo_large.png"), "Large app logo")
STYLE_CSS_PATH_WEB = validate_path(os.path.join(ASSETS_DIR, "style_web_reports.css"), "Global CSS stylesheet")
ESCALATION_PROTOCOLS_JSON_PATH = validate_path(os.path.join(ASSETS_DIR, "escalation_protocols.json"), "Escalation protocols JSON")
PICTOGRAM_MAP_JSON_PATH = validate_path(os.path.join(ASSETS_DIR, "pictogram_map.json"), "Pictogram map JSON")
HAPTIC_PATTERNS_JSON_PATH = validate_path(os.path.join(ASSETS_DIR, "haptic_patterns.json"), "Haptic patterns JSON")
# AUDIO_ALERTS_DIR = validate_path(os.path.join(ASSETS_DIR, "audio_alerts"), "Audio alerts directory", is_dir=True) # Create if used

# Data Source Files
HEALTH_RECORDS_CSV_PATH = validate_path(os.path.join(DATA_SOURCES_DIR, "health_records_expanded.csv"), "Health records CSV")
ZONE_ATTRIBUTES_CSV_PATH = validate_path(os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv"), "Zone attributes CSV")
ZONE_GEOMETRIES_GEOJSON_FILE_PATH = validate_path(os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson"), "Zone geometries GeoJSON") # Renamed for clarity
IOT_CLINIC_ENVIRONMENT_CSV_PATH = validate_path(os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv"), "IoT clinic environment CSV")

# --- II. Health & Operational Thresholds ---
# Vitals
ALERT_SPO2_CRITICAL_LOW_PCT = 90  # Matches glossary use, original was ALERT_SPO2_CRITICAL
ALERT_SPO2_WARNING_LOW_PCT = 94   # Matches glossary use, original was ALERT_SPO2_WARNING
ALERT_BODY_TEMP_FEVER_C = 38.0
ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5
ALERT_HR_TACHYCARDIA_BPM = 100 # Renamed for clarity
ALERT_HR_BRADYCARDIA_BPM = 50 # Renamed for clarity

# Worker/Patient Specific
HEAT_STRESS_RISK_BODY_TEMP_C = 37.5 # For individual, from skin/core temp
HEAT_STRESS_DANGER_BODY_TEMP_C = 38.5 # For individual

# Environmental
ALERT_AMBIENT_CO2_HIGH_PPM = 1500
ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500
ALERT_AMBIENT_PM25_HIGH_UGM3 = 35
ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50
ALERT_AMBIENT_NOISE_HIGH_DBA = 85 # Changed to dBA for standard noise unit
ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32 # From Heat Index calculation
ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41

# AI & System Scores
FATIGUE_INDEX_MODERATE_THRESHOLD = 60 # General threshold
FATIGUE_INDEX_HIGH_THRESHOLD = 80     # General threshold
STRESS_HRV_LOW_THRESHOLD_MS = 20      # For RMSSD or similar HRV metric in ms

RISK_SCORE_LOW_THRESHOLD = 40    # Start of Low Risk for display/categorization
RISK_SCORE_MODERATE_THRESHOLD = 60 # Start of Moderate Risk
RISK_SCORE_HIGH_THRESHOLD = 75   # Start of High Risk

# Clinic Operations
TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10
TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5

# District Level
DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70  # Avg AI risk score for a zone to be "high risk"
DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60
DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10 # Absolute number of TB cases for intervention
DISTRICT_DISEASE_PREVALENCE_HIGH_PERCENTILE = 0.80 # For identifying zones with high prevalence

# Supply Chain
CRITICAL_SUPPLY_DAYS_REMAINING = 7
LOW_SUPPLY_DAYS_REMAINING = 14

# General Activity / Wellness
TARGET_DAILY_STEPS = 8000
RANDOM_SEED = 42 # For reproducible AI simulations

# Age Thresholds (example, can be expanded)
AGE_THRESHOLD_LOW = 5 # e.g., Under 5
AGE_THRESHOLD_MODERATE = 18
AGE_THRESHOLD_HIGH = 60
AGE_THRESHOLD_VERY_HIGH = 75


# --- III. Edge Device Configuration (PED specific) ---
EDGE_APP_DEFAULT_LANGUAGE = "en"
EDGE_APP_SUPPORTED_LANGUAGES = ["en", "sw", "fr"]
# Paths to config files for PEDs are already defined in ASSETS_DIR section
EDGE_MODEL_VITALS_DETERIORATION = "vitals_deterioration_v1.tflite" # Filename example
EDGE_MODEL_FATIGUE_ASSESSMENT = "fatigue_index_v1.tflite"        # Filename example
EDGE_MODEL_ENVIRONMENTAL_ANOMALY = "anomaly_detection_base.tflite" # Filename example
EDGE_DATA_BASELINE_WINDOW_DAYS = 7
EDGE_DATA_PROCESSING_INTERVAL_SECONDS = 60
PED_SQLITE_DB_NAME = "sentinel_ped_local.db"
PED_MAX_LOG_FILE_SIZE_MB = 50
EDGE_DATA_SYNC_PROTOCOLS_SUPPORTED = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE_BYTES = 256 # Small, for single QR codes; larger data needs splitting
SMS_DATA_COMPRESSION_METHOD = "BASE85_ZLIB"

# --- IV. Supervisor Hub & Facility Node Configuration ---
HUB_SQLITE_DB_NAME = "sentinel_supervisor_hub.db"
FACILITY_NODE_DB_TYPE = "POSTGRESQL" # Example, could be SQLite for simpler deployments
FHIR_SERVER_ENDPOINT_LOCAL = "http://localhost:8080/fhir" # Example FHIR endpoint
NODE_REPORTING_INTERVAL_HOURS = 24

# --- V. Data Semantics & Categories ---
KEY_TEST_TYPES_FOR_ANALYSIS = { # Standardized structure for easier parsing
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
    "PulseOx": {"disease_group": "Vitals", "target_tat_days": 0, "critical": False, "display_name": "Pulse Oximetry"},
}
CRITICAL_TESTS = [k for k, v in KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False)]

TARGET_TEST_TURNAROUND_DAYS = 2 # General TAT target for non-specific cases
TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0
OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK = 7 # If specific test TAT not found

KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin', 'Iron-Folate', 'Insulin'] # Used for identifying key drugs

TARGET_MALARIA_POSITIVITY_RATE = 10.0 # Example target %, context-specific

# Symptom clusters for epi signal detection (example)
# Structure: { "Cluster Name": ["symptom1_keyword", "symptom2_keyword", ...], ... }
SYMPTOM_CLUSTERS_CONFIG = {
    "Fever, Cough, Fatigue": ["fever", "cough", "fatigue"],
    "Diarrhea & Vomiting": ["diarrhea", "vomit"],
    "Fever & Rash": ["fever", "rash"]
}

# --- VI. Web Dashboard & Visualization Configuration ---
CACHE_TTL_SECONDS_WEB_REPORTS = 3600 # 1 hour default cache for web reports data
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = 30 # Default days for trend charts
WEB_PLOT_DEFAULT_HEIGHT = 400 # Default height for most Plotly charts
WEB_PLOT_COMPACT_HEIGHT = 320 # For smaller charts
WEB_MAP_DEFAULT_HEIGHT = 600    # Default height for maps

# Map settings
MAPBOX_STYLE_WEB = "carto-positron" # Default open style, can be overridden by env var
DEFAULT_CRS_STANDARD = "EPSG:4326" # Standard WGS84 for GeoJSON
MAP_DEFAULT_CENTER_LAT = -1.286389  # Example: Nairobi, Kenya
MAP_DEFAULT_CENTER_LON = 36.817223
MAP_DEFAULT_ZOOM_LEVEL = 5     # General regional zoom

# --- VII. Color Palette (Consistent with style_web_reports.css) ---
# These are primarily for programmatic use (e.g., Plotly chart color scales or dynamic styling)
# CSS variables in style_web_reports.css are the source of truth for web display styling.
COLOR_RISK_HIGH = "#D32F2F"
COLOR_RISK_MODERATE = "#FBC02D" # Changed from app_config for consistency
COLOR_RISK_LOW = "#388E3C"
COLOR_RISK_NEUTRAL = "#757575"

COLOR_ACTION_PRIMARY = "#1976D2"
COLOR_ACTION_SECONDARY = "#546E7A" # Changed from app_config
COLOR_ACCENT_BRIGHT = "#4D7BF3"  # Example accent

COLOR_POSITIVE_DELTA = "#27AE60"
COLOR_NEGATIVE_DELTA = "#C0392B"

COLOR_TEXT_DARK = "#343a40"
COLOR_TEXT_HEADINGS_MAIN = "#1A2557"
COLOR_TEXT_HEADINGS_SUB = "#2C3E50"
COLOR_TEXT_MUTED = "#6c757d"
COLOR_TEXT_LINK_DEFAULT = COLOR_ACTION_PRIMARY

COLOR_BACKGROUND_PAGE = "#f8f9fa"
COLOR_BACKGROUND_CONTENT = "#ffffff" # Also used for plot paper_bgcolor
COLOR_BACKGROUND_SUBTLE = "#e9ecef"
COLOR_BACKGROUND_WHITE = "#FFFFFF" # Explicit white for clarity

COLOR_BORDER_LIGHT = "#dee2e6"
COLOR_BORDER_MEDIUM = "#ced4da"

# Disease-specific colors (Example, can be expanded or moved to a dedicated theme file)
# These are used if specific colors per disease are needed in plots beyond the standard theme colorway.
LEGACY_DISEASE_COLORS_WEB = { # Renamed from DISEASE_COLORS for clarity of use
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6", "Pneumonia": "#3B82F6",
    "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1", "Hypertension": "#F97316",
    "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16", "Heat Stroke": "#FF6347",
    "Severe Dehydration": "#4682B4", "Sepsis": "#800080", "Diarrheal Diseases (Severe)": "#D2691E",
    "Other": "#6B7280"
}

# --- End of Configuration ---
settings_logger.info(f"Sentinel settings loaded. APP_NAME: {APP_NAME} v{APP_VERSION}")
