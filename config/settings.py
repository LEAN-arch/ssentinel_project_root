# sentinel_project_root/config/settings.py
# Centralized Configuration for "Sentinel Health Co-Pilot"

import os
import logging
from datetime import datetime
from pathlib import Path
# DO NOT import sys here for path manipulation if app.py is handling it.

# --- Base Project Directory ---
# This settings.py file is in sentinel_project_root/config/settings.py
# So, Path(__file__).resolve().parent is the 'config' directory.
# And Path(__file__).resolve().parent.parent is 'ssentinel_project_root'.
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
# The print statement for PROJECT_ROOT_DIR in settings.py should use settings_logger
# or be removed if it's for temporary debugging. Let's remove it to keep settings clean.

# --- Logger for Path Validation ---
settings_logger = logging.getLogger(__name__) # This logger is for settings.py's own use

# --- Path Validation Helper ---
def validate_path(path_obj: Path, description: str, is_dir: bool = False) -> Path:
    abs_path = path_obj.resolve() 
    if not abs_path.exists():
        settings_logger.warning(f"{description} not found at resolved absolute path: {abs_path}")
    elif is_dir and not abs_path.is_dir():
        settings_logger.warning(f"{description} is not a directory: {abs_path}")
    elif not is_dir and not abs_path.is_file():
        settings_logger.warning(f"{description} is not a file: {abs_path}")
    # else:
        # settings_logger.debug(f"{description} validated at: {abs_path}") # Optional debug
    return abs_path

# --- I. Core System & Directory Configuration ---
APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "4.0.3" # Incremented version
ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {datetime.now().year} {ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org"
LOG_LEVEL = os.getenv("SENTINEL_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("SENTINEL_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s")
LOG_DATE_FORMAT = os.getenv("SENTINEL_LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

# Directories (resolved to absolute paths using PROJECT_ROOT_DIR defined above)
ASSETS_DIR = validate_path(PROJECT_ROOT_DIR / "assets", "Assets directory", is_dir=True)
DATA_SOURCES_DIR = validate_path(PROJECT_ROOT_DIR / "data_sources", "Data sources directory", is_dir=True)

# Key Asset Files (now strings of absolute paths)
APP_LOGO_SMALL_PATH = str(validate_path(ASSETS_DIR / "sentinel_logo_small.png", "Small app logo"))
APP_LOGO_LARGE_PATH = str(validate_path(ASSETS_DIR / "sentinel_logo_large.png", "Large app logo"))
STYLE_CSS_PATH_WEB = str(validate_path(ASSETS_DIR / "style_web_reports.css", "Global CSS stylesheet"))
ESCALATION_PROTOCOLS_JSON_PATH = str(validate_path(ASSETS_DIR / "escalation_protocols.json", "Escalation protocols JSON"))
PICTOGRAM_MAP_JSON_PATH = str(validate_path(ASSETS_DIR / "pictogram_map.json", "Pictogram map JSON"))
HAPTIC_PATTERNS_JSON_PATH = str(validate_path(ASSETS_DIR / "haptic_patterns.json", "Haptic patterns JSON"))

# Data Source Files (now strings of absolute paths)
HEALTH_RECORDS_CSV_PATH = str(validate_path(DATA_SOURCES_DIR / "health_records_expanded.csv", "Health records CSV"))
ZONE_ATTRIBUTES_CSV_PATH = str(validate_path(DATA_SOURCES_DIR / "zone_attributes.csv", "Zone attributes CSV"))
ZONE_GEOMETRIES_GEOJSON_FILE_PATH = str(validate_path(DATA_SOURCES_DIR / "zone_geometries.geojson", "Zone geometries GeoJSON"))
IOT_CLINIC_ENVIRONMENT_CSV_PATH = str(validate_path(DATA_SOURCES_DIR / "iot_clinic_environment.csv", "IoT clinic environment CSV"))

# --- II. Health & Operational Thresholds (Content remains the same, truncated for brevity here) ---
ALERT_SPO2_CRITICAL_LOW_PCT = 90
ALERT_SPO2_WARNING_LOW_PCT = 94
ALERT_BODY_TEMP_FEVER_C = 38.0
# ... (all other threshold constants remain as previously defined) ...
AGE_THRESHOLD_VERY_HIGH = 75

# --- III. Edge Device Configuration (Content remains the same) ---
EDGE_APP_DEFAULT_LANGUAGE = "en"
# ... (all other edge device constants) ...
SMS_DATA_COMPRESSION_METHOD = "BASE85_ZLIB"

# --- IV. Supervisor Hub & Facility Node Configuration (Content remains the same) ---
HUB_SQLITE_DB_NAME = "sentinel_supervisor_hub.db"
# ... (all other hub/node constants) ...
NODE_REPORTING_INTERVAL_HOURS = 24

# --- V. Data Semantics & Categories (Content remains the same) ---
KEY_TEST_TYPES_FOR_ANALYSIS = {
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    # ... (all other test types) ...
}
CRITICAL_TESTS = [k for k, v in KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False)]
# ... (all other data semantics constants) ...
SYMPTOM_CLUSTERS_CONFIG = {
    "Fever, Cough, Fatigue": ["fever", "cough", "fatigue"],
    "Diarrhea & Vomiting": ["diarrhea", "vomit"],
    "Fever & Rash": ["fever", "rash"]
}

# --- VI. Web Dashboard & Visualization Configuration (Content remains the same) ---
CACHE_TTL_SECONDS_WEB_REPORTS = int(os.getenv("SENTINEL_CACHE_TTL", 3600))
# ... (all other web dashboard constants) ...
MAP_DEFAULT_ZOOM_LEVEL = 5

# --- VII. Color Palette (Content remains the same) ---
COLOR_RISK_HIGH = "#D32F2F"
# ... (all other color constants) ...
LEGACY_DISEASE_COLORS_WEB = {
    "TB": "#EF4444", "Malaria": "#F59E0B", # ... (all other disease colors) ...
    "Other": "#6B7280"
}

# Ensure log level from env var is valid before using it for logging within settings.py if needed
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    # Use print here as settings_logger might not be fully configured if LOG_LEVEL itself is bad for basicConfig
    print(f"WARNING (settings.py): Invalid LOG_LEVEL '{LOG_LEVEL}' from env. Defaulting to INFO for settings_logger.", file=sys.stderr)
    LOG_LEVEL = "INFO" # Reset to a valid default

# Configure the settings_logger specifically if needed for path validation warnings
# This is separate from the main app's logging config, but good practice
# if settings_logger.handlers: # Clear existing handlers if any to avoid duplicate logs on re-import
#     for handler in settings_logger.handlers:
#         settings_logger.removeHandler(handler)
# settings_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
# console_handler_settings = logging.StreamHandler(sys.stderr) # Log settings warnings to stderr
# console_handler_settings.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
# settings_logger.addHandler(console_handler_settings)
# settings_logger.propagate = False # Prevent passing to root logger if configured separately

settings_logger.info(f"Sentinel settings loaded. APP_NAME: {APP_NAME} v{APP_VERSION}. PROJECT_ROOT_DIR in settings: {PROJECT_ROOT_DIR}")
