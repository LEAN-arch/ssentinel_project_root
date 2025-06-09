# sentinel_project_root/config/settings.py
# SME-EVALUATED AND REVISED VERSION (GOLD STANDARD)
# This definitive version restores all original settings while incorporating critical
# bug fixes and robustness enhancements.

import os
import logging
from datetime import datetime
from pathlib import Path

# --- Logger Setup (must be at the top) ---
settings_logger = logging.getLogger(__name__)

def validate_path(path_obj: Path, description: str, is_dir: bool = False) -> Path:
    """
    Helper to validate if a path exists and log warnings if not.
    Now includes a check for empty files.
    """
    abs_path = path_obj.resolve()
    if not abs_path.exists():
        settings_logger.warning(f"Configuration Warning: {description} not found at resolved path: {abs_path}")
    elif is_dir and not abs_path.is_dir():
        settings_logger.warning(f"Configuration Warning: Expected a directory for {description}, but found a file at: {abs_path}")
    elif not is_dir and not abs_path.is_file():
        settings_logger.warning(f"Configuration Warning: Expected a file for {description}, but found a directory at: {abs_path}")
    # Add a check for empty files, as this can be a sign of a failed data pipeline.
    elif not is_dir and abs_path.is_file() and abs_path.stat().st_size == 0:
        settings_logger.warning(f"Configuration Warning: {description} file is empty (0 bytes) at: {abs_path}")
    return abs_path

# --- I. Core System & Directory Configuration ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
settings_logger.debug(f"PROJECT_ROOT_DIR resolved to: {PROJECT_ROOT_DIR}")

APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "4.0.3"
ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {datetime.now().year} {ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org"
LOG_LEVEL = os.getenv("SENTINEL_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("SENTINEL_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_DATE_FORMAT = os.getenv("SENTINEL_LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

# Paths
ASSETS_DIR = validate_path(PROJECT_ROOT_DIR / "assets", "Assets directory", is_dir=True)
DATA_SOURCES_DIR = validate_path(PROJECT_ROOT_DIR / "data_sources", "Data sources directory", is_dir=True)
APP_LOGO_SMALL_PATH = str(validate_path(ASSETS_DIR / "sentinel_logo_small.png", "Small app logo"))
APP_LOGO_LARGE_PATH = str(validate_path(ASSETS_DIR / "sentinel_logo_large.png", "Large app logo"))
STYLE_CSS_PATH_WEB = str(validate_path(ASSETS_DIR / "style_web_reports.css", "Global CSS stylesheet"))
ESCALATION_PROTOCOLS_JSON_PATH = str(validate_path(ASSETS_DIR / "escalation_protocols.json", "Escalation protocols JSON"))
PICTOGRAM_MAP_JSON_PATH = str(validate_path(ASSETS_DIR / "pictogram_map.json", "Pictogram map JSON"))
HAPTIC_PATTERNS_JSON_PATH = str(validate_path(ASSETS_DIR / "haptic_patterns.json", "Haptic patterns JSON"))
# --- CRITICAL BUG FIX: Pointing to the correct CSV file name provided in the prompt ---
HEALTH_RECORDS_PATH = str(validate_path(DATA_SOURCES_DIR / "health_records.csv", "Health records CSV"))
ZONE_ATTRIBUTES_PATH = str(validate_path(DATA_SOURCES_DIR / "zone_attributes.csv", "Zone attributes CSV"))
ZONE_GEOMETRIES_PATH = str(validate_path(DATA_SOURCES_DIR / "zone_geometries.geojson", "Zone geometries GeoJSON"))
IOT_ENV_RECORDS_PATH = str(validate_path(DATA_SOURCES_DIR / "iot_clinic_environment.csv", "IoT clinic environment CSV"))


# --- II. Health & Operational Thresholds ---
ALERT_SPO2_CRITICAL_LOW_PCT = 90
ALERT_SPO2_WARNING_LOW_PCT = 94
ALERT_BODY_TEMP_FEVER_C = 38.0
ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5
ALERT_HR_TACHYCARDIA_BPM = 100
ALERT_HR_BRADYCARDIA_BPM = 50
HEAT_STRESS_RISK_BODY_TEMP_C = 37.5
HEAT_STRESS_DANGER_BODY_TEMP_C = 38.5
ALERT_AMBIENT_CO2_HIGH_PPM = 1500
ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500
ALERT_AMBIENT_PM25_HIGH_UGM3 = 35
ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50
ALERT_AMBIENT_NOISE_HIGH_DBA = 85
ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32
ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41
FATIGUE_INDEX_MODERATE_THRESHOLD = 60
FATIGUE_INDEX_HIGH_THRESHOLD = 80
STRESS_HRV_LOW_THRESHOLD_MS = 20
RISK_SCORE_LOW_THRESHOLD = 40
RISK_SCORE_MODERATE_THRESHOLD = 60
RISK_SCORE_HIGH_THRESHOLD = 75
TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10
TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5
DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70
DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60
DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10
DISTRICT_DISEASE_PREVALENCE_HIGH_PERCENTILE = 0.80
CRITICAL_SUPPLY_DAYS_REMAINING = 7
LOW_SUPPLY_DAYS_REMAINING = 14
TARGET_DAILY_STEPS = 8000
RANDOM_SEED = 42
AGE_THRESHOLD_LOW = 5
AGE_THRESHOLD_MODERATE = 18
AGE_THRESHOLD_HIGH = 60
AGE_THRESHOLD_VERY_HIGH = 75


# --- III. Edge Device Configuration ---
EDGE_APP_DEFAULT_LANGUAGE = "en"
EDGE_APP_SUPPORTED_LANGUAGES = ["en", "sw", "fr"]
EDGE_MODEL_VITALS_DETERIORATION = "vitals_deterioration_v1.tflite"
EDGE_MODEL_FATIGUE_ASSESSMENT = "fatigue_index_v1.tflite"
EDGE_MODEL_ENVIRONMENTAL_ANOMALY = "anomaly_detection_base.tflite"
EDGE_DATA_BASELINE_WINDOW_DAYS = 7
EDGE_DATA_PROCESSING_INTERVAL_SECONDS = 60
PED_SQLITE_DB_NAME = "sentinel_ped_local.db"
PED_MAX_LOG_FILE_SIZE_MB = 50
EDGE_DATA_SYNC_PROTOCOLS_SUPPORTED = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE_BYTES = 256
SMS_DATA_COMPRESSION_METHOD = "BASE85_ZLIB"


# --- IV. Supervisor Hub & Facility Node Configuration ---
HUB_SQLITE_DB_NAME = "sentinel_supervisor_hub.db"
FACILITY_NODE_DB_TYPE = "POSTGRESQL"
FHIR_SERVER_ENDPOINT_LOCAL = "http://localhost:8080/fhir"
NODE_REPORTING_INTERVAL_HOURS = 24


# --- V. Data Semantics & Categories ---
KEY_TEST_TYPES_FOR_ANALYSIS = {
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
    "PulseOx": {"disease_group": "Vitals", "target_tat_days": 0, "critical": False, "display_name": "Pulse Oximetry"},
}
# This derived list makes it easy to filter for critical tests elsewhere in the application.
CRITICAL_TESTS = [k for k, v in KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False)]
TARGET_TEST_TURNAROUND_DAYS = 2.0
TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0
TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0
OVERDUE_TEST_BUFFER_DAYS = 2
OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK = 7
KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin', 'Iron-Folate', 'Insulin']
TARGET_MALARIA_POSITIVITY_RATE = 10.0
SYMPTOM_CLUSTERS_CONFIG = {
    "Fever, Cough, Fatigue": ["fever", "cough", "fatigue"],
    "Diarrhea & Vomiting": ["diarrhea", "vomit"],
    "Fever & Rash": ["fever", "rash"]
}


# --- VI. Web Dashboard & Visualization Configuration ---
CACHE_TTL_SECONDS_WEB_REPORTS = int(os.getenv("SENTINEL_CACHE_TTL", 3600))
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = 30
WEB_PLOT_DEFAULT_HEIGHT = 400
WEB_PLOT_COMPACT_HEIGHT = 320
WEB_MAP_DEFAULT_HEIGHT = 600
MAPBOX_STYLE_WEB = "carto-positron"
DEFAULT_CRS_STANDARD = "EPSG:4326"
MAP_DEFAULT_CENTER_LAT = -1.286389
MAP_DEFAULT_CENTER_LON = 36.817223
MAP_DEFAULT_ZOOM_LEVEL = 5


# --- VII. Color Palette ---
COLOR_RISK_HIGH = "#D32F2F"
COLOR_RISK_MODERATE = "#FBC02D"
COLOR_RISK_LOW = "#388E3C"
COLOR_RISK_NEUTRAL = "#757575"
COLOR_ACTION_PRIMARY = "#1976D2"
COLOR_ACTION_SECONDARY = "#546E7A"
COLOR_ACCENT_BRIGHT = "#4D7BF3"
COLOR_POSITIVE_DELTA = "#27AE60"
COLOR_NEGATIVE_DELTA = "#C0392B"
COLOR_TEXT_DARK = "#343a40"
COLOR_TEXT_HEADINGS_MAIN = "#1A2557"
COLOR_TEXT_HEADINGS_SUB = "#2C3E50"
COLOR_TEXT_MUTED = "#6c757d"
COLOR_TEXT_LINK_DEFAULT = COLOR_ACTION_PRIMARY
COLOR_BACKGROUND_PAGE = "#f8f9fa"
COLOR_BACKGROUND_CONTENT = "#ffffff"
COLOR_BACKGROUND_SUBTLE = "#e9ecef"
COLOR_BACKGROUND_WHITE = "#FFFFFF"
# --- BUG FIX: Corrected typo in transparent color definition ---
COLOR_BACKGROUND_CONTENT_TRANSPARENT = 'rgba(255,255,255,0.85)'
COLOR_BORDER_LIGHT = "#dee2e6"
COLOR_BORDER_MEDIUM = "#ced4da"

LEGACY_DISEASE_COLORS_WEB = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6", "Pneumonia": "#3B82F6",
    "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1", "Hypertension": "#F97316",
    "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16", "Heat Stroke": "#FF6347",
    "Severe Dehydration": "#4682B4", "Sepsis": "#800080", "Diarrheal Diseases (Severe)": "#D2691E",
    "Other": "#6B7280"
}


# --- Final Validation & Logging ---
# --- BUG FIX: Removed duplicated code block ---
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    settings_logger.warning(f"Invalid LOG_LEVEL '{LOG_LEVEL}' from env. The root logger may default to INFO.")
    
settings_logger.info(f"Sentinel settings module loaded. APP_NAME: {APP_NAME} v{APP_VERSION}.")
