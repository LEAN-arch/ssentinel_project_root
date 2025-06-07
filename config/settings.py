# sentinel_project_root/config/settings.py
# Centralized, validated, and environment-aware configuration for Sentinel Health Co-Pilot.

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

settings_logger = logging.getLogger(__name__)

def _get_env(var_name: str, default: Any, var_type: type = str) -> Any:
    value = os.getenv(var_name, str(default))
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 't', 'y', 'yes')
        return var_type(value)
    except (ValueError, TypeError):
        settings_logger.warning(f"Could not cast env var '{var_name}' to {var_type}. Using default: {default}.")
        return default

# --- Foundational Constants (Module-Level) ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

def _validate_path(path_str: Union[str, Path], is_dir: bool = False) -> Path:
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT_DIR / path_obj
    if not path_obj.exists():
        settings_logger.warning(f"Config path check: '{path_obj.name}' not found at: {path_obj.resolve()}")
    elif is_dir and not path_obj.is_dir():
        settings_logger.warning(f"Config path error: '{path_obj.resolve()}' is not a directory.")
    elif not is_dir and not path_obj.is_file():
        settings_logger.warning(f"Config path error: '{path_obj.resolve()}' is not a file.")
    return path_obj

# --- Core System & Directory Configuration ---
APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "6.0.0" # Version bump for stable architecture
ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {datetime.now().year} {ORGANIZATION_NAME}. | v{APP_VERSION}"
LOG_LEVEL = _get_env("SENTINEL_LOG_LEVEL", "INFO")
APP_LAYOUT = "wide"
ASSETS_DIR = _validate_path("assets", is_dir=True)
DATA_SOURCES_DIR = _validate_path("data_sources", is_dir=True)
APP_LOGO_SMALL_PATH = str(_validate_path("assets/sentinel_logo_small.png"))
STYLE_CSS_PATH = str(_validate_path("styles/main.css"))
HEALTH_RECORDS_CSV_PATH = str(_validate_path("data_sources/health_records_expanded.csv"))
ZONE_ATTRIBUTES_CSV_PATH = str(_validate_path("data_sources/zone_attributes.csv"))
ZONE_GEOMETRIES_GEOJSON_FILE_PATH = str(_validate_path("data_sources/zone_geometries.geojson"))
ESCALATION_PROTOCOLS_JSON_PATH = str(_validate_path("config/escalation_protocols.json"))

# --- Health & Operational Thresholds ---
ALERT_SPO2_CRITICAL_LOW_PCT = _get_env("SENTINEL_THRESHOLD_SPO2_CRITICAL_LOW", 90, int)
ALERT_SPO2_WARNING_LOW_PCT = _get_env("SENTINEL_THRESHOLD_SPO2_WARNING_LOW", 94, int)
ALERT_BODY_TEMP_FEVER_C = _get_env("SENTINEL_THRESHOLD_BODY_TEMP_FEVER", 38.0, float)
ALERT_BODY_TEMP_HIGH_FEVER_C = _get_env("SENTINEL_THRESHOLD_BODY_TEMP_HIGH_FEVER", 39.5, float)
RISK_SCORE_HIGH_THRESHOLD = _get_env("SENTINEL_THRESHOLD_RISK_SCORE_HIGH", 75, int)
RISK_SCORE_MODERATE_THRESHOLD = _get_env("SENTINEL_THRESHOLD_RISK_SCORE_MODERATE", 60, int)
FATIGUE_INDEX_HIGH_THRESHOLD = _get_env("SENTINEL_THRESHOLD_FOLLOWUP_PRIORITY_HIGH", 80, int)
FATIGUE_INDEX_MODERATE_THRESHOLD = _get_env("SENTINEL_THRESHOLD_FOLLOWUP_PRIORITY_MODERATE", 60, int)
TASK_PRIORITY_HIGH_THRESHOLD = _get_env("SENTINEL_THRESHOLD_TASK_PRIORITY_HIGH", 80, int)
TASK_PRIORITY_MEDIUM_THRESHOLD = _get_env("SENTINEL_THRESHOLD_TASK_PRIORITY_MEDIUM", 60, int)
AGE_THRESHOLD_CHILD = _get_env("SENTINEL_THRESHOLD_AGE_CHILD", 5, int)
AGE_THRESHOLD_ADULT = _get_env("SENTINEL_THRESHOLD_AGE_ADULT", 18, int)
AGE_THRESHOLD_ELDERLY = _get_env("SENTINEL_THRESHOLD_AGE_ELDERLY", 60, int)

# --- Data Semantics & Categories ---
KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_TEST_TYPES_FOR_ANALYSIS: Dict[str, Dict[str, Union[str, int, bool]]] = {
    "Sputum-GeneXpert": {"group": "TB", "critical": True, "target_tat_days": 1},
    "RDT-Malaria": {"group": "Malaria", "critical": True, "target_tat_days": 0.5},
    "HIV-Rapid": {"group": "HIV", "critical": True, "target_tat_days": 0.25},
}
CRITICAL_TESTS = [k for k, v in KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical")]
SYMPTOM_CLUSTERS_CONFIG: Dict[str, List[str]] = {
    "ILI (Flu-like)": ["fever", "cough", "headache"],
    "Gastrointestinal": ["diarrhea", "vomit", "nausea"],
    "Respiratory Distress": ["cough", "breathless", "short of breath"],
    "Fever & Rash": ["fever", "rash"]
}
MIN_PATIENTS_FOR_SYMPTOM_CLUSTER = _get_env("SENTINEL_SEMANTICS_MIN_CLUSTER_PATIENTS", 2, int)

# --- Web UI & Visualization Configuration ---
CACHE_TTL_SECONDS_WEB_REPORTS = _get_env("SENTINEL_WEBUI_CACHE_TTL_SECONDS", 1800, int)
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = _get_env("SENTINEL_WEBUI_DEFAULT_DATE_RANGE_DAYS", 30, int)
WEB_PLOT_DEFAULT_HEIGHT = _get_env("SENTINEL_WEBUI_PLOT_DEFAULT_HEIGHT", 400, int)
WEB_PLOT_COMPACT_HEIGHT = _get_env("SENTINEL_WEBUI_PLOT_COMPACT_HEIGHT", 320, int)
WEB_MAP_DEFAULT_HEIGHT = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_HEIGHT", 600, int)
MAPBOX_STYLE_WEB = _get_env("SENTINEL_WEBUI_MAPBOX_STYLE", "carto-positron")
MAP_DEFAULT_CENTER_LAT = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_CENTER_LAT", -1.286, float)
MAP_DEFAULT_CENTER_LON = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_CENTER_LON", 36.817, float)
MAP_DEFAULT_ZOOM_LEVEL = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_ZOOM_LEVEL", 5, int)

# --- Color Palette ---
COLOR_ACTION_PRIMARY = "#1976D2"; COLOR_ACTION_SECONDARY = "#546E7A"; COLOR_ACCENT_BRIGHT = "#FFC107"
COLOR_RISK_HIGH = "#D32F2F"; COLOR_RISK_MODERATE = "#FFA000"; COLOR_RISK_LOW = "#388E3C"; COLOR_RISK_NEUTRAL = "#757575"; COLOR_INFO = "#0277BD"
COLOR_POSITIVE_DELTA = "#388E3C"; COLOR_NEGATIVE_DELTA = "#D32F2F"
COLOR_TEXT_DARK = "#212121"; COLOR_TEXT_HEADINGS_MAIN = "#333333"; COLOR_TEXT_MUTED = "#6c757d"
COLOR_BACKGROUND_PAGE = "#F4F6F9"; COLOR_BACKGROUND_CONTENT = "#FFFFFF"; COLOR_BACKGROUND_WHITE = "#FFFFFF"; COLOR_BORDER_LIGHT = "#E0E0E0"
LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {"TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6", "Pneumonia": "#3B82F6", "Other": "#6B7280"}

settings_logger.info(f"Sentinel settings loaded: {APP_NAME} v{APP_VERSION}")
