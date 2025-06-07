# sentinel_project_root/config/settings.py
# Centralized, validated, and environment-aware configuration for Sentinel Health Co-Pilot.

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

settings_logger = logging.getLogger(__name__)

# --- Foundational Constants & Helpers (Module-Level) ---
# THIS IS THE CRITICAL FIX: Define PROJECT_ROOT_DIR at the module level.
# It is now a true constant available before any class definitions.
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

def _get_env(var_name: str, default: Any, var_type: type = str) -> Any:
    """Gets an environment variable, casts it to a type, and provides a default."""
    value = os.getenv(var_name, str(default))
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 't', 'y', 'yes')
        return var_type(value)
    except (ValueError, TypeError):
        settings_logger.warning(
            f"Could not cast env var '{var_name}' (value: '{value}') to {var_type}. "
            f"Using default: {default}."
        )
        return default

def _validate_path(path_str: str, is_dir: bool = False) -> Path:
    """Helper to validate a path relative to the project root."""
    path_obj = Path(path_str)
    # The helper now uses the module-level PROJECT_ROOT_DIR constant.
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT_DIR / path_obj
    
    if not path_obj.exists():
        settings_logger.warning(f"Config path check: '{path_obj.name}' not found at: {path_obj.resolve()}")
    elif is_dir and not path_obj.is_dir():
        settings_logger.warning(f"Config path error: '{path_obj.resolve()}' is not a directory.")
    elif not is_dir and not path_obj.is_file():
        settings_logger.warning(f"Config path error: '{path_obj.resolve()}' is not a file.")
    return path_obj


# --- Configuration Classes ---

class Core:
    """Core application settings and file paths."""
    APP_NAME = "Sentinel Health Co-Pilot"
    APP_VERSION = "5.1.0" # Version bump for corrected architecture
    ORGANIZATION_NAME = "LMIC Health Futures Initiative"
    APP_FOOTER_TEXT = f"© {datetime.now().year} {ORGANIZATION_NAME}. | v{APP_VERSION}"
    LOG_LEVEL = _get_env("SENTINEL_LOG_LEVEL", "INFO")
    
    # Paths now correctly use the module-level helper.
    ASSETS_DIR = _validate_path("assets", is_dir=True)
    DATA_SOURCES_DIR = _validate_path("data_sources", is_dir=True)
    
    APP_LOGO_SMALL_PATH = str(_validate_path("assets/sentinel_logo_small.png"))
    STYLE_CSS_PATH = str(_validate_path("styles/main.css"))
    
    HEALTH_RECORDS_CSV_PATH = str(_validate_path("data_sources/health_records_expanded.csv"))
    ZONE_ATTRIBUTES_CSV_PATH = str(_validate_path("data_sources/zone_attributes.csv"))
    ZONE_GEOMETRIES_GEOJSON_FILE_PATH = str(_validate_path("data_sources/zone_geometries.geojson"))
    ESCALATION_PROTOCOLS_JSON_PATH = str(_validate_path("config/escalation_protocols.json"))


class Thresholds:
    """Defines all critical operational and health-related thresholds."""
    SPO2_CRITICAL_LOW = _get_env("SENTINEL_THRESHOLD_SPO2_CRITICAL_LOW", 90, int)
    SPO2_WARNING_LOW = _get_env("SENTINEL_THRESHOLD_SPO2_WARNING_LOW", 94, int)
    BODY_TEMP_FEVER = _get_env("SENTINEL_THRESHOLD_BODY_TEMP_FEVER", 38.0, float)
    BODY_TEMP_HIGH_FEVER = _get_env("SENTINEL_THRESHOLD_BODY_TEMP_HIGH_FEVER", 39.5, float)
    RISK_SCORE_HIGH = _get_env("SENTINEL_THRESHOLD_RISK_SCORE_HIGH", 75, int)
    RISK_SCORE_MODERATE = _get_env("SENTINEL_THRESHOLD_RISK_SCORE_MODERATE", 60, int)
    FOLLOWUP_PRIORITY_HIGH = _get_env("SENTINEL_THRESHOLD_FOLLOWUP_PRIORITY_HIGH", 80, int)
    FOLLOWUP_PRIORITY_MODERATE = _get_env("SENTINEL_THRESHOLD_FOLLOWUP_PRIORITY_MODERATE", 60, int)
    TASK_PRIORITY_HIGH = _get_env("SENTINEL_THRESHOLD_TASK_PRIORITY_HIGH", 80, int)
    TASK_PRIORITY_MEDIUM = _get_env("SENTINEL_THRESHOLD_TASK_PRIORITY_MEDIUM", 60, int)
    AGE_CHILD = _get_env("SENTINEL_THRESHOLD_AGE_CHILD", 5, int)
    AGE_ADULT = _get_env("SENTINEL_THRESHOLD_AGE_ADULT", 18, int)
    AGE_ELDERLY = _get_env("SENTINEL_THRESHOLD_AGE_ELDERLY", 60, int)


class Semantics:
    """Configuration for data categories, labels, and analytical groupings."""
    KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
    KEY_TEST_TYPES: Dict[str, Dict[str, Union[str, int, bool]]] = {
        "Sputum-GeneXpert": {"group": "TB", "critical": True, "target_tat_days": 1},
        "RDT-Malaria": {"group": "Malaria", "critical": True, "target_tat_days": 0.5},
        "HIV-Rapid": {"group": "HIV", "critical": True, "target_tat_days": 0.25},
    }
    CRITICAL_TESTS = [k for k, v in KEY_TEST_TYPES.items() if v.get("critical")]
    SYMPTOM_CLUSTERS_CONFIG: Dict[str, List[str]] = {
        "ILI (Flu-like)": ["fever", "cough", "headache"],
        "Gastrointestinal": ["diarrhea", "vomit", "nausea"],
        "Respiratory Distress": ["cough", "breathless", "short of breath"],
        "Fever & Rash": ["fever", "rash"]
    }
    MIN_PATIENTS_FOR_SYMPTOM_CLUSTER = _get_env("SENTINEL_SEMANTICS_MIN_CLUSTER_PATIENTS", 2, int)


class WebUI:
    """Settings for the Streamlit web dashboards and visualizations."""
    CACHE_TTL_SECONDS = _get_env("SENTINEL_WEBUI_CACHE_TTL_SECONDS", 1800, int)
    DEFAULT_DATE_RANGE_DAYS = _get_env("SENTINEL_WEBUI_DEFAULT_DATE_RANGE_DAYS", 30, int)
    PLOT_DEFAULT_HEIGHT = _get_env("SENTINEL_WEBUI_PLOT_DEFAULT_HEIGHT", 400, int)
    PLOT_COMPACT_HEIGHT = _get_env("SENTINEL_WEBUI_PLOT_COMPACT_HEIGHT", 320, int)
    MAP_DEFAULT_HEIGHT = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_HEIGHT", 600, int)
    MAPBOX_STYLE = _get_env("SENTINEL_WEBUI_MAPBOX_STYLE", "carto-positron")
    MAP_DEFAULT_CENTER_LAT = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_CENTER_LAT", -1.286, float)
    MAP_DEFAULT_CENTER_LON = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_CENTER_LON", 36.817, float)
    MAP_DEFAULT_ZOOM_LEVEL = _get_env("SENTINEL_WEBUI_MAP_DEFAULT_ZOOM_LEVEL", 5, int)


class ColorPalette:
    """Centralized color definitions for consistent branding and data visualization."""
    ACTION_PRIMARY = "#1976D2"; ACTION_SECONDARY = "#546E7A"; ACCENT_BRIGHT = "#FFC107"
    RISK_HIGH = "#D32F2F"; RISK_MODERATE = "#FFA000"; RISK_LOW = "#388E3C"; RISK_NEUTRAL = "#757575"; INFO = "#0277BD"
    POSITIVE_DELTA = "#388E3C"; NEGATIVE_DELTA = "#D32F2F"
    TEXT_DARK = "#212121"; TEXT_HEADINGS_MAIN = "#333333"; TEXT_MUTED = "#6c757d"
    BACKGROUND_PAGE = "#F4F6F9"; BACKGROUND_CONTENT = "#FFFFFF"; BORDER_LIGHT = "#E0E0E0"
    LEGACY_DISEASE_COLORS_WEB: Dict[str, str] = {"TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6", "Pneumonia": "#3B82F6", "Other": "#6B7280"}


class Settings:
    """A unified settings object that provides easy access to all configuration sections."""
    def __init__(self):
        self.Core = Core()
        self.Thresholds = Thresholds()
        self.Semantics = Semantics()
        self.WebUI = WebUI()
        self.ColorPalette = ColorPalette()
        self.PROJECT_ROOT_DIR = PROJECT_ROOT_DIR # Add the global constant here for easy access.
        
        # --- Create flattened aliases for backward compatibility ---
        all_attrs = {}
        for section in [self.Core, self.Thresholds, self.Semantics, self.WebUI, self.ColorPalette]:
            for attr_name in dir(section):
                if not attr_name.startswith('_') and attr_name.isupper():
                     all_attrs[attr_name] = getattr(section, attr_name)
        
        self.__dict__.update(all_attrs)
        # Add legacy flat names not captured by the loop
        self.CACHE_TTL_SECONDS_WEB_REPORTS = self.WebUI.CACHE_TTL_SECONDS
        self.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = self.WebUI.DEFAULT_DATE_RANGE_DAYS
        self.APP_LAYOUT = "wide"


settings = Settings()

if settings.LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    settings_logger.warning(f"Invalid LOG_LEVEL '{settings.LOG_LEVEL}'. Application logger may default to INFO.")

settings_logger.info(f"Sentinel settings loaded: {settings.APP_NAME} v{settings.APP_VERSION}")
