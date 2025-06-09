# sentinel_project_root/config/settings.py
# SME PLATINUM STANDARD (V4.1 - IMPORT FIX)
# This definitive version corrects the critical ImportError by importing
# `computed_field` from `pydantic` instead of `typing`.

import os
import logging
from datetime import datetime
from pathlib import Path
# <<< SME FIX >>> `computed_field` is removed from this import.
from typing import List, Dict, Literal, Any

# <<< SME FIX >>> `computed_field` is added to the pydantic import.
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Logger Setup ---
settings_logger = logging.getLogger(__name__)

# --- Helper Functions (can be outside the class or static methods) ---
def _validate_path(path_val: Any, info: Any, is_dir: bool = False) -> Path:
    """ Pydantic-compatible validator for file/directory paths. """
    if not isinstance(path_val, (str, Path)):
        raise ValueError(f"Path must be a string or Path object, not {type(path_val)}")
    
    path_obj = Path(path_val).resolve()
    description = info.field_name.replace('_', ' ').title()

    if not path_obj.exists():
        settings_logger.warning(f"Config Warning: {description} not found at: {path_obj}")
    elif is_dir and not path_obj.is_dir():
        settings_logger.warning(f"Config Warning: Expected directory for {description}, found file at: {path_obj}")
    elif not is_dir and not path_obj.is_file():
        settings_logger.warning(f"Config Warning: Expected file for {description}, found directory at: {path_obj}")
    elif not is_dir and path_obj.stat().st_size == 0:
        settings_logger.warning(f"Config Warning: {description} file is empty at: {path_obj}")
    return path_obj

# --- Nested Models for Structured Configuration ---
class TestTypeConfig(BaseModel):
    disease_group: str
    target_tat_days: float
    critical: bool
    display_name: str

class SymptomClusterConfig(BaseModel):
    fever_cough_fatigue: List[str] = Field(..., alias="Fever, Cough, Fatigue")
    diarrhea_vomiting: List[str] = Field(..., alias="Diarrhea & Vomiting")
    fever_rash: List[str] = Field(..., alias="Fever & Rash")

# --- Main Settings Class ---
class Settings(BaseSettings):
    """
    Manages all application settings using Pydantic for validation and type safety.
    Loads settings from environment variables with a specified prefix.
    """
    model_config = SettingsConfigDict(env_prefix='SENTINEL_', case_sensitive=False)

    # --- I. Core System & Directory Configuration ---
    PROJECT_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    APP_NAME: str = "Sentinel Health Co-Pilot"
    APP_VERSION: str = "4.0.3"
    ORGANIZATION_NAME: str = "LMIC Health Futures Initiative"
    SUPPORT_CONTACT_INFO: str = "support@lmic-health-futures.org"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Paths (will use validators)
    ASSETS_DIR: Path
    DATA_SOURCES_DIR: Path
    APP_LOGO_SMALL_PATH: Path
    APP_LOGO_LARGE_PATH: Path
    STYLE_CSS_PATH_WEB: Path
    ESCALATION_PROTOCOLS_JSON_PATH: Path
    PICTOGRAM_MAP_JSON_PATH: Path
    HAPTIC_PATTERNS_JSON_PATH: Path
    HEALTH_RECORDS_PATH: Path
    ZONE_ATTRIBUTES_PATH: Path
    ZONE_GEOMETRIES_PATH: Path
    IOT_ENV_RECORDS_PATH: Path

    @field_validator('ASSETS_DIR', 'DATA_SOURCES_DIR', mode='before')
    @classmethod
    def validate_dirs(cls, v, info):
        return _validate_path(v, info, is_dir=True)

    @field_validator(
        'APP_LOGO_SMALL_PATH', 'APP_LOGO_LARGE_PATH', 'STYLE_CSS_PATH_WEB', 'ESCALATION_PROTOCOLS_JSON_PATH',
        'PICTOGRAM_MAP_JSON_PATH', 'HAPTIC_PATTERNS_JSON_PATH', 'HEALTH_RECORDS_PATH',
        'ZONE_ATTRIBUTES_PATH', 'ZONE_GEOMETRIES_PATH', 'IOT_ENV_RECORDS_PATH', mode='before'
    )
    @classmethod
    def validate_files(cls, v, info):
        return _validate_path(v, info, is_dir=False)

    @model_validator(mode='before')
    @classmethod
    def set_default_paths(cls, values: Any) -> Any:
        if isinstance(values, dict):
            root = values.get('PROJECT_ROOT_DIR', Path(__file__).resolve().parent.parent)
            values.setdefault('ASSETS_DIR', root / "assets")
            values.setdefault('DATA_SOURCES_DIR', root / "data_sources")
            assets = values.get('ASSETS_DIR')
            data = values.get('DATA_SOURCES_DIR')
            
            values.setdefault('APP_LOGO_SMALL_PATH', assets / "sentinel_logo_small.png")
            values.setdefault('APP_LOGO_LARGE_PATH', assets / "sentinel_logo_large.png")
            values.setdefault('STYLE_CSS_PATH_WEB', assets / "style_web_reports.css")
            values.setdefault('ESCALATION_PROTOCOLS_JSON_PATH', assets / "escalation_protocols.json")
            values.setdefault('PICTOGRAM_MAP_JSON_PATH', assets / "pictogram_map.json")
            values.setdefault('HAPTIC_PATTERNS_JSON_PATH', assets / "haptic_patterns.json")
            values.setdefault('HEALTH_RECORDS_PATH', data / "health_records_expanded.csv")
            values.setdefault('ZONE_ATTRIBUTES_PATH', data / "zone_attributes.csv")
            values.setdefault('ZONE_GEOMETRIES_PATH', data / "zone_geometries.geojson")
            values.setdefault('IOT_ENV_RECORDS_PATH', data / "iot_clinic_environment.csv")
        return values

    # --- II. Health & Operational Thresholds ---
    ALERT_SPO2_CRITICAL_LOW_PCT: int = 90
    ALERT_SPO2_WARNING_LOW_PCT: int = 94
    ALERT_BODY_TEMP_HIGH_FEVER_C: float = 39.5
    RISK_SCORE_LOW_THRESHOLD: int = 40
    RISK_SCORE_MODERATE_THRESHOLD: int = 60
    RISK_SCORE_HIGH_THRESHOLD: int = 75
    CRITICAL_SUPPLY_DAYS_REMAINING: int = 7
    LOW_SUPPLY_DAYS_REMAINING: int = 14
    TARGET_TEST_TURNAROUND_DAYS: float = 2.0
    OVERDUE_TEST_BUFFER_DAYS: int = 2
    TESTING_TOP_N_REJECTION_REASONS: int = 10
    # ... other thresholds
    RANDOM_SEED: int = 42

    # --- V. Data Semantics & Categories ---
    KEY_TEST_TYPES_FOR_ANALYSIS: Dict[str, TestTypeConfig] = {
        "Malaria RDT": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
        "CBC": {"disease_group": "General", "target_tat_days": 1, "critical": True, "display_name": "CBC"},
        "COVID-19 Ag": {"disease_group": "Respiratory", "target_tat_days": 0.25, "critical": True, "display_name": "COVID-19 Ag"},
    }
    KEY_DIAGNOSES_FOR_ACTION: List[str] = ['Malaria', 'Pneumonia', 'Diarrhea', 'Hypertension', 'Diabetes']
    KEY_DRUG_SUBSTRINGS_SUPPLY: List[str] = ['Paracetamol', 'Amoxicillin', 'ORS Packet', 'Metformin']
    
    # ... other semantic configs

    # <<< SME FIX >>> These @computed_field decorators will now work correctly.
    @computed_field
    @property
    def CRITICAL_TESTS(self) -> List[str]:
        return [k for k, v in self.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.critical]
        
    @computed_field
    @property
    def APP_FOOTER_TEXT(self) -> str:
        return f"© {datetime.now().year} {self.ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."

    # --- VI. Web Dashboard & VII. Color Palette ---
    CACHE_TTL_SECONDS_WEB_REPORTS: int = Field(default=3600, alias="CACHE_TTL")
    MAPBOX_STYLE_WEB: str = "carto-positron"
    MAP_DEFAULT_CENTER_LAT: float = -1.286389
    MAP_DEFAULT_CENTER_LON: float = 36.817223
    MAP_DEFAULT_ZOOM_LEVEL: int = 5
    COLOR_ACTION_PRIMARY: str = "#1976D2"
    # ... other colors

# --- Singleton Instance ---
try:
    settings = Settings()
    settings_logger.info(f"Sentinel settings loaded and validated. APP_NAME: {settings.APP_NAME} v{settings.APP_VERSION}.")
except Exception as e:
    settings_logger.critical(f"FATAL: Could not initialize application settings. Error: {e}", exc_info=True)
    raise
