# ssentinel_project_root/config/settings.py
# SME PLATINUM STANDARD (V4 - Pydantic Model & Validation)
# This version refactors the configuration into a Pydantic settings model.
# This provides automatic type validation, coercion from environment variables,
# and a structured, self-documenting schema for all application settings.

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Literal, Any, computed_field

# <<< SME REVISION V4 >>> Use Pydantic for settings management and validation.
from pydantic import BaseModel, Field, field_validator, model_validator
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

# <<< SME REVISION V4 >>> Define nested models for structured configuration.
class TestTypeConfig(BaseModel):
    disease_group: str
    target_tat_days: float
    critical: bool
    display_name: str

class SymptomClusterConfig(BaseModel):
    fever_cough_fatigue: List[str] = Field(..., alias="Fever, Cough, Fatigue")
    diarrhea_vomiting: List[str] = Field(..., alias="Diarrhea & Vomiting")
    fever_rash: List[str] = Field(..., alias="Fever & Rash")

# <<< SME REVISION V4 >>> The main settings class.
class Settings(BaseSettings):
    """
    Manages all application settings using Pydantic for validation and type safety.
    Loads settings from environment variables with a specified prefix.
    """
    # Use model_config to define behavior, like reading from .env files or setting prefixes
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

    # <<< SME REVISION V4 >>> Pydantic validators replace the standalone helper function.
    # They run automatically when the Settings object is created.
    @field_validator(
        'ASSETS_DIR', 'DATA_SOURCES_DIR', mode='before'
    )
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

    # <<< SME REVISION V4 >>> model_validator allows setting default paths based on other fields.
    @model_validator(mode='before')
    @classmethod
    def set_default_paths(cls, values: Any) -> Any:
        if isinstance(values, dict):
            root = values.get('PROJECT_ROOT_DIR', Path(__file__).resolve().parent.parent)
            values.setdefault('ASSETS_DIR', root / "assets")
            values.setdefault('DATA_SOURCES_DIR', root / "data_sources")
            assets = values.get('ASSETS_DIR', root / "assets")
            data = values.get('DATA_SOURCES_DIR', root / "data_sources")
            
            # Set defaults for all paths if not provided
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
    RISK_SCORE_MODERATE_THRESHOLD: int = 60
    RISK_SCORE_HIGH_THRESHOLD: int = 75
    # ... (all other simple key-value thresholds would go here with their types) ...
    RANDOM_SEED: int = 42

    # --- III. Edge Device, IV. Supervisor Hub, etc. ---
    # ... (these sections would also be defined here) ...
    
    # --- V. Data Semantics & Categories ---
    KEY_TEST_TYPES_FOR_ANALYSIS: Dict[str, TestTypeConfig] = {
        "Malaria RDT": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
        "CBC": {"disease_group": "General", "target_tat_days": 1, "critical": True, "display_name": "CBC"},
        "COVID-19 Ag": {"disease_group": "Respiratory", "target_tat_days": 0.25, "critical": True, "display_name": "COVID-19 Ag"},
        # ... other tests
    }
    # This is the semantically correct and validated variable name
    KEY_DIAGNOSES_FOR_ACTION: List[str] = ['Malaria', 'Pneumonia', 'Diarrhea', 'Hypertension', 'Diabetes', 'URI', 'Bacterial Infection']
    KEY_DRUG_SUBSTRINGS_SUPPLY: List[str] = ['Paracetamol', 'Amoxicillin', 'ORS Packet', 'Metformin', 'Lisinopril']
    NON_INFORMATIVE_SYMPTOMS: List[str] = ['none', 'n/a', 'asymptomatic', '']
    SYMPTOM_CLUSTERS_CONFIG: SymptomClusterConfig = {
        "Fever, Cough, Fatigue": ["fever", "cough", "fatigue"],
        "Diarrhea & Vomiting": ["diarrhea", "vomit"],
        "Fever & Rash": ["fever", "rash"]
    }

    # <<< SME REVISION V4 >>> Computed fields are the Pydantic way to derive values.
    @computed_field
    @property
    def CRITICAL_TESTS(self) -> List[str]:
        return [k for k, v in self.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.critical]
        
    @computed_field
    @property
    def APP_FOOTER_TEXT(self) -> str:
        return f"© {datetime.now().year} {self.ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."

    # --- VI. Web Dashboard & VII. Color Palette ---
    CACHE_TTL_SECONDS_WEB_REPORTS: int = Field(default=3600, alias="CACHE_TTL") # Can set alias for env var
    COLOR_RISK_HIGH: str = "#D32F2F"
    COLOR_RISK_MODERATE: str = "#FBC02D"
    # ... (all other colors) ...
    COLOR_BORDER_MEDIUM: str = "#ced4da"

# --- Singleton Instance ---
# Create a single, validated instance of the settings for the entire application to import.
try:
    settings = Settings()
    # Now you can convert paths to strings where needed on usage, e.g., str(settings.APP_LOGO_SMALL_PATH)
    settings_logger.info(f"Sentinel settings loaded and validated. APP_NAME: {settings.APP_NAME} v{settings.APP_VERSION}.")
except Exception as e:
    settings_logger.critical(f"FATAL: Could not initialize application settings. Error: {e}", exc_info=True)
    # In a real application, you might want to exit here if settings are critical.
    raise
