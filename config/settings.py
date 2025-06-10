# sentinel_project_root/config/settings.py
# SME PLATINUM STANDARD - CENTRALIZED CONFIGURATION HUB (V2 - FINAL)

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import (BaseModel, DirectoryPath, Field, FilePath,
                      computed_field, model_validator)
from pydantic_settings import BaseSettings, SettingsConfigDict

settings_logger = logging.getLogger(__name__)

# --- Nested Models for Structured Configuration ---
class TestTypeConfig(BaseModel):
    disease_group: str
    target_tat_days: float
    is_critical: bool
    display_name: str

class ModelWeightsConfig(BaseModel):
    base_age_gt_65: float = 10.0; base_age_lt_5: float = 12.0
    symptom_cluster_severity_high: float = 25.0; symptom_cluster_severity_med: float = 15.0
    vital_spo2_critical: float = 35.0; vital_temp_critical: float = 25.0
    comorbidity: float = 15.0; risk_score_multiplier: float = 0.6
    critical_vital_alert: float = 40.0; pending_urgent_referral: float = 30.0
    days_overdue_multiplier: float = 1.5; poor_med_adherence: float = 20.0

class AnalyticsConfig(BaseModel):
    spo2_critical_threshold_pct: int = 90; spo2_warning_threshold_pct: int = 94
    temp_high_fever_threshold_c: float = 39.0; noise_high_threshold_db: int = 80
    risk_score_low_threshold: int = 40; risk_score_moderate_threshold: int = 65
    supply_critical_threshold_days: int = 7; supply_low_threshold_days: int = 14
    test_tat_overdue_buffer_days: int = 2; top_n_rejection_reasons: int = 5
    prophet_forecast_days: int = 90; prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0; random_seed: int = 42

# --- Main Settings Class ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='SENTINEL_', case_sensitive=False, env_file='.env', env_file_encoding='utf-8', extra='ignore')

    PROJECT_ROOT_DIR: DirectoryPath = Path(__file__).resolve().parent.parent
    APP_NAME: str = "Sentinel Health Co-Pilot"; APP_VERSION: str = "5.1.0"
    ORGANIZATION_NAME: str = "Resilient Health Systems Initiative"; SUPPORT_CONTACT_INFO: str = "support@rhsc.org"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    ASSETS_DIR: DirectoryPath; DATA_SOURCES_DIR: DirectoryPath
    APP_LOGO_SMALL_PATH: FilePath; APP_LOGO_LARGE_PATH: FilePath; STYLE_CSS_PATH: FilePath
    ESCALATION_PROTOCOLS_PATH: FilePath; PICTOGRAM_MAP_PATH: FilePath; HAPTIC_PATTERNS_PATH: FilePath
    HEALTH_RECORDS_PATH: FilePath; ZONE_ATTRIBUTES_PATH: FilePath; ZONE_GEOMETRIES_PATH: FilePath; IOT_RECORDS_PATH: FilePath

    @model_validator(mode='before')
    @classmethod
    def set_default_paths(cls, values: Any) -> Any:
        if isinstance(values, dict):
            root = values.get('PROJECT_ROOT_DIR', Path(__file__).resolve().parent.parent)
            assets = root / "assets"; data = root / "data_sources"
            values.setdefault('ASSETS_DIR', assets); values.setdefault('DATA_SOURCES_DIR', data)
            values.setdefault('APP_LOGO_SMALL_PATH', assets / "sentinel_logo_small.png"); values.setdefault('APP_LOGO_LARGE_PATH', assets / "sentinel_logo_large.png")
            values.setdefault('STYLE_CSS_PATH', assets / "style_web_reports.css"); values.setdefault('ESCALATION_PROTOCOLS_PATH', assets / "escalation_protocols.json")
            values.setdefault('PICTOGRAM_MAP_PATH', assets / "pictogram_map.json"); values.setdefault('HAPTIC_PATTERNS_PATH', assets / "haptic_patterns.json")
            values.setdefault('HEALTH_RECORDS_PATH', data / "health_records_expanded.csv"); values.setdefault('ZONE_ATTRIBUTES_PATH', data / "zone_attributes.csv")
            values.setdefault('ZONE_GEOMETRIES_PATH', data / "zone_geometries.geojson"); values.setdefault('IOT_RECORDS_PATH', data / "iot_clinic_environment.csv")
        return values

    KEY_TEST_TYPES: Dict[str, TestTypeConfig] = {
        "Malaria RDT": TestTypeConfig(disease_group="Vector-Borne", target_tat_days=0.5, is_critical=True, display_name="Malaria RDT"),
        "CBC": TestTypeConfig(disease_group="General", target_tat_days=1.0, is_critical=True, display_name="CBC"),
        "COVID-19 Ag": TestTypeConfig(disease_group="Respiratory", target_tat_days=0.25, is_critical=True, display_name="COVID-19 Ag"),
    }
    KEY_DIAGNOSES: List[str] = ['Malaria', 'Pneumonia', 'Diarrhea', 'Tuberculosis', 'Hypertension']
    KEY_SUPPLY_ITEMS: List[str] = ['Paracetamol', 'Amoxicillin', 'ORS', 'Metformin', 'Gloves']
    SYMPTOM_CLUSTERS: Dict[str, List[str]] = {"respiratory_distress": ["difficulty breathing", "chest pain"], "severe_febrile": ["fever", "chills", "stiff neck"]}
    
    @computed_field
    @property
    def CRITICAL_TESTS(self) -> List[str]: return [k for k, v in self.KEY_TEST_TYPES.items() if v.is_critical]

    ANALYTICS: AnalyticsConfig = AnalyticsConfig(); MODEL_WEIGHTS: ModelWeightsConfig = ModelWeightsConfig()

    WEB_CACHE_TTL_SECONDS: int = 3600
    MAPBOX_TOKEN: Optional[str] = Field(None, description="Set via SENTINEL_MAPBOX_TOKEN env var")
    MAPBOX_STYLE: str = "carto-positron"; MAP_DEFAULT_CENTER: Tuple[float, float] = (-1.286389, 36.817223); MAP_DEFAULT_ZOOM: int = 5
    
    COLOR_PRIMARY: str = "#1976D2"; COLOR_SECONDARY: str = "#546E7A"; COLOR_ACCENT: str = "#4D7BF3"
    COLOR_BACKGROUND_PAGE: str = "#F8F9FA"; COLOR_BACKGROUND_CONTENT: str = "#FFFFFF"
    COLOR_TEXT_PRIMARY: str = "#343A40"; COLOR_TEXT_HEADINGS: str = "#1A2557"; COLOR_TEXT_MUTED: str = "#6C757D"
    COLOR_BORDER: str = "#DEE2E6"; COLOR_RISK_HIGH: str = "#D32F2F"; COLOR_RISK_MODERATE: str = "#FBC02D"
    COLOR_RISK_LOW: str = "#388E3C"; COLOR_DELTA_POSITIVE: str = "#27AE60"; COLOR_DELTA_NEGATIVE: str = "#C0392B"
    PLOTLY_COLORWAY: List[str] = [COLOR_PRIMARY, COLOR_DELTA_POSITIVE, COLOR_RISK_MODERATE, COLOR_RISK_HIGH, COLOR_SECONDARY]
    
    @computed_field
    @property
    def APP_FOOTER_TEXT(self) -> str: return f"© {datetime.now().year} {self.ORGANIZATION_NAME}. Actionable Intelligence for Resilient Health Systems."

try:
    settings = Settings()
    settings_logger.info(f"Sentinel settings loaded successfully. App: {settings.APP_NAME} v{settings.APP_VERSION}")
except Exception as e:
    settings_logger.critical(f"FATAL: Could not initialize Pydantic settings. Error: {e}", exc_info=True)
    raise
