# sentinel_project_root/data_processing/__init__.py
# This file makes the 'data_processing' directory a Python package.

from .loaders import (
    load_health_records,
    load_iot_clinic_environment_data,
    load_zone_data,
    load_escalation_protocols,
    load_pictogram_map,
    load_haptic_patterns
)
from .aggregation import (
    get_overall_kpis,
    get_chw_summary_kpis, # Renamed for clarity
    get_clinic_summary_kpis, # Renamed for clarity
    get_clinic_environmental_summary_kpis, # Renamed for clarity
    get_district_summary_kpis,
    get_trend_data # General trend utility
)
from .enrichment import (
    enrich_zone_geodata_with_health_aggregates # Renamed for clarity
)
from .helpers import (
    clean_column_names,
    convert_to_numeric,
    robust_json_load,
    hash_dataframe_safe # Replaces hash_geodataframe
)

__all__ = [
    "load_health_records",
    "load_iot_clinic_environment_data",
    "load_zone_data",
    "load_escalation_protocols",
    "load_pictogram_map",
    "load_haptic_patterns",
    "get_overall_kpis",
    "get_chw_summary_kpis",
    "get_clinic_summary_kpis",
    "get_clinic_environmental_summary_kpis",
    "get_district_summary_kpis",
    "get_trend_data",
    "enrich_zone_geodata_with_health_aggregates",
    "clean_column_names",
    "convert_to_numeric",
    "robust_json_load",
    "hash_dataframe_safe"
]
