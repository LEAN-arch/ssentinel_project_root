# sentinel_project_root/data_processing/__init__.py
# This file makes the 'data_processing' directory a Python package
# and exposes key functions for easier importing.

from .aggregation import (
    get_trend_data,
    # get_overall_kpis, # REMOVED - This function no longer exists
    get_chw_summary_kpis,
    get_clinic_summary_kpis,
    get_clinic_environmental_summary_kpis,
    get_district_summary_kpis
)

from .enrichment import (
    enrich_zone_geodata_with_health_aggregates
)

from .helpers import (
    data_cleaner,
    convert_to_numeric,
    robust_json_load,
    hash_dataframe_safe,
    convert_date_columns,
    standardize_missing_values
)

from .loaders import (
    load_health_records,
    load_iot_clinic_environment_data,
    load_zone_data,
    load_json_config,
    load_escalation_protocols,
    load_pictogram_map,
    load_haptic_patterns
)

# You can define a list of all exposed functions if you want to control 'from data_processing import *'
__all__ = [
    # aggregation
    "get_trend_data",
    # "get_overall_kpis", # REMOVED
    "get_chw_summary_kpis",
    "get_clinic_summary_kpis",
    "get_clinic_environmental_summary_kpis",
    "get_district_summary_kpis",
    # enrichment
    "enrich_zone_geodata_with_health_aggregates",
    # helpers
    "data_cleaner",
    "convert_to_numeric",
    "robust_json_load",
    "hash_dataframe_safe",
    "convert_date_columns",
    "standardize_missing_values",
    # loaders
    "load_health_records",
    "load_iot_clinic_environment_data",
    "load_zone_data",
    "load_json_config",
    "load_escalation_protocols",
    "load_pictogram_map",
    "load_haptic_patterns"
]
