# sentinel_project_root/data_processing/__init__.py
"""
Initializes the data_processing package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.

This file defines the public API of the data_processing package.
"""

# --- Import functions from submodules to expose them publicly ---

# From aggregation.py
from .aggregation import (
    get_trend_data,
    get_clinic_summary_kpis,
    get_clinic_environmental_summary_kpis,
    get_chw_summary_kpis,
    get_district_summary_kpis
)

# From enrichment.py
from .enrichment import (
    enrich_zone_geodata_with_health_aggregates
)

# From helpers.py - Expose the singleton cleaner and key utilities
from .helpers import (
    data_cleaner,
    convert_to_numeric,
    robust_json_load,
    hash_dataframe_safe,
    convert_date_columns
)

# From loaders.py
from .loaders import (
    load_health_records,
    load_iot_clinic_environment_data,
    load_zone_data,
    load_json_config,
    load_escalation_protocols,
    load_pictogram_map,
    load_haptic_patterns
)

# --- Define __all__ for explicit public API definition ---
# This tells tools and developers which names are part of the public API
# and controls 'from data_processing import *' behavior.
__all__ = [
    # aggregation
    "get_trend_data",
    "get_clinic_summary_kpis",
    "get_clinic_environmental_summary_kpis",
    "get_chw_summary_kpis",
    "get_district_summary_kpis",
    
    # enrichment
    "enrich_zone_geodata_with_health_aggregates",
    
    # helpers
    "data_cleaner",
    "convert_to_numeric",
    "robust_json_load",
    "hash_dataframe_safe",
    "convert_date_columns",
    
    # loaders
    "load_health_records",
    "load_iot_clinic_environment_data",
    "load_zone_data",
    "load_json_config",
    "load_escalation_protocols",
    "load_pictogram_map",
    "load_haptic_patterns"
]
