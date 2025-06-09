# sentinel_project_root/data_processing/__init__.py
# SME PLATINUM STANDARD (V4 - MODERN API ALIGNMENT)
# This version updates the package's public API to reflect the architectural
# improvements in its submodules, primarily replacing the old `data_cleaner`
# with the new fluent `DataPipeline` class.

"""
Initializes the data_processing package, making key functions and classes
available at the top level for easier, cleaner imports in other modules.

This __init__.py defines the public API for the package.
"""

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

# <<< SME REVISION >>> Updated imports to reflect the new helpers.py (V4) architecture.
# From helpers.py - Expose the fluent pipeline class and key standalone utilities.
from .helpers import (
    DataPipeline,         # The primary tool for sequential data cleaning.
    convert_to_numeric,
    robust_json_load,
    hash_dataframe_safe
    # REMOVED: data_cleaner (replaced by DataPipeline).
    # REMOVED: convert_date_columns (now a method of DataPipeline).
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

# --- Define the public API for the data_processing package ---
# This list controls what is imported when a user does `from data_processing import *`
# and is considered the canonical list of public-facing components.
__all__ = [
    # aggregation
    "get_trend_data",
    "get_clinic_summary_kpis",
    "get_clinic_environmental_summary_kpis",
    "get_chw_summary_kpis",
    "get_district_summary_kpis",

    # enrichment
    "enrich_zone_geodata_with_health_aggregates",

    # <<< SME REVISION >>> Updated the helpers API.
    # helpers
    "DataPipeline",         # Expose the class for users to instantiate.
    "convert_to_numeric",
    "robust_json_load",
    "hash_dataframe_safe",

    # loaders
    "load_health_records",
    "load_iot_clinic_environment_data",
    "load_zone_data",
    "load_json_config",
    "load_escalation_protocols",
    "load_pictogram_map",
    "load_haptic_patterns"
]
