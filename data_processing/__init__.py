# sentinel_project_root/data_processing/__init__.py
"""
Initializes the data_processing package, defining its public API.

This package provides a suite of robust, high-performance tools for loading,
cleaning, enriching, and aggregating health data for the Sentinel application.
The primary entry points are the loader functions and the `DataPipeline` class.
"""

# --- Core Data Pipeline & Utilities from helpers.py ---
from .helpers import (
    DataPipeline,
    convert_to_numeric,
    robust_json_load,
    hash_dataframe
)

# --- Data Loading from loaders.py ---
from .loaders import (
    load_health_records,
    load_iot_records,
    load_zone_data,
    load_json_asset
)

# --- Data Enrichment from enrichment.py ---
from .enrichment import (
    enrich_health_records_with_kpis,
    enrich_zone_data_with_aggregates
)

# --- Data Aggregation from aggregation.py ---
from .aggregation import (
    calculate_clinic_kpis,
    calculate_environmental_kpis,
    calculate_district_kpis,
    calculate_trend
)

# --- Define the canonical public API for the package ---
__all__ = [
    # helpers.py
    "DataPipeline",
    "convert_to_numeric",
    "robust_json_load",
    "hash_dataframe",

    # loaders.py
    "load_health_records",
    "load_iot_records",
    "load_zone_data",
    "load_json_asset",

    # enrichment.py
    "enrich_health_records_with_kpis",
    "enrich_zone_data_with_aggregates",

    # aggregation.py
    "calculate_clinic_kpis",
    "calculate_environmental_kpis",
    "calculate_district_kpis",
    "calculate_trend",
]
