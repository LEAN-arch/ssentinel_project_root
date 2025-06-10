# sentinel_project_root/data_processing/__init__.py
# SME PLATINUM STANDARD - ROBUST & EXPLICIT PACKAGE API (V4 - FINAL FIX)

"""
Initializes the data_processing package, defining its public API.

This file explicitly exports all public-facing functions from its submodules,
providing a single, consistent import point for the rest of the application.
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

# --- Cached Aggregation Functions from cached.py ---
from .cached import (
    get_cached_clinic_kpis,
    get_cached_environmental_kpis,
    get_cached_trend
)

# --- Pure, Non-cached Logic from logic.py (for backend use) ---
from .logic import (
    calculate_clinic_kpis,
    calculate_environmental_kpis,
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

    # cached.py (for UI)
    "get_cached_clinic_kpis",
    "get_cached_environmental_kpis",
    "get_cached_trend",

    # logic.py (for backend)
    "calculate_clinic_kpis",
    "calculate_environmental_kpis",
    "calculate_trend",
]
