# sentinel_project_root/data_processing/__init__.py
# SME PLATINUM STANDARD - ROBUST & EXPLICIT PACKAGE API (V5 - FINAL FIX)

"""
Initializes the data_processing package, defining its public API.
"""

from .helpers import DataPipeline, convert_to_numeric, robust_json_load, hash_dataframe
from .loaders import load_health_records, load_iot_records, load_zone_data, load_json_asset
from .enrichment import enrich_health_records_with_kpis, enrich_zone_data_with_aggregates
from .cached import get_cached_clinic_kpis, get_cached_environmental_kpis, get_cached_trend
from .logic import calculate_clinic_kpis, calculate_environmental_kpis, calculate_trend

__all__ = [
    "DataPipeline", "convert_to_numeric", "robust_json_load", "hash_dataframe",
    "load_health_records", "load_iot_records", "load_zone_data", "load_json_asset",
    "enrich_health_records_with_kpis", "enrich_zone_data_with_aggregates",
    "get_cached_clinic_kpis", "get_cached_environmental_kpis", "get_cached_trend",
    "calculate_clinic_kpis", "calculate_environmental_kpis", "calculate_trend",
]
