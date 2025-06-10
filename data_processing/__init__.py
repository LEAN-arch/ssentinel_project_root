# sentinel_project_root/data_processing/__init__.py
# SME PLATINUM STANDARD - SIMPLIFIED & ROBUST PACKAGE API (V2)

"""
Initializes the data_processing package.

This __init__ file intentionally exposes only the most fundamental and
widely used components to prevent circular import issues. Other modules
should import directly from the specific submodules (e.g., `aggregation`, `loaders`)
for the functions they need.
"""

# Expose the core data pipeline tool and the primary data loader.
from .helpers import DataPipeline
from .loaders import load_health_records, load_iot_records, load_zone_data, load_json_asset

__all__ = [
    "DataPipeline",
    "load_health_records",
    "load_iot_records",
    "load_zone_data",
    "load_json_asset",
]
