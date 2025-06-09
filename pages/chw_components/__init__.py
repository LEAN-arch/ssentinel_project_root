# ssentinel_project_root/pages/chw_components/__init__.py
"""
This file makes the 'chw_components' directory a Python package.

It also defines the public API of this package by specifying which functions
are intended to be imported by other parts of the application.

SME NOTE: This file has been corrected to import the refactored and correctly
named functions, resolving the critical ImportError that was preventing the
application from starting.
"""

# Import the refactored, correct function from activity_trends.py
from .activity_trends import get_chw_activity_trends

# Import the deprecated wrapper, which is the correct public-facing function for now.
from .alert_generation import generate_chw_alerts

# Import the other components, assuming their function names are correct.
from .epi_signals import extract_chw_epi_signals
from .summary_metrics import calculate_chw_daily_summary_metrics
from .task_processing import generate_chw_tasks

# The __all__ variable defines the public API of the package.
# When someone writes `from pages.chw_components import *`, only these names will be imported.
__all__ = [
    "get_chw_activity_trends",  # Use the new, correct function name
    "generate_chw_alerts",
    "extract_chw_epi_signals",
    "calculate_chw_daily_summary_metrics",
    "generate_chw_tasks"
]
