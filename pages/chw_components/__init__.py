# ssentinel_project_root/pages/chw_components/__init__.py
"""
This file makes the 'chw_components' directory a Python package and defines
its public API.
"""
from .activity_trends import get_chw_activity_trends
from .alert_generation import generate_chw_alerts
from .epi_signals import extract_chw_epi_signals
from .summary_metrics import calculate_chw_daily_summary_metrics
from .task_processing import generate_chw_tasks

__all__ = [
    "get_chw_activity_trends",
    "generate_chw_alerts",
    "extract_chw_epi_signals",
    "calculate_chw_daily_summary_metrics",
    "generate_chw_tasks"
]
