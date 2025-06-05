# sentinel_project_root/pages/clinic_components/__init__.py
# This file makes the 'clinic_components' directory a Python package.

from .env_details import prepare_clinic_environmental_detail_data
from .epi_data import calculate_clinic_epidemiological_data
from .kpi_structuring import ( # Renamed module # This comment indicates the intent
    structure_main_clinic_kpis, 
    structure_disease_specific_clinic_kpis
)
from .patient_focus import prepare_clinic_patient_focus_overview_data
from .supply_forecast import prepare_clinic_supply_forecast_overview_data
from .testing_insights import prepare_clinic_lab_testing_insights_data

__all__ = [
    "prepare_clinic_environmental_detail_data",
    "calculate_clinic_epidemiological_data",
    "structure_main_clinic_kpis",
    "structure_disease_specific_clinic_kpis",
    "prepare_clinic_patient_focus_overview_data",
    "prepare_clinic_supply_forecast_overview_data",
    "prepare_clinic_lab_testing_insights_data"
]
