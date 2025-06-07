# ssentinel_project_root/pages/chw_components/epi_signals.py
"""
[PARTIALLY DEPRECATED] This module contains specific epidemiological signal extraction logic.
While functional, its logic would ideally be moved into a more centralized analytics module
in a future refactoring. It has been cleaned and made robust for current use.
"""
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

logger = logging.getLogger(__name__)

try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in epi_signals.py: {e}", exc_info=True)
    # Define a dummy function to allow module to be imported
    def convert_to_numeric(series, default_value): return series
    class settings: SYMPTOM_CLUSTERS_CONFIG = {}

def _get_setting(attr_name: str, default_value: Any) -> Any:
    return getattr(settings, attr_name, default_value)

# --- Private Helper Functions ---

def _detect_symptom_clusters(
    df_symptoms: pd.DataFrame,
    chw_zone_context: str
) -> List[Dict[str, Any]]:
    """Detects symptom clusters based on configuration in settings.py."""
    symptom_clusters_config = _get_setting("SYMPTOM_CLUSTERS_CONFIG", {})
    min_patients_for_cluster = _get_setting("MIN_PATIENTS_FOR_SYMPTOM_CLUSTER", 2)
    max_clusters_to_report = 3

    if df_symptoms.empty or 'patient_reported_symptoms' not in df_symptoms or 'patient_id' not in df_symptoms:
        return []
    
    symptoms_lower = df_symptoms['patient_reported_symptoms'].astype(str).str.lower()
    detected_clusters = []

    for cluster_name, keywords in symptom_clusters_config.items():
        if not isinstance(keywords, list) or not keywords:
            continue
        
        # Build a regex pattern for the cluster's keywords
        pattern = r'(' + '|'.join([r'\b' + re.escape(kw.lower()) + r'\b' for kw in keywords]) + r')'
        matches = symptoms_lower.str.contains(pattern, na=False, regex=True)
        
        if matches.any():
            patient_count = df_symptoms.loc[matches, 'patient_id'].nunique()
            if patient_count >= min_patients_for_cluster:
                detected_clusters.append({
                    "symptoms_pattern": cluster_name,
                    "patient_count": int(patient_count),
                    "location_hint": chw_zone_context
                })

    return sorted(detected_clusters, key=lambda x: x['patient_count'], reverse=True)[:max_clusters_to_report]

# --- Public API Function (Preserved for Compatibility) ---

def extract_chw_epi_signals(
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context: str,
    chw_daily_encounter_df: Optional[pd.DataFrame] = None,
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.
    """
    try:
        processing_date = pd.to_datetime(for_date, errors='coerce').date()
    except:
        processing_date = pd.Timestamp('now').date()

    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": processing_date.isoformat(),
        "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0,
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": {},
        "detected_symptom_clusters": []
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"EpiSignals: No daily encounter data for {processing_date}. Some signals may be zero.")
        return epi_signals_output

    df = chw_daily_encounter_df.copy()

    # Filter for the specific date
    if 'encounter_date' in df.columns:
        df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
        df = df[df['encounter_date'].dt.date == processing_date]

    if df.empty:
        return epi_signals_output

    # Calculate signals
    key_conditions = _get_setting("KEY_CONDITIONS_FOR_ACTION", [])
    if 'condition' in df.columns and key_conditions:
        key_condition_pattern = '|'.join([re.escape(c) for c in key_conditions])
        epi_signals_output["symptomatic_patients_key_conditions_count"] = df[df['condition'].str.contains(key_condition_pattern, case=False, na=False)]['patient_id'].nunique()

    if 'condition' in df.columns:
        epi_signals_output["newly_identified_malaria_patients_count"] = df[df['condition'].str.contains("malaria", case=False, na=False)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df[df['condition'].str.contains("tb|tuberculosis", case=False, na=False)]['patient_id'].nunique()

    # Detect symptom clusters
    if 'patient_reported_symptoms' in df.columns:
        epi_signals_output["detected_symptom_clusters"] = _detect_symptom_clusters(df, chw_zone_context)

    # Note: Demographics and TB tracing would require more complex logic and data joins,
    # which is better suited for a dedicated analytics module. This implementation
    # focuses on preserving the core functionality of the original file.

    return epi_signals_output
