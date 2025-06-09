# ssentinel_project_root/pages/chw_components/epi_signals.py
"""
SME FINAL VERSION: This component extracts epidemiological signals from CHW daily
data. The function signature has been corrected to align with its usage in the
dashboard, resolving the TypeError.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

logger = logging.getLogger(__name__)

# --- Safe Setting Import ---
try:
    from config import settings
except ImportError:
    class MockSettings:
        # Define defaults for any settings used
        KEY_CONDITIONS_FOR_ACTION = ['malaria', 'tb', 'tuberculosis']
        SYMPTOM_CLUSTERS_CONFIG = {
            "ILI (Influenza-like Illness)": ["fever", "cough", "ache"],
            "GI (Gastrointestinal)": ["diarrhea", "vomit", "nausea"]
        }
        MIN_PATIENTS_FOR_SYMPTOM_CLUSTER = 2
    settings = MockSettings()

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets a configuration value."""
    return getattr(settings, attr_name, default_value)

def _detect_symptom_clusters(
    df_symptoms: pd.DataFrame,
    chw_zone_context: str,
    max_clusters_to_report: int
) -> List[Dict[str, Any]]:
    """Detects symptom clusters based on configuration."""
    if df_symptoms.empty or 'patient_reported_symptoms' not in df_symptoms:
        return []
        
    symptoms_lower = df_symptoms['patient_reported_symptoms'].str.lower().dropna()
    if symptoms_lower.empty:
        return []

    symptom_clusters_config = _get_setting('SYMPTOM_CLUSTERS_CONFIG', {})
    min_patients_for_cluster = _get_setting('MIN_PATIENTS_FOR_SYMPTOM_CLUSTER', 2)
    detected_clusters = []

    for cluster_name, keywords in symptom_clusters_config.items():
        if not keywords: continue
        
        # Create a regex to match all keywords for the cluster
        pattern = r'' + ''.join([f'(?=.*\\b{re.escape(kw.lower())}\\b)' for kw in keywords])
        mask = symptoms_lower.str.contains(pattern, na=False)
        
        if mask.any():
            patient_count = df_symptoms.loc[mask, 'patient_id'].nunique()
            if patient_count >= min_patients_for_cluster:
                detected_clusters.append({
                    "symptoms_pattern": cluster_name, 
                    "patient_count": int(patient_count), 
                    "location_hint": chw_zone_context
                })

    return sorted(detected_clusters, key=lambda x: x['patient_count'], reverse=True)[:max_clusters_to_report]

def extract_chw_epi_signals(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    for_date: Union[str, date_type, datetime],
    chw_zone_context: Optional[str] = "All Zones",
    **kwargs # Accept extra kwargs to prevent errors
) -> Dict[str, Any]:
    """
    Public factory function to extract epidemiological signals.
    
    SME NOTE: The function signature is now corrected to accept `chw_zone_context`
    and optional `**kwargs`, resolving the TypeError from the dashboard. The logic has
    been streamlined and made more robust.
    """
    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        return {}

    try:
        processing_date = pd.to_datetime(for_date).date()
    except (AttributeError, ValueError):
        logger.warning(f"Invalid 'for_date' in extract_chw_epi_signals. Defaulting to today.")
        processing_date = datetime.now().date()
    
    df = chw_daily_encounter_df.copy()

    # Ensure required columns exist
    required_cols = {
        'patient_id': 'object',
        'condition': 'Unknown',
        'patient_reported_symptoms': ''
    }
    for col, default in required_cols.items():
        if col not in df:
            df[col] = default
        df[col].fillna(default, inplace=True)
    
    # Calculate signals
    key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    key_cond_pattern = '|'.join([re.escape(c) for c in key_conditions])
    
    symptomatic_patients = 0
    if key_cond_pattern:
        symptomatic_patients = df[
            df['condition'].str.contains(key_cond_pattern, case=False, na=False)
        ]['patient_id'].nunique()
    
    malaria_cases = df[df['condition'].str.contains('malaria', case=False, na=False)]['patient_id'].nunique()
    
    # Detect clusters
    symptom_clusters = _detect_symptom_clusters(
        df_symptoms=df,
        chw_zone_context=chw_zone_context or "All Zones",
        max_clusters_to_report=3
    )
    
    return {
        "date_of_activity": processing_date.isoformat(),
        "symptomatic_patients_key_conditions_count": int(symptomatic_patients),
        "newly_identified_malaria_patients_count": int(malaria_cases),
        "detected_symptom_clusters": symptom_clusters,
    }
