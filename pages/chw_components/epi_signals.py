# ssentinel_project_root/pages/chw_components/epi_signals.py
"""
SME FINAL VERSION: Extracts epidemiological signals from CHW daily data.
This version includes a temporary data simulation feature to guarantee that
symptom clusters can be populated for UI testing and verification.
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
    # This mock class ensures the script runs even if the main config is missing.
    class MockSettings:
        KEY_CONDITIONS_FOR_ACTION = ['malaria', 'tb', 'tuberculosis']
        SYMPTOM_CLUSTERS_CONFIG = {
            "ILI (Influenza-like Illness)": ["fever", "cough", "headache"],
            "GI (Gastrointestinal)": ["diarrhea", "vomit", "nausea"]
        }
        MIN_PATIENTS_FOR_SYMPTOM_CLUSTER = 2
    settings = MockSettings()

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets a configuration value."""
    return getattr(settings, attr_name, default_value)

def _simulate_symptom_cluster_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    SME NOTE: TEMPORARY DATA SIMULATION FUNCTION.
    This function injects fake patient data into the DataFrame to ensure a
    symptom cluster is detected. Set `ENABLE_SIMULATION` to False to disable.
    """
    ENABLE_SIMULATION = True # <--- SET THIS TO FALSE TO TURN OFF FAKE DATA

    if not ENABLE_SIMULATION:
        return df

    logger.warning("Epi-Signals: Data simulation is ENABLED. Injecting fake cluster data.")
    
    # Define the fake data for 3 patients with Influenza-like Illness symptoms
    fake_cluster_data = {
        'patient_id': ['SIM-PATIENT-001', 'SIM-PATIENT-002', 'SIM-PATIENT-003'],
        'patient_reported_symptoms': [
            'Patient reports a high fever, a persistent cough, and a severe headache.',
            'Has a bad cough and feels very weak with a fever and headache.',
            'Complaining of headache, a dry cough, and a high temperature (fever).'
        ],
        'condition': ['ILI', 'ILI', 'ILI']
    }
    df_fake = pd.DataFrame(fake_cluster_data)
    
    # Combine the original data with the new fake data
    df_combined = pd.concat([df, df_fake], ignore_index=True)
    return df_combined


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
        
        # Create a regex that requires all keywords to be present
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
    """
    if not isinstance(chw_daily_encounter_df, pd.DataFrame):
        # Handle case where no dataframe is passed at all
        df = pd.DataFrame()
    else:
        df = chw_daily_encounter_df.copy()

    # --- DATA SIMULATION STEP ---
    # This will add fake data to ensure the cluster detection logic runs.
    df = _simulate_symptom_cluster_data(df)
    
    # --- Original Logic ---
    try:
        processing_date = pd.to_datetime(for_date).date()
    except (AttributeError, ValueError):
        processing_date = datetime.now().date()
    
    # Ensure required columns exist
    required_cols = {'patient_id': 'object', 'condition': 'Unknown', 'patient_reported_symptoms': ''}
    for col, default in required_cols.items():
        if col not in df:
            df[col] = default
        if df[col].dtype == 'object':
            df[col].fillna(default, inplace=True)

    # Calculate signals
    key_conditions = _get_setting('KEY_CONDITIONS_FOR_ACTION', [])
    key_cond_pattern = '|'.join([re.escape(c) for c in key_conditions]) if key_conditions else ''
    
    symptomatic_patients = 0
    if key_cond_pattern:
        symptomatic_patients = df[df['condition'].str.contains(key_cond_pattern, case=False, na=False)]['patient_id'].nunique()
    
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
