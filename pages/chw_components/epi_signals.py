# sentinel_project_root/pages/chw_components/epi_signals.py
# Extracts epidemiological signals from CHW daily data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric, standardize_missing_values
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in epi_signals.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

# --- Pre-compiled Regex for Performance ---
SYMPTOM_KEYWORDS_PATTERN = re.compile(
    r"\b(fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short\s+of\s+breath|fatigue|dizzy|nausea)\b",
    re.IGNORECASE
)
MALARIA_PATTERN = re.compile(r"\b(malaria|rdt-malaria)\b", re.IGNORECASE)
TB_PATTERN = re.compile(r"\b(tb|tuberculosis)\b", re.IGNORECASE)


def _prepare_epi_dataframe(df: pd.DataFrame, processing_date: date) -> pd.DataFrame:
    """Prepares the DataFrame for epi signal extraction."""
    log_prefix = "EpiSignalsPrep"
    
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    df_prepared = df.copy()

    # Define required columns and their sane defaults
    numeric_defaults = {
        'ai_risk_score': np.nan,
        'age': np.nan,
        'tb_contact_tracing_completed': 0, # Assuming 0 for not done
    }
    string_defaults = {
        'patient_id': f"UPID_Epi_{processing_date.isoformat()}",
        'condition': "UnknownCondition",
        'patient_reported_symptoms': "",
        'gender': "Unknown",
        'referral_reason': "",
        'referral_status': "Unknown",
        'test_type': 'N/A',
        'test_result': 'N/A',
    }
    
    # Use robust helper for standardization
    df_prepared = standardize_missing_values(df_prepared, string_defaults, numeric_defaults)

    if 'encounter_date' not in df_prepared.columns:
        df_prepared['encounter_date'] = pd.NaT
    df_prepared['encounter_date'] = pd.to_datetime(df_prepared['encounter_date'], errors='coerce')
    
    # Filter for the specific processing date
    df_filtered = df_prepared[df_prepared['encounter_date'].dt.date == processing_date].copy()
    if df_filtered.empty:
        logger.info(f"({log_prefix}) No encounters for {processing_date} after date filtering.")
    
    return df_filtered


def _calculate_demographics_high_risk(df_high_risk: pd.DataFrame) -> Dict[str, Any]:
    """Calculates age and gender distribution for a DataFrame of high-risk patients."""
    demographics = {
        "total_high_risk_patients": len(df_high_risk),
        "age_group_distribution": {},
        "gender_distribution": {}
    }
    if df_high_risk.empty:
        return demographics

    # Age Group Distribution
    if 'age' in df_high_risk.columns and df_high_risk['age'].notna().any():
        age_bins = [0, 5, 18, 60, 75, np.inf]
        age_labels = ['0-4', '5-17', '18-59', '60-74', '75+']
        demographics["age_group_distribution"] = pd.cut(
            df_high_risk['age'], bins=age_bins, labels=age_labels, right=False
        ).value_counts().sort_index().to_dict()

    # Gender Distribution
    if 'gender' in df_high_risk.columns:
        demographics["gender_distribution"] = df_high_risk['gender'].value_counts().to_dict()
        
    return demographics


def _detect_symptom_clusters(
    df_symptoms: pd.DataFrame, chw_zone_context: str, max_clusters: int
) -> List[Dict[str, Any]]:
    """Detects symptom clusters based on co-occurrence of keywords from settings."""
    symptom_clusters_config = getattr(settings, 'SYMPTOM_CLUSTERS_CONFIG', {})
    min_patients_for_cluster = getattr(settings, 'MIN_PATIENTS_FOR_SYMPTOM_CLUSTER', 2)

    if df_symptoms.empty or 'patient_reported_symptoms' not in df_symptoms or not symptom_clusters_config:
        return []

    symptoms_series = df_symptoms['patient_reported_symptoms'].str.lower()
    detected_clusters = []

    for cluster_name, keywords in symptom_clusters_config.items():
        if not keywords: continue
        
        # Create a boolean mask for each keyword and combine with logical AND
        combined_mask = pd.Series(True, index=symptoms_series.index)
        for keyword in keywords:
            keyword_regex = r'\b' + re.escape(keyword.lower().strip()) + r'\b'
            combined_mask &= symptoms_series.str.contains(keyword_regex, na=False, regex=True)
        
        if combined_mask.any():
            patient_count = df_symptoms.loc[combined_mask, 'patient_id'].nunique()
            if patient_count >= min_patients_for_cluster:
                detected_clusters.append({
                    "symptoms_pattern": cluster_name,
                    "patient_count": int(patient_count),
                    "location_hint": chw_zone_context
                })

    return sorted(detected_clusters, key=lambda x: x['patient_count'], reverse=True)[:max_clusters]


def extract_chw_epi_signals(
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context: str,
    chw_daily_encounter_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Extracts key epidemiological signals from a CHW's daily data.

    This enhanced function provides counts of key conditions, identifies potential
    symptom clusters, and offers demographic breakdowns of high-risk patients for
    improved situational awareness.
    """
    log_prefix = "CHWEpiSignalExtract"
    
    try:
        processing_date = pd.to_datetime(for_date).date()
    except (ValueError, TypeError):
        logger.warning(f"({log_prefix}) Invalid 'for_date' ('{for_date}'). Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date()

    logger.info(f"({log_prefix}) Extracting signals for date: {processing_date.isoformat()}, context: {chw_zone_context}")

    # Initialize output structure
    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": processing_date.isoformat(),
        "symptomatic_patients_key_conditions_count": 0,
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": {},
        "detected_symptom_clusters": []
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({log_prefix}) No daily encounter data provided.")
        return epi_signals_output

    df = _prepare_epi_dataframe(chw_daily_encounter_df, processing_date)
    if df.empty:
        return epi_signals_output

    # --- Vectorized Calculations for Performance ---
    
    # Count symptomatic patients for key conditions
    key_conditions_regex = '|'.join(getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', []))
    if key_conditions_regex:
        symptoms_present_mask = df['patient_reported_symptoms'].str.contains(SYMPTOM_KEYWORDS_PATTERN, na=False)
        key_condition_mask = df['condition'].str.contains(key_conditions_regex, case=False, na=False)
        epi_signals_output["symptomatic_patients_key_conditions_count"] = df.loc[symptoms_present_mask & key_condition_mask, 'patient_id'].nunique()

    # Count newly identified Malaria and TB cases (from condition or test results)
    malaria_cond_mask = df['condition'].str.contains(MALARIA_PATTERN, na=False)
    malaria_test_mask = (df['test_type'].str.contains(MALARIA_PATTERN, na=False)) & (df['test_result'].str.lower() == 'positive')
    epi_signals_output["newly_identified_malaria_patients_count"] = df.loc[malaria_cond_mask | malaria_test_mask, 'patient_id'].nunique()

    tb_cond_mask = df['condition'].str.contains(TB_PATTERN, na=False)
    tb_test_mask = (df['test_type'].str.contains(TB_PATTERN, na=False)) & (df['test_result'].str.lower() == 'positive')
    epi_signals_output["newly_identified_tb_patients_count"] = df.loc[tb_cond_mask | tb_test_mask, 'patient_id'].nunique()

    # Count pending TB contact tracing tasks
    tb_contact_mask = (
        df['referral_reason'].str.contains("contact trac", case=False, na=False) &
        (df['referral_status'].str.lower() == 'pending')
    )
    epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df.loc[tb_contact_mask, 'patient_id'].nunique()

    # Calculate demographics of high AI risk patients
    risk_threshold = getattr(settings, 'RISK_SCORE_HIGH_THRESHOLD', 75)
    high_risk_df = df.loc[df['ai_risk_score'] >= risk_threshold].drop_duplicates(subset=['patient_id'])
    epi_signals_output["demographics_of_high_ai_risk_patients_today"] = _calculate_demographics_high_risk(high_risk_df)
    
    # Detect symptom clusters
    symptoms_df = df.loc[df['patient_reported_symptoms'] != ''].copy()
    if not symptoms_df.empty:
        epi_signals_output["detected_symptom_clusters"] = _detect_symptom_clusters(
            symptoms_df, chw_zone_context, max_clusters=3
        )

    logger.info(f"({log_prefix}) CHW local epi signals extracted successfully.")
    return epi_signals_output
