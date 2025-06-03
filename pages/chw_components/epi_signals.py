# sentinel_project_root/pages/chw_components/epi_signals.py
# Extracts epidemiological signals from CHW daily data for Sentinel Health Co-Pilot.
# Renamed from epi_signal_extractor.py

import pandas as pd
import numpy as np
import logging
import re # For regex based symptom keyword matching
from typing import Dict, Any, Optional, List
from datetime import date # For type hinting and date operations

from config import settings # Use new settings module
from data_processing.helpers import convert_to_numeric # Local import

logger = logging.getLogger(__name__)

def extract_chw_epi_signals( # Renamed function
    chw_daily_encounter_df: Optional[pd.DataFrame],
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None, # E.g., can pass worker fatigue index
    for_date: Any, # Expects a date or date-like string for context
    chw_zone_context: str, # Zone context for these signals (for logging/display)
    max_symptom_clusters_to_report: int = 3
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.
    This includes symptomatic patient counts, specific disease identifications,
    pending tasks (like TB contact tracing), high-risk demographics, and symptom clusters.
    """
    module_log_prefix = "CHWEpiSignalExtract" # Renamed for clarity

    # Standardize for_date
    try:
        processing_date = pd.to_datetime(for_date).date() if for_date else pd.Timestamp('now').date()
    except Exception:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}). Defaulting to current date.")
        processing_date = pd.Timestamp('now').date()
    processing_date_str = processing_date.isoformat()

    logger.info(f"({module_log_prefix}) Extracting CHW local epi signals for date: {processing_date_str}, context: {chw_zone_context}")

    # Initialize output structure with defaults
    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": processing_date_str,
        "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0,
        "symptom_keywords_for_monitoring": "", # Will be populated by regex pattern
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": { # Nested dict for clarity
            "total_high_risk_patients_count": 0,
            "age_group_distribution": {}, # E.g., {'0-4 yrs': 2, '50+ yrs': 1}
            "gender_distribution": {}    # E.g., {'Male': 1, 'Female': 2}
        },
        "detected_symptom_clusters": [] # List of dicts: {symptoms_pattern, patient_count, location_hint}
    }

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data provided for {processing_date_str}. Some signals might be zero or based on pre_calculated_kpis.")
        # Populate from pre_calculated_chw_kpis if available, especially for task counts
        if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict):
            # Example: if TB contact tracing count is passed directly
            epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
                convert_to_numeric(pd.Series([pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 0)]), default_value=0).iloc[0]
            )
        return epi_signals_output

    df_epi_src = chw_daily_encounter_df.copy()
    
    # Ensure 'encounter_date' is datetime and filter for the target_date
    if 'encounter_date' in df_epi_src.columns:
        df_epi_src['encounter_date'] = pd.to_datetime(df_epi_src['encounter_date'], errors='coerce')
        df_epi_src = df_epi_src[df_epi_src['encounter_date'].dt.date == processing_date]
    else:
        logger.warning(f"({module_log_prefix}) 'encounter_date' column missing. Epi signals may be inaccurate for {processing_date_str}.")
        return epi_signals_output # Cannot proceed without encounter_date for daily filtering

    if df_epi_src.empty:
        logger.info(f"({module_log_prefix}) No CHW data for date {processing_date_str} to extract epi signals from.")
        # Still try to populate TB tasks from pre_calculated_kpis if available
        if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict):
            epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
                convert_to_numeric(pd.Series([pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count', 0)]), default_value=0).iloc[0]
            )
        return epi_signals_output
        
    # Define essential columns and their safe defaults for this component's logic
    essential_cols_config = {
        'patient_id': {"default": f"UnknownPID_EpiSgnl_{processing_date_str}", "type": str},
        'condition': {"default": "UnknownCondition", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str}, # Default to empty string
        'ai_risk_score': {"default": np.nan, "type": float},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "", "type": str}, # For TB contact tracing
        'referral_status': {"default": "Unknown", "type": str} # For TB contact tracing
    }
    common_na_values_epi = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL', 'unknown']

    for col_name, config in essential_cols_config.items():
        if col_name not in df_epi_src.columns:
            df_epi_src[col_name] = config["default"]
        
        if config["type"] == float: # Includes np.nan default
            df_epi_src[col_name] = convert_to_numeric(df_epi_src[col_name], default_value=config["default"])
        elif config["type"] == str:
            df_epi_src[col_name] = df_epi_src[col_name].astype(str).fillna(str(config["default"]))
            df_epi_src[col_name] = df_epi_src[col_name].replace(common_na_values_epi, str(config["default"]), regex=False).str.strip()

    # 1. Symptomatic Patients with Key Conditions
    # Define key conditions that are typically symptomatic for this count
    key_symptomatic_conditions_list = list(set(settings.KEY_CONDITIONS_FOR_ACTION) & 
                                           {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", 
                                            "Diarrheal Diseases (Severe)", "Heat Stroke"}) # Intersect with known symptomatic ones
    
    # General regex for common symptom keywords
    # This can be expanded or made configurable
    general_symptom_keywords_pattern = re.compile(
        r"\b(fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short\s+of\s+breath|fatigue|dizzy|nausea)\b",
        re.IGNORECASE
    )
    # Store the keywords used for monitoring for display/info purposes
    epi_signals_output["symptom_keywords_for_monitoring"] = general_symptom_keywords_pattern.pattern.replace(r"\b", "").replace(r"\s+", " ").replace("|", ", ")


    if 'patient_reported_symptoms' in df_epi_src.columns and 'condition' in df_epi_src.columns:
        symptoms_present_mask = df_epi_src['patient_reported_symptoms'].astype(str).str.contains(general_symptom_keywords_pattern, na=False)
        
        key_condition_present_mask = df_epi_src['condition'].astype(str).apply(
            lambda x_cond_str: any(key_c.lower() in x_cond_str.lower() for key_c in key_symptomatic_conditions_list)
        )
        
        symptomatic_key_condition_df = df_epi_src[symptoms_present_mask & key_condition_present_mask]
        if 'patient_id' in symptomatic_key_condition_df.columns:
            epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_key_condition_df['patient_id'].nunique()

    # 2. Specific Disease Counts (Newly Identified Today)
    if 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        condition_series_lower = df_epi_src['condition'].astype(str).str.lower()
        epi_signals_output["newly_identified_malaria_patients_count"] = df_epi_src[
            condition_series_lower.str.contains("malaria", na=False)
        ]['patient_id'].nunique()
        
        epi_signals_output["newly_identified_tb_patients_count"] = df_epi_src[
            condition_series_lower.str.contains(r"\btb\b|tuberculosis", na=False, regex=True) # More specific TB match
        ]['patient_id'].nunique()

    # 3. Pending TB Contact Tracing Tasks
    # Prefer pre_calculated_kpis if available and contains this, else calculate from df_epi_src
    if pre_calculated_chw_kpis and isinstance(pre_calculated_chw_kpis, dict) and \
       'pending_tb_contact_tracing_tasks_count' in pre_calculated_chw_kpis and \
       pd.notna(pre_calculated_chw_kpis['pending_tb_contact_tracing_tasks_count']):
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(
            convert_to_numeric(pd.Series([pre_calculated_chw_kpis['pending_tb_contact_tracing_tasks_count']]), default_value=0).iloc[0]
        )
    elif all(col_tb in df_epi_src.columns for col_tb in ['condition', 'referral_status', 'referral_reason', 'patient_id']):
        # Calculate if not provided: CHW is expected to log a referral task for contact tracing
        tb_contact_tracing_mask = (
            df_epi_src['condition'].astype(str).str.contains(r"\btb\b|tuberculosis", case=False, na=False, regex=True) &
            df_epi_src['referral_reason'].astype(str).str.contains("contact trac", case=False, na=False) & # Flexible match
            (df_epi_src['referral_status'].astype(str).str.lower() == 'pending')
        )
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_epi_src[tb_contact_tracing_mask]['patient_id'].nunique()

    # 4. Demographics of High AI Risk Patients Encountered Today
    if 'ai_risk_score' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        high_risk_patients_df = df_epi_src[
            convert_to_numeric(df_epi_src['ai_risk_score'], default_value=0.0) >= settings.RISK_SCORE_HIGH_THRESHOLD
        ].drop_duplicates(subset=['patient_id']) # Count unique patients
        
        if not high_risk_patients_df.empty:
            demo_stats_dict = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats_dict["total_high_risk_patients_count"] = len(high_risk_patients_df)
            
            # Age group distribution
            if 'age' in high_risk_patients_df.columns and high_risk_patients_df['age'].notna().any():
                age_bins = [0, settings.AGE_THRESHOLD_LOW, settings.AGE_THRESHOLD_MODERATE, settings.AGE_THRESHOLD_HIGH, settings.AGE_THRESHOLD_VERY_HIGH, np.inf]
                age_labels = [f'0-{settings.AGE_THRESHOLD_LOW-1}', f'{settings.AGE_THRESHOLD_LOW}-{settings.AGE_THRESHOLD_MODERATE-1}', 
                              f'{settings.AGE_THRESHOLD_MODERATE}-{settings.AGE_THRESHOLD_HIGH-1}', f'{settings.AGE_THRESHOLD_HIGH}-{settings.AGE_THRESHOLD_VERY_HIGH-1}', 
                              f'{settings.AGE_THRESHOLD_VERY_HIGH}+']
                
                high_risk_patients_df['age_group_demo'] = pd.cut(
                    convert_to_numeric(high_risk_patients_df['age']),
                    bins=age_bins, labels=age_labels, right=False, include_lowest=True
                )
                demo_stats_dict["age_group_distribution"] = high_risk_patients_df['age_group_demo'].value_counts().sort_index().to_dict()
            
            # Gender distribution
            if 'gender' in high_risk_patients_df.columns and high_risk_patients_df['gender'].notna().any():
                # Simple normalization for gender
                gender_map_func = lambda g_str: "Male" if str(g_str).strip().lower() in ['m', 'male'] else \
                                              "Female" if str(g_str).strip().lower() in ['f', 'female'] else "Other/Unknown"
                high_risk_patients_df['gender_normalized_demo'] = high_risk_patients_df['gender'].apply(gender_map_func)
                gender_counts_map = high_risk_patients_df[
                    high_risk_patients_df['gender_normalized_demo'].isin(["Male", "Female"]) # Focus on Male/Female for this summary
                ]['gender_normalized_demo'].value_counts().to_dict()
                demo_stats_dict["gender_distribution"] = gender_counts_map

    # 5. Symptom Cluster Detection (Basic example, can be enhanced with NLP or more complex rules)
    if 'patient_reported_symptoms' in df_epi_src.columns and \
       df_epi_src['patient_reported_symptoms'].str.strip().astype(bool).any(): # Check if any non-empty strings
        
        symptoms_series_lower = df_epi_src['patient_reported_symptoms'].astype(str).str.lower()
        
        # Symptom patterns from config (e.g., {"Cluster Name": ["keyword1", "keyword2"], ...})
        configured_symptom_patterns = settings.SYMPTOM_CLUSTERS_CONFIG
        min_patients_for_cluster_alert = 2 # Configurable: min patients to trigger a cluster alert

        detected_clusters_list = []
        for cluster_name, keywords_list in configured_symptom_patterns.items():
            if not keywords_list: continue # Skip if no keywords for this cluster name
            
            # Create a mask where all keywords for this cluster are present in symptoms
            # This uses a sequence of .str.contains() calls
            current_cluster_mask = pd.Series([True] * len(symptoms_series_lower), index=symptoms_series_lower.index)
            for keyword_val in keywords_list:
                current_cluster_mask &= symptoms_series_lower.str.contains(keyword_val.lower(), na=False)
            
            if current_cluster_mask.any(): # If any rows match all keywords for this cluster
                patients_in_cluster_count = df_epi_src[current_cluster_mask]['patient_id'].nunique()
                if patients_in_cluster_count >= min_patients_for_cluster_alert:
                    detected_clusters_list.append({
                        "symptoms_pattern": cluster_name, # User-friendly name of the cluster
                        "patient_count": int(patients_in_cluster_count),
                        "location_hint": chw_zone_context # The zone where CHW is operating
                    })
        
        # Sort clusters by patient count (descending) and take top N
        if detected_clusters_list:
            epi_signals_output["detected_symptom_clusters"] = sorted(
                detected_clusters_list, key=lambda x_cluster: x_cluster['patient_count'], reverse=True
            )[:max_symptom_clusters_to_report]
            # Could trigger escalation protocol for significant clusters
            # if epi_signals_output["detected_symptom_clusters"] and epi_signals_output["detected_symptom_clusters"][0]['patient_count'] >= 3: # Example
            #     execute_escalation_protocol("SUSPECTED_COMMUNITY_OUTBREAK_SYMPTOM_CLUSTER", 
            #                                 {"CLUSTER_DETAILS": epi_signals_output["detected_symptom_clusters"][0], "ZONE_ID": chw_zone_context})


    num_clusters_found = len(epi_signals_output["detected_symptom_clusters"])
    logger.info(f"({module_log_prefix}) CHW local epi signals extracted for {processing_date_str}. Symptom clusters found: {num_clusters_found}")
    return epi_signals_output
