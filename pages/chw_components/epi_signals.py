# sentinel_project_root/pages/chw_components/epi_signals.py
# Extracts epidemiological signals from CHW daily data for Sentinel Health Co-Pilot.

import pandas as pd
import numpy as np
import logging
import re # For regex based symptom keyword matching
from typing import Dict, Any, Optional, List
from datetime import date as date_type # For type hinting and date operations

from config import settings
from data_processing.helpers import convert_to_numeric

logger = logging.getLogger(__name__)

def extract_chw_epi_signals(
    chw_daily_encounter_df: Optional[pd.DataFrame],
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None,
    for_date: Any, 
    chw_zone_context: str,
    max_symptom_clusters_to_report: int = 3
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.
    """
    module_log_prefix = "CHWEpiSignalExtract"

    try:
        processing_date = pd.to_datetime(for_date, errors='coerce').date()
        if pd.isna(processing_date): raise ValueError("Invalid for_date for epi signals.")
    except Exception as e_date_parse:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ({for_date}): {e_date_parse}. Defaulting to current system date.")
        processing_date = pd.Timestamp('now').date()
    processing_date_str = processing_date.isoformat()

    logger.info(f"({module_log_prefix}) Extracting CHW local epi signals for date: {processing_date_str}, context: {chw_zone_context}")

    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": processing_date_str, "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0, "symptom_keywords_for_monitoring": "",
        "newly_identified_malaria_patients_count": 0, "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": {
            "total_high_risk_patients_count": 0, "age_group_distribution": {}, "gender_distribution": {}
        },
        "detected_symptom_clusters": []
    }

    # Populate TB tasks from pre_calculated_kpis if available and valid
    if isinstance(pre_calculated_chw_kpis, dict):
        pending_tb_tasks_precalc = pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count')
        if pd.notna(pending_tb_tasks_precalc):
            try:
                epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(pd.to_numeric(pending_tb_tasks_precalc, errors='raise'))
            except (ValueError, TypeError):
                logger.warning(f"({module_log_prefix}) Could not convert pre-calculated 'pending_tb_contact_tracing_tasks_count' to int.")


    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data for {processing_date_str}. Signals based on pre_calculated_kpis or defaults.")
        return epi_signals_output

    df_epi_src = chw_daily_encounter_df.copy()
    
    if 'encounter_date' in df_epi_src.columns:
        df_epi_src['encounter_date'] = pd.to_datetime(df_epi_src['encounter_date'], errors='coerce')
        df_epi_src = df_epi_src[df_epi_src['encounter_date'].dt.date == processing_date] # Filter to processing_date
    else:
        logger.warning(f"({module_log_prefix}) 'encounter_date' missing. Epi signals for {processing_date_str} may be inaccurate."); return epi_signals_output
    if df_epi_src.empty:
        logger.info(f"({module_log_prefix}) No CHW data for {processing_date_str} after date filtering."); return epi_signals_output
        
    essential_cols_config_epi = {
        'patient_id': {"default": f"UPID_EpiSgnl_{processing_date_str}", "type": str},
        'condition': {"default": "UnknownCondition", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str},
        'ai_risk_score': {"default": np.nan, "type": float},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "", "type": str},
        'referral_status': {"default": "Unknown", "type": str}
    }
    common_na_epi_sig = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    na_regex_epi = r'^(?:' + '|'.join(re.escape(s) for s in common_na_epi_sig if s) + r')$'


    for col_name, config in essential_cols_config_epi.items():
        if col_name not in df_epi_src.columns: df_epi_src[col_name] = config["default"]
        if config["type"] == float: df_epi_src[col_name] = convert_to_numeric(df_epi_src[col_name], default_value=config["default"])
        elif config["type"] == str:
            df_epi_src[col_name] = df_epi_src[col_name].astype(str).fillna(str(config["default"]))
            if any(common_na_epi_sig): df_epi_src[col_name] = df_epi_src[col_name].replace(na_regex_epi, str(config["default"]), regex=True)
            df_epi_src[col_name] = df_epi_src[col_name].str.strip()

    # 1. Symptomatic Patients with Key Conditions
    key_sympt_conds = list(set(settings.KEY_CONDITIONS_FOR_ACTION) & 
                           {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", "Diarrheal Diseases (Severe)", "Heat Stroke"})
    sympt_keywords_pattern = re.compile(r"\b(fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short\s+of\s+breath|fatigue|dizzy|nausea)\b", re.IGNORECASE)
    epi_signals_output["symptom_keywords_for_monitoring"] = sympt_keywords_pattern.pattern.replace(r"\b", "").replace(r"\s+", " ").replace("|", ", ")

    if 'patient_reported_symptoms' in df_epi_src.columns and 'condition' in df_epi_src.columns:
        symptoms_mask = df_epi_src['patient_reported_symptoms'].astype(str).str.contains(sympt_keywords_pattern, na=False)
        key_cond_mask = df_epi_src['condition'].astype(str).apply(lambda x: any(re.search(r'\b' + re.escape(kc.lower()) + r'\b', x.lower()) for kc in key_sympt_conds))
        sympt_key_cond_df = df_epi_src[symptoms_mask & key_cond_mask]
        if 'patient_id' in sympt_key_cond_df.columns:
            epi_signals_output["symptomatic_patients_key_conditions_count"] = sympt_key_cond_df['patient_id'].nunique()

    # 2. Specific Disease Counts
    if 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        cond_lower = df_epi_src['condition'].astype(str).str.lower()
        epi_signals_output["newly_identified_malaria_patients_count"] = df_epi_src[cond_lower.str.contains(r"\bmalaria\b", na=False, regex=True)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_epi_src[cond_lower.str.contains(r"\btb\b|tuberculosis", na=False, regex=True)]['patient_id'].nunique()

    # 3. Pending TB Contact Tracing Tasks (if not already populated from pre_calculated_chw_kpis)
    if epi_signals_output["pending_tb_contact_tracing_tasks_count"] == 0 and \
       all(c in df_epi_src.columns for c in ['condition', 'referral_status', 'referral_reason', 'patient_id']):
        tb_contact_mask = (df_epi_src['condition'].astype(str).str.contains(r"\btb\b|tuberculosis", case=False, na=False, regex=True)) & \
                          (df_epi_src['referral_reason'].astype(str).str.contains("contact trac", case=False, na=False)) & \
                          (df_epi_src['referral_status'].astype(str).str.lower() == 'pending')
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_epi_src[tb_contact_mask]['patient_id'].nunique()

    # 4. Demographics of High AI Risk Patients
    if 'ai_risk_score' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        high_risk_df = df_epi_src[df_epi_src['ai_risk_score'] >= settings.RISK_SCORE_HIGH_THRESHOLD].drop_duplicates(subset=['patient_id'])
        if not high_risk_df.empty:
            demo_stats = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats["total_high_risk_patients_count"] = len(high_risk_df)
            if 'age' in high_risk_df.columns and high_risk_df['age'].notna().any():
                age_bins = [0, settings.AGE_THRESHOLD_LOW, settings.AGE_THRESHOLD_MODERATE, settings.AGE_THRESHOLD_HIGH, settings.AGE_THRESHOLD_VERY_HIGH, np.inf]
                age_lbls = [f'0-{settings.AGE_THRESHOLD_LOW-1}', f'{settings.AGE_THRESHOLD_LOW}-{settings.AGE_THRESHOLD_MODERATE-1}', 
                            f'{settings.AGE_THRESHOLD_MODERATE}-{settings.AGE_THRESHOLD_HIGH-1}', f'{settings.AGE_THRESHOLD_HIGH}-{settings.AGE_THRESHOLD_VERY_HIGH-1}', 
                            f'{settings.AGE_THRESHOLD_VERY_HIGH}+']
                demo_stats["age_group_distribution"] = pd.cut(high_risk_df['age'], bins=age_bins, labels=age_lbls, right=False, include_lowest=True).value_counts().sort_index().to_dict()
            if 'gender' in high_risk_df.columns and high_risk_df['gender'].notna().any():
                gender_map = lambda g: "Male" if str(g).lower() in ['m', 'male'] else ("Female" if str(g).lower() in ['f', 'female'] else "Other/Unknown")
                demo_stats["gender_distribution"] = high_risk_df['gender'].apply(gender_map).value_counts().to_dict()

    # 5. Symptom Cluster Detection
    if 'patient_reported_symptoms' in df_epi_src.columns and df_epi_src['patient_reported_symptoms'].str.strip().astype(bool).any():
        symptoms_lower = df_epi_src['patient_reported_symptoms'].astype(str).str.lower()
        sympt_patterns = settings.SYMPTOM_CLUSTERS_CONFIG
        min_patients_cluster = 2 
        detected_clusters: List[Dict[str, Any]] = []
        for cluster_name, keywords in sympt_patterns.items():
            if not keywords: continue
            current_mask = pd.Series([True] * len(symptoms_lower), index=symptoms_lower.index)
            for keyword in keywords:
                current_mask &= symptoms_lower.str.contains(re.escape(keyword.lower()), na=False, regex=True) # Use regex with escape
            if current_mask.any():
                patients_in_cluster = df_epi_src[current_mask]['patient_id'].nunique()
                if patients_in_cluster >= min_patients_cluster:
                    detected_clusters.append({"symptoms_pattern": cluster_name, "patient_count": int(patients_in_cluster), "location_hint": chw_zone_context})
        if detected_clusters:
            epi_signals_output["detected_symptom_clusters"] = sorted(detected_clusters, key=lambda x: x['patient_count'], reverse=True)[:max_symptom_clusters_to_report]

    logger.info(f"({module_log_prefix}) CHW local epi signals extracted for {processing_date_str}. Clusters: {len(epi_signals_output['detected_symptom_clusters'])}")
    return epi_signals_output
