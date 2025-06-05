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

    if isinstance(pre_calculated_chw_kpis, dict):
        pending_tb_tasks_precalc = pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count')
        if pd.notna(pending_tb_tasks_precalc):
            try:
                # Ensure the input to Series is a list-like, even for a single value
                epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(convert_to_numeric(pd.Series([pending_tb_tasks_precalc]), default_value=0).iloc[0])
            except (ValueError, TypeError):
                logger.warning(f"({module_log_prefix}) Could not convert pre-calculated 'pending_tb_contact_tracing_tasks_count' to int.")


    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data for {processing_date_str}. Signals based on pre_calculated_kpis or defaults.")
        return epi_signals_output

    df_epi_src = chw_daily_encounter_df.copy()
    
    if 'encounter_date' in df_epi_src.columns:
        df_epi_src['encounter_date'] = pd.to_datetime(df_epi_src['encounter_date'], errors='coerce')
        df_epi_src.dropna(subset=['encounter_date'], inplace=True) # Drop rows if date conversion failed
        df_epi_src = df_epi_src[df_epi_src['encounter_date'].dt.date == processing_date]
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
    common_na_epi_sig_list = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown']
    # Ensure na_regex_epi_pattern is only created if common_na_epi_sig_list has non-empty strings
    valid_na_strings_for_regex = [s for s in common_na_epi_sig_list if s]
    na_regex_epi_pattern = r'^(?:' + '|'.join(re.escape(s) for s in valid_na_strings_for_regex) + r')$' if valid_na_strings_for_regex else None


    for col_name, config in essential_cols_config_epi.items():
        if col_name not in df_epi_src.columns: df_epi_src[col_name] = config["default"]
        
        current_col_dtype = df_epi_src[col_name].dtype
        if config["type"] == float: 
            if pd.api.types.is_object_dtype(current_col_dtype) and na_regex_epi_pattern:
                df_epi_src[col_name] = df_epi_src[col_name].replace(na_regex_epi_pattern, np.nan, regex=True)
            df_epi_src[col_name] = convert_to_numeric(df_epi_src[col_name], default_value=config["default"])
        elif config["type"] == str:
            df_epi_src[col_name] = df_epi_src[col_name].astype(str).fillna(str(config["default"]))
            if na_regex_epi_pattern:
                 df_epi_src[col_name] = df_epi_src[col_name].replace(na_regex_epi_pattern, str(config["default"]), regex=True)
            df_epi_src[col_name] = df_epi_src[col_name].str.strip()

    # 1. Symptomatic Patients with Key Conditions
    key_symptomatic_conditions_list_val = list(set(settings.KEY_CONDITIONS_FOR_ACTION) & 
                                               {"TB", "Pneumonia", "Malaria", "Dengue", "Sepsis", 
                                                "Diarrheal Diseases (Severe)", "Heat Stroke"}) # Renamed variable
    symptom_keywords_re_pattern_val = re.compile(r"\b(fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short\s+of\s+breath|fatigue|dizzy|nausea)\b", re.IGNORECASE) # Renamed
    epi_signals_output["symptom_keywords_for_monitoring"] = symptom_keywords_re_pattern_val.pattern.replace(r"\b", "").replace(r"\s+", " ").replace("|", ", ")

    if 'patient_reported_symptoms' in df_epi_src.columns and 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        symptoms_present_mask_val_epi = df_epi_src['patient_reported_symptoms'].astype(str).str.contains(symptom_keywords_re_pattern_val, na=False)
        
        key_condition_regex_parts_val = [r'\b' + re.escape(kc.lower()) + r'\b' for kc in key_symptomatic_conditions_list_val]
        key_condition_full_regex_val = '|'.join(key_condition_regex_parts_val) if key_condition_regex_parts_val else '$^' # Match nothing if list is empty
        
        key_condition_present_mask_val_epi = df_epi_src['condition'].astype(str).str.lower().str.contains(key_condition_full_regex_val, na=False, regex=True)
        
        symptomatic_key_condition_df_val_epi = df_epi_src[symptoms_present_mask_val_epi & key_condition_present_mask_val_epi]
        epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_key_condition_df_val_epi['patient_id'].nunique()

    # 2. Specific Disease Counts
    if 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        condition_lower_series_val_epi = df_epi_src['condition'].astype(str).str.lower()
        epi_signals_output["newly_identified_malaria_patients_count"] = df_epi_src[condition_lower_series_val_epi.str.contains(r"\bmalaria\b", na=False, regex=True)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_epi_src[condition_lower_series_val_epi.str.contains(r"\btb\b|tuberculosis", na=False, regex=True)]['patient_id'].nunique()

    # 3. Pending TB Contact Tracing Tasks
    if epi_signals_output.get("pending_tb_contact_tracing_tasks_count", 0) == 0 and \
       all(c_tb_check_epi in df_epi_src.columns for c_tb_check_epi in ['condition', 'referral_status', 'referral_reason', 'patient_id']):
        tb_contact_tracing_mask_val_epi = (df_epi_src['condition'].astype(str).str.contains(r"\btb\b|tuberculosis", case=False, na=False, regex=True)) & \
                                          (df_epi_src['referral_reason'].astype(str).str.contains("contact trac", case=False, na=False)) & \
                                          (df_epi_src['referral_status'].astype(str).str.lower() == 'pending')
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_epi_src[tb_contact_tracing_mask_val_epi]['patient_id'].nunique()

    # 4. Demographics of High AI Risk Patients
    if 'ai_risk_score' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        high_risk_df_val_epi = df_epi_src[convert_to_numeric(df_epi_src['ai_risk_score'], default_value=-1.0) >= settings.RISK_SCORE_HIGH_THRESHOLD].drop_duplicates(subset=['patient_id'])
        if not high_risk_df_val_epi.empty:
            demo_stats_dict_val_epi = epi_signals_output["demographics_of_high_ai_risk_patients_today"]
            demo_stats_dict_val_epi["total_high_risk_patients_count"] = len(high_risk_df_val_epi)
            if 'age' in high_risk_df_val_epi.columns and high_risk_df_val_epi['age'].notna().any():
                age_bins_val_epi = [0, settings.AGE_THRESHOLD_LOW, settings.AGE_THRESHOLD_MODERATE, settings.AGE_THRESHOLD_HIGH, settings.AGE_THRESHOLD_VERY_HIGH, np.inf]
                age_lbls_val_epi = [f'0-{settings.AGE_THRESHOLD_LOW-1}', f'{settings.AGE_THRESHOLD_LOW}-{settings.AGE_THRESHOLD_MODERATE-1}', 
                                    f'{settings.AGE_THRESHOLD_MODERATE}-{settings.AGE_THRESHOLD_HIGH-1}', f'{settings.AGE_THRESHOLD_HIGH}-{settings.AGE_THRESHOLD_VERY_HIGH-1}', 
                                    f'{settings.AGE_THRESHOLD_VERY_HIGH}+']
                age_series_for_cut_val_epi = convert_to_numeric(high_risk_df_val_epi['age'], default_value=np.nan)
                if age_series_for_cut_val_epi.notna().any():
                    demo_stats_dict_val_epi["age_group_distribution"] = pd.cut(age_series_for_cut_val_epi.dropna(), bins=age_bins_val_epi, labels=age_lbls_val_epi, right=False, include_lowest=True).value_counts().sort_index().to_dict()
            if 'gender' in high_risk_df_val_epi.columns and high_risk_df_val_epi['gender'].notna().any():
                gender_map_func_val_epi = lambda g_str_val_epi: "Male" if str(g_str_val_epi).lower() in ['m', 'male'] else ("Female" if str(g_str_val_epi).lower() in ['f', 'female'] else "Other/Unknown")
                gender_counts_val_epi = high_risk_df_val_epi['gender'].apply(gender_map_func_val_epi).value_counts().to_dict()
                demo_stats_dict_val_epi["gender_distribution"] = {k_gen_epi:v_gen_epi for k_gen_epi,v_gen_epi in gender_counts_val_epi.items() if k_gen_epi in ["Male", "Female"]}

    # 5. Symptom Cluster Detection
    if 'patient_reported_symptoms' in df_epi_src.columns and df_epi_src['patient_reported_symptoms'].astype(str).str.strip().str.len().gt(0).any():
        symptoms_lower_series_val_epi = df_epi_src['patient_reported_symptoms'].astype(str).str.lower()
        sympt_patterns_config_val_epi = settings.SYMPTOM_CLUSTERS_CONFIG
        min_patients_for_cluster_val_epi = 2 
        detected_clusters_final_list_epi: List[Dict[str, Any]] = []

        if isinstance(sympt_patterns_config_val_epi, dict):
            for cluster_name_item_epi, keywords_list_item_epi in sympt_patterns_config_val_epi.items():
                if not isinstance(keywords_list_item_epi, list) or not keywords_list_item_epi: continue
                
                current_cluster_mask_val_epi = pd.Series([True] * len(symptoms_lower_series_val_epi), index=symptoms_lower_series_val_epi.index)
                all_keywords_valid_for_cluster = True # Flag to check if all keywords in a pattern are valid
                for keyword_val_item_epi in keywords_list_item_epi:
                    if not isinstance(keyword_val_item_epi, str) or not keyword_val_item_epi.strip():
                        all_keywords_valid_for_cluster = False; break
                    keyword_regex_epi = r'\b' + re.escape(keyword_val_item_epi.lower().strip()) + r'\b'
                    current_cluster_mask_val_epi &= symptoms_lower_series_val_epi.str.contains(keyword_regex_epi, na=False, regex=True)
                
                if not all_keywords_valid_for_cluster: continue

                if current_cluster_mask_val_epi.any():
                    patients_in_cluster_num_epi = df_epi_src[current_cluster_mask_val_epi]['patient_id'].nunique()
                    if patients_in_cluster_num_epi >= min_patients_for_cluster_val_epi:
                        detected_clusters_final_list_epi.append({"symptoms_pattern": cluster_name_item_epi, "patient_count": int(patients_in_cluster_num_epi), "location_hint": chw_zone_context})
            
            if detected_clusters_final_list_epi:
                epi_signals_output["detected_symptom_clusters"] = sorted(detected_clusters_final_list_epi, key=lambda x_cluster_item_epi: x_cluster_item_epi['patient_count'], reverse=True)[:max_symptom_clusters_to_report]

    logger.info(f"({module_log_prefix}) CHW local epi signals extracted for {processing_date_str}. Clusters: {len(epi_signals_output['detected_symptom_clusters'])}")
    return epi_signals_output
