import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import date as date_type, datetime

# --- Logger Setup ---
# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Module Imports ---
try:
    from config import settings
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    # Log a critical error if essential modules cannot be imported
    logger.error(f"Critical import error in epi_signals.py: {e}. Ensure paths/dependencies are correct.", exc_info=True)
    raise

# --- Constants ---
# Common NA strings for robust replacement
COMMON_NA_STRINGS_EPI = frozenset(['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])

# FIXED: Reconstructed and valid regex pattern from the corrupted original.
_pattern_parts_epi = [re.escape(s) for s in COMMON_NA_STRINGS_EPI if s]
NA_REGEX_EPI_PATTERN = (
    r'^\s*(?:' + '|'.join(_pattern_parts_epi) + r')\s*$'
    if _pattern_parts_epi
    else ''
)

# Pre-compile common regex patterns
SYMPTOM_KEYWORDS_PATTERN_EPI = re.compile(
    r"\b(fever|cough|chills|headache|ache|pain|diarrhea|vomit|rash|breathless|short\s+of\s+breath|fatigue|dizzy|nausea)\b",
    re.IGNORECASE
)
MALARIA_PATTERN_EPI = re.compile(r"\bmalaria\b", re.IGNORECASE)
TB_PATTERN_EPI = re.compile(r"\btb\b|tuberculosis", re.IGNORECASE)


def _prepare_epi_dataframe(
    df: pd.DataFrame,
    cols_config: Dict[str, Dict[str, Any]],
    log_prefix: str
) -> pd.DataFrame:
    """Prepares the DataFrame for epi signal extraction: ensures columns exist, correct types, and handles NAs."""
    df_prepared = df.copy()
    for col_name, config in cols_config.items():
        # FIXED: Corrected IndentationError for the entire block below
        default_value = config["default"]
        target_type_str = config["type"]

        if col_name not in df_prepared.columns:
            if target_type_str == "datetime" and default_value is pd.NaT:
                 df_prepared[col_name] = pd.NaT
            elif isinstance(default_value, (list, dict)):
                 df_prepared[col_name] = [default_value.copy() for _ in range(len(df_prepared))]
            else:
                 df_prepared[col_name] = default_value
        
        if target_type_str in [float, int, "datetime"] and pd.api.types.is_object_dtype(df_prepared[col_name].dtype):
            if NA_REGEX_EPI_PATTERN:
                try:
                    df_prepared[col_name].replace(NA_REGEX_EPI_PATTERN, np.nan, regex=True, inplace=True)
                except Exception as e_regex:
                    logger.warning(f"({log_prefix}) Regex NA replacement failed for '{col_name}': {e_regex}. Proceeding.")
        
        try:
            if target_type_str == "datetime":
                df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
            elif target_type_str == float:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=float)
            elif target_type_str == int:
                df_prepared[col_name] = convert_to_numeric(df_prepared[col_name], default_value=default_value, target_type=int)
            elif target_type_str == str:
                series = df_prepared[col_name].fillna(str(default_value))
                df_prepared[col_name] = series.astype(str).str.strip()
        except Exception as e_conv:
            logger.error(f"({log_prefix}) Error converting column '{col_name}' to {target_type_str}: {e_conv}. Using defaults.", exc_info=True)
            if target_type_str == "datetime" and default_value is pd.NaT:
                df_prepared[col_name] = pd.NaT
            else:
                df_prepared[col_name] = default_value
            
    return df_prepared


def _calculate_demographics_high_risk(
    df_high_risk: pd.DataFrame,
    log_prefix: str
) -> Dict[str, Any]:
    """Calculates age and gender distribution for high-risk patients."""
    demographics = {
        "total_high_risk_patients_count": len(df_high_risk),
        "age_group_distribution": {},
        "gender_distribution": {}
    }
    if df_high_risk.empty:
        return demographics

    if 'age' in df_high_risk.columns and df_high_risk['age'].notna().any():
        age_bins = [
            0,
            getattr(settings, 'AGE_THRESHOLD_LOW', 5),
            getattr(settings, 'AGE_THRESHOLD_MODERATE', 18),
            getattr(settings, 'AGE_THRESHOLD_HIGH', 60),
            getattr(settings, 'AGE_THRESHOLD_VERY_HIGH', 75),
            np.inf
        ]
        # Explicitly cast settings values to int to ensure clean labels (e.g., "5-17" not "5.0-17.0").
        age_labels = [
            f'0-{int(age_bins[1]) - 1}',
            f'{int(age_bins[1])}-{int(age_bins[2]) - 1}',
            f'{int(age_bins[2])}-{int(age_bins[3]) - 1}',
            f'{int(age_bins[3])}-{int(age_bins[4]) - 1}',
            f'{int(age_bins[4])}+'
        ]
        age_series_for_cut = convert_to_numeric(df_high_risk['age'], default_value=np.nan).dropna()
        if not age_series_for_cut.empty:
            try:
                demographics["age_group_distribution"] = pd.cut(
                    age_series_for_cut,
                    bins=age_bins, labels=age_labels, right=False, include_lowest=True
                ).value_counts().sort_index().to_dict()
            except Exception as e_age_cut:
                 logger.warning(f"({log_prefix}) Could not create age group distribution: {e_age_cut}")

    if 'gender' in df_high_risk.columns and df_high_risk['gender'].notna().any():
        def map_gender(g_str: Any) -> str:
            g_lower = str(g_str).lower().strip()
            if g_lower in ['m', 'male']: return "Male"
            if g_lower in ['f', 'female']: return "Female"
            return "Other/Unknown"
        
        gender_counts = df_high_risk['gender'].apply(map_gender).value_counts().to_dict()
        demographics["gender_distribution"] = {
            k: v for k, v in gender_counts.items() if k in ["Male", "Female", "Other/Unknown"]
        }
    return demographics


def _detect_symptom_clusters(
    df_symptoms: pd.DataFrame,
    symptom_clusters_config: Dict[str, List[str]],
    chw_zone_context: str,
    max_clusters_to_report: int,
    min_patients_for_cluster: int = 2,
    log_prefix: str = "SymptomClusterDetection"
) -> List[Dict[str, Any]]:
    """Detects symptom clusters based on configuration."""
    if df_symptoms.empty or 'patient_reported_symptoms' not in df_symptoms or 'patient_id' not in df_symptoms:
        return []
    
    symptoms_lower_series = df_symptoms['patient_reported_symptoms'].astype(str).str.lower()
    detected_clusters_list: List[Dict[str, Any]] = []

    if not isinstance(symptom_clusters_config, dict):
        logger.warning(f"({log_prefix}) Symptom cluster configuration is not a dictionary. Skipping cluster detection.")
        return []

    for cluster_name, keywords_list in symptom_clusters_config.items():
        if not isinstance(keywords_list, list) or not keywords_list:
            logger.debug(f"({log_prefix}) Invalid or empty keywords for cluster '{cluster_name}'. Skipping.")
            continue
        
        current_cluster_series_mask = pd.Series(True, index=symptoms_lower_series.index)
        for keyword in keywords_list:
            if not isinstance(keyword, str) or not keyword.strip():
                continue
            keyword_regex = r'\b' + re.escape(keyword.lower().strip()) + r'\b'
            current_cluster_series_mask &= symptoms_lower_series.str.contains(keyword_regex, na=False, regex=True)
        
        if current_cluster_series_mask.any():
            patients_in_cluster_count = df_symptoms.loc[current_cluster_series_mask, 'patient_id'].nunique()
            if patients_in_cluster_count >= min_patients_for_cluster:
                detected_clusters_list.append({
                    "symptoms_pattern": cluster_name,
                    "patient_count": int(patients_in_cluster_count),
                    "location_hint": chw_zone_context
                })

    if detected_clusters_list:
        return sorted(detected_clusters_list, key=lambda x: x['patient_count'], reverse=True)[:max_clusters_to_report]
    return []


def extract_chw_epi_signals(
    for_date: Union[str, pd.Timestamp, date_type, datetime],
    chw_zone_context: str,
    chw_daily_encounter_df: Optional[pd.DataFrame] = None,
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None,
    max_symptom_clusters_to_report: int = 3
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from a CHW's daily data.
    """
    module_log_prefix = "CHWEpiSignalExtract"
    try:
        processing_date_dt = pd.to_datetime(for_date, errors='coerce')
        # FIXED: Use robust pd.isna() for NaT checking
        if pd.isna(processing_date_dt):
            raise ValueError(f"Invalid 'for_date' ({for_date}) for epi signals.")
        processing_date = processing_date_dt.date()
    except Exception as e_date_parse:
        logger.warning(f"({module_log_prefix}) Invalid 'for_date' ('{for_date}'): {e_date_parse}. Defaulting to current system date.", exc_info=True)
        processing_date = pd.Timestamp('now').date()

    processing_date_str = processing_date.isoformat()
    logger.info(f"({module_log_prefix}) Extracting CHW local epi signals for date: {processing_date_str}, context: {chw_zone_context}")

    epi_signals_output: Dict[str, Any] = {
        "date_of_activity": processing_date_str,
        "operational_context": chw_zone_context,
        "symptomatic_patients_key_conditions_count": 0,
        "symptom_keywords_for_monitoring": SYMPTOM_KEYWORDS_PATTERN_EPI.pattern.replace(r"\b", "").replace(r"\s+", " ").replace("|", ", "),
        "newly_identified_malaria_patients_count": 0,
        "newly_identified_tb_patients_count": 0,
        "pending_tb_contact_tracing_tasks_count": 0,
        "demographics_of_high_ai_risk_patients_today": {
            "total_high_risk_patients_count": 0, "age_group_distribution": {}, "gender_distribution": {}
        },
        "detected_symptom_clusters": []
    }

    if isinstance(pre_calculated_chw_kpis, dict):
        pending_tb_tasks_val = pre_calculated_chw_kpis.get('pending_tb_contact_tracing_tasks_count')
        if pd.notna(pending_tb_tasks_val):
            try:
                epi_signals_output["pending_tb_contact_tracing_tasks_count"] = int(convert_to_numeric(pending_tb_tasks_val, default_value=0, target_type=int))
            except (ValueError, TypeError) as e_conv_tb:
                logger.warning(f"({module_log_prefix}) Could not convert pre-calculated 'pending_tb_contact_tracing_tasks_count' ('{pending_tb_tasks_val}') to int: {e_conv_tb}.")

    if not isinstance(chw_daily_encounter_df, pd.DataFrame) or chw_daily_encounter_df.empty:
        logger.warning(f"({module_log_prefix}) No daily encounter data for {processing_date_str}. Signals will be based on pre_calculated_kpis or defaults only.")
        return epi_signals_output

    essential_cols_config_epi = {
        'patient_id': {"default": f"UPID_EpiSgnl_{processing_date_str}", "type": str},
        'encounter_date': {"default": pd.NaT, "type": "datetime"},
        'condition': {"default": "UnknownCondition", "type": str},
        'patient_reported_symptoms': {"default": "", "type": str},
        'ai_risk_score': {"default": np.nan, "type": float},
        'age': {"default": np.nan, "type": float},
        'gender': {"default": "Unknown", "type": str},
        'referral_reason': {"default": "", "type": str},
        'referral_status': {"default": "Unknown", "type": str}
    }
    df_epi_src = _prepare_epi_dataframe(chw_daily_encounter_df, essential_cols_config_epi, module_log_prefix)

    if 'encounter_date' in df_epi_src.columns and df_epi_src['encounter_date'].notna().any():
        df_epi_src = df_epi_src[df_epi_src['encounter_date'].dt.date == processing_date]

    if df_epi_src.empty:
        logger.info(f"({module_log_prefix}) No CHW data for {processing_date_str} after date filtering. Signals based on pre_calculated_kpis or defaults only.")
        return epi_signals_output
    
    key_symptomatic_conditions = getattr(settings, 'KEY_CONDITIONS_FOR_ACTION', [])

    if 'patient_reported_symptoms' in df_epi_src.columns and 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        symptoms_present_mask = df_epi_src['patient_reported_symptoms'].str.contains(SYMPTOM_KEYWORDS_PATTERN_EPI, na=False)
        
        if key_symptomatic_conditions:
            key_condition_regex = '|'.join([r'\b' + re.escape(c.lower()) + r'\b' for c in key_symptomatic_conditions])
            key_condition_present_mask = df_epi_src['condition'].str.lower().str.contains(key_condition_regex, na=False, regex=True)
            symptomatic_key_condition_df = df_epi_src[symptoms_present_mask & key_condition_present_mask]
            epi_signals_output["symptomatic_patients_key_conditions_count"] = symptomatic_key_condition_df['patient_id'].nunique()

    if 'condition' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        condition_lower_series = df_epi_src['condition'].str.lower()
        epi_signals_output["newly_identified_malaria_patients_count"] = df_epi_src[condition_lower_series.str.contains(MALARIA_PATTERN_EPI, na=False)]['patient_id'].nunique()
        epi_signals_output["newly_identified_tb_patients_count"] = df_epi_src[condition_lower_series.str.contains(TB_PATTERN_EPI, na=False)]['patient_id'].nunique()

    if epi_signals_output.get("pending_tb_contact_tracing_tasks_count", 0) == 0 and \
       all(c in df_epi_src.columns for c in ['condition', 'referral_status', 'referral_reason', 'patient_id']):
        
        tb_contact_tracing_mask = (
            df_epi_src['condition'].str.contains(TB_PATTERN_EPI, na=False) &
            df_epi_src['referral_reason'].str.contains("contact trac", case=False, na=False) &
            (df_epi_src['referral_status'].str.lower() == 'pending')
        )
        epi_signals_output["pending_tb_contact_tracing_tasks_count"] = df_epi_src[tb_contact_tracing_mask]['patient_id'].nunique()

    if 'ai_risk_score' in df_epi_src.columns and 'patient_id' in df_epi_src.columns:
        risk_score_high_thresh = getattr(settings, 'RISK_SCORE_HIGH_THRESHOLD', 75.0)
        df_epi_src['ai_risk_score'] = convert_to_numeric(df_epi_src['ai_risk_score'], default_value=np.nan)
        high_risk_df = df_epi_src[df_epi_src['ai_risk_score'] >= risk_score_high_thresh].drop_duplicates(subset=['patient_id'])
        if not high_risk_df.empty:
            epi_signals_output["demographics_of_high_ai_risk_patients_today"] = _calculate_demographics_high_risk(high_risk_df, module_log_prefix)

    if 'patient_reported_symptoms' in df_epi_src.columns:
        symptoms_df = df_epi_src[['patient_id', 'patient_reported_symptoms']].copy()
        symptoms_df.dropna(subset=['patient_reported_symptoms'], inplace=True)
        symptoms_df = symptoms_df[symptoms_df['patient_reported_symptoms'].astype(str).str.strip() != '']
        
        symptom_clusters_config = getattr(settings, 'SYMPTOM_CLUSTERS_CONFIG', {})
        min_patients_for_cluster = getattr(settings, 'MIN_PATIENTS_FOR_SYMPTOM_CLUSTER', 2)

        if not symptoms_df.empty and symptom_clusters_config:
            epi_signals_output["detected_symptom_clusters"] = _detect_symptom_clusters(
                symptoms_df, symptom_clusters_config, chw_zone_context,
                max_symptom_clusters_to_report, min_patients_for_cluster, module_log_prefix
            )

    num_clusters = len(epi_signals_output.get("detected_symptom_clusters", []))
    logger.info(f"({module_log_prefix}) CHW local epi signals extracted for {processing_date_str}. Clusters found: {num_clusters}")
    return epi_signals_output
