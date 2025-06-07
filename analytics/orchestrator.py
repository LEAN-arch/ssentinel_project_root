import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Tuple, Dict, Any

# --- Module Imports ---
try:
    from config import settings
    from .risk_prediction import RiskPredictionModel
    from .followup_prioritization import FollowUpPrioritizer
    from .supply_forecasting import SupplyForecastingModel  # AI-simulated model
    from data_processing.helpers import convert_to_numeric
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in orchestrator.py: {e}. Ensure paths are correct.", exc_info=True)
    raise

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)


def apply_ai_models(
    health_df_input: Optional[pd.DataFrame],
    current_supply_status_df: Optional[pd.DataFrame] = None,  # For AI supply forecast
    use_ai_supply_model: bool = False,  # Flag to switch supply model
    source_context: str = "AIModelOrchestrator"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Applies a sequence of AI simulation models (Risk, Prioritization, and optionally
    AI Supply Forecasting) to the provided health data.

    Args:
        health_df_input: DataFrame containing health records.
        current_supply_status_df: Optional DataFrame with current supply levels.
        use_ai_supply_model: If True, uses the AI-simulated supply forecasting model.
        source_context: Logging context string.

    Returns:
        A tuple containing the enriched health DataFrame and an optional supply forecast DataFrame.
    """
    logger.info(f"({source_context}) Starting AI model application process.")

    ai_added_cols = ['ai_risk_score', 'ai_followup_priority_score']

    if not isinstance(health_df_input, pd.DataFrame):
        logger.error(f"({source_context}) Input is not a DataFrame (type: {type(health_df_input)}).")
        return pd.DataFrame(columns=ai_added_cols), None
    
    if health_df_input.empty:
        logger.warning(f"({source_context}) Input health_df_input is empty. Returning empty enriched DataFrame.")
        final_cols = list(set(health_df_input.columns.tolist() + ai_added_cols))
        return pd.DataFrame(columns=final_cols), None

    df_enriched = health_df_input.copy()

    # --- Pre-processing before applying models ---
    numeric_cols_defaults = {
        'age': 30, 'chronic_condition_flag': 0, 'min_spo2_pct': 98.0,
        'vital_signs_temperature_celsius': 37.0, 'max_skin_temp_celsius': 37.0,
        'fall_detected_today': 0, 'ambient_heat_index_c': 25.0, 'ppe_compliant_flag': 1,
        'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': 0.0,
        'hrv_rmssd_ms': 50.0, 'tb_contact_traced': 0, 'days_task_overdue': 0,
    }
    string_cols_defaults = {
        'condition': "UnknownCondition", 'medication_adherence_self_report': "Unknown",
        'referral_status': "Unknown", 'referral_reason': "N/A"
    }

    for col, default in numeric_cols_defaults.items():
        if col not in df_enriched.columns:
            df_enriched[col] = default
        df_enriched[col] = convert_to_numeric(df_enriched[col], default_value=default)

    # FIXED: Replaced buggy string cleaning with a single, robust regex approach.
    common_na_strings = frozenset(['nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'])
    # This regex matches an empty/whitespace string OR any of the other specified NA strings.
    na_regex = r'^\s*$|^(?:' + '|'.join(re.escape(s) for s in common_na_strings) + r')$'

    for col, default in string_cols_defaults.items():
        if col not in df_enriched.columns:
            df_enriched[col] = default
        
        # First, fill actual np.nan values, then ensure string type and use regex for textual NAs.
        df_enriched[col] = df_enriched[col].fillna(default)
        df_enriched[col] = df_enriched[col].astype(str).str.replace(
            na_regex, str(default), case=False, regex=True
        ).str.strip()


    # --- 1. Apply Risk Prediction Model ---
    try:
        risk_model = RiskPredictionModel()
        df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
        logger.info(f"({source_context}) Applied RiskPredictionModel. 'ai_risk_score' column updated.")
    except Exception as e_risk:
        logger.error(f"({source_context}) Error applying RiskPredictionModel: {e_risk}", exc_info=True)
        if 'ai_risk_score' not in df_enriched.columns:
            df_enriched['ai_risk_score'] = np.nan


    # --- 2. Apply Follow-up Prioritization Model ---
    try:
        prioritizer = FollowUpPrioritizer()
        df_enriched['ai_followup_priority_score'] = prioritizer.generate_followup_priorities(df_enriched)
        logger.info(f"({source_context}) Applied FollowUpPrioritizer. 'ai_followup_priority_score' column updated.")
    except Exception as e_prio:
        logger.error(f"({source_context}) Error applying FollowUpPrioritizer: {e_prio}", exc_info=True)
        if 'ai_followup_priority_score' not in df_enriched.columns:
            df_enriched['ai_followup_priority_score'] = np.nan


    # --- 3. Apply AI-Simulated Supply Forecasting Model (Conditional) ---
    supply_forecast_df_output: Optional[pd.DataFrame] = None
    if use_ai_supply_model:
        if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
            required_supply_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
            if all(col in current_supply_status_df.columns for col in required_supply_cols):
                try:
                    ai_supply_forecaster = SupplyForecastingModel()
                    supply_forecast_df_output = ai_supply_forecaster.forecast_supply_levels_advanced(
                        current_supply_levels_df=current_supply_status_df.copy(),
                        forecast_days_out=settings.LOW_SUPPLY_DAYS_REMAINING * 4
                    )
                    logger.info(f"({source_context}) Applied AI-Simulated SupplyForecastingModel. Forecast records: {len(supply_forecast_df_output) if supply_forecast_df_output is not None else 0}")
                except Exception as e_supply_ai:
                    logger.error(f"({source_context}) Error applying AI-Simulated SupplyForecastingModel: {e_supply_ai}", exc_info=True)
            else:
                missing = [col for col in required_supply_cols if col not in current_supply_status_df.columns]
                logger.warning(f"({source_context}) AI supply forecast skipped: missing required columns: {missing}.")
        else:
            logger.info(f"({source_context}) AI supply forecast skipped: current_supply_status_df not provided or empty.")

    logger.info(f"({source_context}) AI model application process completed. Enriched DataFrame shape: {df_enriched.shape}")
    return df_enriched, supply_forecast_df_output
