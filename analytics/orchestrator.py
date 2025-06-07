# sentinel_project_root/analytics/orchestrator.py
# Orchestrates the sequential application of AI/Analytics models to health data.

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

# --- Core Imports ---
try:
    from config import settings
    from .risk_prediction import RiskPredictionModel
    from .followup_prioritization import FollowUpPrioritizer
    from .supply_forecasting import SupplyForecastingModel
    from data_processing.helpers import standardize_missing_values
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in orchestrator.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

def apply_ai_models(
    health_df_input: Optional[pd.DataFrame],
    current_supply_status_df: Optional[pd.DataFrame] = None,
    use_ai_supply_model: bool = False,
    source_context: str = "AIModelOrchestrator"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Applies a sequence of AI simulation models (Risk, Prioritization, and optionally
    AI Supply Forecasting) to the provided health data.

    This orchestrator ensures data is properly prepared and flows through each
    analytical model, enriching the dataset with predictive scores.

    Args:
        health_df_input: DataFrame containing health records.
        current_supply_status_df: Optional DataFrame with current supply levels for AI forecasting.
        use_ai_supply_model: If True, uses the advanced AI-simulated supply forecasting model.
        source_context: Logging context string.

    Returns:
        A tuple containing:
            - df_enriched: Health DataFrame enriched with AI scores.
            - supply_forecast_df: DataFrame with AI supply forecast results, or None.
    """
    logger.info(f"({source_context}) Starting AI model application process.")

    if not isinstance(health_df_input, pd.DataFrame) or health_df_input.empty:
        logger.warning(f"({source_context}) Input health_df_input is empty or invalid. Returning empty structures.")
        # Return an empty DataFrame with expected AI columns for schema consistency
        return pd.DataFrame(columns=['ai_risk_score', 'ai_followup_priority_score']), None

    df_enriched = health_df_input.copy()

    # --- Robust Pre-processing Step ---
    # Ensures all columns needed by downstream models exist and have sane defaults.
    # This prevents errors from missing columns or incorrect data types.
    numeric_defaults = {
        'age': 30, 'chronic_condition_flag': 0, 'min_spo2_pct': 98.0,
        'vital_signs_temperature_celsius': 37.0, 'fall_detected_today': 0,
        'signs_of_fatigue_observed_flag': 0, 'hrv_rmssd_ms': 50.0,
        'tb_contact_traced': 0, 'days_task_overdue': 0,
        'ai_risk_score': 0.0, 'ai_followup_priority_score': 0.0
    }
    string_defaults = {
        'condition': "UnknownCondition", 'medication_adherence_self_report': "Unknown",
        'referral_status': "Unknown", 'referral_reason': "N/A"
    }
    df_enriched = standardize_missing_values(df_enriched, string_defaults, numeric_defaults)
    logger.info(f"({source_context}) Pre-processing complete. DataFrame shape: {df_enriched.shape}")

    # --- 1. Apply Risk Prediction Model ---
    try:
        risk_model = RiskPredictionModel()
        df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
        logger.info(f"({source_context}) Applied RiskPredictionModel successfully.")
    except Exception as e:
        logger.error(f"({source_context}) Failed to apply RiskPredictionModel: {e}", exc_info=True)
        # Ensure column exists with NaNs if model fails, maintaining schema
        if 'ai_risk_score' not in df_enriched.columns:
            df_enriched['ai_risk_score'] = np.nan

    # --- 2. Apply Follow-up Prioritization Model ---
    try:
        prioritizer = FollowUpPrioritizer()
        # The new prioritizer is vectorized and highly efficient.
        df_enriched['ai_followup_priority_score'] = prioritizer.generate_followup_priorities(df_enriched)
        logger.info(f"({source_context}) Applied FollowUpPrioritizer successfully.")
    except Exception as e:
        logger.error(f"({source_context}) Failed to apply FollowUpPrioritizer: {e}", exc_info=True)
        if 'ai_followup_priority_score' not in df_enriched.columns:
            df_enriched['ai_followup_priority_score'] = np.nan

    # --- 3. Apply AI-Simulated Supply Forecasting Model (Conditional) ---
    supply_forecast_df_output: Optional[pd.DataFrame] = None
    if use_ai_supply_model:
        logger.info(f"({source_context}) AI supply forecasting is enabled. Checking inputs.")
        if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
            required_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
            if all(col in current_supply_status_df.columns for col in required_cols):
                try:
                    ai_forecaster = SupplyForecastingModel()
                    supply_forecast_df_output = ai_forecaster.forecast_supply_levels_advanced(
                        current_supply_levels_df=current_supply_status_df,
                        forecast_days_out=getattr(settings.Thresholds, 'SUPPLY_FORECAST_HORIZON_DAYS', 56)
                    )
                    logger.info(f"({source_context}) AI SupplyForecastingModel applied. Forecast records: {len(supply_forecast_df_output) if supply_forecast_df_output is not None else 0}")
                except Exception as e:
                    logger.error(f"({source_context}) Error applying AI SupplyForecastingModel: {e}", exc_info=True)
            else:
                missing_cols = [c for c in required_cols if c not in current_supply_status_df.columns]
                logger.warning(f"({source_context}) AI supply forecast skipped: input DF missing columns: {missing_cols}.")
        else:
            logger.warning(f"({source_context}) AI supply forecast skipped: input DF not provided or empty.")

    logger.info(f"({source_context}) AI model application process completed. Enriched DataFrame shape: {df_enriched.shape}")
    return df_enriched, supply_forecast_df_output
