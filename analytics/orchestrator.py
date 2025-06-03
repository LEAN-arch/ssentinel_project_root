# sentinel_project_root/analytics/orchestrator.py
# Orchestrates the application of various AI/Analytics models to health data.

import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any

from config import settings # Use new settings module
from .risk_prediction import RiskPredictionModel
from .followup_prioritization import FollowUpPrioritizer
from .supply_forecasting import SupplyForecastingModel, generate_simple_supply_forecast
from data_processing.helpers import convert_to_numeric, standardize_missing_values # For pre-processing

logger = logging.getLogger(__name__)

def apply_ai_models(
    health_df_input: Optional[pd.DataFrame],
    current_supply_status_df: Optional[pd.DataFrame] = None, # Optional: for AI supply forecast
    use_ai_supply_model: bool = False, # Flag to switch supply model
    source_context: str = "AIModelOrchestrator"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Applies a sequence of AI simulation models (Risk, Prioritization, Supply Forecasting)
    to the provided health data.

    Args:
        health_df_input: DataFrame containing health records.
        current_supply_status_df: Optional DataFrame with current supply levels for AI forecasting.
                                   Expected columns: 'item', 'current_stock',
                                   'avg_daily_consumption_historical', 'last_stock_update_date'.
        use_ai_supply_model: If True and current_supply_status_df is provided, uses the
                             AI-simulated supply forecasting model. Otherwise, uses simple model.
        source_context: Logging context string.

    Returns:
        Tuple containing:
            - df_enriched: Health DataFrame enriched with AI scores.
            - supply_forecast_df: DataFrame with supply forecast results, or None.
    """
    logger.info(f"({source_context}) Starting AI model application process.")

    # --- Initialize Output DataFrames ---
    # Define base columns that should exist even if input is empty, plus AI-added columns
    # This ensures a consistent schema for the enriched DataFrame.
    base_health_cols_schema = [ # Example subset of critical columns for schema definition
        'encounter_id', 'patient_id', 'encounter_date', 'condition', 'min_spo2_pct',
        'vital_signs_temperature_celsius', 'age', 'gender', 'zone_id',
        'item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'days_task_overdue'
    ]
    ai_added_cols_schema = ['ai_risk_score', 'ai_followup_priority_score']
    final_enriched_cols_schema = list(set(base_health_cols_schema + ai_added_cols_schema))


    if not isinstance(health_df_input, pd.DataFrame):
        logger.error(f"({source_context}) Input health_df_input is not a DataFrame (type: {type(health_df_input)}). Cannot apply AI models.")
        # Return empty DataFrame with expected AI columns
        return pd.DataFrame(columns=final_enriched_cols_schema), None
        
    if health_df_input.empty:
        logger.warning(f"({source_context}) Input health_df_input is empty. Returning empty enriched DataFrame.")
        # Merge input columns with AI columns to return a consistently schemed empty DF
        existing_cols = health_df_input.columns.tolist()
        return pd.DataFrame(columns=list(set(existing_cols + ai_added_cols_schema))), None

    df_enriched = health_df_input.copy()

    # --- Pre-processing before applying models (ensure critical columns exist and are typed) ---
    # This step is crucial for robustness if upstream data loading isn't perfectly standardized.
    # Define numeric columns needed by models and their safe defaults
    numeric_cols_needed = {
        'age': 30, 'chronic_condition_flag': 0, 'min_spo2_pct': 98.0,
        'vital_signs_temperature_celsius': 37.0, 'max_skin_temp_celsius': 37.0,
        'fall_detected_today': 0, 'ambient_heat_index_c': 25.0, 'ppe_compliant_flag': 1,
        'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': 0.0,
        'hrv_rmssd_ms': 50.0, 'tb_contact_traced': 0, 'days_task_overdue': 0
    }
    string_cols_needed = { # And their safe string defaults
        'condition': "Unknown", 'medication_adherence_self_report': "Unknown",
        'referral_status': "Unknown", 'referral_reason': "N/A"
        # 'worker_task_priority': 'Normal' # If used by prioritizer
    }

    for col, default in numeric_cols_needed.items():
        if col not in df_enriched.columns:
            df_enriched[col] = default
        df_enriched[col] = convert_to_numeric(df_enriched[col], default_value=default)

    for col, default in string_cols_needed.items():
        if col not in df_enriched.columns:
            df_enriched[col] = default
        df_enriched[col] = df_enriched[col].astype(str).fillna(default)
        # Further clean common NA strings for string columns
        common_na_strings = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT', '<NA>', 'null', 'NULL']
        df_enriched[col] = df_enriched[col].replace(common_na_strings, default, regex=False)


    # --- 1. Apply Risk Prediction Model ---
    try:
        risk_model = RiskPredictionModel()
        df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
        logger.info(f"({source_context}) Applied RiskPredictionModel. 'ai_risk_score' column added/updated.")
        if len(df_enriched) < 10: # Log sample scores for small DFs for quick check
            logger.debug(f"({source_context}) Sample AI risk scores: {df_enriched['ai_risk_score'].head(min(5, len(df_enriched))).tolist()}")
    except Exception as e_risk:
        logger.error(f"({source_context}) Error applying RiskPredictionModel: {e_risk}", exc_info=True)
        df_enriched['ai_risk_score'] = np.nan # Fallback: Add column with NaN if model fails


    # --- 2. Apply Follow-up Prioritization Model ---
    try:
        prioritizer = FollowUpPrioritizer()
        # 'days_task_overdue' is handled by pre-processing above now
        df_enriched['ai_followup_priority_score'] = prioritizer.generate_followup_priorities(df_enriched)
        logger.info(f"({source_context}) Applied FollowUpPrioritizer. 'ai_followup_priority_score' column added/updated.")
        if len(df_enriched) < 10:
            logger.debug(f"({source_context}) Sample AI followup priority scores: {df_enriched['ai_followup_priority_score'].head(min(5, len(df_enriched))).tolist()}")
    except Exception as e_prio:
        logger.error(f"({source_context}) Error applying FollowUpPrioritizer: {e_prio}", exc_info=True)
        df_enriched['ai_followup_priority_score'] = np.nan # Fallback


    # --- 3. Apply Supply Forecasting Model (Conditional) ---
    supply_forecast_df_output: Optional[pd.DataFrame] = None
    
    if use_ai_supply_model:
        if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
            required_supply_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
            if all(col in current_supply_status_df.columns for col in required_supply_cols):
                try:
                    ai_supply_forecaster = SupplyForecastingModel()
                    supply_forecast_df_output = ai_supply_forecaster.forecast_supply_levels_advanced(
                        current_supply_levels_df=current_supply_status_df,
                        forecast_days_out=settings.LOW_SUPPLY_DAYS_REMAINING * 4 # e.g., forecast 4 weeks out
                    )
                    logger.info(f"({source_context}) Applied AI-Simulated SupplyForecastingModel. Forecast records: {len(supply_forecast_df_output) if supply_forecast_df_output is not None else 0}")
                except Exception as e_supply_ai:
                    logger.error(f"({source_context}) Error applying AI-Simulated SupplyForecastingModel: {e_supply_ai}", exc_info=True)
                    supply_forecast_df_output = None # Fallback to None
            else:
                missing_supply_cols = [col for col in required_supply_cols if col not in current_supply_status_df.columns]
                logger.warning(f"({source_context}) AI supply forecast skipped: current_supply_status_df missing required columns: {missing_supply_cols}.")
        else:
            logger.info(f"({source_context}) AI supply forecast skipped: current_supply_status_df not provided or empty.")
    # else: # If not using AI model, the simple linear forecast might be called by the component itself using health_df
    # logger.info(f"({source_context}) AI supply model not selected. Simple forecast can be generated separately if needed.")
    # The simple model `generate_simple_supply_forecast` takes `health_df_input` directly.
    # If we want to *always* run a simple forecast if AI is not run, it could be done here.
    # For now, let's assume the component will call the simple one if `use_ai_supply_model` is False
    # and `current_supply_status_df` is not the right input for it.

    logger.info(f"({source_context}) AI model application process completed. Enriched DataFrame shape: {df_enriched.shape}")
    return df_enriched, supply_forecast_df_output
