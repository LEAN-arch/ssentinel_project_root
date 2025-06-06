# sentinel_project_root/analytics/orchestrator.py
# Orchestrates the application of various AI/Analytics models to health data.

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Tuple, Dict, Any

from config import settings
from .risk_prediction import RiskPredictionModel
from .followup_prioritization import FollowUpPrioritizer
from .supply_forecasting import SupplyForecastingModel # AI-simulated model
# Simple supply forecast `generate_simple_supply_forecast` is also in analytics.supply_forecasting
# It's typically called directly by UI components if `use_ai_supply_model` is False.
from data_processing.helpers import convert_to_numeric # For pre-processing

logger = logging.getLogger(__name__)

def apply_ai_models(
    health_df_input: Optional[pd.DataFrame],
    current_supply_status_df: Optional[pd.DataFrame] = None, # For AI supply forecast
    use_ai_supply_model: bool = False, # Flag to switch supply model
    source_context: str = "AIModelOrchestrator"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Applies a sequence of AI simulation models (Risk, Prioritization, and optionally
    AI Supply Forecasting) to the provided health data.

    Args:
        health_df_input: DataFrame containing health records.
        current_supply_status_df: Optional DataFrame with current supply levels for AI forecasting.
                                   Expected columns: 'item', 'current_stock',
                                   'avg_daily_consumption_historical', 'last_stock_update_date'.
        use_ai_supply_model: If True and current_supply_status_df is provided, uses the
                             AI-simulated supply forecasting model.
        source_context: Logging context string.

    Returns:
        Tuple containing:
            - df_enriched: Health DataFrame enriched with AI scores.
            - supply_forecast_df: DataFrame with AI supply forecast results, or None if not run/failed.
    """
    logger.info(f"({source_context}) Starting AI model application process.")

    # Define base columns that should ideally exist even if input is empty, plus AI-added columns.
    # This helps ensure a consistent schema for the enriched DataFrame.
    base_health_cols_schema_example = [ # Example subset of critical columns for schema definition
        'encounter_id', 'patient_id', 'encounter_date', 'condition', 'min_spo2_pct',
        'vital_signs_temperature_celsius', 'age', 'gender', 'zone_id'
        # Note: supply-related columns like 'item', 'item_stock_agg_zone' are not part of base health schema
        # but are used by supply models. 'days_task_overdue' is often from health data.
    ]
    ai_added_cols_schema = ['ai_risk_score', 'ai_followup_priority_score']
    # This final_enriched_cols_schema is more of a conceptual guide for what an empty DF might look like.
    # The actual columns will depend on health_df_input and what models add.
    final_enriched_cols_schema = list(set(base_health_cols_schema_example + ai_added_cols_schema))


    if not isinstance(health_df_input, pd.DataFrame):
        logger.error(f"({source_context}) Input health_df_input is not a DataFrame (type: {type(health_df_input)}). Cannot apply AI models.")
        return pd.DataFrame(columns=final_enriched_cols_schema), None
        
    if health_df_input.empty:
        logger.warning(f"({source_context}) Input health_df_input is empty. Returning empty enriched DataFrame.")
        # Merge input columns (if any) with AI columns to return a consistently schemed empty DF
        existing_cols = health_df_input.columns.tolist()
        return pd.DataFrame(columns=list(set(existing_cols + ai_added_cols_schema))), None

    df_enriched = health_df_input.copy() # Work on a copy

    # --- Pre-processing before applying models ---
    # Ensure critical columns exist and are typed correctly. This enhances robustness.
    # Define numeric columns needed by models and their safe defaults
    numeric_cols_defaults_orchestrator = {
        'age': 30, 'chronic_condition_flag': 0, 'min_spo2_pct': 98.0,
        'vital_signs_temperature_celsius': 37.0, 'max_skin_temp_celsius': 37.0,
        'fall_detected_today': 0, 'ambient_heat_index_c': 25.0, 'ppe_compliant_flag': 1,
        'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': 0.0,
        'hrv_rmssd_ms': 50.0, 'tb_contact_traced': 0, 'days_task_overdue': 0,
        'ai_risk_score': 0.0, 'ai_followup_priority_score': 0.0 # Initialize AI scores to default
    }
    string_cols_defaults_orchestrator = { # And their safe string defaults
        'condition': "UnknownCondition", 'medication_adherence_self_report': "Unknown",
        'referral_status': "Unknown", 'referral_reason': "N/A"
    }
    common_na_strings_orchestrator = ['', 'nan', 'none', 'n/a', '#n/a', 'np.nan', 'nat', '<na>', 'null', 'nu', 'unknown'] # Expanded list

    for col_name, default_val in numeric_cols_defaults_orchestrator.items():
        if col_name not in df_enriched.columns:
            df_enriched[col_name] = default_val # Add column with default if missing
        # Convert to numeric, ensuring NaNs from conversion are filled with the default
        df_enriched[col_name] = convert_to_numeric(df_enriched[col_name], default_value=default_val)

    # CORRECTED: Replaced inefficient and buggy string cleaning loop with a robust, single-regex approach.
    # Build a single regex for all NA strings for efficiency and correctness.
    na_strings_for_regex = [s for s in common_na_strings_orchestrator if s]
    # This regex matches empty/whitespace strings OR any of the other specified NA strings.
    na_regex = r'^\s*$' + (r'|^(?:' + '|'.join(re.escape(s) for s in na_strings_for_regex) + r')$' if na_strings_for_regex else '')

    for col_name, default_val in string_cols_defaults_orchestrator.items():
        if col_name not in df_enriched.columns:
            df_enriched[col_name] = default_val
        
        # First, fill any actual np.nan/pd.NA values with the intended default.
        df_enriched[col_name] = df_enriched[col_name].fillna(default_val)
        
        # Now, ensure the column is string type and replace all textual NA representations
        # (including empty strings) using the single regex for efficiency and correctness.
        df_enriched[col_name] = df_enriched[col_name].astype(str).str.replace(
            na_regex, str(default_val), case=False, regex=True
        )
        df_enriched[col_name] = df_enriched[col_name].str.strip()


    # --- 1. Apply Risk Prediction Model ---
    try:
        risk_model = RiskPredictionModel()
        # predict_bulk_risk_scores should handle internal defaults if columns are missing from df_enriched,
        # but pre-processing above makes it more robust.
        df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
        logger.info(f"({source_context}) Applied RiskPredictionModel. 'ai_risk_score' column added/updated.")
    except Exception as e_risk:
        logger.error(f"({source_context}) Error applying RiskPredictionModel: {e_risk}", exc_info=True)
        # Ensure column exists with NaN if model fails and column wasn't added/filled by pre-processing
        if 'ai_risk_score' not in df_enriched.columns: df_enriched['ai_risk_score'] = np.nan
        df_enriched['ai_risk_score'] = df_enriched['ai_risk_score'].fillna(np.nan) # Fill any new NaNs if model produced them


    # --- 2. Apply Follow-up Prioritization Model ---
    try:
        prioritizer = FollowUpPrioritizer()
        # 'days_task_overdue' should be handled by pre-processing numeric_cols_defaults_orchestrator
        df_enriched['ai_followup_priority_score'] = prioritizer.generate_followup_priorities(df_enriched)
        logger.info(f"({source_context}) Applied FollowUpPrioritizer. 'ai_followup_priority_score' column added/updated.")
    except Exception as e_prio:
        logger.error(f"({source_context}) Error applying FollowUpPrioritizer: {e_prio}", exc_info=True)
        if 'ai_followup_priority_score' not in df_enriched.columns: df_enriched['ai_followup_priority_score'] = np.nan
        df_enriched['ai_followup_priority_score'] = df_enriched['ai_followup_priority_score'].fillna(np.nan)


    # --- 3. Apply AI-Simulated Supply Forecasting Model (Conditional) ---
    supply_forecast_df_output: Optional[pd.DataFrame] = None
    
    if use_ai_supply_model:
        if isinstance(current_supply_status_df, pd.DataFrame) and not current_supply_status_df.empty:
            required_supply_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
            if all(col in current_supply_status_df.columns for col in required_supply_cols):
                try:
                    ai_supply_forecaster = SupplyForecastingModel() # Instantiate from analytics
                    supply_forecast_df_output = ai_supply_forecaster.forecast_supply_levels_advanced(
                        current_supply_levels_df=current_supply_status_df.copy(), # Pass a copy
                        forecast_days_out=settings.LOW_SUPPLY_DAYS_REMAINING * 4 # e.g., forecast 4 weeks
                    )
                    logger.info(f"({source_context}) Applied AI-Simulated SupplyForecastingModel. Forecast records: {len(supply_forecast_df_output) if supply_forecast_df_output is not None else 0}")
                except Exception as e_supply_ai:
                    logger.error(f"({source_context}) Error applying AI-Simulated SupplyForecastingModel: {e_supply_ai}", exc_info=True)
                    supply_forecast_df_output = None # Fallback to None on error
            else:
                missing_supply_cols = [col for col in required_supply_cols if col not in current_supply_status_df.columns]
                logger.warning(f"({source_context}) AI supply forecast skipped: current_supply_status_df missing required columns: {missing_supply_cols}.")
        else:
            logger.info(f"({source_context}) AI supply forecast skipped: current_supply_status_df not provided or empty.")
    # If not using AI model, the simple linear forecast (`generate_simple_supply_forecast`)
    # would typically be called by the UI component itself, using `health_df_input` (or df_enriched).

    logger.info(f"({source_context}) AI model application process completed. Enriched DataFrame shape: {df_enriched.shape}")
    return df_enriched, supply_forecast_df_output
