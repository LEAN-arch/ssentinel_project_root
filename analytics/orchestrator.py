# sentinel_project_root/analytics/orchestrator.py
# Orchestrates the application of various AI/Analytics models to health data.

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

try:
    from config import settings
    from .risk_prediction import calculate_risk_score
    from .followup_prioritization import calculate_followup_priority
    from data_processing.helpers import data_cleaner
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.critical(f"Critical import error in orchestrator.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class AIModelOrchestrator:
    """
    A pipeline class to sequentially apply AI simulation models (Risk, Prioritization)
    to a health data DataFrame. It ensures data is properly prepared and handles
    model execution failures gracefully.
    """
    def __init__(self, health_df: pd.DataFrame, source_context: str):
        """
        Initializes the orchestrator with the data to be processed.

        Args:
            health_df (pd.DataFrame): The input DataFrame of health records.
            source_context (str): A string for logging context.
        """
        self.df = health_df.copy()
        self.source_context = source_context
        self.notes: list[str] = []

    def _prepare_dataframe(self) -> None:
        """
        Ensures the DataFrame has all necessary columns with correct data types
        and default values, as defined in the application settings. This step is
        critical for the robustness of downstream models.
        """
        if self.df.empty: return

        try:
            # Use declarative configurations from settings for maintainability
            numeric_defaults = getattr(settings, 'RISK_MODEL_NUMERIC_DEFAULTS', {})
            string_defaults = getattr(settings, 'RISK_MODEL_STRING_DEFAULTS', {})
            
            # The standardize_missing_values helper now handles both creation and filling
            self.df = data_cleaner.standardize_missing_values(
                self.df,
                string_cols_defaults=string_defaults,
                numeric_cols_defaults=numeric_defaults
            )
            logger.info(f"({self.source_context}) Data pre-processing and standardization complete.")
        except Exception as e:
            self.notes.append("Failed during data pre-processing.")
            logger.error(f"({self.source_context}) Error in _prepare_dataframe: {e}", exc_info=True)

    def _apply_risk_model(self) -> None:
        """Applies the risk prediction model."""
        if self.df.empty:
            self.df['ai_risk_score'] = np.nan
            return
        
        try:
            # The risk model function is now responsible for its own logic
            self.df = calculate_risk_score(self.df)
            logger.info(f"({self.source_context}) Applied Risk Prediction model.")
        except Exception as e:
            self.notes.append("Risk prediction model failed to execute.")
            logger.error(f"({self.source_context}) Error applying risk model: {e}", exc_info=True)
            self.df['ai_risk_score'] = np.nan # Ensure column exists even on failure

    def _apply_prioritization_model(self) -> None:
        """Applies the follow-up prioritization model."""
        if self.df.empty:
            self.df['ai_followup_priority_score'] = np.nan
            return

        try:
            # The prioritization function encapsulates its own logic
            self.df = calculate_followup_priority(self.df)
            logger.info(f"({self.source_context}) Applied Follow-up Prioritization model.")
        except Exception as e:
            self.notes.append("Follow-up prioritization model failed to execute.")
            logger.error(f"({self.source_context}) Error applying prioritization model: {e}", exc_info=True)
            self.df['ai_followup_priority_score'] = np.nan # Ensure column exists
            self.df['priority_reasons'] = [[] for _ in range(len(self.df))]

    def run_pipeline(self) -> Tuple[pd.DataFrame, list[str]]:
        """
        Executes the full AI model pipeline in sequence.

        Returns:
            Tuple[pd.DataFrame, list[str]]: A tuple containing the enriched
            DataFrame and a list of any processing notes or errors.
        """
        logger.info(f"({self.source_context}) Starting AI model orchestration pipeline.")
        
        self._prepare_dataframe()
        self._apply_risk_model()
        self._apply_prioritization_model()
        
        logger.info(f"({self.source_context}) AI model pipeline completed. Enriched DataFrame shape: {self.df.shape}")
        return self.df, self.notes


def apply_ai_models(
    health_df_input: Optional[pd.DataFrame],
    source_context: str = "AIModelOrchestrator"
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Public factory function to apply a sequence of AI simulation models
    (Risk, Prioritization) to the provided health data.

    This function provides a clean, stable API, encapsulating the complex
    orchestration logic within the `AIModelOrchestrator` class.

    Args:
        health_df_input (Optional[pd.DataFrame]): A DataFrame containing health records.
        source_context (str): A string for logging context.

    Returns:
        Tuple[pd.DataFrame, list[str]]: A tuple containing:
            - An enriched DataFrame with new columns (e.g., `ai_risk_score`, `ai_followup_priority_score`).
              Returns an empty, consistently-schemed DataFrame if the input is invalid or empty.
            - A list of processing notes or errors encountered during the pipeline execution.
    """
    if not isinstance(health_df_input, pd.DataFrame):
        logger.error(f"({source_context}) Input is not a DataFrame (type: {type(health_df_input)}). Cannot apply AI models.")
        return pd.DataFrame(columns=['ai_risk_score', 'ai_followup_priority_score']), ["Invalid input: not a DataFrame."]
    
    if health_df_input.empty:
        logger.warning(f"({source_context}) Input DataFrame is empty. Returning an empty enriched DataFrame.")
        # Ensure consistent schema for empty output
        final_cols = list(health_df_input.columns) + ['ai_risk_score', 'ai_followup_priority_score']
        return pd.DataFrame(columns=list(set(final_cols))), ["Input DataFrame was empty."]

    orchestrator = AIModelOrchestrator(health_df_input, source_context)
    return orchestrator.run_pipeline()
