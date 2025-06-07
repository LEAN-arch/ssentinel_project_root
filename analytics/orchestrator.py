# ssentinel_project_root/analytics/orchestrator.py
"""
Orchestrates the application of various AI/ML models to health data.
This is the central point for applying analytics to raw dataframes, ensuring
models are run in the correct sequence.
"""
import pandas as pd
import logging
from typing import Tuple, Dict, Any

try:
    # Import the model classes, not instances, to be instantiated by the orchestrator.
    from .risk_prediction import RiskPredictionModel
    from .followup_prioritization import FollowUpPrioritizer
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in orchestrator.py: {e}. Ensure all model files exist in the 'analytics' package.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class AIModelOrchestrator:
    """
    Manages and applies a sequence of analytical models to health data.
    This class ensures a consistent and correct order of operations for data enrichment.
    """
    def __init__(self):
        """
        Initializes the orchestrator with instances of all required enrichment models.
        The order of initialization and application is critical.
        """
        self.module_log_prefix = self.__class__.__name__
        logger.info(f"({self.module_log_prefix}) Initializing AI model chain...")
        
        # Instantiate models to be used in the pipeline
        self.risk_model = RiskPredictionModel()
        self.followup_model = FollowUpPrioritizer()
        
        # Define the sequence of models to apply. Order matters.
        self.models_to_apply = [
            ("RiskPredictionModel", self.risk_model.calculate_risk_scores, "ai_risk_score"),
            ("FollowUpPrioritizer", self.followup_model.generate_priority_scores, "ai_followup_priority_score")
            # To add a new model, simply add a new tuple to this list.
        ]
        logger.info(f"({self.module_log_prefix}) AI orchestrator initialized with {len(self.models_to_apply)} model(s).")

    def apply_all_enrichment_models(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies the full sequence of data enrichment models to the input DataFrame.
        Assumes the input DataFrame has been cleaned by the data loading process.
        
        Args:
            df: The input pandas DataFrame.
            
        Returns:
            A tuple containing:
            - The enriched DataFrame with new AI-generated columns.
            - A metadata dictionary detailing the process.
        """
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(), {"notes": ["Input was not a pandas DataFrame."], "models_applied": []}
        if df.empty:
            return df.copy(), {"notes": ["Input DataFrame is empty."], "models_applied": []}

        enriched_df = df.copy()
        metadata = {"notes": [], "models_applied": []}
        logger.info(f"({self.module_log_prefix}) Starting AI model application process on DataFrame of shape {df.shape}.")

        for model_name, model_func, output_col_name in self.models_to_apply:
            try:
                logger.info(f"({self.module_log_prefix}) Applying {model_name}...")
                enriched_df = model_func(enriched_df)
                
                if output_col_name not in enriched_df.columns:
                     raise ValueError(f"Model function did not return the expected output column '{output_col_name}'.")

                logger.info(f"({self.module_log_prefix}) Successfully applied {model_name}. Column '{output_col_name}' created/updated.")
                metadata["models_applied"].append(model_name)
            except Exception as e:
                note = f"FATAL: Failed to apply model '{model_name}': {e}"
                logger.error(f"({self.module_log_prefix}) {note}", exc_info=True)
                metadata["notes"].append(note)
                
                # Ensure the output column exists even if the model fails, to prevent
                # downstream errors in components that expect the column.
                if output_col_name not in enriched_df.columns:
                    enriched_df[output_col_name] = pd.NA

        logger.info(f"({self.module_log_prefix}) AI model application process completed. {len(metadata['models_applied'])} models applied.")
        return enriched_df, metadata

# --- Public Factory Function ---

def apply_ai_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    A simple, public-facing function to instantiate and run the AIModelOrchestrator.
    This is the primary entry point for enriching a health data DataFrame.
    
    Args:
        df: The pandas DataFrame to be enriched.
        
    Returns:
        A tuple of (enriched_df, metadata).
    """
    orchestrator = AIModelOrchestrator()
    return orchestrator.apply_all_enrichment_models(df)
