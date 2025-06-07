# sentinel_project_root/analytics/orchestrator.py
"""
Orchestrates the application of various AI/ML models to health data.
This is the central point for applying analytics to raw dataframes.
"""
import pandas as pd
import logging
from typing import Tuple, Dict, Any

try:
    from config import settings
    # FIX: This now correctly imports the required model classes.
    # The old 'SupplyForecastingModel' is no longer imported because it doesn't exist.
    from .risk_prediction import RiskPredictionModel
    from .followup_prioritization import FollowUpPrioritizer
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in orchestrator.py: {e}. Ensure paths are correct.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class AIModelOrchestrator:
    """
    A class to manage and apply a sequence of analytical models to health data.
    Its sole responsibility is to run enrichment models, not to clean data.
    """
    def __init__(self):
        """
        Initializes the orchestrator with instances of all required enrichment models.
        """
        self.module_log_prefix = self.__class__.__name__
        logger.info(f"({self.module_log_prefix}) Initializing AI model chain...")
        
        self.risk_model = RiskPredictionModel()
        self.followup_model = FollowUpPrioritizer()
        
        self.models_to_apply = [
            ("RiskPredictionModel", self.risk_model.calculate_risk_scores, "ai_risk_score"),
            ("FollowUpPrioritizer", self.followup_model.generate_priority_scores, "ai_followup_priority_score")
        ]
        logger.info(f"({self.module_log_prefix}) AI orchestrator initialized.")

    def apply_all_enrichment_models(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies a sequence of data enrichment models to the input DataFrame.
        Assumes the input DataFrame has already been cleaned by the data loading process.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(), {"notes": ["Input DataFrame is empty."]}

        enriched_df = df.copy()
        metadata = {"notes": [], "models_applied": []}
        logger.info(f"({self.module_log_prefix}) Starting AI model application process on DataFrame of shape {df.shape}.")

        for model_name, model_func, output_col_name in self.models_to_apply:
            try:
                logger.info(f"({self.module_log_prefix}) Applying {model_name}...")
                enriched_df = model_func(enriched_df)
                logger.info(f"({self.module_log_prefix}) Applied {model_name}. Column '{output_col_name}' updated.")
                metadata["models_applied"].append(model_name)
            except Exception as e:
                note = f"Failed to apply {model_name}: {e}"
                logger.error(f"({self.module_log_prefix}) {note}", exc_info=True)
                metadata["notes"].append(note)
                if output_col_name not in enriched_df.columns:
                    enriched_df[output_col_name] = pd.NA

        logger.info(f"({self.module_log_prefix}) AI model application process completed.")
        return enriched_df, metadata

# --- Public Factory Function ---

def apply_ai_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    A simple, public-facing function to instantiate and run the AIModelOrchestrator.
    This is the primary entry point for enriching a health data DataFrame.
    """
    orchestrator = AIModelOrchestrator()
    return orchestrator.apply_all_enrichment_models(df)
