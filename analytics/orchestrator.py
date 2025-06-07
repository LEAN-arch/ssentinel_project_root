# sentinel_project_root/analytics/orchestrator.py
"""
Orchestrates the application of various AI/ML models to health data.
"""
import pandas as pd
import logging
from typing import Tuple, Dict, Any

try:
    from config import settings
    from .risk_prediction import RiskPredictionModel
    from .followup_prioritization import FollowUpPrioritizer
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in orchestrator.py: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class AIModelOrchestrator:
    """
    Manages and applies a sequence of analytical models to health data.
    """
    def __init__(self):
        self.module_log_prefix = self.__class__.__name__
        logger.info(f"({self.module_log_prefix}) Initializing AI model chain...")
        
        self.risk_model = RiskPredictionModel()
        self.followup_model = FollowUpPrioritizer()
        
        # This list now correctly points to the existing methods.
        self.models_to_apply = [
            ("RiskPredictionModel", self.risk_model.calculate_risk_scores, "ai_risk_score"),
            ("FollowUpPrioritizer", self.followup_model.generate_priority_scores, "ai_followup_priority_score")
        ]
        logger.info(f"({self.module_log_prefix}) AI orchestrator initialized.")

    def apply_all_enrichment_models(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

def apply_ai_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Public factory function to run the AIModelOrchestrator.
    NOTE: The signature is now clean and only accepts the dataframe.
    """
    orchestrator = AIModelOrchestrator()
    return orchestrator.apply_all_enrichment_models(df)
