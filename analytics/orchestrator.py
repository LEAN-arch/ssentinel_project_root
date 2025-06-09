# sentinel_project_root/analytics/orchestrator.py
# SME PLATINUM STANDARD - AI MODEL ORCHESTRATION PIPELINE

import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from config import settings
from data_processing.helpers import DataPipeline
from .risk_prediction import calculate_risk_score
from .followup_prioritization import calculate_followup_priority

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """
    A pipeline class to sequentially apply all AI/ML models (Risk, Prioritization)
    to a health data DataFrame. It ensures data is properly prepared and handles
    model execution gracefully.
    """
    def __init__(self, df: pd.DataFrame, source_context: str = "default"):
        self.df = df.copy()
        self.source_context = source_context
        self.errors: List[str] = []

    def _prepare_data(self) -> 'AIOrchestrator':
        """Prepares the DataFrame using the modern DataPipeline helper."""
        if self.df.empty:
            return self
        
        # This step is now implicitly handled by the models themselves,
        # but could be used for pre-model-agnostic cleaning if needed.
        # For now, we rely on each model to prepare its own required columns.
        logger.debug(f"({self.source_context}) Data preparation step in orchestrator.")
        return self

    def _apply_risk_model(self) -> 'AIOrchestrator':
        """Applies the vectorized risk prediction model."""
        if self.df.empty:
            return self
        try:
            self.df = calculate_risk_score(self.df)
            logger.info(f"({self.source_context}) Applied Risk Prediction model.")
        except Exception as e:
            msg = "Risk prediction model failed to execute."
            self.errors.append(msg)
            logger.error(f"({self.source_context}) {msg}: {e}", exc_info=True)
            self.df['ai_risk_score'] = np.nan
        return self

    def _apply_prioritization_model(self) -> 'AIOrchestrator':
        """Applies the vectorized follow-up prioritization model."""
        if self.df.empty:
            return self
        try:
            self.df = calculate_followup_priority(self.df)
            logger.info(f"({self.source_context}) Applied Follow-up Prioritization model.")
        except Exception as e:
            msg = "Follow-up prioritization model failed to execute."
            self.errors.append(msg)
            logger.error(f"({self.source_context}) {msg}: {e}", exc_info=True)
            self.df['ai_followup_priority_score'] = np.nan
            self.df['priority_reasons'] = "Error"
        return self

    def run(self) -> Tuple[pd.DataFrame, List[str]]:
        """Executes the full AI model pipeline in a fluent sequence."""
        logger.info(f"({self.source_context}) Starting AI model orchestration pipeline for {len(self.df)} records.")
        
        (self
            ._prepare_data()
            ._apply_risk_model()
            ._apply_prioritization_model()
        )
        
        logger.info(f"({self.source_context}) AI model pipeline completed. Enriched shape: {self.df.shape}")
        return self.df, self.errors


def apply_ai_models(
    df_input: Optional[pd.DataFrame],
    source_context: str = "AIOrchestrator"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Public factory function to apply a sequence of AI models to health data.
    This is the primary entry point for enriching raw data with AI-driven insights.
    """
    if not isinstance(df_input, pd.DataFrame):
        return pd.DataFrame(), ["Invalid input: not a DataFrame."]
    
    if df_input.empty:
        # Return a correctly schemed empty DataFrame
        return df_input.assign(
            ai_risk_score=np.nan,
            ai_followup_priority_score=np.nan,
            priority_reasons=""
        ), ["Input DataFrame was empty."]

    orchestrator = AIOrchestrator(df_input, source_context)
    return orchestrator.run()
