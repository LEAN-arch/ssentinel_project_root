# sentinel_project_root/analytics/orchestrator.py
# SME PLATINUM STANDARD - AI MODEL ORCHESTRATION PIPELINE

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .followup_prioritization import calculate_followup_priority
from .risk_prediction import calculate_risk_score

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """
    A pipeline class to sequentially apply all AI/ML models (Risk, Prioritization)
    to a health data DataFrame.
    """
    def __init__(self, df: pd.DataFrame, source_context: str = "default"):
        self.df = df.copy()
        self.source_context = source_context
        self.errors: List[str] = []

    def _apply_risk_model(self) -> 'AIOrchestrator':
        if self.df.empty: return self
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
        if self.df.empty: return self
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
        logger.info(f"({self.source_context}) Starting AI pipeline for {len(self.df)} records.")
        (self
            ._apply_risk_model()
            ._apply_prioritization_model()
        )
        logger.info(f"({self.source_context}) AI pipeline completed. Shape: {self.df.shape}")
        return self.df, self.errors

def apply_ai_models(
    df_input: Optional[pd.DataFrame],
    source_context: str = "AIOrchestrator"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Public factory function to apply a sequence of AI models to health data.
    """
    if not isinstance(df_input, pd.DataFrame):
        return pd.DataFrame(), ["Invalid input: not a DataFrame."]
    
    if df_input.empty:
        return df_input.assign(
            ai_risk_score=np.nan,
            ai_followup_priority_score=np.nan,
            priority_reasons=""
        ), ["Input DataFrame was empty."]

    return AIOrchestrator(df_input, source_context).run()
