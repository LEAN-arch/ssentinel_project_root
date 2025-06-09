# ssentinel_project_root/pages/chw_components/activity_trends.py
"""
Component for calculating and preparing CHW activity trend data using a robust,
class-based approach.
"""
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CHWActivityTrendPreparer:
    """Encapsulates the logic for aggregating CHW activity data into trends."""
    def __init__(self, chw_trend_df: Optional[pd.DataFrame]):
        self.df = pd.DataFrame()
        if isinstance(chw_trend_df, pd.DataFrame) and not chw_trend_df.empty:
            if 'encounter_date' in chw_trend_df.columns:
                self.df = chw_trend_df.copy()
                self.df['encounter_date'] = pd.to_datetime(self.df['encounter_date'], errors='coerce')
                self.df.dropna(subset=['encounter_date'], inplace=True)
                self.df.set_index('encounter_date', inplace=True)

    def _resample_and_agg(self, value_col: str, agg_func: str, freq: str = 'D') -> pd.Series:
        """Helper to perform resampling on the internal dataframe."""
        if self.df.empty or value_col not in self.df.columns:
            return pd.Series(dtype='float')
        
        try:
            return self.df[value_col].resample(freq).agg(agg_func).fillna(0)
        except Exception as e:
            logger.error(f"Error during resampling for column '{value_col}': {e}", exc_info=True)
            return pd.Series(dtype='float')

    def prepare(self) -> Dict[str, pd.Series]:
        """Calculates all relevant CHW activity trends."""
        if self.df.empty:
            return {"patient_visits_trend": pd.Series(dtype='int'), "high_priority_followups_trend": pd.Series(dtype='int')}

        visits_trend = self._resample_and_agg(value_col='patient_id', agg_func='nunique')
        
        high_prio_trend = pd.Series(dtype='int')
        if 'ai_followup_priority_score' in self.df.columns:
            try:
                high_prio_df = self.df[self.df['ai_followup_priority_score'] >= 80]
                if not high_prio_df.empty:
                     high_prio_trend = high_prio_df['patient_id'].resample('D').nunique().fillna(0)
            except Exception as e:
                logger.error(f"Could not calculate high priority trend: {e}")
        
        return {
            "patient_visits_trend": visits_trend,
            "high_priority_followups_trend": high_prio_trend,
        }

def get_chw_activity_trends(chw_trend_df: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Factory function to calculate all CHW activity trends. This is the single,
    correct function to import and use in the dashboard.
    """
    preparer = CHWActivityTrendPreparer(chw_trend_df)
    return preparer.prepare()
