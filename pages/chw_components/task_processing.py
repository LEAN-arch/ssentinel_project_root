# ssentinel_project_root/pages/chw_components/task_processing.py
"""
SME FINAL VERSION: This component generates a prioritized list of tasks for CHWs
based on daily encounter data. The function signature has been corrected to
align with its usage in the dashboard, resolving the TypeError.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import date as date_type, datetime, timedelta

logger = logging.getLogger(__name__)

# --- Safe Setting Import ---
try:
    from config import settings
except ImportError:
    # Define a mock settings class for resilience if the real one isn't found
    class MockSettings:
        TASK_PRIORITY_HIGH_THRESHOLD = 70.0
    settings = MockSettings()

def _get_setting(attr_name: str, default_value: Any) -> Any:
    """Safely gets a configuration value."""
    return getattr(settings, attr_name, default_value)

class TaskGenerator:
    """
    Encapsulates the logic for generating tasks from patient data.
    This class-based approach keeps the rules and preparation logic organized.
    """
    def __init__(self, daily_df: pd.DataFrame, for_date: date_type):
        self.df = daily_df.copy() if isinstance(daily_df, pd.DataFrame) else pd.DataFrame()
        self.for_date = for_date

    def _prepare_dataframe(self):
        """Ensures necessary columns exist and are in the correct format."""
        if self.df.empty:
            return
            
        required_cols = {
            'patient_id': 'Unknown',
            'assigned_chw_id': 'Unassigned',
            'zone_id': 'Unknown',
            'referral_status': 'Unknown',

        }
        for col, default in required_cols.items():
            if col not in self.df:
                self.df[col] = default
            self.df[col].fillna(default, inplace=True)

    def _generate_referral_followup_tasks(self) -> List[Dict[str, Any]]:
        """Generates tasks for patients with pending referrals."""
        tasks = []
        if 'referral_status' not in self.df.columns:
            return tasks
            
        pending_referrals = self.df[self.df['referral_status'].str.lower() == 'pending']
        
        for _, row in pending_referrals.iterrows():
            task = {
                "task_id": f"TASK_REF_{row['patient_id']}_{self.for_date.isoformat()}",
                "task_description": f"Follow up on pending referral for '{row.get('condition', 'N/A')}'",
                "patient_id": row['patient_id'],
                "assigned_chw_id": row['assigned_chw_id'],
                "zone_id": row['zone_id'],
                "due_date": (self.for_date + timedelta(days=1)).isoformat(),
                "priority_score": 85.0, # High priority
                "key_patient_context": f"Patient was referred but status is still pending."
            }
            tasks.append(task)
        return tasks
        
    def _generate_high_risk_monitoring_tasks(self) -> List[Dict[str, Any]]:
        """Generates tasks for monitoring high-risk patients."""
        tasks = []
        if 'ai_risk_score' not in self.df.columns:
            return tasks

        high_risk_threshold = _get_setting('RISK_SCORE_HIGH_THRESHOLD', 75.0)
        high_risk_patients = self.df[self.df['ai_risk_score'] >= high_risk_threshold]

        for _, row in high_risk_patients.iterrows():
            task = {
                "task_id": f"TASK_RISK_{row['patient_id']}_{self.for_date.isoformat()}",
                "task_description": f"Conduct wellness check for high-risk patient",
                "patient_id": row['patient_id'],
                "assigned_chw_id": row['assigned_chw_id'],
                "zone_id": row['zone_id'],
                "due_date": (self.for_date + timedelta(days=2)).isoformat(),
                "priority_score": 75.0,
                "key_patient_context": f"Patient has a high AI risk score of {row['ai_risk_score']:.0f}."
            }
            tasks.append(task)
        return tasks

    def generate_tasks(self) -> List[Dict[str, Any]]:
        """Orchestrates the generation of all task types."""
        if self.df.empty:
            return []
            
        self._prepare_dataframe()
        
        all_tasks = []
        all_tasks.extend(self._generate_referral_followup_tasks())
        all_tasks.extend(self._generate_high_risk_monitoring_tasks())
        
        # Deduplicate tasks to ensure one task of a certain type per patient per day
        if not all_tasks:
            return []
            
        tasks_df = pd.DataFrame(all_tasks)
        tasks_df.sort_values('priority_score', ascending=False, inplace=True)
        tasks_df.drop_duplicates(subset=['task_id'], keep='first', inplace=True)
        
        return tasks_df.to_dict('records')


def generate_chw_tasks(
    daily_df: Optional[pd.DataFrame],
    for_date: Union[str, date_type, datetime],
    chw_id: Optional[str] = None,
    zone_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Public factory function to generate a prioritized list of tasks for CHWs.
    
    SME NOTE: The function signature is now corrected to accept `chw_id` and `zone_id`
    as optional keyword arguments, resolving the TypeError from the dashboard.
    """
    if not isinstance(daily_df, pd.DataFrame) or daily_df.empty:
        return []

    try:
        processing_date = pd.to_datetime(for_date).date()
    except (AttributeError, ValueError):
        logger.warning(f"Invalid 'for_date' in generate_chw_tasks. Defaulting to today.")
        processing_date = datetime.now().date()
        
    # The filtering is now correctly assumed to have been done in the calling script.
    # This function's responsibility is purely to generate tasks from the provided slice of data.
    task_generator = TaskGenerator(daily_df, processing_date)
    return task_generator.generate_tasks()
