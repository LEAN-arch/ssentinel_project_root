# sentinel_project_root/analytics/protocol_executor.py
# SME PLATINUM STANDARD - BATCH-ENABLED PROTOCOL ENGINE

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from data_processing.loaders import load_json_asset

logger = logging.getLogger(__name__)

# --- Protocol Loading and Caching ---
@lru_cache(maxsize=1)
def get_escalation_protocols() -> Dict[str, Any]:
    """Loads and caches escalation protocols from the JSON file specified in settings."""
    logger.info("Loading and caching escalation protocols for the first time.")
    default_structure = {"protocols": [], "contacts": {}, "message_templates": {}}
    try:
        loaded_data = load_json_asset('escalation_protocols')
        if loaded_data and isinstance(loaded_data.get("protocols"), list):
            logger.info("Escalation protocols loaded and validated successfully.")
            return loaded_data
        
        logger.error("Failed to load or validate escalation protocols. Using empty structure.")
        return default_structure
    except Exception as e:
        logger.critical(f"Exception loading protocols: {e}. Escalations will be non-functional.", exc_info=True)
        return default_structure

@lru_cache(maxsize=128)
def get_protocol_for_event(event_code: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific protocol by its event code, with caching."""
    protocols_data = get_escalation_protocols()
    for protocol in protocols_data.get("protocols", []):
        if isinstance(protocol, dict) and protocol.get("trigger_event_code") == event_code:
            return protocol
    logger.warning(f"No escalation protocol found for event_code: {event_code}")
    return None

# --- Message Formatting ---
def _format_message(template_code: str, context_data: Dict[str, Any]) -> str:
    protocols_data = get_escalation_protocols()
    template_string = protocols_data.get("message_templates", {}).get(template_code)
    
    if not isinstance(template_string, str):
        logger.warning(f"Message template '{template_code}' not found. Using code as message.")
        return template_code

    context_lower = {str(k).lower(): str(v) for k, v in context_data.items()}

    def replace_match(match: re.Match) -> str:
        placeholder = match.group(1).lower()
        return context_lower.get(placeholder, match.group(0))

    return re.sub(r"\[([a-zA-Z0-9_]+)\]", replace_match, template_string)

# --- Protocol Execution Logic ---
def execute_protocol_for_event(event_code: str, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Executes all steps for a single event, simulating actions."""
    protocol = get_protocol_for_event(event_code)
    if not protocol:
        return [{"action": "NO_PROTOCOL_FOUND", "status": "failed", "details": f"No protocol for {event_code}"}]

    logger.info(f"Executing protocol for event '{event_code}' with context: {list(context_data.keys())}")
    
    results = []
    steps = sorted(protocol.get("steps", []), key=lambda s: s.get("sequence", 99))

    for step in steps:
        action_code = step.get("action_code", "UNKNOWN")
        
        if "NOTIFY" in action_code:
            template = step.get("message_template_code")
            message = _format_message(template, context_data) if template else "No message template."
            details = f"Simulated notification to {step.get('escalation_target_role', 'N/A')}. Message: '{message}'"
        elif "GUIDE" in action_code:
            details = f"Simulated display of guidance '{step.get('guidance_pictogram_code', 'N/A')}'."
        else:
            details = f"Simulated execution of action '{action_code}'."
        
        logger.info(f"  -> Step {step.get('sequence')}: {details}")
        results.append({"action": action_code, "status": "simulated_success", "details": details})
        
    return results

def execute_protocols_for_alerts(alerts_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Batch-processes a DataFrame of alerts, executing protocols for each.
    This is the primary entry point for the alerting engine.
    """
    alerts_with_protocols = alerts_df.dropna(subset=['protocol_id'])
    if alerts_with_protocols.empty:
        return {"total_triggered": 0, "results": {}}

    logger.info(f"Executing protocols for {len(alerts_with_protocols)} triggered alerts.")
    
    all_results = {}
    for index, alert_row in alerts_with_protocols.iterrows():
        event_code = alert_row['protocol_id']
        context = alert_row.to_dict()
        
        execution_summary = execute_protocol_for_event(event_code, context)
        all_results[index] = execution_summary
        
    return {"total_triggered": len(all_results), "results": all_results}
