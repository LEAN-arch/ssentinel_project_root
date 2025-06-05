# sentinel_project_root/analytics/protocol_executor.py
# Handles parsing and execution logic for escalation protocols.

import logging
import json
from typing import Dict, Any, Optional, List, Tuple

from config import settings
from data_processing.loaders import load_escalation_protocols # Relative import likely to cause issues if not structured well

logger = logging.getLogger(__name__)

_LOADED_ESCALATION_PROTOCOLS: Optional[Dict[str, Any]] = None

def _get_loaded_protocols() -> Dict[str, Any]:
    """Ensures escalation protocols are loaded, loading them if necessary."""
    global _LOADED_ESCALATION_PROTOCOLS
    if _LOADED_ESCALATION_PROTOCOLS is None:
        logger.info("First-time access or protocols not loaded; loading escalation_protocols.json.")
        # Use the path from settings for robustness
        _LOADED_ESCALATION_PROTOCOLS = load_escalation_protocols(settings.ESCALATION_PROTOCOLS_JSON_PATH)
        if not _LOADED_ESCALATION_PROTOCOLS or not isinstance(_LOADED_ESCALATION_PROTOCOLS.get("protocols"), list):
             logger.error("Failed to load or validate escalation protocols. Escalations may not function as expected.")
             _LOADED_ESCALATION_PROTOCOLS = {"protocols": [], "contacts": {}, "message_templates": {}}
    return _LOADED_ESCALATION_PROTOCOLS

def get_protocol_for_event(event_code: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific escalation protocol based on the event code.
    """
    protocols_data = _get_loaded_protocols()
    if not protocols_data.get("protocols"): # Check after ensuring protocols_data is a dict
        return None
        
    for protocol in protocols_data["protocols"]:
        if protocol.get("trigger_event_code") == event_code:
            return protocol
    logger.warning(f"No escalation protocol found for event_code: {event_code}")
    return None

def format_escalation_message(template_code: str, context_data: Dict[str, Any]) -> str:
    """
    Formats an escalation message using a template and context data.
    Placeholders are like [PLACEHOLDER_NAME] (case-insensitive matching for keys in context_data).
    """
    protocols_data = _get_loaded_protocols()
    message_templates = protocols_data.get("message_templates", {})
    
    template_string = message_templates.get(template_code)
    if not template_string:
        logger.warning(f"Message template_code '{template_code}' not found. Using code as message.")
        return template_code

    formatted_message = template_string
    # Create a case-insensitive mapping for context_data keys for robust placeholder replacement
    context_data_lower_keys = {k.lower(): v for k, v in context_data.items()}

    import re
    def replace_placeholder(match):
        placeholder_key = match.group(1).lower() # Get key inside brackets and lowercase it
        return str(context_data_lower_keys.get(placeholder_key, match.group(0))) # Replace or keep original if key not found

    formatted_message = re.sub(r"\[([A-Za-z0-9_]+)\]", replace_placeholder, formatted_message)
    
    if "[" in formatted_message and "]" in formatted_message: # Basic check for unreplaced placeholders
        logger.debug(f"Message template '{template_code}' may have unreplaced placeholders: {formatted_message}")
        
    return formatted_message


def execute_escalation_protocol(
    event_code: str,
    triggering_data: Dict[str, Any],
    additional_context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Executes the steps defined in an escalation protocol for a given event.
    This is a simulation; actual actions (SMS, calls) are logged.
    """
    module_log_prefix = "EscalationExecutor"
    logger.info(f"({module_log_prefix}) Executing escalation protocol for event_code: {event_code}")

    protocol = get_protocol_for_event(event_code)
    if not protocol:
        logger.warning(f"({module_log_prefix}) No protocol found for event '{event_code}'. Escalation aborted.")
        return [{"action_code": "NO_PROTOCOL_FOUND", "status": "failed", "details": f"No protocol for {event_code}"}]

    executed_steps_results: List[Dict[str, Any]] = []
    
    full_context_data = triggering_data.copy()
    if additional_context:
        full_context_data.update(additional_context)

    # Ensure all keys in context are strings for consistent formatting if they originate from various sources
    full_context_data_str_keys = {str(k): v for k, v in full_context_data.items()}

    for step in sorted(protocol.get("steps", []), key=lambda x: x.get("sequence", 0)):
        action_code = step.get("action_code", "UNKNOWN_ACTION")
        description = step.get("description", "No description.")
        action_result = {"action_code": action_code, "description": description, "status": "simulated_success", "details": ""}

        logger.info(f"({module_log_prefix}) ==> Simulating Step {step.get('sequence')}: {action_code} - {description}")

        if "NOTIFY" in action_code.upper():
            contact_method = step.get("contact_method", "LOG_ONLY")
            message_template_code = step.get("message_template_code")
            target_role = step.get("escalation_target_role")
            
            message_content = "No message template specified."
            if message_template_code:
                message_content = format_escalation_message(message_template_code, full_context_data_str_keys)
            
            action_result["details"] = f"Simulated notification via {contact_method}. Target: {target_role or 'N/A'}. Message: '{message_content}'"
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            
            protocols_data_contacts = _get_loaded_protocols().get("contacts", {})
            if target_role and f"{target_role}_PHONE" in protocols_data_contacts:
                 logger.info(f"    L_ Simulated contact to {target_role} using number: {protocols_data_contacts[f'{target_role}_PHONE']}")

        elif "GUIDE" in action_code.upper() or "GUIDANCE" in step.get("guidance_pictogram_code", "").upper():
            pictogram_code = step.get("guidance_pictogram_code", "INFO_ICON")
            action_result["details"] = f"Simulated: Displayed guidance pictogram '{pictogram_code}' and JIT instructions for '{description}'."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        elif "SOS" in action_code.upper():
            action_result["details"] = f"Simulated: SOS function ({action_code}) activated on PED."
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            if "CHW_OWN_CRITICAL_HEAT_STRESS" in event_code:
                # Recursion guard might be needed if protocols can chain indefinitely
                logger.info(f"({module_log_prefix}) Chaining SOS event: CHW_SOS_ACTIVATED_INTERNAL")
                # Avoid direct recursion in simulation for now, or implement a depth limit
                # execute_escalation_protocol("CHW_SOS_ACTIVATED_INTERNAL", full_context_data_str_keys)
        else:
            action_result["details"] = f"Simulated: General action '{action_code}' performed as per protocol description."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        executed_steps_results.append(action_result)

    logger.info(f"({module_log_prefix}) Protocol execution for '{event_code}' finished. {len(executed_steps_results)} steps simulated.")
    return executed_steps_results
