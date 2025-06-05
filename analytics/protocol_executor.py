# sentinel_project_root/analytics/protocol_executor.py
# Handles parsing and execution logic for escalation protocols.

import logging
import json
import re # For robust placeholder replacement in messages
from typing import Dict, Any, Optional, List

from config import settings
from data_processing.loaders import load_escalation_protocols # This import is problematic if loaders.py uses st.cache_data

logger = logging.getLogger(__name__)

_LOADED_ESCALATION_PROTOCOLS: Optional[Dict[str, Any]] = None
_PROTOCOLS_LOADED_SUCCESSFULLY: bool = False


def _get_loaded_protocols() -> Dict[str, Any]:
    """
    Ensures escalation protocols are loaded once, loading them if necessary.
    Uses a flag to prevent re-attempting load on failure.
    """
    global _LOADED_ESCALATION_PROTOCOLS, _PROTOCOLS_LOADED_SUCCESSFULLY
    
    if _LOADED_ESCALATION_PROTOCOLS is None and not _PROTOCOLS_LOADED_SUCCESSFULLY: # Only attempt load if not already tried or failed
        logger.info("First-time access or protocols not loaded; loading escalation_protocols.json.")
        try:
            # Critical: `load_escalation_protocols` from `loaders.py` uses `st.cache_data`.
            # This analytics module should ideally not depend on Streamlit specific caching.
            # For now, assuming it works or a non-Streamlit version of load_escalation_protocols is available.
            # If this becomes a problem, `load_escalation_protocols` needs to be refactored
            # to not use Streamlit caching, or protocols loaded differently here.
            # For this fix, assuming `robust_json_load` can be used directly if `loaders` is an issue.
            # loaded_data = robust_json_load(settings.ESCALATION_PROTOCOLS_JSON_PATH) # Alternative direct load
            loaded_data = load_escalation_protocols(settings.ESCALATION_PROTOCOLS_JSON_PATH) # Current structure

            if loaded_data and isinstance(loaded_data.get("protocols"), list):
                _LOADED_ESCALATION_PROTOCOLS = loaded_data
                _PROTOCOLS_LOADED_SUCCESSFULLY = True
                logger.info("Escalation protocols loaded successfully.")
            else:
                _LOADED_ESCALATION_PROTOCOLS = {"protocols": [], "contacts": {}, "message_templates": {}}
                _PROTOCOLS_LOADED_SUCCESSFULLY = False # Mark as failed load attempt
                logger.error("Failed to load or validate escalation protocols. Content missing or malformed. Escalations may not function.")
        except Exception as e:
            _LOADED_ESCALATION_PROTOCOLS = {"protocols": [], "contacts": {}, "message_templates": {}}
            _PROTOCOLS_LOADED_SUCCESSFULLY = False
            logger.error(f"Exception loading escalation protocols: {e}. Escalations will be non-functional.", exc_info=True)
    
    # Return the current state of loaded protocols (could be the empty default on failure)
    return _LOADED_ESCALATION_PROTOCOLS if _LOADED_ESCALATION_PROTOCOLS is not None else \
           {"protocols": [], "contacts": {}, "message_templates": {}}


def get_protocol_for_event(event_code: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific escalation protocol based on the event code.
    """
    protocols_data = _get_loaded_protocols()
    # Ensure 'protocols' key exists and is a list
    if not isinstance(protocols_data.get("protocols"), list):
        logger.error(f"Protocol data is malformed: 'protocols' key missing or not a list. Cannot find protocol for {event_code}.")
        return None
        
    for protocol in protocols_data["protocols"]:
        if isinstance(protocol, dict) and protocol.get("trigger_event_code") == event_code:
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
    
    if not isinstance(message_templates, dict): # Ensure message_templates is a dict
        logger.error("Message templates in protocol data is not a dictionary. Cannot format message.")
        return template_code # Fallback

    template_string = message_templates.get(template_code)
    if not isinstance(template_string, str): # Ensure template_string is actually a string
        logger.warning(f"Message template_code '{template_code}' not found or is not a string. Using code as message.")
        return template_code

    formatted_message = template_string
    
    # Create a case-insensitive mapping for context_data keys for robust placeholder replacement
    # Also ensure all values in context_data are stringified for replacement.
    context_data_lower_keys_str_vals = {str(k).lower(): str(v) for k, v in context_data.items()}

    def replace_placeholder(match_obj: re.Match) -> str:
        placeholder_key_in_brackets = match_obj.group(1) # Key inside brackets, e.g., "PATIENT_ID"
        return context_data_lower_keys_str_vals.get(placeholder_key_in_brackets.lower(), match_obj.group(0)) # Replace or keep original

    # Regex to find placeholders like [KEY_NAME]
    formatted_message = re.sub(r"\[([A-Za-z0-9_]+)\]", replace_placeholder, formatted_message)
    
    if "[" in formatted_message and "]" in formatted_message: # Basic check for unreplaced placeholders after substitution
        logger.debug(f"Message template '{template_code}' may have unreplaced placeholders after formatting: '{formatted_message[:200]}...'") # Log snippet
        
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
    
    # Prepare combined context for message formatting
    # Prioritize additional_context for overrides if keys overlap with triggering_data
    full_context_data = triggering_data.copy() # Start with a copy
    if isinstance(additional_context, dict): # Ensure additional_context is a dict
        full_context_data.update(additional_context)

    # Ensure all keys in the combined context are strings for consistent formatting by format_escalation_message
    full_context_data_str_keys_final = {str(k): v for k, v in full_context_data.items()}

    protocol_steps = protocol.get("steps", [])
    if not isinstance(protocol_steps, list): # Ensure steps is a list
        logger.error(f"({module_log_prefix}) Protocol '{event_code}' has malformed 'steps' (not a list). Escalation aborted.")
        return [{"action_code": "MALFORMED_PROTOCOL_STEPS", "status": "failed", "details": f"Steps not a list for {event_code}"}]

    for step in sorted(protocol_steps, key=lambda x_step: x_step.get("sequence", 0) if isinstance(x_step, dict) else 0):
        if not isinstance(step, dict): # Skip malformed steps
            logger.warning(f"({module_log_prefix}) Skipping malformed step in protocol '{event_code}': {step}")
            continue

        action_code = step.get("action_code", "UNKNOWN_ACTION")
        description = step.get("description", "No description.")
        action_result: Dict[str, Any] = {"action_code": action_code, "description": description, "status": "simulated_success", "details": ""}

        logger.info(f"({module_log_prefix}) ==> Simulating Step {step.get('sequence')}: {action_code} - {description}")

        if "NOTIFY" in action_code.upper():
            contact_method = step.get("contact_method", "LOG_ONLY")
            message_template_code = step.get("message_template_code")
            target_role = step.get("escalation_target_role")
            
            message_content = "No message template specified or template code missing."
            if message_template_code and isinstance(message_template_code, str):
                message_content = format_escalation_message(message_template_code, full_context_data_str_keys_final)
            
            action_result["details"] = f"Simulated notification via {contact_method}. Target: {target_role or 'N/A'}. Message: '{message_content}'"
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            
            protocols_data_loaded = _get_loaded_protocols() # Re-fetch safely
            contacts_info = protocols_data_loaded.get("contacts", {})
            if isinstance(contacts_info, dict) and target_role and f"{target_role}_PHONE" in contacts_info:
                 logger.info(f"    L_ Simulated contact to {target_role} using number: {contacts_info[f'{target_role}_PHONE']}")

        elif "GUIDE" in action_code.upper() or "GUIDANCE" in str(step.get("guidance_pictogram_code", "")).upper():
            pictogram_code = step.get("guidance_pictogram_code", "INFO_ICON")
            action_result["details"] = f"Simulated: Displayed guidance pictogram '{pictogram_code}' and JIT instructions for '{description}'."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        elif "SOS" in action_code.upper():
            action_result["details"] = f"Simulated: SOS function ({action_code}) activated on PED."
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            # Example of chained protocol execution - consider recursion depth or alternative handling for complex chains.
            if "CHW_OWN_CRITICAL_HEAT_STRESS" in event_code: 
                logger.info(f"({module_log_prefix}) Event '{event_code}' might trigger a chained SOS protocol. For simulation, this secondary trigger is noted but not auto-executed here to prevent loops.")
                # execute_escalation_protocol("CHW_SOS_ACTIVATED_INTERNAL", full_context_data_str_keys_final) # Example of chaining if desired

        else: # Default handling for other action codes
            action_result["details"] = f"Simulated: General action '{action_code}' performed as per protocol description."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        executed_steps_results.append(action_result)

    logger.info(f"({module_log_prefix}) Protocol execution for '{event_code}' finished. {len(executed_steps_results)} steps simulated.")
    return executed_steps_results
