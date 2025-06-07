# sentinel_project_root/analytics/protocol_executor.py
# Handles parsing and execution logic for escalation protocols.

import logging
import json
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# --- Core Imports ---
try:
    from config import settings
    from data_processing.helpers import robust_json_load
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in protocol_executor.py: {e}. Check project structure.")
    raise

logger = logging.getLogger(__name__)

# --- Module-level Caching for Protocols ---
# This simple in-memory cache prevents re-reading the file on every call
# and is framework-agnostic (i.e., does not depend on Streamlit).
_LOADED_PROTOCOLS: Optional[Dict[str, Any]] = None
_PROTOCOLS_LOADED_SUCCESSFULLY: bool = False


def _get_protocols() -> Dict[str, Any]:
    """
    Ensures escalation protocols are loaded once from file, validating the structure.
    Uses a module-level cache to avoid repeated file I/O.
    """
    global _LOADED_PROTOCOLS, _PROTOCOLS_LOADED_SUCCESSFULLY
    
    # Only load if the cache is empty and the last attempt wasn't a failure.
    if _LOADED_PROTOCOLS is None and not _PROTOCOLS_LOADED_SUCCESSFULLY:
        logger.info("First-time access: loading escalation protocols from file.")
        default_structure = {"protocols": [], "contacts": {}, "message_templates": {}}
        
        try:
            protocol_path = getattr(settings.Core, 'ESCALATION_PROTOCOLS_JSON_PATH', None)
            if not protocol_path:
                raise FileNotFoundError("ESCALATION_PROTOCOLS_JSON_PATH not configured in settings.")
            
            loaded_data = robust_json_load(Path(protocol_path))

            # Validate the structure of the loaded data
            if (isinstance(loaded_data, dict) and 
                isinstance(loaded_data.get("protocols"), list) and
                isinstance(loaded_data.get("contacts"), dict) and
                isinstance(loaded_data.get("message_templates"), dict)):
                
                _LOADED_PROTOCOLS = loaded_data
                _PROTOCOLS_LOADED_SUCCESSFULLY = True
                logger.info(f"Escalation protocols loaded successfully with {len(loaded_data['protocols'])} protocols.")
            else:
                _LOADED_PROTOCOLS = default_structure
                logger.error("Failed to validate escalation protocols: content missing or malformed.")
        
        except Exception as e:
            _LOADED_PROTOCOLS = default_structure
            _PROTOCOLS_LOADED_SUCCESSFULLY = False # Mark as a failed attempt
            logger.error(f"Exception loading escalation protocols: {e}. Escalations will be non-functional.", exc_info=True)
            
    return _LOADED_PROTOCOLS or {"protocols": [], "contacts": {}, "message_templates": {}}


def get_protocol_for_event(event_code: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific escalation protocol based on the event code."""
    protocols_data = _get_protocols()
    for protocol in protocols_data.get("protocols", []):
        if isinstance(protocol, dict) and protocol.get("trigger_event_code") == event_code:
            return protocol
    logger.warning(f"No escalation protocol found for event_code: {event_code}")
    return None


def format_escalation_message(template_code: str, context_data: Dict[str, Any]) -> str:
    """
    Formats an escalation message using a template and context data.
    Placeholders like [PLACEHOLDER_NAME] are replaced case-insensitively.
    """
    protocols_data = _get_protocols()
    template_string = protocols_data.get("message_templates", {}).get(template_code)

    if not isinstance(template_string, str):
        logger.warning(f"Message template '{template_code}' not found or is not a string. Using code as message.")
        return template_code

    # Create a case-insensitive mapping of context data for robust replacement
    context_lower_keys = {str(k).lower(): str(v) for k, v in context_data.items()}

    def replace_placeholder(match: re.Match) -> str:
        placeholder = match.group(1).lower()
        return context_lower_keys.get(placeholder, match.group(0))

    # Regex to find placeholders like [KEY_NAME]
    return re.sub(r"\[([A-Za-z0-9_]+)\]", replace_placeholder, template_string)


def execute_escalation_protocol(
    event_code: str,
    triggering_data: Dict[str, Any],
    additional_context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Executes the steps defined in an escalation protocol for a given event.
    This is a simulation; actual actions (e.g., SMS, API calls) are logged.
    """
    log_prefix = "EscalationExecutor"
    logger.info(f"({log_prefix}) Executing escalation for event: '{event_code}'")

    protocol = get_protocol_for_event(event_code)
    if not protocol:
        return [{"action_code": "NO_PROTOCOL_FOUND", "status": "failed", "details": f"No protocol for {event_code}"}]

    # Combine context data, with additional_context overriding triggering_data on key collision
    full_context = {**triggering_data, **(additional_context or {})}
    
    executed_steps: List[Dict[str, Any]] = []
    protocol_steps = sorted(protocol.get("steps", []), key=lambda s: s.get("sequence", 0))

    for step in protocol_steps:
        if not isinstance(step, dict):
            logger.warning(f"({log_prefix}) Skipping malformed step in protocol '{event_code}': {step}")
            continue

        action_code = step.get("action_code", "UNKNOWN_ACTION")
        description = step.get("description", "No description provided.")
        action_result = {"action_code": action_code, "description": description, "status": "simulated_success", "details": ""}

        logger.info(f"({log_prefix}) ==> Simulating Step {step.get('sequence')}: {action_code}")

        if "NOTIFY" in action_code:
            contact_method = step.get("contact_method", "LOG_ONLY").upper()
            template_code = step.get("message_template_code")
            target_role = step.get("escalation_target_role")
            
            message = format_escalation_message(template_code, full_context) if template_code else "No message template specified."
            
            action_result["details"] = f"Simulated '{contact_method}' to '{target_role}'. Message: '{message}'"
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        elif "GUIDE" in action_code:
            pictogram = step.get("guidance_pictogram_code", "INFO_ICON")
            action_result["details"] = f"Simulated: Displayed guidance pictogram '{pictogram}' and instructions for '{description}'."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        else: # Default handling for other action codes
            action_result["details"] = f"Simulated: General action '{action_code}' performed as per description."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        executed_steps.append(action_result)

    logger.info(f"({log_prefix}) Protocol '{event_code}' finished with {len(executed_steps)} simulated steps.")
    return executed_steps
