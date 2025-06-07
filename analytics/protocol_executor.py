import logging
import json
import re
from typing import Dict, Any, Optional, List

# --- Module Imports ---
try:
    from config import settings
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Critical import error in protocol_executor.py: {e}. Ensure paths are correct.", exc_info=True)
    raise

# FIXED: Use the correct `__name__` magic variable.
logger = logging.getLogger(__name__)

# --- Globals for Caching ---
_LOADED_ESCALATION_PROTOCOLS: Optional[Dict[str, Any]] = None
_PROTOCOLS_LOADED_SUCCESSFULLY: bool = False


def _load_protocols_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely loads and parses a JSON file from the given path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Escalation protocols file not found at path: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from escalation protocols file at {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
        return None


def _get_loaded_protocols() -> Dict[str, Any]:
    """
    Ensures escalation protocols are loaded once, caching the result.
    Uses a flag to prevent re-attempting load on persistent failure.
    """
    global _LOADED_ESCALATION_PROTOCOLS, _PROTOCOLS_LOADED_SUCCESSFULLY

    # FIXED: This logic is refactored to be a robust, load-once pattern.
    if _LOADED_ESCALATION_PROTOCOLS is None:
        logger.info("First-time access; loading escalation_protocols.json.")
        default_structure = {"protocols": [], "contacts": {}, "message_templates": {}}
        try:
            loaded_data = _load_protocols_from_file(settings.ESCALATION_PROTOCOLS_JSON_PATH)

            if loaded_data and isinstance(loaded_data.get("protocols"), list):
                _LOADED_ESCALATION_PROTOCOLS = loaded_data
                _PROTOCOLS_LOADED_SUCCESSFULLY = True
                logger.info("Escalation protocols loaded and validated successfully.")
            else:
                _LOADED_ESCALATION_PROTOCOLS = default_structure
                _PROTOCOLS_LOADED_SUCCESSFULLY = False
                logger.error("Failed to load or validate escalation protocols. File content may be malformed.")
        except Exception as e:
            _LOADED_ESCALATION_PROTOCOLS = default_structure
            _PROTOCOLS_LOADED_SUCCESSFULLY = False
            logger.error(f"An exception occurred loading escalation protocols: {e}. Escalations will be non-functional.", exc_info=True)

    return _LOADED_ESCALATION_PROTOCOLS


def get_protocol_for_event(event_code: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific escalation protocol based on the event code.
    """
    protocols_data = _get_loaded_protocols()
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
    Placeholders are like [PLACEHOLDER_NAME] (case-insensitive key matching).
    """
    protocols_data = _get_loaded_protocols()
    message_templates = protocols_data.get("message_templates", {})
    if not isinstance(message_templates, dict):
        logger.error("Message templates in protocol data is not a dictionary. Cannot format message.")
        return template_code

    template_string = message_templates.get(template_code)
    if not isinstance(template_string, str):
        logger.warning(f"Message template_code '{template_code}' not found or is not a string. Using code as message.")
        return template_code

    # Create a case-insensitive mapping of stringified context data for robust replacement.
    context_lower_keys = {str(k).lower(): str(v) for k, v in context_data.items()}

    def replace_placeholder(match: re.Match) -> str:
        placeholder = match.group(1).lower()
        return context_lower_keys.get(placeholder, match.group(0))

    formatted_message = re.sub(r"\[([A-Za-z0-9_]+)\]", replace_placeholder, template_string)
    
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
        logger.warning(f"({module_log_prefix}) No protocol for event '{event_code}'. Escalation aborted.")
        return [{"action_code": "NO_PROTOCOL_FOUND", "status": "failed", "details": f"No protocol for {event_code}"}]

    executed_steps_results: List[Dict[str, Any]] = []

    # FIXED: Removed redundant key stringification. `format_escalation_message` handles it.
    full_context_data = triggering_data.copy()
    if isinstance(additional_context, dict):
        full_context_data.update(additional_context)

    protocol_steps = protocol.get("steps", [])
    if not isinstance(protocol_steps, list):
        logger.error(f"({module_log_prefix}) Protocol '{event_code}' has malformed 'steps' (not a list).")
        return [{"action_code": "MALFORMED_PROTOCOL_STEPS", "status": "failed", "details": f"Steps not a list for {event_code}"}]

    for step in sorted(protocol_steps, key=lambda s: s.get("sequence", 0) if isinstance(s, dict) else 0):
        if not isinstance(step, dict):
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
            
            message_content = format_escalation_message(message_template_code, full_context_data) if message_template_code else "No message template specified."
            
            action_result["details"] = f"Simulated notification via {contact_method}. Target: {target_role or 'N/A'}. Message: '{message_content}'"
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            
        elif "GUIDE" in action_code.upper() or "GUIDANCE" in str(step.get("guidance_pictogram_code", "")).upper():
            pictogram_code = step.get("guidance_pictogram_code", "INFO_ICON")
            action_result["details"] = f"Simulated: Displayed guidance pictogram '{pictogram_code}' and instructions for '{description}'."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        elif "SOS" in action_code.upper():
            action_result["details"] = f"Simulated: SOS function ({action_code}) activated."
            logger.info(f"    L_ Action Detail: {action_result['details']}")
        
        else:
            action_result["details"] = f"Simulated: General action '{action_code}' performed."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        executed_steps_results.append(action_result)

    logger.info(f"({module_log_prefix}) Protocol execution for '{event_code}' finished. {len(executed_steps_results)} steps simulated.")
    return executed_steps_results
