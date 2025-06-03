# sentinel_project_root/analytics/protocol_executor.py
# Handles parsing and execution logic for escalation protocols.

import logging
import json
from typing import Dict, Any, Optional, List

from config import settings # Use new settings module
from data_processing.loaders import load_escalation_protocols # To load protocols if not already loaded

logger = logging.getLogger(__name__)

# Global variable to store loaded protocols to avoid reloading from file repeatedly
_LOADED_ESCALATION_PROTOCOLS: Optional[Dict[str, Any]] = None

def _get_loaded_protocols() -> Dict[str, Any]:
    """Ensures escalation protocols are loaded, loading them if necessary."""
    global _LOADED_ESCALATION_PROTOCOLS
    if _LOADED_ESCALATION_PROTOCOLS is None:
        logger.info("First-time access or protocols not loaded; loading escalation_protocols.json.")
        _LOADED_ESCALATION_PROTOCOLS = load_escalation_protocols() # Uses path from settings
        if not _LOADED_ESCALATION_PROTOCOLS or not _LOADED_ESCALATION_PROTOCOLS.get("protocols"):
             logger.error("Failed to load or validate escalation protocols. Escalations will not function.")
             _LOADED_ESCALATION_PROTOCOLS = {"protocols": [], "contacts": {}, "message_templates": {}} # Safe default
    return _LOADED_ESCALATION_PROTOCOLS

def get_protocol_for_event(event_code: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific escalation protocol based on the event code.

    Args:
        event_code: The code identifying the trigger event (e.g., "PATIENT_CRITICAL_SPO2_LOW").

    Returns:
        The protocol dictionary if found, else None.
    """
    protocols_data = _get_loaded_protocols()
    if not protocols_data.get("protocols"):
        return None
        
    for protocol in protocols_data["protocols"]:
        if protocol.get("trigger_event_code") == event_code:
            return protocol
    logger.warning(f"No escalation protocol found for event_code: {event_code}")
    return None

def format_escalation_message(template_code: str, context_data: Dict[str, Any]) -> str:
    """
    Formats an escalation message using a template and context data.

    Args:
        template_code: The code for the message template (e.g., "MSG_CRIT_SPO2_SUP").
        context_data: A dictionary containing data to fill into the template placeholders.
                      Placeholders are like [PLACEHOLDER_NAME].

    Returns:
        The formatted message string, or the template_code itself if not found.
    """
    protocols_data = _get_loaded_protocols()
    message_templates = protocols_data.get("message_templates", {})
    
    template_string = message_templates.get(template_code)
    if not template_string:
        logger.warning(f"Message template_code '{template_code}' not found. Using code as message.")
        return template_code

    formatted_message = template_string
    for placeholder, value in context_data.items():
        # Ensure placeholder format matches (e.g., [PLACEHOLDER])
        formatted_message = formatted_message.replace(f"[{placeholder.upper()}]", str(value))
    
    # Log if some placeholders were not filled, which might indicate missing context_data
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

    Args:
        event_code: The code of the event that triggered the escalation.
        triggering_data: Data from the event source (e.g., patient record, sensor reading).
        additional_context: Optional extra data to merge into context for message formatting.

    Returns:
        A list of simulated action results (dictionaries).
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
    full_context_data = triggering_data.copy()
    if additional_context:
        full_context_data.update(additional_context)


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
                message_content = format_escalation_message(message_template_code, full_context_data)
            
            action_result["details"] = f"Simulated notification via {contact_method}. Target: {target_role or 'N/A'}. Message: '{message_content}'"
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            
            # Example of using contacts from protocols_data if needed for simulation detail
            protocols_data = _get_loaded_protocols()
            contacts_info = protocols_data.get("contacts", {})
            if target_role and f"{target_role}_PHONE" in contacts_info: # Example contact key format
                 logger.info(f"    L_ Simulated contact to {target_role} using number: {contacts_info[f'{target_role}_PHONE']}")


        elif "GUIDE" in action_code.upper() or "GUIDANCE" in step.get("guidance_pictogram_code", "").upper():
            pictogram_code = step.get("guidance_pictogram_code", "INFO_ICON")
            action_result["details"] = f"Simulated: Displayed guidance pictogram '{pictogram_code}' and JIT instructions for '{description}'."
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            # In a real PED, this would trigger UI change based on pictogram_code

        elif "SOS" in action_code.upper():
            action_result["details"] = f"Simulated: SOS function ({action_code}) activated on PED."
            logger.info(f"    L_ Action Detail: {action_result['details']}")
            if "CHW_OWN_CRITICAL_HEAT_STRESS" in event_code: # Specific context for CHW SOS
                execute_escalation_protocol("CHW_SOS_ACTIVATED_INTERNAL", full_context_data) # Chain to another protocol if needed


        # Add more simulated actions here based on action_codes
        else:
            action_result["details"] = f"Simulated: General action '{action_code}' performed as per protocol description."
            logger.info(f"    L_ Action Detail: {action_result['details']}")

        executed_steps_results.append(action_result)

    logger.info(f"({module_log_prefix}) Protocol execution for '{event_code}' finished. {len(executed_steps_results)} steps simulated.")
    return executed_steps_results

# Example of how this might be triggered (e.g., from RiskPredictionModel or an alerting function)
# if __name__ == '__main__':
#     # This is for testing the executor module directly
#     logging.basicConfig(level=logging.INFO, format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)
#     test_patient_data = {
#         "PATIENT_ID": "SPID_TEST_007", "ZONE_ID": "ZoneC", "GPS_COORDS_OR_LANDMARK": "Near Market Square",
#         "SPO2_VALUE": 88, "CHW_ID": "CHW003", "CHW_PHONE_NUMBER": "+15551237777", "CHW_OBSERVED_ACTION_TAKEN": "Positioned patient"
#     }
#     results = execute_escalation_protocol("PATIENT_CRITICAL_SPO2_LOW", test_patient_data)
#     print("\n--- Protocol Execution Results ---")
#     for res_step in results:
#         print(f"  - Action: {res_step['action_code']}, Status: {res_step['status']}, Details: {res_step['details']}")

#     test_chw_data = {
#         "CHW_ID": "CHW004", "CURRENT_TEMP_READING": 39.8, "SYMPTOMS": "Confusion, dizziness"
#     }
#     results_chw_heat = execute_escalation_protocol("CHW_OWN_CRITICAL_HEAT_STRESS", test_chw_data)
#     print("\n--- CHW Heat Stress Protocol Results ---")
#     for res_step_chw in results_chw_heat:
#         print(f"  - Action: {res_step_chw['action_code']}, Status: {res_step_chw['status']}, Details: {res_step_chw['details']}")
