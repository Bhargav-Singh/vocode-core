# utilities/state_utils.py
from typing import Any, Dict

def get_reset_state_update(status: str) -> Dict[str, Any]:
    """
    Returns a state update that clears all business data and temp slots,
    but PRESERVES 'flow_name' and sets the new 'dialogue_status'.
    """
    return {
        "dialogue_status": status,
        
        # Reset Logic: Explicitly set to None/Empty
        "member_id": None,
        "dob": None,
        "npi": None,
        "user_role": None,
        
        "temp_member_id": None,
        "temp_dob": None,
        "temp_npi": None,
        
        "retries": {}, # Reset retry counter
        "last_user_input": None, # Clear last input
        "intent": None,
        
        # NOTE: 'flow_name' is NOT included here, so LangGraph will keep its current value.
    }