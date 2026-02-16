from typing import TypedDict, Optional, Literal, Dict

class IVRState(TypedDict):
    # --- Control Flags ---
    # The text the TTS engine should speak
    message_to_play: str
    # How the system should listen: "varied" (speech), "single" (DTMF/short), "none" (hangup/processing)
    input_type: Literal["varied", "single", "none"]
    # The raw input captured from the user
    last_user_input: Optional[str]
    # Current active intent (CLAIM, AUTH, ELIGIBILITY, TRANSFER)
    intent: Optional[str]
    # Status to help Parent Graph decide next step after a Subgraph ends
    dialogue_status: Literal["active", "transfer", "main_menu", "complete"]

    # NEW: Tracks which flow is active (CLAIM, AUTH, etc.) - Preserved during reset
    flow_name: Optional[str]

    # --- Shared Business Slots ---
    member_id: Optional[str]
    dob: Optional[str]

    # --- Auth Flow Specific Slots ---
    npi: Optional[str]
    user_role: Optional[Literal["PATIENT", "PROVIDER"]]

    # --- Temporary Slots (For Confirmation Loops) ---
    temp_member_id: Optional[str]
    temp_dob: Optional[str]
    temp_npi: Optional[str]

    # --- RETRY TRACKER ---
    # Stores counts like {"member_id": 1, "dob": 2}
    retries: Optional[Dict[str, int]]