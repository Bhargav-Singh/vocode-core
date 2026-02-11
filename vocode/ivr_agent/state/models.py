from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Intent(str, Enum):
    CLAIM_STATUS = "CLAIM_STATUS"
    AUTH_STATUS = "AUTH_STATUS"
    ELIGIBILITY = "ELIGIBILITY"
    UNKNOWN = "UNKNOWN"
    TEST_FLOW = "TEST_FLOW"


class CallerType(str, Enum):
    PATIENT = "PATIENT"
    PROVIDER = "PROVIDER"
    UNKNOWN = "UNKNOWN"


class FlowPhase(str, Enum):
    CALL_START = "CALL_START"
    INTENT_CAPTURE = "INTENT_CAPTURE"
    CLAIM_FLOW = "CLAIM_FLOW"
    AUTH_FLOW = "AUTH_FLOW"
    ELIGIBILITY_FLOW = "ELIGIBILITY_FLOW"
    TEST_FLOW = "TEST_FLOW"
    RESULT_MENU = "RESULT_MENU"
    TRANSFER = "TRANSFER"
    END = "END"


@dataclass
class SlotState:
    # Shared slots
    member_id: Optional[str] = None
    dob: Optional[str] = None
    service_date: Optional[str] = None
    caller_type: Optional[CallerType] = None
    npi_tax_id: Optional[str] = None

    # Test flow slots
    full_name: Optional[str] = None
    birthdate: Optional[str] = None


@dataclass
class ConversationState:
    phase: FlowPhase = FlowPhase.CALL_START
    intent: Intent = Intent.UNKNOWN
    slots: SlotState = field(default_factory=SlotState)
    retries: Dict[str, int] = field(default_factory=dict)
    last_prompt: Optional[str] = None
    last_summary: Optional[str] = None
    last_confirmed_field: Optional[str] = None
    last_user_utterance: Optional[str] = None
    current_field: Optional[str] = None
    pending_value: Optional[str] = None
    awaiting_confirmation: bool = False

    def reset_slots(self) -> None:
        self.slots = SlotState()

    def reset_retries(self) -> None:
        self.retries = {}
