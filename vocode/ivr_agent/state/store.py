from __future__ import annotations

from typing import Optional

from vocode.ivr_agent.state.models import ConversationState, FlowPhase, Intent


DEFAULT_RETRY_LIMIT = 3


def new_state() -> ConversationState:
    return ConversationState()


def reset_for_main_menu(state: ConversationState) -> None:
    state.reset_slots()
    state.reset_retries()
    state.intent = Intent.UNKNOWN
    state.phase = FlowPhase.INTENT_CAPTURE
    state.last_confirmed_field = None
    state.last_prompt = None
    state.last_summary = None
    state.current_field = None
    state.pending_value = None
    state.awaiting_confirmation = False


def set_intent(state: ConversationState, intent: Intent) -> None:
    state.intent = intent
    if intent == Intent.CLAIM_STATUS:
        state.phase = FlowPhase.CLAIM_FLOW
    elif intent == Intent.AUTH_STATUS:
        state.phase = FlowPhase.AUTH_FLOW
    elif intent == Intent.ELIGIBILITY:
        state.phase = FlowPhase.ELIGIBILITY_FLOW
    elif intent == Intent.TEST_FLOW:
        state.phase = FlowPhase.TEST_FLOW
    else:
        state.phase = FlowPhase.INTENT_CAPTURE


def increment_retry(state: ConversationState, field_name: str) -> int:
    state.retries[field_name] = state.retries.get(field_name, 0) + 1
    return state.retries[field_name]


def retries_exceeded(
    state: ConversationState, field_name: str, limit: int = DEFAULT_RETRY_LIMIT
) -> bool:
    return state.retries.get(field_name, 0) >= limit


def record_confirmation(state: ConversationState, field_name: str, value: Optional[str]) -> None:
    setattr(state.slots, field_name, value)
    state.last_confirmed_field = field_name
