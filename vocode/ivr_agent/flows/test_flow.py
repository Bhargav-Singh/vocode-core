from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from vocode.ivr_agent.state.models import ConversationState, FlowPhase
from vocode.ivr_agent.state.store import increment_retry, record_confirmation, retries_exceeded


AFFIRMATIVE = ("yes", "yep", "yeah", "correct", "right", "sure")
NEGATIVE = ("no", "nope", "incorrect", "wrong")


@dataclass
class FlowResponse:
    prompt: str
    is_done: bool = False
    should_transfer: bool = False
    result_payload: Optional[Dict[str, int]] = None


def _is_affirmative(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    tokens = re.findall(r"[a-z']+", cleaned)
    return any(token in tokens for token in AFFIRMATIVE)


def _is_negative(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    tokens = re.findall(r"[a-z']+", cleaned)
    return any(token in tokens for token in NEGATIVE)


def _validate_full_name(value: str) -> bool:
    return len(value.strip().split()) >= 2


def _validate_birthdate(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", value.strip()))


def _field_prompt(field_name: str) -> str:
    if field_name == "full_name":
        return "Please tell me your full name."
    if field_name == "birthdate":
        return "Please provide your date of birth in MM/DD/YYYY format."
    return "Please provide the requested information."


def _confirm_prompt(value: str) -> str:
    return f"You said {value}. Is that correct?"


def _get_marks(full_name: str, birthdate: str, dry_run: bool = True) -> Dict[str, int]:
    # Dry run for now; return static data for testing.
    if dry_run:
        return {"math": 88, "english": 92, "science": 85}
    # Placeholder for future real data source.
    return {"math": 0, "english": 0, "science": 0}


def start_test_flow(state: ConversationState) -> FlowResponse:
    state.phase = FlowPhase.TEST_FLOW
    state.current_field = "full_name"
    state.awaiting_confirmation = False
    state.pending_value = None
    return FlowResponse(prompt=_field_prompt(state.current_field))


def handle_test_flow_input(
    state: ConversationState, user_text: str, dry_run: bool = True
) -> FlowResponse:
    if state.phase != FlowPhase.TEST_FLOW:
        state.phase = FlowPhase.TEST_FLOW

    if state.current_field is None:
        state.current_field = "full_name"

    if state.awaiting_confirmation:
        if _is_affirmative(user_text):
            record_confirmation(state, state.current_field, state.pending_value)
            state.pending_value = None
            state.awaiting_confirmation = False
            if state.current_field == "full_name":
                state.current_field = "birthdate"
                return FlowResponse(prompt=_field_prompt(state.current_field))
            marks = _get_marks(
                full_name=state.slots.full_name or "",
                birthdate=state.slots.birthdate or "",
                dry_run=dry_run,
            )
            state.phase = FlowPhase.RESULT_MENU
            summary = (
                "Thanks. Your marks are: "
                f"Math {marks['math']}, English {marks['english']}, Science {marks['science']}."
            )
            state.last_summary = summary
            return FlowResponse(prompt=summary, is_done=True, result_payload=marks)

        if _is_negative(user_text):
            increment_retry(state, state.current_field)
            if retries_exceeded(state, state.current_field):
                state.phase = FlowPhase.TRANSFER
                return FlowResponse(
                    prompt="I am transferring you to a representative for further help.",
                    should_transfer=True,
                )
            state.awaiting_confirmation = False
            state.pending_value = None
            return FlowResponse(prompt=_field_prompt(state.current_field))

        # Unknown confirmation response
        increment_retry(state, state.current_field)
        if retries_exceeded(state, state.current_field):
            state.phase = FlowPhase.TRANSFER
            return FlowResponse(
                prompt="I am transferring you to a representative for further help.",
                should_transfer=True,
            )
        return FlowResponse(prompt=_confirm_prompt(state.pending_value or "that"))

    # Collect value for current field
    candidate = (user_text or "").strip()
    if not candidate:
        increment_retry(state, state.current_field)
        if retries_exceeded(state, state.current_field):
            state.phase = FlowPhase.TRANSFER
            return FlowResponse(
                prompt="I am transferring you to a representative for further help.",
                should_transfer=True,
            )
        return FlowResponse(prompt=_field_prompt(state.current_field))

    if state.current_field == "full_name" and not _validate_full_name(candidate):
        increment_retry(state, state.current_field)
        if retries_exceeded(state, state.current_field):
            state.phase = FlowPhase.TRANSFER
            return FlowResponse(
                prompt="I am transferring you to a representative for further help.",
                should_transfer=True,
            )
        return FlowResponse(prompt="Please provide both first and last name.")

    if state.current_field == "birthdate" and not _validate_birthdate(candidate):
        increment_retry(state, state.current_field)
        if retries_exceeded(state, state.current_field):
            state.phase = FlowPhase.TRANSFER
            return FlowResponse(
                prompt="I am transferring you to a representative for further help.",
                should_transfer=True,
            )
        return FlowResponse(prompt="Please use MM/DD/YYYY format, for example 01/31/1990.")

    state.pending_value = candidate
    state.awaiting_confirmation = True
    return FlowResponse(prompt=_confirm_prompt(candidate))
