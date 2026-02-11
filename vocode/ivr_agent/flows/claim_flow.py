from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from vocode.ivr_agent.agent.utterance_parser import ParsedUtterance
from vocode.ivr_agent.state.models import CallerType, ConversationState, FlowPhase
from vocode.ivr_agent.state.store import increment_retry, record_confirmation, retries_exceeded


@dataclass
class FlowResponse:
    prompt: str
    is_done: bool = False
    should_transfer: bool = False
    result_payload: Optional[Dict[str, str]] = None


def _field_prompt(field_name: str) -> str:
    if field_name == "member_id":
        return "Please provide the member ID. Numbers only."
    if field_name == "dob":
        return "Please provide the member's date of birth."
    if field_name == "service_date":
        return "Please provide the claim service date."
    if field_name == "caller_type":
        return "Are you the patient or a provider?"
    if field_name == "npi_tax_id":
        return "Please provide the NPI or Tax ID associated with the claim."
    return "Please provide the requested information."


def _confirm_prompt(value: str) -> str:
    return f"You said {value}. Is that correct?"


def _next_field(state: ConversationState) -> Optional[str]:
    if state.current_field == "member_id":
        return "dob"
    if state.current_field == "dob":
        return "service_date"
    if state.current_field == "service_date":
        return "caller_type"
    if state.current_field == "caller_type":
        if state.slots.caller_type == CallerType.PROVIDER:
            return "npi_tax_id"
        return None
    if state.current_field == "npi_tax_id":
        return None
    return None


def _get_candidate(field_name: str, parsed: ParsedUtterance) -> Optional[str]:
    if field_name == "member_id":
        return parsed.member_id
    if field_name == "dob":
        return parsed.dob
    if field_name == "service_date":
        return parsed.service_date
    if field_name == "caller_type":
        if parsed.caller_type == CallerType.UNKNOWN:
            return None
        return parsed.caller_type.value
    if field_name == "npi_tax_id":
        return parsed.npi_tax_id
    return None


def _get_claims(
    member_id: str,
    dob: str,
    service_date: str,
    caller_type: CallerType,
    npi_tax_id: Optional[str],
    dry_run: bool = True,
) -> List[Dict[str, str]]:
    _ = dob, caller_type, npi_tax_id
    if not dry_run:
        return []
    if member_id.endswith("0"):
        return []
    claim_1 = {
        "claim_id": "CLM12345",
        "status": "Processed",
        "payment_status": "Paid",
        "service_date": service_date,
        "amount": "$123.45",
    }
    claim_2 = {
        "claim_id": "CLM98765",
        "status": "Pending",
        "payment_status": "Not paid",
        "service_date": service_date,
        "amount": "$56.78",
    }
    if service_date.endswith("01"):
        return [claim_1, claim_2]
    return [claim_1]


def _format_claims_summary(claims: List[Dict[str, str]]) -> str:
    if not claims:
        return "I couldn't find any records with the information provided."
    if len(claims) == 1:
        claim = claims[0]
        return (
            "I found one claim. "
            f"Claim {claim['claim_id']} is {claim['status']}. "
            f"Payment status is {claim['payment_status']}. "
            f"Service date {claim['service_date']}. Amount {claim['amount']}."
        )
    summaries = []
    for idx, claim in enumerate(claims, start=1):
        summaries.append(
            f"Claim {idx}: {claim['claim_id']} is {claim['status']}, "
            f"payment status {claim['payment_status']}, service date {claim['service_date']}, "
            f"amount {claim['amount']}."
        )
    return "I found multiple claims. " + " ".join(summaries)


def start_claim_flow(state: ConversationState) -> FlowResponse:
    state.phase = FlowPhase.CLAIM_FLOW
    state.current_field = "member_id"
    state.awaiting_confirmation = False
    state.pending_value = None
    return FlowResponse(prompt=_field_prompt(state.current_field))


def handle_claim_flow_input(
    state: ConversationState, parsed: ParsedUtterance, dry_run: bool = True
) -> FlowResponse:
    if state.phase != FlowPhase.CLAIM_FLOW:
        state.phase = FlowPhase.CLAIM_FLOW

    if state.current_field is None:
        state.current_field = "member_id"

    if state.awaiting_confirmation:
        if parsed.confirmation == "YES":
            record_confirmation(state, state.current_field, state.pending_value)
            if state.current_field == "caller_type":
                state.slots.caller_type = CallerType(state.pending_value or "UNKNOWN")
            state.pending_value = None
            state.awaiting_confirmation = False
            state.current_field = _next_field(state)
            if state.current_field:
                return FlowResponse(prompt=_field_prompt(state.current_field))
            claims = _get_claims(
                member_id=state.slots.member_id or "",
                dob=state.slots.dob or "",
                service_date=state.slots.service_date or "",
                caller_type=state.slots.caller_type or CallerType.UNKNOWN,
                npi_tax_id=state.slots.npi_tax_id,
                dry_run=dry_run,
            )
            summary = _format_claims_summary(claims)
            state.last_summary = summary
            state.phase = FlowPhase.RESULT_MENU
            return FlowResponse(prompt=summary, is_done=True)

        # Treat NO/UNKNOWN as missing input (counts toward retry)
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

    candidate = _get_candidate(state.current_field, parsed)
    if not candidate:
        increment_retry(state, state.current_field)
        if retries_exceeded(state, state.current_field):
            state.phase = FlowPhase.TRANSFER
            return FlowResponse(
                prompt="I am transferring you to a representative for further help.",
                should_transfer=True,
            )
        return FlowResponse(prompt=_field_prompt(state.current_field))

    state.pending_value = candidate
    state.awaiting_confirmation = True
    return FlowResponse(prompt=_confirm_prompt(candidate))
