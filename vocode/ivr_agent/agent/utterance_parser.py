from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from vocode.ivr_agent.state.models import CallerType, FlowPhase, Intent


PARSER_SYSTEM_PROMPT = (
    "You classify and extract structured data from a call-center utterance. "
    "Return ONLY JSON with these keys:\n"
    "- intent: CLAIM_STATUS | AUTH_STATUS | ELIGIBILITY | TEST_FLOW | UNKNOWN\n"
    "- command: START_OVER | MAIN_MENU | TRANSFER | REPEAT | NONE\n"
    "- confirmation: YES | NO | UNKNOWN\n"
    "- caller_type: PATIENT | PROVIDER | UNKNOWN\n"
    "- member_id: string or null\n"
    "- dob: MM/DD/YYYY or null\n"
    "- service_date: MM/DD/YYYY or null\n"
    "- npi_tax_id: string or null\n"
    "- full_name: string or null\n"
    "- birthdate: MM/DD/YYYY or null\n"
    "- result_action: REPEAT | ANOTHER | MAIN_MENU | TRANSFER | NONE\n"
    "Use null when a field is not provided. Only set result_action when the user "
    "is responding to a post-result menu. Convert dates to MM/DD/YYYY when possible."
)


@dataclass
class ParsedUtterance:
    intent: Intent = Intent.UNKNOWN
    command: str = "NONE"
    confirmation: str = "UNKNOWN"
    caller_type: CallerType = CallerType.UNKNOWN
    member_id: Optional[str] = None
    dob: Optional[str] = None
    service_date: Optional[str] = None
    npi_tax_id: Optional[str] = None
    full_name: Optional[str] = None
    birthdate: Optional[str] = None
    result_action: str = "NONE"


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {}


def _normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip()
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", cleaned):
        return cleaned
    return None


def _normalize_digits(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    return digits or None


def _extract_date(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if not match:
        return None
    month, day, year = match.groups()
    if len(year) != 4:
        return None
    return f"{int(month):02d}/{int(day):02d}/{year}"


def _extract_digits(text: str, min_len: int = 4) -> Optional[str]:
    match = re.search(rf"\b\d{{{min_len},}}\b", text)
    return match.group(0) if match else None


def _fallback_yes_no(text: str) -> str:
    cleaned = text.strip().lower()
    tokens = re.findall(r"[a-z']+", cleaned)
    if any(t in ("yes", "yep", "yeah", "correct", "right", "sure") for t in tokens):
        return "YES"
    if any(t in ("no", "nope", "incorrect", "wrong") for t in tokens):
        return "NO"
    return "UNKNOWN"


def _fallback_command(text: str) -> str:
    lowered = text.lower()
    if "start over" in lowered:
        return "START_OVER"
    if "main menu" in lowered:
        return "MAIN_MENU"
    if "agent" in lowered or "representative" in lowered:
        return "TRANSFER"
    if "repeat" in lowered or "say it again" in lowered or "again" in lowered:
        return "REPEAT"
    return "NONE"


def _fallback_result_action(text: str) -> str:
    lowered = text.lower()
    if "repeat" in lowered or "say it again" in lowered or "again" in lowered:
        return "REPEAT"
    if "another" in lowered or "different" in lowered or "new" in lowered:
        return "ANOTHER"
    if "main menu" in lowered or "start over" in lowered:
        return "MAIN_MENU"
    if "agent" in lowered or "representative" in lowered:
        return "TRANSFER"
    return "NONE"


class LLMUtteranceParser:
    def __init__(self, client: AsyncOpenAI | AsyncAzureOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    async def parse(
        self,
        text: str,
        phase: FlowPhase,
        current_field: Optional[str] = None,
        awaiting_confirmation: bool = False,
    ) -> ParsedUtterance:
        user_context = (
            f"Phase: {phase.value}\n"
            f"Current field: {current_field or 'none'}\n"
            f"Awaiting confirmation: {awaiting_confirmation}\n"
            f"User: {text}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=250,
                messages=[
                    {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_context},
                ],
            )
            content = response.choices[0].message.content if response.choices else ""
        except Exception:
            logger.exception("Failed to parse utterance with LLM.")
            content = ""

        data = _safe_json_loads(content or "{}")

        intent_value = (data.get("intent") or "UNKNOWN").upper()
        intent = Intent.UNKNOWN
        if intent_value in Intent.__members__:
            intent = Intent[intent_value]

        caller_type_value = (data.get("caller_type") or "UNKNOWN").upper()
        caller_type = CallerType.UNKNOWN
        if caller_type_value in CallerType.__members__:
            caller_type = CallerType[caller_type_value]

        parsed = ParsedUtterance(
            intent=intent,
            command=str(data.get("command") or "NONE").upper(),
            confirmation=str(data.get("confirmation") or "UNKNOWN").upper(),
            caller_type=caller_type,
            member_id=_normalize_digits(data.get("member_id")),
            dob=_normalize_date(data.get("dob")),
            service_date=_normalize_date(data.get("service_date")),
            npi_tax_id=_normalize_digits(data.get("npi_tax_id")),
            full_name=(data.get("full_name") or None),
            birthdate=_normalize_date(data.get("birthdate")),
            result_action=str(data.get("result_action") or "NONE").upper(),
        )

        # Fallback extraction when the LLM misses a field.
        if not parsed.member_id and current_field == "member_id":
            parsed.member_id = _normalize_digits(_extract_digits(text, min_len=4))
        if not parsed.dob and current_field == "dob":
            parsed.dob = _normalize_date(_extract_date(text))
        if not parsed.service_date and current_field == "service_date":
            parsed.service_date = _normalize_date(_extract_date(text))
        if not parsed.npi_tax_id and current_field == "npi_tax_id":
            parsed.npi_tax_id = _normalize_digits(_extract_digits(text, min_len=5))
        if parsed.caller_type == CallerType.UNKNOWN and current_field == "caller_type":
            lowered = text.lower()
            if "patient" in lowered:
                parsed.caller_type = CallerType.PATIENT
            elif "provider" in lowered:
                parsed.caller_type = CallerType.PROVIDER

        if parsed.confirmation == "UNKNOWN" and awaiting_confirmation:
            parsed.confirmation = _fallback_yes_no(text)

        if parsed.command == "NONE":
            parsed.command = _fallback_command(text)

        if parsed.result_action == "NONE" and phase == FlowPhase.RESULT_MENU:
            parsed.result_action = _fallback_result_action(text)

        return parsed
