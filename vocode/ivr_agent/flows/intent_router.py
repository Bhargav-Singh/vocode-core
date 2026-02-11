from __future__ import annotations

import re
from typing import Iterable

from vocode.ivr_agent.state.models import Intent


CLAIM_KEYWORDS = ("claim", "payment", "remit", "service date")
AUTH_KEYWORDS = ("authorization", "pre approval", "pre-approval", "auth")
ELIGIBILITY_KEYWORDS = ("eligibility", "coverage", "covered", "benefits", "member covered")
TEST_KEYWORDS = ("test", "marks", "score", "grade")


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _looks_like_member_id(text: str) -> bool:
    return bool(re.fullmatch(r"\d{6,}", text.strip()))


def classify_intent(text: str, allow_test_flow: bool = False) -> Intent:
    """Lightweight intent router. LLM-based classification can replace this later."""
    cleaned = (text or "").strip()
    if not cleaned:
        return Intent.UNKNOWN

    if allow_test_flow and _contains_any(cleaned, TEST_KEYWORDS):
        return Intent.TEST_FLOW

    if _contains_any(cleaned, CLAIM_KEYWORDS):
        return Intent.CLAIM_STATUS
    if _contains_any(cleaned, AUTH_KEYWORDS):
        return Intent.AUTH_STATUS
    if _contains_any(cleaned, ELIGIBILITY_KEYWORDS):
        return Intent.ELIGIBILITY

    # If caller starts by giving a numeric ID, push toward intent capture rather than UNKNOWN
    if _looks_like_member_id(cleaned):
        return Intent.UNKNOWN

    return Intent.UNKNOWN
