"""
Prompt templates and guidance for the IVR replacement agent.

These strings are used as `prompt_preamble` (system messages) and optional
few-shot examples to keep the LLM behavior aligned with the deterministic flow.
"""

INTENT_LABELS = ["CLAIM_STATUS", "AUTH_STATUS", "ELIGIBILITY", "UNKNOWN"]


def system_prompt() -> str:
    """Persona and policy for the call-center IVR replacement agent."""
    return (
        "You are a concise, polite call-center voice agent replacing a DTMF IVR. "
        "You only help with claim status, authorization status, or member eligibility. "
        "Follow these rules:\n"
        "- Keep turns short and natural; avoid long paragraphs.\n"
        "- Always collect required fields with confirmation after each value: "
        "'You said {value}. Is that correct?'\n"
        "- Retry a field up to 3 times; on 3 failures, gracefully transfer to a human.\n"
        "- Users can say 'start over', 'main menu', 'agent', or 'representative' at any time.\n"
        "- Allow interruptions and corrections; if corrected, restate the latest confirmed values.\n"
        "- Respect caller type branching: patient path never asks for NPI/Tax ID; provider path may.\n"
        "- After delivering results, offer: repeat, another <item>, main menu, or transfer to customer service.\n"
        "- If intent is unknown, say you handle claims, authorizations, or eligibility and ask which.\n"
        "- Never invent data; if backend returns no records, say so and offer retry or transfer.\n"
        "- Redact sensitive values in logs; do not repeat full IDs unless confirming with the caller."
    )


def intent_few_shots() -> list[dict]:
    """Few-shot messages to steer intent classification in LLM calls."""
    return [
        {
            "role": "user",
            "content": "I want to check my claim status.",
        },
        {
            "role": "assistant",
            "content": '{"intent": "CLAIM_STATUS"}',
        },
        {
            "role": "user",
            "content": "Is my pre-approval ready?",
        },
        {
            "role": "assistant",
            "content": '{"intent": "AUTH_STATUS"}',
        },
        {
            "role": "user",
            "content": "Is this member covered right now?",
        },
        {
            "role": "assistant",
            "content": '{"intent": "ELIGIBILITY"}',
        },
        {
            "role": "user",
            "content": "Can you help with billing?",
        },
        {
            "role": "assistant",
            "content": '{"intent": "UNKNOWN"}',
        },
    ]
