from loguru import logger
import re

THINKING_PATTERNS = [
    r"m+-?hm+",
    r"m+",
    r"oh+",
    r"ah+",
    r"um+",
    r"uh+",
]

ACKNOWLEDGMENT_PATTERNS = [
    "yes",
    "sure",
    "quite",
    "right",
    "really",
    "good heavens",
    "i see",
    "of course",
    "oh dear",
    "oh god",
    "thats nice",
    "thats not bad",
    "thats right",
    r"yeah+",
    "makes sense",
]

BACKCHANNEL_PATTERNS = THINKING_PATTERNS + ACKNOWLEDGMENT_PATTERNS

def is_backchannel(message: str) -> bool:
    cleaned = re.sub(r"[^\w\s]", "", message).strip().lower()
    for regex in BACKCHANNEL_PATTERNS:
        if re.fullmatch(regex, cleaned):
            logger.debug(f"Matched backchannel pattern '{regex}' for message '{message}'")
            return True
    return False

def is_thinking(message: str) -> bool:
    cleaned = re.sub(r"[^\w\s]", "", message).strip().lower()
    return any(re.fullmatch(regex, cleaned) for regex in THINKING_PATTERNS)
