from __future__ import annotations

from enum import Enum
from typing import Optional

# from vocode.ivr_agent.state.models import ConversationState, FlowPhase
# from vocode.ivr_agent.state.store import reset_for_main_menu


class CommandType(str, Enum):
    START_OVER = "START_OVER"
    MAIN_MENU = "MAIN_MENU"
    TRANSFER = "TRANSFER"
    NONE = "NONE"


# def detect_command(text: str) -> CommandType:
#     cleaned = (text or "").strip().lower()
#     if not cleaned:
#         return CommandType.NONE
#     if "start over" in cleaned:
#         return CommandType.START_OVER
#     if "main menu" in cleaned:
#         return CommandType.MAIN_MENU
#     if "agent" in cleaned or "representative" in cleaned:
#         return CommandType.TRANSFER
#     return CommandType.NONE


# def apply_command(state: ConversationState, command: CommandType) -> Optional[FlowPhase]:
#     if command == CommandType.START_OVER or command == CommandType.MAIN_MENU:
#         reset_for_main_menu(state)
#         return state.phase
#     if command == CommandType.TRANSFER:
#         state.phase = FlowPhase.TRANSFER
#         return state.phase
#     return None


number_mappings = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "*": "star",
    "#": "hash"
}