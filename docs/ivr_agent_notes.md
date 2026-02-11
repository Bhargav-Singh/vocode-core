# IVR Agent Notes (Working Log)

Use this file to capture decisions, TODOs, and follow-ups across sessions. Keep it concise and append new notes with dates when useful.

## 2024-XX-XX
- Phase 0 scaffold created:
  - Package `vocode/ivr_agent/` with subfolders (`state`, `flows`, `policies`, `actions`, `rag`, `tests`, `telemetry`).
  - Local runner stub `quickstarts/ivr_agent_local.py` (copy of `streaming_deepgram_openai_elevenlabs.py`) to be swapped to IVR agent wiring in later phases.
- Next planned steps (Phase 1–2):
  - Add prompt/policy module and state models.
  - Start replacing the local runner to import the new IVR agent coordinator once ready.

## 2024-XX-XX
- Phase 1 start:
  - Added `vocode/ivr_agent/policies/prompt_templates.py` with system persona and intent few-shots.
- Testing note:
  - We will build a test flow that validates name + birthdate and returns “marks”; for now use hardcoded test data and a `dry_run` flag to return static values (no backend calls).

## 2024-XX-XX
- Phase 2 implementation:
  - Added state models in `vocode/ivr_agent/state/models.py` (intent/phase enums, slots, conversation state).
  - Added state helpers in `vocode/ivr_agent/state/store.py` (new state, intent mapping, retries, confirmations).
  - Added command detection in `vocode/ivr_agent/state/commands.py` (start over, main menu, transfer).
- Scope note:
  - We will stop once a basic flow is testable via CLI (target around Phase 5) before full telephony wiring.

## 2024-XX-XX
- Phase 3 implementation:
  - Added a lightweight intent router in `vocode/ivr_agent/flows/intent_router.py` (keyword-based; supports optional test flow).

## 2024-XX-XX
- Phase 4 implementation:
  - Added test flow controller in `vocode/ivr_agent/flows/test_flow.py` (name + birthdate capture, confirmation, dry-run marks).
  - Extended `ConversationState` with `current_field`, `pending_value`, `awaiting_confirmation` for step tracking.

## 2024-XX-XX
- Phase 5 implementation:
  - Added CLI test runner in `quickstarts/ivr_agent_local.py` (`--cli` or `IVR_CLI=1`) to exercise the test flow without telephony.
  - CLI uses intent router + test flow and supports start over / main menu / transfer keywords.

## 2024-XX-XX
- Phase 6 implementation:
  - Added optional RAG config scaffolding in `vocode/ivr_agent/rag/config.py` with bounded context formatting.
  - Left retrieval wiring as a placeholder to be hooked to a vector DB later.

## 2024-XX-XX
- Adjustments:
  - CLI repeat now replays the last summary (stored in `ConversationState.last_summary`).
  - Confirmation parsing now tokenizes input to avoid false negatives from substrings.

## 2024-XX-XX
- Audio test flow:
  - Added `IVRTestFlowAgent` in `vocode/ivr_agent/agent/test_flow_agent.py` to run the test flow in the streaming pipeline with optional LLM rephrasing.
  - Updated `quickstarts/ivr_agent_local.py` to use the new agent in audio mode; CLI path unchanged.

## 2024-XX-XX
- Claim flow start:
  - Added LLM-based utterance parser in `vocode/ivr_agent/agent/utterance_parser.py` for intent/command/slot extraction (no keyword-only routing).
  - Added claim flow controller in `vocode/ivr_agent/flows/claim_flow.py` with dry-run claim data and confirmations.
  - Added `IVRFlowAgent` in `vocode/ivr_agent/agent/ivr_flow_agent.py` and wired audio mode to use it in `quickstarts/ivr_agent_local.py`.

## 2024-XX-XX
- Parser resilience:
  - Added lightweight regex fallbacks in `vocode/ivr_agent/agent/utterance_parser.py` for member ID, dates, confirmations, and result actions when LLM parsing fails.
