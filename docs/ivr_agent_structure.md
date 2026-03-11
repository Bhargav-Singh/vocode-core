# IVR Replacement Agent – Project Structure (Reference)

Goal: house all IVR/call-center agent logic inside this repo with clear separation between local testing, telephony wiring, prompts/policies, backend integrations, and RAG.

## Proposed Layout
- `quickstarts/ivr_agent_local.py`  
  - Local mic/speaker runner (starts as a copy of `streaming_deepgram_openai_elevenlabs.py` wired to the new agent + state manager).
- `apps/ivr_telephony_app/`  
  - FastAPI entrypoint for Twilio/Vonage; mirrors `apps/telephony_app` but imports the IVR agent stack.  
  - `config.py` (settings/env), `main.py` (routes), `dependencies.py` (wiring agent/synth/transcriber), `twilio_handlers.py` (webhooks/tests).
- `vocode/ivr_agent/` (new package)
  - `__init__.py`
  - `flows/` – state machines for intents/slots; pure logic (no IO). Each file = flow (e.g., `support_flow.py`, `balance_flow.py`).
  - `state/` – conversation state models, slot store, transcript summarization helpers.
  - `policies/` – prompt templates/persona, guardrails, end-of-call rules.
  - `actions/` – thin adapters for 3rd-party APIs (validation, lookup, ticket creation). Keep interfaces + fake stubs for local dev/tests.
  - `rag/` – optional retrievers/vector config if knowledge grounding is needed (plugs into ChatGPT/Groq configs).
  - `telemetry/` – logging/metrics hooks, Sentry tags, call outcome events.
  - `tests/` – unit/contract tests for flows/actions; fixtures for transcripts and mock API responses.
- `docs/ivr_agent_structure.md` (this file)  
  - Living reference; update when flows/actions change.
- `talkbot.env` or `.env.example`  
  - Include Twilio keys, OpenAI/Groq, Deepgram, ElevenLabs, backend API base URL/token.

## Wiring Notes
- Agent: wrap `ChatGPTAgent` (or `GroqAgent`) with a small coordinator that reads/writes `state/` and injects flow directives into prompts. Keep interrupt/backchannel settings in config.
- Transcriber/Synthesizer: keep Deepgram + ElevenLabs defaults; expose overrides via settings/env for telephony (narrowband models, 8–24 kHz).
- Backend calls: implement interfaces in `actions/` with stub fakes for offline dev; add retries/timeouts and clear error messages to user.
- RAG: only if needed—place retrieval config in `vocode/ivr_agent/rag/` and pass vector results into agent prompts.
- Telephony: reuse `StreamingConversation` with Twilio/Vonage output devices; keep local runner identical except for IO devices.

## Next Steps (suggested)
1) Clone `quickstarts/streaming_deepgram_openai_elevenlabs.py` into `quickstarts/ivr_agent_local.py` and swap in the IVR agent coordinator + state store.
2) Create `vocode/ivr_agent/state` + `flows/support_flow.py` for a first intent (e.g., account lookup → problem type → resolution/transfer).
3) Add `apps/ivr_telephony_app/` that imports the same agent stack and exposes Twilio webhook routes; mirror the existing telephony app structure.
4) Add minimal tests under `vocode/ivr_agent/tests` for flow progression and fake backend validation calls.
