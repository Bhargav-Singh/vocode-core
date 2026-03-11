# IVR Replacement Agent – Implementation Plan (Phased)

Goal: implement the conversational IVR replacement described in `IVR Guide flow` using the Vocode stack (Twilio → Deepgram → ChatGPT/Groq → ElevenLabs) with backend claim/auth/eligibility lookups. This plan maps *what/why/how/where* so multiple contributors can work in parallel.

## Scope Baseline
- Flows covered: Claim Status, Authorization Status, Eligibility, plus unknown/fallback.
- Shared rules: confirmation after every captured field, max 3 retries per field → transfer; “start over/main menu/agent” works anytime; interruptions allowed.
- Data sources: backend REST APIs (claims/auth/eligibility); stub first, real later.

## Phase 0 — Scaffolding & Entry Points
- Why: isolate IVR logic from quickstarts and telephony app to keep changes contained.
- Where/How:
  - Copy `quickstarts/streaming_deepgram_openai_elevenlabs.py` → `quickstarts/ivr_agent_local.py`; swap in IVR agent wiring (see Phase 1–2).
  - Add package `vocode/ivr_agent/` with subpackages:
    - `state/` (conversation state, slot store, retry counters).
    - `flows/` (claim/auth/eligibility flow logic).
    - `policies/` (prompts/persona/confirmation templates).
    - `actions/` (API interfaces + stubs).
    - `rag/` (optional retrieval wiring).
    - `tests/`.
  - Reference doc: keep `docs/ivr_agent_structure.md` updated.

## Phase 1 — Agent Persona, Prompts, and Interrupt Settings
- Why: ensure LLM behavior matches call-center tone and rules.
- Where/How:
  - Create `vocode/ivr_agent/policies/prompt_templates.py` with:
    - System prompt: purpose (claims/auth/eligibility only), confirmation pattern, retry rules, allow corrections/start over, concise style.
    - User-intent primer/examples for classification (CLAIM_STATUS, AUTH_STATUS, ELIGIBILITY, UNKNOWN).
    - Post-result menu options (repeat/another/main menu/transfer).
  - In `ivr_agent_local.py`, set `ChatGPTAgentConfig` (or `GroqAgentConfig`): `prompt_preamble` from policy, `backchannel_probability` tuned low, `interrupt_sensitivity` medium/high.

## Phase 2 — State & Slot Management
- Why: enforce IVR-like deterministic flow; guard against LLM drift.
- Where/How:
  - `vocode/ivr_agent/state/models.py`: data classes for call state (phase, intent), slots (member_id, dob, service_date, caller_type, npi_tax_id), retry counters, last_confirmed values.
  - `vocode/ivr_agent/state/store.py`: helper to reset slots on start over; increment retry; check transfer threshold.
  - `vocode/ivr_agent/state/commands.py`: small API to update slots, confirm values, and route to flow handlers.
  - Attach state manager into agent (e.g., wrap `ChatGPTAgent` or build a thin coordinator that preprocesses user text, updates state, and feeds structured context into the LLM).

## Phase 3 — Intent Classification & Router
- Why: map first user turn (or at any “main menu”) to one of three flows.
- Where/How:
  - `vocode/ivr_agent/flows/intent_router.py`: classify using LLM (few-shot examples) or regex fallback; on UNKNOWN trigger fallback prompt with 3-attempt limit then transfer.
  - Provide structured intent in prompt context (e.g., JSON summary injected into `messages` before LLM response).

## Phase 4 — Flow Implementations (Deterministic Controllers)
- Why: replicate IVR paths with confirmations/retries.
- Where/How (one file per flow under `vocode/ivr_agent/flows/`):
  - `claim_flow.py`: steps for member_id → dob → service_date → caller_type → (provider? ask npi_tax_id) → call API stub → render results → post-result menu.
  - `auth_flow.py`: member_id → dob → caller_type → (provider? npi_tax_id) → API → handle multiple authorizations by date selection → post-result menu.
  - `eligibility_flow.py`: member_id → dob → API → post-result menu.
  - Each flow exposes:
    - `next_prompt(state, user_text)` → (prompt, updates, maybe api_call_request)
    - `handle_backend_result(...)` to format claims/auth/eligibility responses concisely.
  - Enforce shared confirmation template (“You said {value}. Is that correct?”) and retry limit (3 → transfer).

## Phase 5 — Backend Actions (Stubs → Real)
- Why: decouple API calls for easy swap.
- Where/How:
  - `vocode/ivr_agent/actions/interfaces.py`: define `ClaimsClient`, `AuthClient`, `EligibilityClient` with method signatures returning typed results.
  - `vocode/ivr_agent/actions/stub_clients.py`: deterministic fake responses for local dev/tests (single/multiple/no records cases).
  - `vocode/ivr_agent/actions/http_clients.py`: real HTTP implementations (later), with timeouts/retries and redaction of PII in logs.

## Phase 6 — RAG (Optional)
- Why: only if policy/coverage explanations need grounding.
- Where/How:
  - `vocode/ivr_agent/rag/config.py`: vector DB config (source path, embedding model).
  - Inject retrieved context into agent prompts when intent = ELIGIBILITY (or as needed).

## Phase 7 — Telephony Integration
- Why: production call handling.
- Where/How:
  - `apps/ivr_telephony_app/`:
    - `config.py` (settings/env: Twilio creds, backend URLs, model keys).
    - `main.py` (FastAPI routes; inbound call webhook; transfer handler).
    - `dependencies.py` (wire up IVR agent, transcriber, synthesizer).
    - `twilio_handlers.py` (transfer `<Redirect>`, error handling).
  - Output device: Twilio (reuse existing telephony output devices).
  - Keep `quickstarts/ivr_agent_local.py` for mic/speaker parity with same agent wiring.

## Phase 8 — UX/Behavior Hardening
- Why: production readiness.
- Where/How:
  - Configure interruption/backchannel thresholds in `ChatGPTAgentConfig`.
  - Add “start over” and “main menu” intercept in state manager (reset slots, return to intent capture).
  - Implement “agent/representative” keyword → transfer immediately.
  - Add polite end-call message and hangup hook.

## Phase 9 — Testing & QA
- Why: catch regressions and flow breaks.
- Where/How:
  - `vocode/ivr_agent/tests/`:
    - Unit tests for state transitions and retry/confirmation logic.
    - Stubbed backend tests covering single/multiple/no-record cases.
    - Prompt snapshot tests for key flows (golden outputs).
  - Manual call test checklist (Twilio): intent routing, retries, transfer, interruption handling.

## Phase 10 — Logging, Metrics, and Redaction
- Why: observability without leaking PII.
- Where/How:
  - `vocode/ivr_agent/telemetry/` with helpers to:
    - Tag Sentry spans per intent/flow outcome.
    - Redact member IDs/DOB/NPI in logs.
    - Emit call outcome events (resolved/transfer/no-records).

## Deliverables by Phase
- Phase 0–2: new package scaffolding, local runner, state models, prompts.
- Phase 3–4: intent router + full claim/auth/eligibility controllers with confirmations/retries.
- Phase 5: stub API clients + formatting helpers.
- Phase 6: optional RAG wiring.
- Phase 7: telephony app using same agent stack.
- Phase 8–10: UX polish, tests, telemetry, redaction.
