# Vocode – Development Index

Purpose: fast navigation and runbooks for two goals:
- Local streaming agent using system mic/speakers
- Telephony agent reachable via a phone number (Twilio/Vonage)

----------------

## 0) Prereqs & Env

- Python >= 3.10, Poetry installed
- Install repo deps: `poetry install`
- Optional extras by feature (pyproject): `-E telephony -E synthesizers -E transcribers`
- Keys (minimal for examples):
  - `OPENAI_API_KEY`
  - `DEEPGRAM_API_KEY` (streaming ASR quickstart)
  - `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`
  - Telephony: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, plus `BASE_URL` (or use pyngrok)

Quick set: copy `.env.template` under `apps/telephony_app/` to `.env` and fill it.

----------------

## 1) Runbooks

### A) Local Streaming Conversation (mic/speaker)

- Command: `make streaming_conversation`
- Entry example: `quickstarts/streaming_conversation.py`
- Customization knobs:
  - Switch ASR provider via `DeepgramTranscriberConfig` → others in `vocode/streaming/transcriber/`
  - Switch TTS provider via `AzureSynthesizerConfig` → others in `vocode/streaming/synthesizer/`
  - Agent (LLM): `ChatGPTAgentConfig` or Anthropic, etc.

If device auto-select fails: set `use_default_devices=False` and pick from the prompt, or pass `input_device_name`/`output_device_name` to `create_streaming_microphone_input_and_speaker_output`.

### B) Telephony – Inbound (answer a phone call)

- App: `apps/telephony_app/`
- Start (Poetry):
  1) `cd apps/telephony_app`
  2) `poetry install`
  3) Ensure Redis is running (default `localhost:6379`)
  4) `poetry run uvicorn main:app --port 3000`
  5) If `BASE_URL` missing, app uses pyngrok automatically (optional `NGROK_AUTH_TOKEN`).
- Configure Twilio number webhook: `https://<BASE_URL>/inbound_call`
- Agent/config lives in: `apps/telephony_app/main.py` (uses `ChatGPTAgentConfig` by default; swap to `SpellerAgentConfig` as example)

### C) Telephony – Outbound (place a call)

- Script: `apps/telephony_app/outbound_call.py`
- Set `to_phone`/`from_phone` and env keys; run: `poetry run python outbound_call.py`

----------------

## 2) Code Entry Map

- Helpers (devices): `vocode/helpers.py`
- Streaming engine: `vocode/streaming/streaming_conversation.py`
- Agents: `vocode/streaming/agent/` (e.g., `chat_gpt_agent.py`, `base_agent.py`, `default_factory.py`)
- Transcribers: `vocode/streaming/transcriber/` (Deepgram, Whisper, Google, Azure, etc.)
- Synthesizers: `vocode/streaming/synthesizer/` (Azure, ElevenLabs, Google, PlayHT, Cartesia, etc.)
- Output devices: `vocode/streaming/output_device/` (speaker, twilio, vonage, websocket)
- Telephony server: `vocode/streaming/telephony/server/`
- Telephony clients: `vocode/streaming/telephony/client/`
- Telephony conversations: `vocode/streaming/telephony/conversation/`
- Telephony templating: `vocode/streaming/telephony/templates/`
- Quickstarts: `quickstarts/`

High-signal files to open first:
- `quickstarts/streaming_conversation.py` – minimal local streaming wiring
- `apps/telephony_app/main.py` – FastAPI + inbound Twilio config
- `apps/telephony_app/outbound_call.py` – programmatic outbound
- `vocode/streaming/streaming_conversation.py` – orchestration logic, interrupts, filler audio
- `vocode/streaming/telephony/server/base.py` – inbound routes, config persistence

----------------

## 3) Config Objects (common)

- Agent: `ChatGPTAgentConfig` (OpenAI), Anthropic variants
- Transcriber: `DeepgramTranscriberConfig`, `AzureTranscriberConfig`, `GoogleTranscriberConfig`, etc.
- Synthesizer: `AzureSynthesizerConfig`, `ElevenLabsSynthesizerConfig`, `GoogleSynthesizerConfig`, etc.
- Telephony inbound:
  - `TwilioInboundCallConfig` or `VonageInboundCallConfig`
  - `TwilioCallConfig`/`VonageCallConfig` saved to Redis via `ConfigManager`

Provider swap pattern:
1) Replace config object in quickstart or telephony wiring
2) Import matching class in `transcriber/` or `synthesizer/`
3) Ensure extra dep installed and env key provided

----------------

## 4) Fast Navigation (ripgrep cheats)

- Find conversation core: `rg -n "class StreamingConversation|AudioPipeline" vocode/streaming`
- Agent interface/impls: `rg -n "class .*Agent\(|AgentConfig\(" vocode/streaming/agent`
- Transcribers: `rg -n "class .*Transcriber\(" vocode/streaming/transcriber`
- Synthesizers: `rg -n "class .*Synthesizer\(" vocode/streaming/synthesizer`
- Telephony wiring: `rg -n "TelephonyServer|InboundCallConfig|CallsRouter" vocode/streaming/telephony`
- Output devices: `rg -n "class .*OutputDevice" vocode/streaming/output_device`

----------------

## 5) Validation Checklist

Local streaming
- Keys present (OpenAI, Deepgram, Azure)
- Headphones or echo-cancel setup
- `make streaming_conversation` runs; bot responds, interruption works

Telephony inbound
- Redis running
- `uvicorn` app started; logged inbound TwiML URL
- Twilio webhook set to `/inbound_call`
- Call connects; bot speaks; hangup ends cleanly

Telephony outbound
- `BASE_URL` reachable (ngrok or public)
- `to_phone`/`from_phone` valid and authorized in Twilio
- Script starts call and audio flows both ways

----------------

## 6) Troubleshooting Notes

- No audio device prompt: set `use_default_devices=False` or pass names to helpers in `vocode/helpers.py`.
- Echo/feedback: use headphones; or route through a virtual device with echo cancellation.
- Twilio 11200/11205 errors: verify webhook URL, TLS, and request method (POST). Check server logs in `apps/telephony_app`.
- Redis not found: start Redis locally or run a container `docker run -dp 6379:6379 redis/redis-stack:latest`.
- Missing provider deps: install extras `poetry install -E telephony -E synthesizers -E transcribers`.

----------------

## 7) Next Steps (TODOs)

- [ ] Swap transcriber/synthesizer providers and compare latency
- [ ] Add simple custom `BaseAgent` demo (echo/speller) to streaming flow
- [ ] Enable recording events and verify uploads via events manager
- [ ] Add LiveKit output device test, if desired
