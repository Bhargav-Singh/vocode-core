# Vocode Core – Dev Notes

## Overview

Vocode is a Python SDK for real‑time, voice-first LLM apps. It orchestrates transcription, LLM reasoning, and speech synthesis with pluggable providers, and includes telephony support (Twilio/Vonage), a streaming conversation engine, and quickstarts.

## Setup

- Python: >= 3.10
- Install (library users): `pip install vocode`
- Dev install (repo):
  - Install Poetry
  - `poetry install`
  - Useful targets: `make streaming_conversation`, `make turn_based_conversation`, `make test`, `make lint`

## Key Paths

- Core package: `vocode/`
  - Streaming: `vocode/streaming/` – conversations, agents, transcribers, synthesizers, devices
  - Turn-based: `vocode/turn_based/`
  - Telephony: `vocode/streaming/telephony/`
  - Utilities: `vocode/streaming/utils/`, `vocode/utils/`
- Quickstarts: `quickstarts/` (e.g., `streaming_conversation.py`)
- Self-hosted Telephony App: `apps/telephony_app/` (FastAPI server)
- Docs content: `docs/` (Mintlify MDX, plus OpenAPI spec)
- Project config: `pyproject.toml`

## Core Concepts

- Conversation: central orchestrator combining 5 components
  - Transcriber → ASR provider wrapper
  - Agent → LLM or custom logic
  - Synthesizer → TTS provider wrapper
  - Input Device → audio in (e.g., microphone)
  - Output Device → audio out (e.g., speaker, telephony)

Key implementation files:

- Helpers to pick devices: `vocode/helpers.py`
- Streaming engine: `vocode/streaming/streaming_conversation.py`
- Agents: `vocode/streaming/agent/`
- Transcribers: `vocode/streaming/transcriber/`
- Synthesizers: `vocode/streaming/synthesizer/`
- Output devices: `vocode/streaming/output_device/`
- Events & state: `vocode/streaming/utils/`

## Quickstarts

- Streaming conversation example: `quickstarts/streaming_conversation.py`
  - Uses Deepgram (ASR), OpenAI (agent), Azure TTS by default
- Turn-based example: `quickstarts/turn_based_conversation.py`
- Make targets: `make streaming_conversation`, `make turn_based_conversation`

## Telephony Server (Self-hosted)

- Path: `apps/telephony_app/`
- Docs: `docs/open-source/telephony.mdx`
- Run options:
  1) Docker: build image then `docker-compose up`
  2) Poetry: `poetry install`, run Redis, then `poetry run uvicorn main:app --port 3000`
- Configure Twilio number webhook → `https://<BASE_URL>/inbound_call`

## Docs Index (handy entries)

- Top-level README: `README.md`
- Open-source intro/how it works: `docs/open-source/how-it-works.mdx`
- Python quickstart: `docs/open-source/python-quickstart.mdx`
- Telephony guide: `docs/open-source/telephony.mdx`
- Agent reference: `docs/open-source/agent-reference.mdx`
- Additional open-source guides: `docs/open-source/*.mdx`

## Running Tests & Tooling

- Tests: `pytest` via `make test` (configured in `pyproject.toml`)
- Typecheck: `make typecheck` (mypy)
- Format: `make lint` (black + isort)

## Environment Keys (common)

- OpenAI: `OPENAI_API_KEY`
- Deepgram: `DEEPGRAM_API_KEY`
- Azure Speech: `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`
- Telephony: Twilio/Vonage keys (see `apps/telephony_app/.env.template`)

## Notes

- Many providers are optional extras in `pyproject.toml`; install only what you need (e.g., `poetry install -E telephony -E synthesizers -E transcribers`).
- System audio path has no built-in echo cancellation; prefer headphones or a virtual device with echo suppression.
