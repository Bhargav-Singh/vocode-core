# IVR Agent Local Test Flow (CLI)

This quickstart runs the IVR test flow locally, either through a CLI or the
full streaming audio pipeline (STT → LLM → TTS). It is intended to validate
the state machine (name + birthdate capture, confirmation, retries, and summary)
before wiring telephony or full backend integrations.

## What this does
- Starts a CLI-based "call" loop or the full audio pipeline.
- Lets you run the claim status flow (member ID, DOB, service date, caller type).
- Includes the test flow (name + birthdate) for internal validation.
- Confirms inputs and enforces a 3-retry limit per field.
- Uses static data for claims and marks when `dry_run=True`.

## Requirements
- Python >= 3.10
- Project dependencies installed (recommended: `poetry install`)
- CLI mode: no API keys required.
- Audio mode: requires `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `GOOGLE_API_KEY`.

### 📞 Local Audio Setup for Call Transfers

1. Download the [Office telephone ring](https://mixkit.co/free-sound-effects/phone-ring/) audio file.
2. Open a terminal in the audio file's directory and start a local server:
   ```bash
   python -m http.server 8000
   ```
3. In a new terminal window, expose the server using ngrok:
   ```bash
   ngrok http 8000
   ```
4. Copy the ngrok HTTPS URL, append your filename (e.g., https://<url>.ngrok.app/ring.wav), and paste it into the transfer_handoff_node function inside vocode-core/vocode/ivr_agent/flows/intent_router.py.

## Run (CLI Test Flow)
From the repo root:

```bash
python quickstarts/ivr_agent_local.py --cli
```

Alternative:

```bash
IVR_CLI=1 python quickstarts/ivr_agent_local.py
```

## Run (Audio Test Flow: STT → LLM → TTS)
```bash
python quickstarts/ivr_agent_local.py
```

Ensure your audio devices are available and the required API keys are set.

## How to use the CLI
- Start with: `test` (or say "marks"/"score") to enter the test flow.
- Provide full name and birthdate (MM/DD/YYYY).
- Confirm each value when prompted.
- After the summary, say:
  - `repeat` to hear the last summary again
  - `start over` or `main menu` to reset
  - `exit` to quit

## Notes
- The test flow returns static marks for now; this will be replaced with real
  backend calls later.
- The claim flow uses a dry-run response (static claims). To simulate “no records,”
  try a member ID ending in `0`.
