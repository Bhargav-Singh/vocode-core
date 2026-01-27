# streaming_deepgram_openai_elevenlabs Quickstart

Run-through for `quickstarts/streaming_deepgram_openai_elevenlabs.py`, the sample that transcribes microphone input with Deepgram, sends it to an OpenAI ChatGPT-style agent, and speaks responses via ElevenLabs.

## Requirements
- Python 3.11 (recommended)
- Isolated environment (Conda, `venv`, or similar)
- Microphone and speakers/headphones available to the OS
- API keys:
  - `OPENAI_API_KEY`
  - `DEEPGRAM_API_KEY`
  - `ELEVENLABS_API_KEY`
- Optional: `python-dotenv` for automatic `.env` loading

## Environment & Dependencies
1. Create and activate a Python 3.11 environment.
   ```bash
   conda create -n vocode python=3.11
   conda activate vocode
   ```
   You can substitute your preferred tool (`pyenv`, `uv`, `python -m venv`, etc.).

2. Install Poetry inside the environment and resolve dependencies.
   ```bash
   pip install --upgrade pip
   pip install poetry
   poetry install
   ```
   Poetry reads `pyproject.toml` / `poetry.lock` to install the exact dependency set. If you cannot use Poetry, fallback to `pip install -e .` after activating the environment.

## Configure Secrets & Options
Populate a configuration file (`talkbot.env` or `.env` in the repo root) or export variables directly. The script loads in this order: `talkbot.env`, `.env`, system environment.

Minimum:
```
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
```

Optional tweaks:
```
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_BASE_URL=https://api.openai.com/v1
DEEPGRAM_LANGUAGE=en-US
DEEPGRAM_MODEL=general
DEEPGRAM_TIER=enhanced
DEEPGRAM_VERSION=latest
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OPT_LATENCY=3
ELEVENLABS_SAMPLING_RATE=22050
USE_DEFAULT_DEVICES=false
INPUT_DEVICE_NAME=Your Microphone Name
OUTPUT_DEVICE_NAME=Your Speaker Name
MIC_SAMPLING_RATE=16000
```
Omit anything unnecessary—the defaults from the script will apply.

## Run the Demo
```bash
python quickstarts/streaming_deepgram_openai_elevenlabs.py
```
The script prints the available input/output devices; set `USE_DEFAULT_DEVICES=true` to rely on OS defaults or specify device names via variables above. Press `Ctrl+C` to stop. Microphone level logs appear every ~0.5 seconds so you can confirm capture.

## Troubleshooting
- **Missing required setting**: ensure the three API keys are present.
- **Audio permission errors**: grant microphone access to your terminal (macOS/Linux) or run as administrator (Windows) if required.
- **Choppy or delayed playback**: lower `ELEVENLABS_OPT_LATENCY` (0–4) or set `ELEVENLABS_SAMPLING_RATE` to `16000`.
- **Different config location**: export keys in the shell before running (`export OPENAI_API_KEY=...`).

Happy hacking!
