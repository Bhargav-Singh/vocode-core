import asyncio
import os
import signal
import sys
import time
from math import log10
from pathlib import Path

import numpy as np
import sounddevice as sd
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure repo root is on sys.path for local imports when running from quickstarts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.ivr_agent.agent.ivr_flow_agent import IVRFlowAgent
from vocode.ivr_agent.flows.intent_router import classify_intent
from vocode.ivr_agent.flows.test_flow import handle_test_flow_input, start_test_flow
from vocode.ivr_agent.policies.prompt_templates import system_prompt
from vocode.ivr_agent.state.commands import apply_command, detect_command
from vocode.ivr_agent.state.models import FlowPhase, Intent
from vocode.ivr_agent.state.store import new_state, set_intent
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber


def load_env_if_available(*paths: str) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        for p in paths:
            if p:
                load_dotenv(p, override=False)
    except Exception:
        # silently continue if python-dotenv not installed
        pass


class Settings(BaseSettings):
    """Environment-backed settings for streaming conversation.

    Create a .env (or a custom env file) containing your keys and optional device names.
    """

    # Keys
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model_name: str | None = None
    openai_temperature: float | None = None
    deepgram_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    # Deepgram options
    deepgram_language: str | None = None
    deepgram_model: str | None = None
    deepgram_tier: str | None = None
    deepgram_version: str | None = None

    # ElevenLabs options
    elevenlabs_voice_id: str | None = None  # defaults to Adam if None
    elevenlabs_model_id: str | None = None
    elevenlabs_opt_latency: int | None = 4  # 0..4
    elevenlabs_sampling_rate: int | None = 22050  # prefer 16000/22050/24000 to avoid 44100 restriction

    # Device selection (optional)
    use_default_devices: bool = False
    input_device_name: str | None = None
    output_device_name: str | None = None
    mic_sampling_rate: int | None = 16000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def require(name: str, value: str | None):
    if not value:
        print(f"Missing required setting: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def run_cli(dry_run: bool = True) -> None:
    state = new_state()
    state.phase = FlowPhase.INTENT_CAPTURE

    def say(text: str) -> None:
        state.last_prompt = text
        print(f"Agent: {text}")

    print("IVR Agent CLI (test flow only). Type 'exit' to quit.")
    say(
        "Thank you for calling. I can help you check claim status, authorization status, "
        "or member eligibility. How can I assist you today?"
    )

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("exit", "quit"):
            say("Thank you for calling. Have a great day.")
            break

        state.last_user_utterance = user_text
        command = detect_command(user_text)
        applied_phase = apply_command(state, command)
        if applied_phase == FlowPhase.TRANSFER:
            say("Transferring you to a representative now.")
            break
        if applied_phase == FlowPhase.INTENT_CAPTURE:
            say("How can I assist you today?")
            continue

        if state.phase in (FlowPhase.CALL_START, FlowPhase.INTENT_CAPTURE):
            intent = classify_intent(user_text, allow_test_flow=True)
            if intent == Intent.TEST_FLOW:
                set_intent(state, intent)
                response = start_test_flow(state)
                say(response.prompt)
            else:
                say(
                    "Claim, authorization, and eligibility flows are not wired yet. "
                    "For now, say 'test' to run the test flow."
                )
            continue

        if state.phase == FlowPhase.TEST_FLOW:
            response = handle_test_flow_input(state, user_text, dry_run=dry_run)
            say(response.prompt)
            if response.should_transfer:
                break
            if response.is_done:
                say("Say 'repeat', 'start over', or 'exit'.")
            continue

        if state.phase == FlowPhase.RESULT_MENU:
            lowered = user_text.lower()
            if "repeat" in lowered and state.last_summary:
                say(state.last_summary)
                say("Say 'repeat', 'start over', or 'exit'.")
            else:
                say("Say 'start over' to return to the main menu or 'exit' to quit.")
            continue

        if state.phase == FlowPhase.TRANSFER:
            say("Transferring you to a representative now.")
            break


async def main():
    # Load env files if present (optional second file)
    load_env_if_available("talkbot.env", ".env", "/home/hellfire/vocode-core/talkbot.env")

    settings = Settings()

    # Resolve API keys (validate explicitly to avoid late failures)
    openai_key = require("OPENAI_API_KEY", settings.openai_api_key or os.getenv("OPENAI_API_KEY"))
    deepgram_key = require("DEEPGRAM_API_KEY", settings.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY"))
    eleven_key = require("ELEVENLABS_API_KEY", settings.elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY"))

    # List devices for visibility
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        output_devices = [d for d in devices if d.get("max_output_channels", 0) > 0]
        print("Available input devices:")
        for idx, d in enumerate(input_devices):
            print(f"  {idx}: {d['name']}")
        print("Available output devices:")
        for idx, d in enumerate(output_devices):
            print(f"  {idx}: {d['name']}")
    except Exception:
        pass

    # Select audio devices
    microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=settings.use_default_devices,
        input_device_name=settings.input_device_name,
        output_device_name=settings.output_device_name,
        mic_sampling_rate=settings.mic_sampling_rate,
        speaker_sampling_rate=settings.elevenlabs_sampling_rate,
    )

    # Build the conversation pipeline
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=deepgram_key,
                language=settings.deepgram_language,
                model=settings.deepgram_model,
                tier=settings.deepgram_tier,
                version=settings.deepgram_version,
            ),
        ),
        agent=IVRFlowAgent(
            ChatGPTAgentConfig(
                openai_api_key=openai_key,
                base_url_override=(settings.openai_base_url or os.getenv("OPENAI_BASE_URL")),
                model_name=(
                    settings.openai_model_name
                    or os.getenv("OPENAI_MODEL_NAME")
                    or "gpt-3.5-turbo-1106"
                ),
                temperature=(
                    settings.openai_temperature
                    if settings.openai_temperature is not None
                    else float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
                ),
                initial_message=BaseMessage(
                    text=(
                        "Thank you for calling. I can help you check claim status, "
                        "authorization status, or member eligibility. How can I assist you today?"
                    )
                ),
                prompt_preamble=system_prompt(),
            ),
            dry_run=True,
            use_llm_rephrase=True,
        ),
        synthesizer=ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=eleven_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id=settings.elevenlabs_model_id,
                optimize_streaming_latency=settings.elevenlabs_opt_latency,
            ),
        ),
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))

    # Periodic mic level debug to confirm input activity
    last_level_print = 0.0
    while conversation.is_active():
        chunk = await microphone_input.get_audio()

        now = time.time()
        if now - last_level_print > 0.5 and chunk:
            try:
                samples = np.frombuffer(chunk, dtype=np.int16)
                if samples.size:
                    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
                    if rms > 0:
                        dbfs = 20.0 * log10(rms / 32768.0)
                        print(f"[mic] level: {dbfs:.1f} dBFS")
                    else:
                        print("[mic] silence")
                else:
                    print("[mic] empty buffer")
            except Exception:
                pass
            last_level_print = now

        conversation.receive_audio(chunk)


if __name__ == "__main__":
    configure_pretty_logging()
    if "--cli" in sys.argv or os.getenv("IVR_CLI") == "1":
        run_cli(dry_run=True)
    else:
        asyncio.run(main())
