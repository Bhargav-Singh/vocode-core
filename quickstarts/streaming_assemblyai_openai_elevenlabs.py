import asyncio
import os
import signal
import sys
import time
from math import log10

from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriber
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

configure_pretty_logging()


class Settings(BaseSettings):
    """Environment-backed settings for streaming conversation.

    You can override these by creating a .env file next to this script
    containing lines like OPENAI_API_KEY=... etc.
    """

    # Keys are read from environment or .env (no hardcoded defaults)
    openai_api_key: str | None = None
    assembly_ai_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    # Optional ElevenLabs parameters
    elevenlabs_voice_id: str | None = None  # defaults to Adam if None
    elevenlabs_model_id: str | None = None  # e.g. "eleven_turbo_v2"
    elevenlabs_opt_latency: int | None = 4  # 0..4; lower latency trades quality
    # Some ElevenLabs plans do not allow 44100 Hz PCM. Use 16000/22050/24000.
    elevenlabs_sampling_rate: int | None = 22050

    # Optional device selection (by name). If unset, you'll be prompted unless
    # use_default_devices=True, in which case system defaults are used.
    use_default_devices: bool = False
    input_device_name: str | None = None
    output_device_name: str | None = None
    mic_sampling_rate: int | None = 16000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def _ensure_env_loaded():
    # Load .env early so pydantic can pick them up


    def load_env(dotenv_path: str = ".env") -> bool:
        """Load environment variables from a .env file if python-dotenv is available.

        Returns True if a .env file was successfully loaded, False otherwise.
        Safe to call even if python-dotenv is not installed.
        """
        try:
             # type: ignore

            return bool(load_dotenv(dotenv_path))
        except Exception:
            return False

    print(load_env("/home/hellfire/vocode-core/talkbot.env"))


settings = Settings()  # reads from .env via model_config and process env


async def main():
    _ensure_env_loaded()
    # List devices (helps verify names/choices)
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

    # Resolve API keys (env first, then settings values)
    eleven_api_key = settings.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not eleven_api_key:
        print("Missing ELEVENLABS_API_KEY in environment or .env", file=sys.stderr)
        sys.exit(1)

    # Wire conversation
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=AssemblyAITranscriber(
            AssemblyAITranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
            ),
            api_key=settings.assembly_ai_api_key,  # falls back to env inside transcriber if None
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,  # falls back to OPENAI_API_KEY env if None
                initial_message=BaseMessage(text="Hello!"),
                prompt_preamble=(
                    "You are a helpful real-time voice assistant. Be concise and natural."
                ),
            )
        ),
        synthesizer=ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=eleven_api_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id=settings.elevenlabs_model_id,
                optimize_streaming_latency=settings.elevenlabs_opt_latency,
            ),
        ),
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    last_level_print = 0.0
    while conversation.is_active():
        chunk = await microphone_input.get_audio()

        # Debug: show mic level periodically to confirm input activity
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
                # Non-fatal; continue even if level calc fails
                pass
            last_level_print = now

        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
