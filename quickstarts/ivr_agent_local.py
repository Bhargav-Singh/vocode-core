import asyncio
import os
import signal
import sys
import time
from math import log10
from pathlib import Path
import uuid

import numpy as np
import sounddevice as sd
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure repo root is on sys.path for local imports when running from quickstarts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.ivr_agent.agent.ivr_flow_agent import IVRFlowAgent
from langgraph.types import Command
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig, IVRAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.ivr_agent.flows.intent_router import IntentRouter 
from vocode.streaming.synthesizer.google_synthesizer import GoogleSynthesizer
from vocode.streaming.models.synthesizer import GoogleSynthesizerConfig
from vocode.streaming.transcriber.google_transcriber import GoogleTranscriber
from vocode.streaming.models.transcriber import GoogleTranscriberConfig
from vocode.ivr_agent.utilities.llm_initializer import Gemini

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
    google_api_key: str | None = None

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


async def run_cli(dry_run: bool = True) -> None:
    """
    Runs the IVR agent in a local command-line loop using LangGraph.
    """

    # Load env files
    load_env_if_available("talkbot.env", ".env")

    settings = Settings()

    google_key = require("GOOGLE_API_KEY", settings.google_api_key or os.getenv("GOOGLE_API_KEY"))

    gemini = Gemini(api_key=google_key)
    llm = gemini.get_llm()

    intent_router = IntentRouter(llm)
    parent_graph = await intent_router.build_graph()
    
    # 1. Setup State
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("IVR Agent CLI (LangGraph). Type 'exit' to quit.")

    # 2. Initialize Graph (Start the flow)
    # This runs 'greet_and_listen' and pauses at interrupt()
    await parent_graph.ainvoke({"dialogue_status": "active"}, config)

    # 3. Fetch Initial Greeting
    snapshot = await parent_graph.aget_state(config)
    greeting = "Hello?"
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        greeting = snapshot.tasks[0].interrupts[0].value.get("message_to_play", "Hello?")
    
    print(f"Agent: {greeting}")

    # 4. Conversation Loop
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("exit", "quit", "stop"):
            print("Conversation ended.")
            break

        # Resume the graph with user input
        await parent_graph.ainvoke(Command(resume=user_text), config)

        # Retrieve the new state (Bot Response)
        snapshot = await parent_graph.aget_state(config)

        # Check if flow has ended (reached END node)
        if not snapshot.next:
            print("Agent: (Call Ended)")
            break

        # Check for interrupt payload (Message to play)
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            payload = snapshot.tasks[0].interrupts[0].value

            msg = payload.get("message_to_play", "")
            print(f"Agent: {msg}")

            # Check if this node signals a hangup/transfer
            if payload.get("input_type") == "none":
                print("(System: Termination signal received)")
                break


async def main():
    # Load env files
    load_env_if_available("talkbot.env", ".env")

    settings = Settings()

    # Resolve API keys
    google_key = require("GOOGLE_API_KEY", settings.google_api_key or os.getenv("GOOGLE_API_KEY"))

    # Select audio devices
    microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=settings.use_default_devices,
        input_device_name=settings.input_device_name,
        output_device_name=settings.output_device_name,
        mic_sampling_rate=settings.mic_sampling_rate,
        speaker_sampling_rate=16000,
    )

    ivr_agent = await IVRFlowAgent(
            IVRAgentConfig(
                # Note: The initial message here is handled by StreamingConversation. 
                # The Graph also produces a greeting. Typically, you align them or let the Graph handle logic.
                initial_message=BaseMessage(
                    text=(
                        "Thank you for calling. I can help you check claim status, "
                        "authorization status, or member eligibility. How can I assist you today?"
                    )
                ),
                prompt_preamble="You are a helpful IVR assistant.", # Simplistic preamble, real logic is in graph
                interrupt_sensitivity="high",
                num_check_human_present_times=4,
                allowed_idle_time_seconds=10,
                google_api_key=google_key,
            ),
            dry_run=True,
            use_llm_rephrase=False, # Disable rephrase to reduce latency for now
        )

    # Build the conversation pipeline
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=GoogleTranscriber(
            GoogleTranscriberConfig.from_input_device(
                microphone_input,
                api_key=google_key,
            ),
        ),
        agent=ivr_agent,
        synthesizer=GoogleSynthesizer(
            GoogleSynthesizerConfig.from_output_device(
                speaker_output,
                language_code="en-US",
                voice_name="en-US-Neural2-D",
                pitch=0,
                speaking_rate=1.00,
                api_key=google_key,
            ),
        ),
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))

    # Periodic mic level debug
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
            except Exception:
                pass
            last_level_print = now

        conversation.receive_audio(chunk)


if __name__ == "__main__":
    configure_pretty_logging()
    if "--cli" in sys.argv or os.getenv("IVR_CLI") == "1":
        asyncio.run(run_cli(dry_run=True))
    else:
        asyncio.run(main())
