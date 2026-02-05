import asyncio
import json
import os
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.input_device.microphone_input import MicrophoneInput
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    PunctuationEndpointingConfig,
    Transcription,
)
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriber
from vocode.streaming.utils.worker import QueueConsumer


async def stt_smoke(
    api_key: Optional[str],
    input_device_name: Optional[str] = None,
    mic_sampling_rate: int = 16000,
):
    # Select mic (prompt if name not provided)
    mic_input, _ = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False if input_device_name is None else True,
        input_device_name=input_device_name,
        output_device_name=None,
        mic_sampling_rate=mic_sampling_rate,
    )

    assert isinstance(mic_input, MicrophoneInput)

    # Build transcriber + a simple consumer
    transcriber = AssemblyAITranscriber(
        AssemblyAITranscriberConfig.from_input_device(
            mic_input,
            endpointing_config=PunctuationEndpointingConfig(),
        ),
        api_key=api_key,
    )
    consumer: QueueConsumer[Transcription] = QueueConsumer()
    transcriber.consumer = consumer
    transcriber.start()

    print("STT smoke running. Speak into the selected mic. Ctrl+C to stop.")

    async def printer():
        while True:
            t: Transcription = await consumer.input_queue.get()
            kind = "FINAL" if t.is_final else "PART"
            print(f"[AAI:{kind}] conf={t.confidence:.2f} text={t.message}")

    printer_task = asyncio.create_task(printer())

    try:
        while True:
            chunk = await mic_input.get_audio()
            transcriber.send_audio(chunk)
    except KeyboardInterrupt:
        pass
    finally:
        printer_task.cancel()
        await transcriber.terminate()


if __name__ == "__main__":
    configure_pretty_logging()
    # Load env files if present
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
    # load_env("talkbot.env")
    load_env(".env")
    api_key = os.getenv("ASSEMBLY_AI_API_KEY")
    if not api_key:
        print("Missing ASSEMBLY_AI_API_KEY in environment/.env", file=sys.stderr)
    asyncio.run(stt_smoke(api_key=api_key))
