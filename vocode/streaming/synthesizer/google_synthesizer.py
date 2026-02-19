import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google.api_core.client_options import ClientOptions
import google.auth
from google.cloud import texttospeech_v1beta1 as tts  # type: ignore

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import GoogleSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult


class GoogleSynthesizer(BaseSynthesizer[GoogleSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: GoogleSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        if synthesizer_config.api_key:
            client_options = ClientOptions(api_key=synthesizer_config.api_key)
        else:
            client_options = None
            google.auth.default()

        # Instantiates a client
        self.client = tts.TextToSpeechClient(client_options=client_options)

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        self.voice = tts.VoiceSelectionParams(
            language_code=synthesizer_config.language_code,
            name=synthesizer_config.voice_name,
        )

        # Select the type of audio file you want returned
        self.audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=synthesizer_config.sampling_rate,
            speaking_rate=synthesizer_config.speaking_rate,
            pitch=synthesizer_config.pitch,
            effects_profile_id=["telephony-class-application"],
        )
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: GoogleSynthesizerConfig) -> str:
        return synthesizer_config.voice_name

    def synthesize(self, message: str) -> Any:
        synthesis_input = tts.SynthesisInput(ssml=f"<speak>{message}</speak>")

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        return self.client.synthesize_speech(
            request=tts.SynthesizeSpeechRequest(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config,
                enable_time_pointing=[1],  # SSML_MARK or [tts.SynthesizeSpeechRequest.TimepointType.SSML_MARK]
            )
        )

    # TODO: make this nonblocking, see speech.TextToSpeechAsyncClient
    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        response: tts.SynthesizeSpeechResponse = (  # type: ignore
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor, self.synthesize, message.text
            )
        )
        output_sample_rate = response.audio_config.sample_rate_hertz

        output_bytes_io = io.BytesIO()
        in_memory_wav = wave.open(output_bytes_io, "wb")
        in_memory_wav.setnchannels(1)
        in_memory_wav.setsampwidth(2)
        in_memory_wav.setframerate(output_sample_rate)
        in_memory_wav.writeframes(response.audio_content[44:])
        output_bytes_io.seek(0)

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )
        return result
