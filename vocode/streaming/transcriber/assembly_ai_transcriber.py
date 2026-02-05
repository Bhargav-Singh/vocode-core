import asyncio
import audioop
import json
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import websockets
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
    Transcription,
)
from vocode.streaming.models.websocket import AudioMessage
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

ASSEMBLY_AI_URL = "wss://api.assemblyai.com/v2/realtime/ws"
ASSEMBLY_AI_V3_URL = "wss://streaming.assemblyai.com/v3/ws"


class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("ASSEMBLY_AI_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set ASSEMBLY_AI_API_KEY environment variable or pass it as a parameter"
            )
        # Allow overriding model and ws URL via config or env
        self.model = (
            transcriber_config.model
            if hasattr(transcriber_config, "model")
            else None
        ) or getenv("ASSEMBLY_AI_MODEL")
        self.ws_url = (
            transcriber_config.ws_url
            if hasattr(transcriber_config, "ws_url")
            else None
        ) or getenv("ASSEMBLY_AI_WS_URL") or ASSEMBLY_AI_URL
        # Detect v3 endpoint usage
        self._is_v3 = (
            "streaming.assemblyai.com" in self.ws_url or "/v3/ws" in self.ws_url
        )
        self._ended = False
        self.buffer = bytearray()
        self.audio_cursor = 0
        # Debug toggles (set VOCODE_STT_DEBUG=true/1 to enable verbose logs)
        debug_env = str(getenv("VOCODE_STT_DEBUG", "")).lower()
        self.debug: bool = debug_env in ("1", "true", "yes", "on")
        self._debug_send_count = 0
        self._last_close_code: Optional[int] = None
        self._last_close_reason: Optional[str] = None
        self._resample_state = None

        if isinstance(
            self.transcriber_config.endpointing_config,
            (TimeEndpointingConfig, PunctuationEndpointingConfig),
        ):
            self.transcriber_config.end_utterance_silence_threshold_milliseconds = int(
                self.transcriber_config.endpointing_config.time_cutoff_seconds * 1000
            )
        self.terminate_msg = json.dumps({"terminate_session": True})
        self.end_utterance_silence_threshold_msg = (
            None
            if self.transcriber_config.end_utterance_silence_threshold_milliseconds is None
            else json.dumps(
                {
                    "end_utterance_silence_threshold": self.transcriber_config.end_utterance_silence_threshold_milliseconds
                }
            )
        )

    async def ready(self):
        return True

    async def _run_loop(self):
        await self.process()

    def _effective_sampling_rate(self) -> int:
        return self.transcriber_config.downsampling or self.transcriber_config.sampling_rate

    def send_audio(self, chunk):
        if self.debug:
            logger.debug(f"[AAI] enqueue audio chunk bytes={len(chunk)}")
        if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            sample_width = 1
            if isinstance(chunk, np.ndarray):
                chunk = chunk.astype(np.int16)
                chunk = chunk.tobytes()
            chunk = audioop.ulaw2lin(chunk, sample_width)

        # Optional downsampling for LINEAR16
        target_rate = self.transcriber_config.downsampling
        if (
            target_rate
            and self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16
            and target_rate != self.transcriber_config.sampling_rate
        ):
            try:
                resampled, self._resample_state = audioop.ratecv(
                    chunk,
                    2,
                    1,
                    self.transcriber_config.sampling_rate,
                    target_rate,
                    self._resample_state,
                )
                chunk = resampled
            except Exception as e:
                if self.debug:
                    logger.debug(f"[AAI] downsample failed (non-fatal): {e}")

        self.buffer.extend(chunk)

        if (
            len(self.buffer) / (2 * self.transcriber_config.sampling_rate)
        ) >= self.transcriber_config.buffer_size_seconds:
            self.consume_nonblocking(self.buffer)
            self.buffer = bytearray()

    async def terminate(self):
        self._ended = True
        await super().terminate()

    def get_assembly_ai_url(self):
        sr = self._effective_sampling_rate()
        if self._is_v3:
            params = {"sample_rate": str(sr)}
            # encoding mapping for v3
            if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
                params["encoding"] = "pcm_s16le"
            elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
                params["encoding"] = "pcm_mulaw"
            # Optional tuning params if configured
            eot = getattr(self.transcriber_config, "end_of_turn_confidence_threshold", None)
            if eot is not None:
                params["end_of_turn_confidence_threshold"] = str(eot)
            fmt = getattr(self.transcriber_config, "format_turns", None)
            if fmt is not None:
                params["format_turns"] = "true" if fmt else "false"
            min_sil = getattr(
                self.transcriber_config, "min_end_of_turn_silence_when_confident_ms", None
            )
            if min_sil is not None:
                params["min_end_of_turn_silence_when_confident"] = str(min_sil)
            max_sil = getattr(self.transcriber_config, "max_turn_silence_ms", None)
            if max_sil is not None:
                params["max_turn_silence"] = str(max_sil)
            base = self.ws_url or ASSEMBLY_AI_V3_URL
            return base + f"?{urlencode(params)}"
        else:
            url_params = {"sample_rate": sr}
            if self.model:
                url_params.update({"model": self.model})
            if self.transcriber_config.word_boost:
                url_params.update({"word_boost": json.dumps(self.transcriber_config.word_boost)})
            base = self.ws_url or ASSEMBLY_AI_URL
            return base + f"?{urlencode(url_params)}"

    async def process(self):
        backoff = 1.0
        self.audio_cursor = 0
        while not self._ended:
            self._last_close_code = None
            self._last_close_reason = None
            URL = self.get_assembly_ai_url()
            if self.debug:
                logger.debug(
                    f"[AAI] connecting websocket url={URL} ping_interval=5 ping_timeout=20"
                )
            try:
                async with websockets.connect(
                    URL,
                    additional_headers=(
                        (
                            "Authorization",
                            ("Bearer " + self.api_key)
                            if str(getenv("ASSEMBLY_AI_AUTH_SCHEME", "")).lower()
                            == "bearer"
                            else self.api_key,
                        ),
                    ),
                    ping_interval=5,
                    ping_timeout=20,
                ) as ws:
                    await asyncio.sleep(0.1)

                    if not self._is_v3:
                        # v2: Send config message
                        try:
                            config_payload = {
                                "config": {
                                    "sample_rate": self._effective_sampling_rate(),
                                }
                            }
                            if (
                                self.transcriber_config.end_utterance_silence_threshold_milliseconds
                                is not None
                            ):
                                config_payload["config"][
                                    "end_utterance_silence_threshold"
                                ] = self.transcriber_config.end_utterance_silence_threshold_milliseconds
                            await ws.send(json.dumps(config_payload))
                            if self.debug:
                                logger.debug(f"[AAI] sent config: {config_payload}")
                        except Exception as e:
                            if self.debug:
                                logger.debug(f"[AAI] failed to send config (non-fatal): {e}")

                    # Fallback legacy message
                    if (
                        self.end_utterance_silence_threshold_msg
                        and "end_utterance_silence_threshold" in self.end_utterance_silence_threshold_msg
                    ):
                        await ws.send(self.end_utterance_silence_threshold_msg)
                        if self.debug:
                            logger.debug(
                                f"[AAI] sent end_utterance_silence_threshold: {self.end_utterance_silence_threshold_msg}"
                            )

                    async def sender(ws):  # sends audio to websocket
                        while not self._ended:
                            try:
                                data = await asyncio.wait_for(self._input_queue.get(), 5)
                            except asyncio.exceptions.TimeoutError:
                                break
                            num_channels = 1
                            sample_width = 2
                            self.audio_cursor += len(data) / (
                                self._effective_sampling_rate() * num_channels * sample_width
                            )
                            if self._is_v3:
                                await ws.send(data)
                            else:
                                payload = json.dumps(
                                    {"audio_data": AudioMessage.from_bytes(data).data}
                                )
                                await ws.send(payload)
                            if self.debug:
                                self._debug_send_count += 1
                                if self._debug_send_count % 10 == 0:
                                    logger.debug(
                                        f"[AAI] sent audio frame #{self._debug_send_count} bytes={len(data)} cursor={self.audio_cursor:.2f}s"
                                    )
                        if not self._is_v3:
                            await ws.send(self.terminate_msg)
                        logger.debug("Terminating AssemblyAI transcriber sender")
                        if self.debug:
                            logger.debug("[AAI] terminate requested")

                    async def receiver(ws):
                        while not self._ended:
                            try:
                                result_str = await ws.recv()
                                if not self._is_v3:
                                    data = json.loads(result_str)
                                    if "error" in data and data["error"]:
                                        raise Exception(data["error"])
                            except websockets.exceptions.ConnectionClosedError as e:
                                self._last_close_code = getattr(e, "code", None)
                                self._last_close_reason = getattr(e, "reason", None)
                                logger.debug(e)
                                break

                            if self._is_v3:
                                # v3 message handling
                                try:
                                    data = json.loads(result_str)
                                except Exception:
                                    if self.debug:
                                        logger.debug("[AAI] v3 non-JSON message; ignoring")
                                    continue
                                mtype = data.get("type")
                                if self.debug:
                                    preview = result_str[:200].replace("\n", " ")
                                    logger.debug(f"[AAI] recv raw: {preview}")
                                if mtype == "Begin":
                                    continue
                                elif mtype == "Turn":
                                    transcript = data.get("transcript", "")
                                    end_of_turn = bool(data.get("end_of_turn", False))
                                    conf = float(data.get("end_of_turn_confidence", 0.0))
                                    if transcript:
                                        self.produce_nonblocking(
                                            Transcription(
                                                message=transcript,
                                                confidence=conf,
                                                is_final=end_of_turn,
                                            )
                                        )
                                elif mtype == "Termination":
                                    break
                            else:
                                data = json.loads(result_str)
                                is_final = (
                                    "message_type" in data and data["message_type"] == "FinalTranscript"
                                )
                                if self.debug:
                                    preview = result_str[:200].replace("\n", " ")
                                    logger.debug(f"[AAI] recv raw: {preview}")
                                if "text" in data and data["text"]:
                                    if self.debug:
                                        logger.debug(
                                            f"[AAI] recv transcript final={is_final} text='{data.get('text','')}' conf={data.get('confidence')}"
                                        )
                                    self.produce_nonblocking(
                                        Transcription(
                                            message=data["text"],
                                            confidence=data.get("confidence", 0.0),
                                            is_final=is_final,
                                        )
                                    )

                    await asyncio.gather(sender(ws), receiver(ws))
            except Exception as e:
                if self.debug:
                    logger.debug(f"[AAI] websocket/connect error: {e}")

            if self._ended:
                break
            # Auto-retry on model deprecation
            if self._last_close_code == 4105:
                if not self.model or self.model != "universal":
                    self.model = "universal"
                    if self.debug:
                        logger.debug(
                            "[AAI] retrying with model=universal due to 4105 close code"
                        )
                    continue
                # If still 4105 and using v2, try v3 endpoint
                if not self._is_v3:
                    self.ws_url = ASSEMBLY_AI_V3_URL
                    self._is_v3 = True
                    if self.debug:
                        logger.debug(
                            f"[AAI] retrying with v3 endpoint url={self.get_assembly_ai_url()}"
                        )
                    continue
            # Exponential backoff before reconnect
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
