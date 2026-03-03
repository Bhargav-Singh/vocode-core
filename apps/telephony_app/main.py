from pathlib import Path
import sys
# Ensure repo root is on sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Standard library imports
import os
import sys
import uvicorn

from dotenv import load_dotenv

# Third-party imports
from fastapi import FastAPI
from loguru import logger
from pyngrok import ngrok

# Local application/library specific imports
from speller_agent import SpellerAgentFactory

from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig, IVRAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.synthesizer.google_synthesizer import GoogleSynthesizer
from vocode.streaming.models.synthesizer import GoogleSynthesizerConfig
from vocode.streaming.transcriber.google_transcriber import GoogleTranscriber
from vocode.streaming.models.transcriber import GoogleTranscriberConfig, DeepgramTranscriberConfig, AssemblyAITranscriberConfig, GladiaTranscriberConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramEndpointingConfig, TimeSilentConfig

# if running from python, this will load the local .env
# docker-compose will load the .env file by itself
load_dotenv()

configure_pretty_logging()

app = FastAPI()

config_manager = RedisConfigManager()

BASE_URL = os.getenv("BASE_URL")

if not BASE_URL:
    ngrok_auth = os.environ.get("NGROK_AUTH_TOKEN")
    if ngrok_auth is not None:
        ngrok.set_auth_token(ngrok_auth)
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 3000

    # Open a ngrok tunnel to the dev server
    BASE_URL = ngrok.connect(port).public_url.replace("https://", "")
    logger.info('ngrok tunnel "{}" -> "http://127.0.0.1:{}"'.format(BASE_URL, port))

if not BASE_URL:
    raise ValueError("BASE_URL must be set in environment if not using pyngrok")

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=IVRAgentConfig(
                initial_message=BaseMessage(
                    text=(
                        "Thank you for calling. I can help you check claim status, "
                        "authorization status, or member eligibility. How can I assist you today?"
                    )
                ),
                prompt_preamble="You are a helpful IVR assistant.", # Simplistic preamble, real logic is in graph
                allow_initial_message_to_be_cut_off=True,
                interrupt_sensitivity="high",
                num_check_human_present_times=4,
                allowed_idle_time_seconds=10,
                google_api_key=os.environ["GOOGLE_API_KEY"],
            ),
            # transcriber_config=GoogleTranscriberConfig.from_telephone_input_device(
            #             api_key=os.environ["GOOGLE_API_KEY"],
            #         ),
            
            # transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
            #     model="nova-2", # Optimized for 8kHz telephony
            #     language="en-us",
            #     endpointing_config=DeepgramEndpointingConfig(
            #         vad_threshold_ms=1000,
            #         utterance_cutoff_ms=1000,
            #         time_silent_config=TimeSilentConfig(
            #             time_cutoff_seconds=1.0,
            #             post_punctuation_time_seconds=0.8,
            #         ),
            #     ),
            #     keywords=["claim", "status", "auth", "status", "eligibility", "eligible", "feb", "august", "double", "patient", "provider", "date of birth", "service date", "member id", "npi", "tax", "id"],
            #     api_key=os.environ["DEEPGRAM_API_KEY"],
            # ),

            transcriber_config=AssemblyAITranscriberConfig.from_telephone_input_device(
                # Newer AssemblyAI realtime requires selecting a model (e.g., "universal")
                model="universal",
                # v3 Streaming optional params
                end_of_turn_confidence_threshold=0.95,
                # The confidence threshold for triggering an "end of turn" signal.
                ws_url="wss://streaming.assemblyai.com/v3/ws",
                format_turns=False,    
                # Format Turns becomes false because for formatting the transcription used the LLM model which takes more time.
                min_end_of_turn_silence_when_confident_ms=1000,
                # The minimum amount of silence required to trigger an "end of turn" signal once the confidence threshold is met.
                mute_during_speech=False,
                max_turn_silence_ms=1200,
                keyterms_prompt=["claim", "status", "auth", "authorization", "eligibility", "eligible", "feb", "august", "double", "patient", "provider", "date of birth", "service date", "member", "npi", "tax", "id", "august", "february", "december", "January"],
            ),

            synthesizer_config=GoogleSynthesizerConfig.from_telephone_output_device(
                        language_code="en-US",
                        voice_name="en-US-Neural2-D",
                        pitch=0,
                        speaking_rate=1.00,
                        api_key=os.environ["GOOGLE_API_KEY"],
                    ),

            # uncomment this to use the speller agent instead
            # agent_config=SpellerAgentConfig(
            #     initial_message=BaseMessage(
            #         text="im a speller agent, say something to me and ill spell it out for you"
            #     ),
            #     generate_responses=False,
            # ),
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
        )
    ],
    agent_factory=SpellerAgentFactory(),
)

app.include_router(telephony_server.get_router())


if __name__ == "__main__":
    # This starts the server and stops the script from exiting
    uvicorn.run(app, host="0.0.0.0", port=3000)