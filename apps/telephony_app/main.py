# Standard library imports
import os
import sys

from dotenv import load_dotenv

# Third-party imports
from fastapi import FastAPI
from loguru import logger
from pyngrok import ngrok
from openai import AsyncOpenAI

# Local application/library specific imports
from speller_agent import SpellerAgentFactory

from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.models.agent import AgentConfig

import tiktoken
from vocode.streaming.agent import token_utils

load_dotenv()

"""
===================================
Helps to use the custom OpenAI endpoint.
This overrides the library's internal client creation. 
Even if Vocode tries to connect to OpenAI, this redirects it to Nebius.
===================================
"""

original_init = AsyncOpenAI.__init__

def patched_init(self, *args, **kwargs):
    # Force the base_url to your custom endpoint
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    
    # Force the api_key to your custom key
    if os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    print(f"DEBUG: Intercepted AsyncOpenAI init. Redirecting to: {kwargs.get('base_url')}")
    original_init(self, *args, **kwargs)

AsyncOpenAI.__init__ = patched_init


"""
===================================
Helps to solve this Error:
num_tokens_from_messages() is not implemented for model meta-llama/Llama-3.3-70B-Instruct. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.
===================================
"""

original_get_tokenizer_info = token_utils.get_tokenizer_info

def patched_get_tokenizer_info(model: str):
    if "llama" in model.lower():
        # Use OpenAI's cl100k_base encoding as an approximation for Llama
        return token_utils.TokenizerInfo(
            encoding=tiktoken.get_encoding("cl100k_base"),
            tokens_per_message=3,
            tokens_per_name=1,
        )
    return original_get_tokenizer_info(model)

token_utils.get_tokenizer_info = patched_get_tokenizer_info



"""
===================================
As it is Code just change the agent_config
===================================
"""

configure_pretty_logging()

app = FastAPI(docs_url=None)

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
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello, How can i assist you today?"),
                prompt_preamble="Have a pleasant conversation about life",
                generate_responses=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model_name=os.getenv("OPENAI_MODEL_NAME"),
                base_url_override=os.getenv("OPENAI_BASE_URL"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE")),
            ),
            
            synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device(
                api_key=os.getenv("ELEVEN_LABS_API_KEY"),
            ),
            
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
        )
    ],
    agent_factory=SpellerAgentFactory(),
)

app.include_router(telephony_server.get_router())
