from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from vocode.ivr_agent.agent.utterance_parser import LLMUtteranceParser
from vocode.ivr_agent.flows.claim_flow import handle_claim_flow_input, start_claim_flow
from vocode.ivr_agent.flows.test_flow import handle_test_flow_input, start_test_flow
from vocode.ivr_agent.state.models import FlowPhase, Intent
from vocode.ivr_agent.state.store import increment_retry, new_state, reset_for_main_menu, set_intent
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent
from vocode.streaming.agent.chat_gpt_agent import instantiate_openai_client
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage

REPHRASE_SYSTEM_PROMPT = (
    "You rewrite short call-center responses. Keep the exact meaning and keep all numbers, "
    "names, and dates unchanged. Do not add new questions or details. Return only the "
    "rewritten response."
)


class IVRFlowAgent(RespondAgent[ChatGPTAgentConfig]):
    def __init__(
        self,
        agent_config: ChatGPTAgentConfig,
        dry_run: bool = True,
        use_llm_rephrase: bool = True,
        **kwargs,
    ):
        super().__init__(agent_config=agent_config, **kwargs)
        self.state = new_state()
        self.state.phase = FlowPhase.INTENT_CAPTURE
        self.dry_run = dry_run
        self.use_llm_rephrase = use_llm_rephrase
        self.openai_client: Optional[AsyncOpenAI | AsyncAzureOpenAI] = None
        self.parser: Optional[LLMUtteranceParser] = None
        self.openai_client = instantiate_openai_client(agent_config)
        if not self.openai_client.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.parser = LLMUtteranceParser(self.openai_client, self._get_model_name())

    def _get_model_name(self) -> str:
        if self.agent_config.azure_params:
            return self.agent_config.azure_params.deployment_name
        return self.agent_config.model_name

    async def _rephrase(self, text: str) -> str:
        if not self.use_llm_rephrase or not self.openai_client:
            return text
        try:
            response = await self.openai_client.chat.completions.create(
                model=self._get_model_name(),
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": REPHRASE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            content = response.choices[0].message.content if response.choices else None
            return content.strip() if content else text
        except Exception:
            logger.exception("LLM rephrase failed; using original prompt.")
            return text

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        _ = conversation_id, is_interrupt, bot_was_in_medias_res
        prompt = "How can I assist you today?"

        parsed = await self.parser.parse(
            human_input,
            phase=self.state.phase,
            current_field=self.state.current_field,
            awaiting_confirmation=self.state.awaiting_confirmation,
        )

        if parsed.command in ("START_OVER", "MAIN_MENU"):
            reset_for_main_menu(self.state)
            prompt = "How can I assist you today?"
        elif parsed.command == "TRANSFER":
            self.state.phase = FlowPhase.TRANSFER
            prompt = "Transferring you to a representative now."
        else:
            if self.state.phase in (FlowPhase.CALL_START, FlowPhase.INTENT_CAPTURE):
                if parsed.intent == Intent.CLAIM_STATUS:
                    set_intent(self.state, parsed.intent)
                    response = start_claim_flow(self.state)
                    prompt = response.prompt
                elif parsed.intent == Intent.TEST_FLOW:
                    set_intent(self.state, parsed.intent)
                    response = start_test_flow(self.state)
                    prompt = response.prompt
                else:
                    retries = increment_retry(self.state, "intent")
                    if retries >= 3:
                        self.state.phase = FlowPhase.TRANSFER
                        prompt = "Transferring you to a representative now."
                    else:
                        prompt = (
                            "I can help with claim status, authorization status, or eligibility. "
                            "Which would you like?"
                        )
            elif self.state.phase == FlowPhase.CLAIM_FLOW:
                response = handle_claim_flow_input(self.state, parsed, dry_run=self.dry_run)
                prompt = response.prompt
            elif self.state.phase == FlowPhase.TEST_FLOW:
                response = handle_test_flow_input(self.state, human_input, dry_run=self.dry_run)
                prompt = response.prompt
            elif self.state.phase == FlowPhase.RESULT_MENU:
                action = parsed.result_action
                if action == "NONE" and parsed.command == "REPEAT":
                    action = "REPEAT"
                if action == "REPEAT" and self.state.last_summary:
                    prompt = f"{self.state.last_summary} Say 'repeat' or 'start over'."
                elif action == "ANOTHER":
                    if self.state.intent == Intent.CLAIM_STATUS:
                        response = start_claim_flow(self.state)
                        prompt = response.prompt
                    elif self.state.intent == Intent.TEST_FLOW:
                        response = start_test_flow(self.state)
                        prompt = response.prompt
                    else:
                        reset_for_main_menu(self.state)
                        prompt = "How can I assist you today?"
                elif action == "MAIN_MENU":
                    reset_for_main_menu(self.state)
                    prompt = "How can I assist you today?"
                elif action == "TRANSFER":
                    self.state.phase = FlowPhase.TRANSFER
                    prompt = "Transferring you to a representative now."
                else:
                    prompt = (
                        "Would you like me to repeat that, check another claim, go to the main menu, "
                        "or transfer you to customer service?"
                    )
            elif self.state.phase == FlowPhase.TRANSFER:
                prompt = "Transferring you to a representative now."

        if self.use_llm_rephrase:
            prompt = await self._rephrase(prompt)

        yield GeneratedResponse(message=BaseMessage(text=prompt), is_interruptible=True)

    def update_last_bot_message_on_cut_off(self, message: str):
        _ = message
        return
