from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from vocode.ivr_agent.flows.intent_router import classify_intent
from vocode.ivr_agent.flows.test_flow import handle_test_flow_input, start_test_flow
from vocode.ivr_agent.state.commands import apply_command, detect_command
from vocode.ivr_agent.state.models import FlowPhase, Intent
from vocode.ivr_agent.state.store import new_state, set_intent
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent
from vocode.streaming.agent.chat_gpt_agent import instantiate_openai_client
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage

REPHRASE_SYSTEM_PROMPT = (
    "You rewrite short call-center responses. Keep the exact meaning and keep all numbers, "
    "names, and dates unchanged. Do not add new questions or details. Return only the "
    "rewritten response."
)


class IVRTestFlowAgent(RespondAgent[ChatGPTAgentConfig]):
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
        if self.use_llm_rephrase:
            self.openai_client = instantiate_openai_client(agent_config)
            if not self.openai_client.api_key:
                raise ValueError("OPENAI_API_KEY must be set in environment or passed in")

    def _get_model_name(self) -> str:
        if self.agent_config.azure_params:
            return self.agent_config.azure_params.deployment_name
        return self.agent_config.model_name

    async def _rephrase(self, text: str) -> str:
        if not self.openai_client:
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

        command = detect_command(human_input)
        applied_phase = apply_command(self.state, command)
        if applied_phase == FlowPhase.TRANSFER:
            prompt = "Transferring you to a representative now."
        elif applied_phase == FlowPhase.INTENT_CAPTURE:
            prompt = "How can I assist you today?"
        else:
            if self.state.phase in (FlowPhase.CALL_START, FlowPhase.INTENT_CAPTURE):
                intent = classify_intent(human_input, allow_test_flow=True)
                if intent == Intent.TEST_FLOW:
                    set_intent(self.state, intent)
                    response = start_test_flow(self.state)
                    prompt = response.prompt
                else:
                    prompt = (
                        "Claim, authorization, and eligibility flows are not wired yet. "
                        "For now, say 'test' to run the test flow."
                    )
            elif self.state.phase == FlowPhase.TEST_FLOW:
                response = handle_test_flow_input(self.state, human_input, dry_run=self.dry_run)
                prompt = response.prompt
            elif self.state.phase == FlowPhase.RESULT_MENU:
                lowered = human_input.lower()
                if "repeat" in lowered and self.state.last_summary:
                    prompt = f"{self.state.last_summary} Say 'repeat' or 'start over'."
                else:
                    prompt = "Say 'repeat' to hear that again, or 'start over' to return to the main menu."
            elif self.state.phase == FlowPhase.TRANSFER:
                prompt = "Transferring you to a representative now."

        if self.use_llm_rephrase:
            prompt = await self._rephrase(prompt)

        yield GeneratedResponse(message=BaseMessage(text=prompt), is_interruptible=True)

    def update_last_bot_message_on_cut_off(self, message: str):
        _ = message
        return
