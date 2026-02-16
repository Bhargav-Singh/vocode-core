from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from vocode.ivr_agent.flows.intent_router import parent_graph
from langgraph.types import Command

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
        # 1. Intitialize the parent class
        super().__init__(agent_config=agent_config, **kwargs)

        # 2. Logical "RAM" of the agent
        # self.state = new_state()
        # self.state.phase = FlowPhase.INTENT_CAPTURE

        # 3. Specific flags
        self.dry_run = dry_run     # If true, don't actually call real backends
        self.use_llm_rephrase = use_llm_rephrase  # "Polish" the text?
        self.google_api_key = google_api_key

        # 4. The "Ears" and "Understanding"
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
        
        # 1. Setup LangGraph Config with Conversation ID
        config = {"configurable": {"thread_id": conversation_id}}
        
        logger.info(f"Generating response for thread {conversation_id}: {human_input}")

        # 2. Check Graph State (Prime if New)
        # If this thread has no history, we must initialize it.
        # The initialization runs the graph up to the first 'interrupt' (Greeting Node).
        current_state = await parent_graph.aget_state(config)
        
        if not current_state.next:
            logger.info("Initializing new graph session")
            # This executes 'greet_and_listen' and pauses at the interrupt
            await parent_graph.ainvoke({"dialogue_status": "active"}, config)

        # 3. Resume Graph with User Input
        # We pass the user's text to the node currently waiting at 'interrupt'
        await parent_graph.ainvoke(Command(resume=human_input), config)

        # 4. Retrieve the Bot's Response
        # The graph has now run to the NEXT interrupt (or finished).
        # We inspect the current state to find the 'message_to_play'.
        snapshot = await parent_graph.aget_state(config)
        
        message_text = "Sorry i don't getting your answer." # Fallback
        
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            # Retrieve the payload passed to interrupt({...}) in the node
            payload = snapshot.tasks[0].interrupts[0].value
            message_text = payload.get("message_to_play", "")
            
            # Check for termination signal
            if payload.get("input_type") == "none":
                logger.info("Graph signaled end of conversation.")
                # We yield the final message, but we might want to signal termination logic here
        
        elif not snapshot.next:
            logger.info("Graph execution completed (END reached).")
            # Handle implicit end of flow if necessary

        # 5. Optional Polish
        if self.use_llm_rephrase and message_text:
            message_text = await self._rephrase(message_text)

        # 6. Yield Response
        yield GeneratedResponse(
            message=BaseMessage(text=message_text), 
            is_interruptible=True
        )

    def update_last_bot_message_on_cut_off(self, message: str):
        _ = message
        return
