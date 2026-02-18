from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from vocode.ivr_agent.flows.intent_router import IntentRouter
from langgraph.types import Command

from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent
from vocode.streaming.agent.chat_gpt_agent import instantiate_openai_client
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from vocode.ivr_agent.utilities.llm_initializer import Gemini

# ----- Intialize Rephrase System prompt template -----

REPHRASE_SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You rewrite short call-center responses. Keep the exact meaning and keep all numbers, names, and dates unchanged. Do not add new questions or details. Return only the rewritten response."""),
    ("human", "{user_input}"),
])


class AsyncMixin:
    def __init__(self, *args, **kwargs):
        """
        Standard constructor used for arguments pass
        Do not override. Use __ainit__ instead
        """
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __ainit__(self, *args, **kwargs):
        """Async constructor, you should implement this"""

    async def __initobj(self):
        """Crutch used for __await__ after spawning"""
        assert not self.async_initialized
        self.async_initialized = True
        # pass the parameters to __ainit__ that passed to __init__
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self):
        return self.__initobj().__await__()


class IVRFlowAgent(AsyncMixin, RespondAgent[ChatGPTAgentConfig]):
    async def __ainit__(
        self,
        agent_config: ChatGPTAgentConfig,
        google_api_key: str,
        dry_run: bool = True,
        use_llm_rephrase: bool = True,
        **kwargs,
    ):
        # 1. Initialize the Parent Class Manually
        # Since the Mixin stopped the normal super() chain, we call this explicitly.
        RespondAgent.__init__(self, agent_config=agent_config, **kwargs)

        # 2. Set attributes (Logic moved from __init__)
        self.google_api_key = google_api_key
        self.dry_run = dry_run
        
        self.use_llm_rephrase = use_llm_rephrase

        # 3. initialize the gemini client
        self.gemini = Gemini(api_key=self.google_api_key)
        self.llm = self.gemini.get_llm()
        
        ## Get the parent graph
        self.intent_router = IntentRouter(self.llm)
        self.parent_graph = await self.intent_router.build_graph()

    def _get_model_name(self) -> str:
        return self.gemini.model_name()

    async def _rephrase(self, text: str) -> str:
        if not self.use_llm_rephrase or not self.gemini:
            return text
        try:
            rephrase_chain = REPHRASE_SYSTEM_PROMPT | self.llm

            response = await rephrase_chain.ainvoke({"user_input": text})

            content = response.content
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

        # 1. Setup LangGraph Config
        config = {"configurable": {"thread_id": conversation_id}}
        logger.info(f"Generating response for thread {conversation_id}: {human_input}")

        # 2. Check Graph State
        current_state = await self.parent_graph.aget_state(config)
        is_new_session = not current_state.next

        # 3. Handle Initialization (The Fix)
        if is_new_session:
            logger.info("Initializing new graph session")
            # Run until the first interrupt (The Greeting)
            await self.parent_graph.ainvoke({"dialogue_status": "active"}, config)
            
            # If the user hasn't said anything yet (just connecting), 
            # we stop here and return the Greeting.
            if not human_input.strip():
                snapshot = await self.parent_graph.aget_state(config)
                # Logic to extract greeting is shared below, so we can skip the 'resume' step
            else:
                # If user provided input immediately (barge-in), we resume immediately
                await self.parent_graph.ainvoke(Command(resume=human_input), config)
                snapshot = await self.parent_graph.aget_state(config)

        else:
            # 4. Standard Resume (Existing Session)
            # We pass the user's text to the node currently waiting at 'interrupt'
            await self.parent_graph.ainvoke(Command(resume=human_input), config)
            snapshot = await self.parent_graph.aget_state(config)


        # 5. Retrieve the Bot's Response (Matches your CLI logic)
        message_text = "Sorry I didn't get that." # Fallback

        if snapshot.tasks and snapshot.tasks[0].interrupts:
            # Retrieve the payload passed to interrupt({...}) in the node
            payload = snapshot.tasks[0].interrupts[0].value
            message_text = payload.get("message_to_play", "Sorry I didn't get that.")
            
            # Check for termination signal
            if payload.get("input_type") == "none":
                logger.info("Graph signaled end of conversation.")
                # We yield the final message, but we might want to signal termination logic here
        
        elif not snapshot.next:
            logger.info("Graph execution completed (END reached).")
            message_text = snapshot.values.get("message_to_play", "Thanks for using our service. Now end the call. Have a great day!")
            # Handle implicit end of flow if necessary

        # 6. Optional Polish
        if self.use_llm_rephrase and message_text:
            message_text = await self._rephrase(message_text)

        # 7. Yield Response
        yield GeneratedResponse(
            message=BaseMessage(text=message_text), 
            is_interruptible=True
        )

    def update_last_bot_message_on_cut_off(self, message: str):
        _ = message
        return
