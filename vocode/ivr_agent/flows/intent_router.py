# Intent Router Flow (Parent Graph Flow)

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from vocode.ivr_agent.state.store import IVRState
from vocode.ivr_agent.policies.prompt_templates import PROMPT
from vocode.ivr_agent.utilities.state_utils import get_reset_state_update

# --- Import Compiled Subgraphs ---
from vocode.ivr_agent.flows.claim_flow import ClaimFlow
from vocode.ivr_agent.flows.eligibility_flow import EligibilityFlow
from vocode.ivr_agent.flows.auth_flow import AuthFlow

# --- Import Schema ---
# (Assuming you put the class above in your schemas file)
from vocode.ivr_agent.state.models import IntentClassifierOutputSchema, CustomerServiceOutputSchema

import asyncio


class IntentRouter:
    def __init__(self, LLM):
        self.LLM = LLM
        self.MAX_RETRIES = 3

        ## Initialize Subgraphs objects
        self.claim_flow = ClaimFlow(LLM)
        self.eligibility_flow = EligibilityFlow(LLM)
        self.auth_flow = AuthFlow(LLM)

    # --- Helper Functions ---
    def get_retries(self, state: IVRState, key: str) -> int: 
        return state.get("retries", {}).get(key, 0)
    
    def increment_retries(self, state: IVRState, key: str) -> dict:
        curr = state.get("retries", {}).copy()
        curr[key] = curr.get(key, 0) + 1
        return {"retries": curr}

    # --- Building Graph ---
    async def build_graph(self):
        builder = StateGraph(IVRState)

        ## Get Subgraphs
        self.claims_graph = await self.claim_flow.build_graph()
        self.eligibility_graph = await self.eligibility_flow.build_graph()
        self.auth_graph = await self.auth_flow.build_graph()

        # --- Nodes ---

        async def greet_and_listen(state: IVRState):
            """
            The Entry Point. Plays the menu and listens for the first command.
            """
            # Logic: If returning from Main Menu, maybe say "What else?" 
            # For now, we keep it standard.

            retry_count = self.get_retries(state, "greet")

            # 1. Determine Message based on retries
            if retry_count == 0:
                msg = "Hello, I am your AI assistant. I can help with Claims, Eligibility, or Authorization. How can I help you today?"
            else:
                msg = "Sorry, I didn't quite catch that. <break time='300ms'/> Please tell me if you need help with Claims, Eligibility, or Authorization."

            # 2. Wait for user input
            user_input = interrupt({
                "message_to_play": msg,
                "input_type": "varied"
            })

            # 3. Check for Empty Input / Silence (Simple Validation)
            # If input is empty, treat as failure and loop back
            if not user_input or not user_input.strip():
                new_retries = self.increment_retries(state, "greet")
                
                # Check Max Retries
                if new_retries["retries"]["greet"] > self.MAX_RETRIES:
                    # Too many failures -> Transfer
                    return Command(goto="transfer_handoff")
                
                # Loop back to self with incremented retry count
                return Command(goto="greet_and_listen", update=new_retries)

            # 4. Valid Input Received -> Move to Classification
            # Reset retries on success so next time we come here (e.g. from main menu) it's fresh
            return Command(
                goto="classify_intent", 
                update={"last_user_input": user_input, "dialogue_status": "active"}
            )
            
        async def classify_intent(state: IVRState):
            """
            Analyzes the user's input to decide which Flow (Subgraph) to activate.
            """
            user_input = state.get("last_user_input", "")
            
            # 1. LLM Classification
            structured_llm = self.LLM.with_structured_output(IntentClassifierOutputSchema, include_raw=True)
            chain = PROMPT['INTENT_CLASSIFIER_PROMPT'] | structured_llm
            
            result = await chain.ainvoke({"user_input": user_input})
            parsed = result['parsed']
            
            intent = parsed.intent # CLAIM, ELIGIBILITY, AUTH, TRANSFER, UNKNOWN

            # 2. Handle Unknown Intent as a "Retry" 
            # If the LLM says "UNKNOWN", we treat it as a failed attempt at the greeting stage
            if intent == "UNKNOWN":
                new_retries = self.increment_retries(state, "greet")
                if new_retries["retries"]["greet"] > self.MAX_RETRIES:
                     return Command(goto="transfer_handoff")
                return Command(goto="greet_and_listen", update=new_retries)
            
            # 3. Valid Intent -> Update State and Route
            return Command(
                goto="route_launch",
                update={
                    "intent": intent, 
                    "flow_name": intent if intent in ["CLAIM", "ELIGIBILITY", "AUTH"] else None,
                    "retries": {}
                }
            )

        async def transfer_handoff_node(state: IVRState):
            """
            Final node if the user wants a human.
            """

            # user_input = interrupt({
            #     "message_to_play": "Please hold while I transfer you to a representative.....",
            #     "input_type": "none", # 'none' signal to telephony to hangup/transfer
            # })

            message = """
            Please hold while I transfer you to a representative..... 
            <break time='1s'/> 

            <audio src="https://jazlynn-nonfictive-wade.ngrok-free.dev/mixkit-office-telephone-ring-1350.wav">
                Transferring now.
            </audio>

            <break time='500ms'/>
            Hello, I am John from QuickCap. How can i help you?
            """

            return Command(goto="execute_transfer", update={"last_user_input": message})

        
        async def execute_transfer(state: IVRState):
            """
            Executes the transfer to a human agent.
            """

            message_text = state.get("last_user_input", "Hello, I am John from QuickCap. How can i help you?")

            user_input = interrupt({
                "message_to_play": message_text,
                "input_type": "varied",
            })

            structured_llm = self.LLM.with_structured_output(CustomerServiceOutputSchema, include_raw=True)
            chain = PROMPT['CUSTOMER_SERVICE_PROMPT'] | structured_llm
            
            result = await chain.ainvoke({"user_input": user_input})
            parsed = result['parsed']

            ## check user wants to exit
            is_exit = parsed.is_exit

            if is_exit == "1" or is_exit == "True":
                return Command(goto="final_node")
            
            else:
                return Command(goto="execute_transfer", update={"last_user_input": parsed.response})

        async def final_node(state: IVRState):
            """
            Final node if the user wants to exit.
            """
            return {
                "message_to_play": "Thank you for calling QuickCap. Goodbye!",
                "input_type": "none", # 'none' signal to telephony to hangup/transfer
                "dialogue_status": "complete"
            }

        async def fallback_node(state: IVRState):
            """
            Handles 'UNKNOWN' intent.
            """
            return {
                "message_to_play": "I'm sorry, I didn't understand that. You can say check claim status, check eligibility, or authorization status.",
                "input_type": "none" # The router will loop this back to greet_and_listen usually
            }

        # --- Routers ---

        async def route_launch(state: IVRState):
            """
            Decides which subgraph to enter based on the classified intent.
            """
            intent = state.get("intent")
            
            if intent == "CLAIM":
                return Command(goto="claims_flow")
            elif intent == "ELIGIBILITY":
                return Command(goto="eligibility_flow")
            elif intent == "AUTH":
                return Command(goto="auth_flow")
            elif intent == "TRANSFER":
                return Command(goto="transfer_handoff")
            
            # If Unknown, loop back (or you could route to fallback_node first)
            return Command(goto="greet_and_listen")

        async def route_after_subgraph(state: IVRState) -> Literal["greet_and_listen", "transfer_handoff", END]:
            """
            Decides what to do when a Subgraph finishes.
            It checks the 'dialogue_status' set by the subgraph's cleanup node.
            """
            status = state.get("dialogue_status")
            
            print(f"DEBUG: Subgraph finished with status: {status}")
            
            if status == "main_menu":
                # Loop back to start
                return "greet_and_listen"
            
            if status == "transfer":
                # Go to transfer logic
                return "transfer_handoff"
                
            # Default: End call if unknown status
            return END

   
        # --- Add Nodes ---
        builder.add_node("greet_and_listen", greet_and_listen)
        builder.add_node("classify_intent", classify_intent)
        builder.add_node("transfer_handoff", transfer_handoff_node)
        builder.add_node("route_launch", route_launch)
        builder.add_node("execute_transfer", execute_transfer)
        builder.add_node("final_node", final_node)
        
        # --- Add Subgraphs as Nodes ---
        builder.add_node("claims_flow", self.claims_graph)
        builder.add_node("eligibility_flow", self.eligibility_graph)
        builder.add_node("auth_flow", self.auth_graph)
        
        # --- Edges ---
        # 1. Start -> Greet
        builder.add_edge(START, "greet_and_listen")
                
        # 2. Subgraphs -> Router (Return from Subgraph)
        builder.add_conditional_edges("claims_flow", route_after_subgraph)
        builder.add_conditional_edges("eligibility_flow", route_after_subgraph)
        builder.add_conditional_edges("auth_flow", route_after_subgraph)
        
        # 3. Transfer -> End
        builder.add_edge("final_node", END)
        
        # --- Compile ---
        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)
