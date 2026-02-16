from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from vocode.ivr_agent.state.store import IVRState
from vocode.ivr_agent.policies.prompt_templates import PROMPT
from vocode.ivr_agent.utilities.llm_initializer import LLM
from vocode.ivr_agent.utilities.state_utils import get_reset_state_update

# --- Import Compiled Subgraphs ---
from vocode.ivr_agent.flows.claim_flow import claims_graph
from vocode.ivr_agent.flows.eligibility_flow import eligibility_graph
from vocode.ivr_agent.flows.auth_flow import auth_graph

# --- Import Schema ---
# (Assuming you put the class above in your schemas file)
from vocode.ivr_agent.state.models import IntentClassifierOutputSchema

import asyncio

# ==========================================
# 1. NODES
# ==========================================

async def greet_and_listen(state: IVRState):
    """
    The Entry Point. Plays the menu and listens for the first command.
    """
    # Logic: If returning from Main Menu, maybe say "What else?" 
    # For now, we keep it standard.
    
    payload = {
        "message_to_play": "Hello, I am your AI assistant. I can help with Claims, Eligibility, or Authorization. How can I help you today?",
        "input_type": "varied"
    }
    
    # 1. Wait for user input
    user_input = interrupt(payload)
    
    return {"last_user_input": user_input, "dialogue_status": "active"}

async def classify_intent_node(state: IVRState):
    """
    Analyzes the user's input to decide which Flow (Subgraph) to activate.
    """
    user_input = state.get("last_user_input", "")
    
    # 1. LLM Classification
    structured_llm = LLM.with_structured_output(IntentClassifierOutputSchema, include_raw=True)
    chain = PROMPT['INTENT_CLASSIFIER_PROMPT'] | structured_llm
    
    result = await chain.ainvoke({"user_input": user_input})
    parsed = result['parsed']
    
    intent = parsed.intent # CLAIM, ELIGIBILITY, AUTH, TRANSFER, UNKNOWN
    
    # 2. Update State
    # We set 'flow_name' so we know where we are, but 'intent' drives the immediate router
    return {
        "intent": intent, 
        "flow_name": intent if intent in ["CLAIM", "ELIGIBILITY", "AUTH"] else None
    }

async def transfer_handoff_node(state: IVRState):
    """
    Final node if the user wants a human.
    """
    return {
        "message_to_play": "Please hold while I transfer you to a representative.",
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

# ==========================================
# 2. ROUTERS
# ==========================================

async def route_launch(state: IVRState) -> Literal["claims_flow", "eligibility_flow", "auth_flow", "transfer_handoff", "greet_and_listen"]:
    """
    Decides which subgraph to enter based on the classified intent.
    """
    intent = state.get("intent")
    
    if intent == "CLAIM":
        return "claims_flow"
    elif intent == "ELIGIBILITY":
        return "eligibility_flow"
    elif intent == "AUTH":
        return "auth_flow"
    elif intent == "TRANSFER":
        return "transfer_handoff"
    
    # If Unknown, loop back (or you could route to fallback_node first)
    return "greet_and_listen"

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

# ==========================================
# 3. GRAPH CONSTRUCTION
# ==========================================

async def build_parent_graph():
    builder = StateGraph(IVRState)
    
    # --- Add Nodes ---
    builder.add_node("greet_and_listen", greet_and_listen)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("transfer_handoff", transfer_handoff_node)
    
    # --- Add Subgraphs as Nodes ---
    # When execution hits these nodes, it enters the compiled subgraph
    builder.add_node("claims_flow", claims_graph)
    builder.add_node("eligibility_flow", eligibility_graph)
    builder.add_node("auth_flow", auth_graph)
    
    # --- Edges ---
    # 1. Start -> Greet
    builder.add_edge(START, "greet_and_listen")
    
    # 2. Greet -> Classify
    builder.add_edge("greet_and_listen", "classify_intent")
    
    # 3. Classify -> Router (Enter Subgraph)
    builder.add_conditional_edges(
        "classify_intent", 
        route_launch
    )
    
    # 4. Subgraphs -> Router (Return from Subgraph)
    # All subgraphs share the same exit logic router
    builder.add_conditional_edges("claims_flow", route_after_subgraph)
    builder.add_conditional_edges("eligibility_flow", route_after_subgraph)
    builder.add_conditional_edges("auth_flow", route_after_subgraph)
    
    # 5. Transfer -> End
    builder.add_edge("transfer_handoff", END)
    
    # --- Compile ---
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)

# Create the executable graph
parent_graph = asyncio.run(build_parent_graph())