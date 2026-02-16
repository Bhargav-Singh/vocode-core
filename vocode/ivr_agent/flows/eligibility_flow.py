# subgraphs/eligibility.py
from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from vocode.ivr_agent.state.store import IVRState
from vocode.ivr_agent.state.models import (
    BinaryConfirmationOutputSchema, 
    DOBExtractorOutputSchema, 
    MultiClassValidatorOutputSchema,
    MemberIDExtractorOutputSchema
)
from vocode.ivr_agent.policies.prompt_templates import PROMPT
from vocode.ivr_agent.utilities.state_utils import get_reset_state_update
from vocode.ivr_agent.utilities.llm_initializer import LLM
import asyncio

MAX_RETRIES = 3

# --- Helper Functions ---
def get_retries(state: IVRState, key: str) -> int: 
    return state.get("retries", {}).get(key, 0)

def increment_retries(state: IVRState, key: str) -> dict:
    curr = state.get("retries", {}).copy()
    curr[key] = curr.get(key, 0) + 1
    return {"retries": curr}

# --- Nodes ---

async def ask_member_id(state: IVRState):
    retry_count = get_retries(state, "member_id")
    msg = "Eligibility Check. Please say the Member ID." if retry_count == 0 else "Please say the Member ID again."
    
    user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
    
    chain = PROMPT['MEMBER_ID_EXTRACTOR_SCHEMA_PROMPT'] | LLM.with_structured_output(MemberIDExtractorOutputSchema, include_raw=True)
    before_parsed = await chain.ainvoke({"user_input": user_input})
    parsed = before_parsed['parsed']

    # Global Commands
    if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
    if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

    # Retry Logic
    if not parsed.extracted_member_id:
        new_retries = increment_retries(state, "member_id")
        if new_retries["retries"]["member_id"] > MAX_RETRIES: 
            return Command(goto="set_status_transfer")
        return Command(goto="ask_member_id", update=new_retries)

    # Success -> Goto Confirm Member ID
    return Command(
        goto="confirm_member_id", 
        update={"last_user_input": user_input, "temp_member_id": parsed.extracted_member_id}
    )

async def confirm_member_id(state: IVRState):
    member_id = state.get("temp_member_id")
    # Safety check
    if not member_id: return Command(goto="ask_member_id")
    
    confirmation = interrupt({"message_to_play": f"ID is {member_id}. Correct?", "input_type": "single"})
    
    chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
    before_parsed = await chain.ainvoke({"user_input": confirmation})
    parsed = before_parsed['parsed']

    if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
    if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
    
    if parsed.confirmation == "1": 
        # Success -> Goto Ask DOB
        return Command(
            goto="ask_dob", 
            update={"last_user_input": "1", "member_id": member_id, "retries": {}}
        )
    
    # Retry Logic (Go back to Ask ID)
    new_retries = increment_retries(state, "member_id")
    if new_retries["retries"]["member_id"] > MAX_RETRIES: 
        return Command(goto="set_status_transfer")
    return Command(goto="ask_member_id", update=new_retries)

async def ask_dob(state: IVRState):
    retry_count = get_retries(state, "dob")
    msg = "Date of Birth?" if retry_count == 0 else "Please say Date of Birth again."
    
    user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
    
    chain = PROMPT['DOB_EXTRACTOR_SCHEMA_PROMPT'] | LLM.with_structured_output(DOBExtractorOutputSchema, include_raw=True)
    before_parsed = await chain.ainvoke({"user_input": user_input})
    parsed = before_parsed['parsed']
    
    if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
    if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
    
    # Retry Logic
    if not parsed.extracted_dob:
        new_retries = increment_retries(state, "dob")
        if new_retries["retries"]["dob"] > MAX_RETRIES: 
            return Command(goto="set_status_transfer")
        return Command(goto="ask_dob", update=new_retries)
    
    # Success -> Goto Confirm DOB
    return Command(
        goto="confirm_dob", 
        update={"last_user_input": parsed.extracted_dob, "temp_dob": parsed.extracted_dob}
    )

async def confirm_dob(state: IVRState):
    dob = state.get("temp_dob")
    if not dob: return Command(goto="ask_dob")
    
    confirmation = interrupt({"message_to_play": f"DOB is {dob}. Correct?", "input_type": "single"})
    
    chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
    before_parsed = await chain.ainvoke({"user_input": confirmation})
    parsed = before_parsed['parsed']
    
    if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
    if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

    if parsed.confirmation == "1": 
        # Success -> Goto Fetch Details
        return Command(
            goto="fetch_details", 
            update={"last_user_input": "1", "dob": dob, "retries": {}}
        )

    # Retry Logic (Go back to Ask DOB)
    new_retries = increment_retries(state, "dob")
    if new_retries["retries"]["dob"] > MAX_RETRIES: 
        return Command(goto="set_status_transfer")
    return Command(goto="ask_dob", update=new_retries)

async def fetch_details(state: IVRState):
    # Simulate DB lookup
    msg = "Member is Active. Now, what do you want are you want to 'repeat' or 'check another' or 'transfer the call to customer service' or 'main menu'."
    
    # Directly transition to input handler
    return Command(
        goto="handle_final",
        update={"message_to_play": msg}
    )

async def handle_final(state: IVRState):
    choice = interrupt({"message_to_play": state["message_to_play"], "input_type": "varied"})
    
    chain = PROMPT['MULTI_CLASS_VALIDATOR_SYSTEM_PROMPT'] | LLM.with_structured_output(MultiClassValidatorOutputSchema, include_raw=True)
    before_parsed = await chain.ainvoke({"user_input": choice})
    parsed = before_parsed['parsed']
    
    if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
    if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
    
    sel = parsed.selection
    
    if sel == "9": # Repeat
        new_retries = increment_retries(state, "fetch_details")
        if new_retries["retries"]["fetch_details"] > MAX_RETRIES: 
            return Command(goto="set_status_transfer")
        return Command(goto="fetch_details", update=new_retries)
    if sel == "8": # Check Another (Restart)
        return Command(goto="ask_member_id", update={
            "member_id": None, "dob": None, "npi": None, "user_role": None,
            "temp_member_id": None, "temp_dob": None, "temp_npi": None,
            "retries": {}
        })
    if sel == "4": # Transfer
        return Command(goto="set_status_transfer")
    
    # Default
    return Command(goto="set_status_main_menu")

# --- Cleanup Nodes ---

async def set_status_transfer(state): return get_reset_state_update("transfer")
async def set_status_main_menu(state): return get_reset_state_update("main_menu")

# --- Graph Construction ---

async def build_eligibility_subgraph():
    wf = StateGraph(IVRState)
    
    # Add Nodes
    wf.add_node("ask_member_id", ask_member_id)
    wf.add_node("confirm_member_id", confirm_member_id)
    wf.add_node("ask_dob", ask_dob)
    wf.add_node("confirm_dob", confirm_dob)
    wf.add_node("fetch_details", fetch_details)
    wf.add_node("handle_final", handle_final)
    
    wf.add_node("set_status_transfer", set_status_transfer)
    wf.add_node("set_status_main_menu", set_status_main_menu)
    
    # Entry Point
    wf.set_entry_point("ask_member_id")
    
    # Clean End Edges
    # (Complex logic removed because Command handles routing)
    wf.add_edge("set_status_transfer", END)
    wf.add_edge("set_status_main_menu", END)
    
    return wf.compile()      ## If anything issue occurred the graph state not changed then give the 'checkpointer=True' in the compile method like this graph.compile(checkpointer=True).

eligibility_graph = asyncio.run(build_eligibility_subgraph())