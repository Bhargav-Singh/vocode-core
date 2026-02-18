# Authorization Flow

from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from vocode.ivr_agent.state.store import IVRState
from vocode.ivr_agent.state.models import (
    BinaryConfirmationOutputSchema, 
    DOBExtractorOutputSchema, 
    MultiClassValidatorOutputSchema,
    MemberIDExtractorOutputSchema,
    NPIExtractorOutputSchema,
    RoleClassifierOutputSchema
)
from vocode.ivr_agent.policies.prompt_templates import PROMPT
from vocode.ivr_agent.utilities.state_utils import get_reset_state_update
import asyncio



class AuthFlow:
    def __init__(self, LLM):
        self.LLM = LLM
        self.MAX_RETRIES = 3

    # --- Helper Functions ---
    def get_retries(self, state: IVRState, key: str) -> int: 
        return state.get("retries", {}).get(key, 0)
    
    def increment_retries(self, state: IVRState, key: str) -> dict:
        curr = state.get("retries", {}).copy()
        curr[key] = curr.get(key, 0) + 1
        return {"retries": curr}

    async def build_graph(self):
        wf = StateGraph(IVRState)


        # --- Nodes ---

        async def ask_member_id(state: IVRState):
            retry_count = self.get_retries(state, "member_id")
            msg = "Auth Status. Please tell me the Member ID." if retry_count == 0 else "I didn't quite catch that. Please say the Member ID again, numbers only."
            
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
            
            chain = PROMPT['MEMBER_ID_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(MemberIDExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            # Global Commands
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

            # Retry Logic
            if not parsed.extracted_member_id:
                new_retries = self.increment_retries(state, "member_id")
                if new_retries["retries"]["member_id"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_member_id", update=new_retries)

            # Success -> Goto Confirm ID
            return Command(
                goto="confirm_member_id", 
                update={"last_user_input": user_input, "temp_member_id": parsed.extracted_member_id}
            )

        async def confirm_member_id(state: IVRState):
            member_id = state.get("temp_member_id")
            if not member_id: return Command(goto="ask_member_id")

            confirmation = interrupt({"message_to_play": f"Member ID is {member_id}. Correct?", "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
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
            new_retries = self.increment_retries(state, "member_id")
            if new_retries["retries"]["member_id"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="ask_member_id", update=new_retries)

        async def ask_dob(state: IVRState):
            retry_count = self.get_retries(state, "dob")
            msg = "Please give me the Date of Birth." if retry_count == 0 else "Please say the Date of Birth again."

            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})

            chain = PROMPT['DOB_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(DOBExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            
            # Retry Logic
            if not parsed.extracted_dob:
                new_retries = self.increment_retries(state, "dob")
                if new_retries["retries"]["dob"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_dob", update=new_retries)
            
            # Success -> Goto Confirm DOB
            return Command(
                goto="confirm_dob", 
                update={"last_user_input": parsed.extracted_dob, "temp_dob": parsed.extracted_dob}
            )

        async def confirm_dob(state: IVRState):
            dob_date = state.get("temp_dob")
            if not dob_date: return Command(goto="ask_dob") 

            confirmation = interrupt({"message_to_play": f"DOB is {dob_date}. Correct?", "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

            if parsed.confirmation == "1":
                # Success -> Goto Patient/Provider Check
                return Command(
                    goto="ask_patient_provider", 
                    update={"last_user_input": "1", "dob": dob_date, "retries": {}}
                )
            
            # Retry Logic (Go back to Ask DOB)
            new_retries = self.increment_retries(state, "dob")
            if new_retries["retries"]["dob"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="ask_dob", update=new_retries)

        async def ask_patient_provider(state: IVRState):
            # Ask Role
            user_input = interrupt({"message_to_play": "Are you a patient or a provider?", "input_type": "varied"})

            chain = PROMPT['ROLE_CLASSIFIER_LOGIC_PROMPT'] | self.LLM.with_structured_output(RoleClassifierOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

            role = parsed.user_role # "1" (Patient) or "2" (Provider)
            
            if role == "1": # Patient -> Skip NPI, go to Fetch
                return Command(
                    goto="fetch_auth_details", 
                    update={"user_role": role, "npi": None, "retries": {}}
                )
            elif role == "2": # Provider -> Ask NPI
                return Command(
                    goto="ask_npi", 
                    update={"user_role": role, "retries": {}}
                )
            
            # Unclear -> Retry this question (Simple Loop)
            new_retries = self.increment_retries(state, "patient_provider")
            if new_retries["retries"]["patient_provider"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="ask_patient_provider", update=new_retries)

        async def ask_npi(state: IVRState):
            retry_count = self.get_retries(state, "npi")
            msg = "Please say the NPI Number." if retry_count == 0 else "Please say the NPI again."
            
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
            
            chain = PROMPT['NPI_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(NPIExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

            # Retry Logic
            if not parsed.extracted_npi:
                new_retries = self.increment_retries(state, "npi")
                if new_retries["retries"]["npi"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_npi", update=new_retries)

            # Success -> Goto Confirm NPI
            return Command(
                goto="confirm_npi", 
                update={"last_user_input": parsed.extracted_npi, "temp_npi": parsed.extracted_npi}
            )

        async def confirm_npi(state: IVRState):
            npi = state.get("temp_npi")
            if not npi: return Command(goto="ask_npi")

            confirmation = interrupt({"message_to_play": f"NPI is {npi}. Correct?", "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

            if parsed.confirmation == "1": 
                # Success -> Goto Fetch
                return Command(
                    goto="fetch_auth_details", 
                    update={"last_user_input": "1", "npi": npi, "retries": {}}
                )

            # Retry Logic (Go back to Ask NPI)
            new_retries = self.increment_retries(state, "npi")
            if new_retries["retries"]["npi"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="ask_npi", update=new_retries)

        async def fetch_auth_details(state: IVRState):
            # Simulate DB Logic
            status = "Pending" 
            msg = f"Your auth status is {status}. Now, what do you want are you want to 'repeat' or 'check another' or 'transfer the call to customer service' or 'main menu'."
            
            # Transition to input handler
            return Command(
                goto="handle_final_input",
                update={"message_to_play": msg}
            )

        async def handle_final_input(state: IVRState):
            choice = interrupt({"message_to_play": state.get("message_to_play"), "input_type": "varied"})

            chain = PROMPT['MULTI_CLASS_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(MultiClassValidatorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": choice})
            parsed = before_parsed['parsed']

            # Global Commands
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")

            # Local Menu Logic
            selection = parsed.selection 
            
            if selection == "9": # Repeat
                new_retries = self.increment_retries(state, "fetch_details")
                if new_retries["retries"]["fetch_details"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="fetch_auth_details", update=new_retries)
            elif selection == "8": # Check Another (Restart Flow)
                return Command(goto="ask_member_id", update={
                    "member_id": None, "dob": None, "npi": None, "user_role": None,
                    "temp_member_id": None, "temp_dob": None, "temp_npi": None,
                    "retries": {}
                })
            elif selection == "4": # Transfer
                return Command(goto="set_status_transfer")
            
            # Default
            return Command(goto="set_status_main_menu")

        # --- Cleanup Nodes ---

        async def set_status_transfer(state: IVRState):
            return get_reset_state_update("transfer")

        async def set_status_main_menu(state: IVRState):
            return get_reset_state_update("main_menu")

    
        # Add Nodes
        wf.add_node("ask_member_id", ask_member_id)
        wf.add_node("confirm_member_id", confirm_member_id)
        wf.add_node("ask_dob", ask_dob)
        wf.add_node("confirm_dob", confirm_dob)
        wf.add_node("ask_patient_provider", ask_patient_provider)
        wf.add_node("ask_npi", ask_npi)     
        wf.add_node("confirm_npi", confirm_npi) 
        wf.add_node("fetch_auth_details", fetch_auth_details) 
        wf.add_node("handle_final_input", handle_final_input)
        
        wf.add_node("set_status_transfer", set_status_transfer)
        wf.add_node("set_status_main_menu", set_status_main_menu)

        wf.set_entry_point("ask_member_id")

        # Clean End Edges
        wf.add_edge("set_status_transfer", END)
        wf.add_edge("set_status_main_menu", END)

        # Note: We remove `checkpointer=True` from compile here.
        # Checkpointing is usually handled by the Parent Graph.
        # If you need it locally for testing, you can add it back.
        return wf.compile()
