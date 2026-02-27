# Eligibility Flow
import json, os
from typing import Literal, Optional, List, Dict
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from vocode.ivr_agent.state.store import IVRState
from vocode.ivr_agent.state.models import (
    BinaryConfirmationOutputSchema, 
    DATEExtractorOutputSchema, 
    MultiClassValidatorOutputSchema,
    MemberIDExtractorOutputSchema
)
from vocode.ivr_agent.policies.prompt_templates import PROMPT
from vocode.ivr_agent.utilities.state_utils import get_reset_state_update
import asyncio
from vocode.ivr_agent.state.commands import number_mappings


class EligibilityFlow:
    def __init__(self, LLM):
        self.MAX_RETRIES = 3
        self.LLM = LLM
        db_path = os.path.join(os.getcwd(), "vocode/ivr_agent/data/eligibility_db.json")
        with open(db_path, "r") as f:
            self.db = json.load(f)

    # --- Helper Functions ---
    def get_retries(self, state: IVRState, key: str) -> int: 
        return state.get("retries", {}).get(key, 0)

    def increment_retries(self, state: IVRState, key: str) -> dict:
        curr = state.get("retries", {}).copy()
        curr[key] = curr.get(key, 0) + 1
        return {"retries": curr}

    def _get_validated_eligibility(self, member_id: str, dob: str) -> Optional[List[Dict]]:
        """
        Validates the user's inputs against the Eligibility JSON DB.
        Returns the eligibility list if successful, or None if validation fails.
        """
        patient_data = self.db.get(member_id)
        
        # 1. Check if Member ID exists
        if not patient_data:
            return None
            
        # 2. Check if DOB matches exactly
        if patient_data.get("dob") != dob:
            return None

        # If it passes BOTH checks, return the eligibility data
        return patient_data.get("eligibility_data", [])

    # --- Building Graph ----

    async def build_graph(self):
        wf = StateGraph(IVRState)

        # --- Nodes ---

        async def ask_member_id(state: IVRState):
            retry_count = self.get_retries(state, "member_id")
            msg = "Eligibility Check. Can you please tell me your Member ID?" if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say your Member ID one more time?"
            
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
            
            chain = PROMPT['MEMBER_ID_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(MemberIDExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            # Global Commands
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            # Retry Logic
            if not parsed.extracted_member_id:
                new_retries = self.increment_retries(state, "member_id")
                if new_retries["retries"]["member_id"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_member_id", update=new_retries)

            # Success -> Goto Confirm Member ID
            return Command(
                goto="confirm_member_id", 
                update={"last_user_input": user_input, "temp_member_id": parsed.extracted_member_id, "retries": {}}
            )

        async def confirm_member_id(state: IVRState):
            member_id = state.get("temp_member_id")
            # Safety check
            if not member_id: return Command(goto="ask_member_id")

            spoken_chars = []

            for ch in member_id:
                if ch in number_mappings:
                    spoken_chars.append(number_mappings[ch])
                else:
                    spoken_chars.append(ch)

            rephrase_member_id = " <break time='50ms'/> ".join(spoken_chars)
            
            # Retry Logic
            retry_count = self.get_retries(state, "confirm_member_id")
            msg = f"Member ID is {rephrase_member_id}. Correct?" if retry_count == 0 else f"Sorry, I didn't quite catch that. <break time='300ms'/> Could you please confirm your Member ID is {rephrase_member_id} correct?"

            # Get User Input
            confirmation = interrupt({"message_to_play": msg, "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")
            
            if parsed.confirmation == "1": 
                # Success -> Goto Ask DOB
                return Command(
                    goto="ask_dob", 
                    update={"last_user_input": "1", "member_id": member_id, "retries": {}}
                )
            
            # Retry Logic (Go back to Ask ID)
            new_retries = self.increment_retries(state, "confirm_member_id")
            if new_retries["retries"]["confirm_member_id"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="confirm_member_id", update=new_retries)

        async def ask_dob(state: IVRState):
            retry_count = self.get_retries(state, "dob")
            msg = "Please give me your Date of Birth." if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say your Date of Birth one more time?"
            
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
            
            chain = PROMPT['DATE_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(DATEExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']
            
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")
            
            # Retry Logic
            if not parsed.extracted_date:
                new_retries = self.increment_retries(state, "dob")
                if new_retries["retries"]["dob"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_dob", update=new_retries)
            
            # Success -> Goto Confirm DOB
            return Command(
                goto="confirm_dob", 
                update={"last_user_input": parsed.extracted_date, "temp_dob": parsed.extracted_date, "retries": {}}
            )

        async def confirm_dob(state: IVRState):
            dob = state.get("temp_dob")
            if not dob: return Command(goto="ask_dob")
            
            retry_count = self.get_retries(state, "confirm_dob")
            msg = f"Your Date of Birth is <break time='300ms'/> <say-as interpret-as='date' format='mdy'>{dob}</say-as>. Correct?" if retry_count == 0 else f"Sorry, I didn't quite catch that. <break time='300ms'/> Could you please confirm your Date of Birth is <break time='300ms'/> <say-as interpret-as='date' format='mdy'>{dob}</say-as> correct?"

            confirmation = interrupt({"message_to_play": msg, "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']
            
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            if parsed.confirmation == "1": 
                # Success -> Goto Fetch Details
                return Command(
                    goto="fetch_details", 
                    update={"last_user_input": "1", "dob": dob, "retries": {}}
                )

            # Retry Logic (Go back to Ask DOB)
            new_retries = self.increment_retries(state, "confirm_dob")
            if new_retries["retries"]["confirm_dob"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="confirm_dob", update=new_retries)

        async def fetch_details(state: IVRState):
            member_id = state.get("member_id")
            dob = state.get("dob")

            plans = self._get_validated_eligibility(member_id, dob)

            if plans is None:
                return Command(goto="set_status_transfer")
            
            if not plans:
                msg = "I'm sorry, I could not find any eligibility for this account. <break time='1s'/> Now, what would you like to do? You can say 'repeat', 'check another', 'transfer', or 'main menu'."
                return Command(goto="handle_final", update={"message_to_play": msg})

            msg_parts = [f"I found {len(plans)} coverage record{'s' if len(plans) > 1 else ''} on file."]

            for idx, plan in enumerate(plans):
                # Format the Effective Date (YYYY/MM/DD requires 'mdy')
                spoken_effective = f"<say-as interpret-as='date' format='mdy'>{plan['effective_date']}</say-as>"

                plan_text = (
                    f"Coverage type is {plan['coverage_type']}. "
                    f"The plan name is {plan['plan_name']}. "
                    f"The current status is {plan['status']}, with an effective date of {spoken_effective}. "
                )
                
                # Check if the plan is inactive and has a termination date
                if plan['status'].lower() == "inactive" and plan.get('termination_date'):
                    spoken_term = f"<say-as interpret-as='date' format='mdy'>{plan['termination_date']}</say-as>"
                    plan_text += f"The policy was terminated on {spoken_term}."
                    
                # Check if the plan is active and read the copay
                elif plan['status'].lower() == "active" and plan.get('office_visit_copay') is not None:
                    plan_text += f"Your office visit copay is {plan['office_visit_copay']} dollars."

                msg_parts.append(plan_text.strip())

            final_speech = " <break time='1500ms'/> ".join(msg_parts)

            msg = f"{final_speech}. <break time='1s'/> Now, what would you like to do? You can say 'repeat', 'check another', 'transfer', or 'main menu'."

            # Directly transition to input handler
            return Command(
                goto="handle_final",
                update={"message_to_play": msg}
            )

        async def handle_final(state: IVRState):
            choice = interrupt({"message_to_play": state["message_to_play"], "input_type": "varied"})
            
            chain = PROMPT['MULTI_CLASS_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(MultiClassValidatorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": choice})
            parsed = before_parsed['parsed']
            
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")
            
            sel = parsed.selection
            
            if sel == "9": # Repeat
                new_retries = self.increment_retries(state, "fetch_details")
                if new_retries["retries"]["fetch_details"] > self.MAX_RETRIES: 
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
        async def set_status_exit(state): return get_reset_state_update("exit")
    
        # Add Nodes
        wf.add_node("ask_member_id", ask_member_id)
        wf.add_node("confirm_member_id", confirm_member_id)
        wf.add_node("ask_dob", ask_dob)
        wf.add_node("confirm_dob", confirm_dob)
        wf.add_node("fetch_details", fetch_details)
        wf.add_node("handle_final", handle_final)
        
        wf.add_node("set_status_transfer", set_status_transfer)
        wf.add_node("set_status_main_menu", set_status_main_menu)
        wf.add_node("set_status_exit", set_status_exit)
        
        # Entry Point
        wf.set_entry_point("ask_member_id")
        
        # Clean End Edges
        # (Complex logic removed because Command handles routing)
        wf.add_edge("set_status_transfer", END)
        wf.add_edge("set_status_main_menu", END)
        wf.add_edge("set_status_exit", END)
        
        return wf.compile()      ## If anything issue occurred the graph state not changed then give the 'checkpointer=True' in the compile method like this graph.compile(checkpointer=True).
