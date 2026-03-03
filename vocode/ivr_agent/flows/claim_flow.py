# Claims Flow
import json, os
from typing import Literal, Optional, List, Dict
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from vocode.ivr_agent.state.store import IVRState
from vocode.ivr_agent.state.models import (
    BinaryConfirmationOutputSchema, 
    DATEExtractorOutputSchema, 
    MultiClassValidatorOutputSchema,
    MemberIDExtractorOutputSchema,
    NPIExtractorOutputSchema,
    RoleClassifierOutputSchema,
    TaxIDExtractorOutputSchema,
    NPIOrTaxIDOutputSchema
)
from vocode.ivr_agent.policies.prompt_templates import PROMPT
from vocode.ivr_agent.utilities.state_utils import get_reset_state_update
import asyncio
from vocode.ivr_agent.state.commands import number_mappings
from loguru import logger


class ClaimFlow:
    def __init__(self, LLM):
        self.LLM = LLM
        self.MAX_RETRIES = 3
        db_path = os.path.join(os.getcwd(), "vocode/ivr_agent/data/claims_db.json")
        with open(db_path, "r") as f:
            self.db = json.load(f)

    # --- Helper Functions ---

    def get_retries(self, state: IVRState, key: str) -> int: 
        return state.get("retries", {}).get(key, 0)

    def increment_retries(self, state: IVRState, key: str) -> dict:
        curr = state.get("retries", {}).copy()
        curr[key] = curr.get(key, 0) + 1
        return {"retries": curr}

    def _get_validated_claims(self, member_id: str, dob: str, user_role: str, npi: Optional[str], service_date: Optional[str], tax_id: Optional[str]) -> Optional[List[Dict]]:
        """
        Validates the user's inputs against the JSON DB.
        Returns the claims list if successful, or None if validation fails.
        """
        patient_data = self.db.get(member_id)
        
        # 1. Check if Member ID exists
        if not patient_data:
            return None
            
        # 2. Check if DOB matches
        if patient_data.get("dob") != dob:
            return None
            
        # 3. Check Role and NPI
        db_type = patient_data.get("type") # "patient" or "provider"
        
        if user_role == "patient" and db_type != "patient":
            return None
            
        if user_role == "provider":
            if db_type != "provider":
                return None
            if npi and patient_data.get("npi_number") != npi:
                return None
            if tax_id and patient_data.get("tax_id") != tax_id:
                return None

        # If it passes ALL checks, return the claims data
        claims_data = patient_data.get("claims_data", [])

        if claims_data:
            if service_date:
                claims_data = [claim for claim in claims_data if claim.get("date_of_service") == service_date]

        return claims_data

    def _get_total_claims(self, member_id: str, dob: str) -> int:
        claims_data = self.db.get(member_id)
        if not claims_data:
            return 0
        if claims_data.get("dob") != dob:
            return 0
        return len(claims_data.get("claims_data", []))

    # --- Building Graph ---
    async def build_graph(self):
        wf = StateGraph(IVRState)
    
        # --- Nodes ---

        async def ask_member_id(state: IVRState):
            retry_count = self.get_retries(state, "member_id")
            msg = "Claim Status. Can you please tell me your Member ID?" if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say your Member ID one more time?"
        
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
        
            chain = PROMPT['MEMBER_ID_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(MemberIDExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            logger.info(f"Parsed command: {parsed.command}")

            # Commands
            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            # Retry Logic
            if not parsed.extracted_member_id:
                new_retries = self.increment_retries(state, "member_id")
                if new_retries["retries"]["member_id"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_member_id", update=new_retries)

            # Success -> Goto Confirmation
            return Command(
                goto="confirm_member_id", 
                update={"last_user_input": user_input, "temp_member_id": parsed.extracted_member_id, "retries": {}}
            )

        async def confirm_member_id(state: IVRState):
            member_id = state.get("temp_member_id")
            # Safety: If temp slot is empty, go back
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

            # Parse User Input
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

            elif parsed.confirmation == "2":
                return Command(goto="ask_member_id", update={"temp_member_id": "", "retries": {}})
        
            # Retry Logic
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
            dob_date = state.get("temp_dob")
            if not dob_date: return Command(goto="ask_dob") 

            retry_count = self.get_retries(state, "confirm_dob")
            msg = f"Your Date of Birth is <break time='300ms'/> <say-as interpret-as='date' format='mdy'>{dob_date}</say-as>. Correct?" if retry_count == 0 else f"Sorry, I didn't quite catch that. <break time='300ms'/> Could you please confirm your Date of Birth is <break time='300ms'/> <say-as interpret-as='date' format='mdy'>{dob_date}</say-as> correct?"

            confirmation = interrupt({"message_to_play": msg, "input_type": "single"})
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            if parsed.confirmation == "1":
                if self._get_total_claims(member_id=state.get("member_id"), dob=dob_date) > 1:
                    return Command(
                        goto="ask_servicedate",
                        update={"last_user_input": "1", "dob": dob_date, "retries": {}}
                    )

                # Success -> Goto for ask patient or provider
                return Command(
                    goto="ask_patient_provider",
                    update={"last_user_input": "1", "dob": dob_date, "retries": {}}
                )
            
            elif parsed.confirmation == "2":
                return Command(goto="ask_dob", update={"temp_dob": "", "retries": {}})

            # Retry Logic
            new_retries = self.increment_retries(state, "confirm_dob")
            if new_retries["retries"]["confirm_dob"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="confirm_dob", update=new_retries)

        async def ask_servicedate(state: IVRState):
            retry_count = self.get_retries(state, "service_date")
            msg = "There are more than one services in file. Please give me the date of service you want to inquire about." if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say the date of service one more time?"

            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})

            chain = PROMPT['DATE_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(DATEExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")
        
            # Retry Logic
            if not parsed.extracted_date:
                new_retries = self.increment_retries(state, "service_date")
                if new_retries["retries"]["service_date"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_servicedate", update=new_retries)
        
            # Success -> Goto Confirm Service Date
            return Command(
                goto="confirm_servicedate",
                update={"last_user_input": parsed.extracted_date, "temp_service_date": parsed.extracted_date, "retries": {}}
            )

        async def confirm_servicedate(state: IVRState):
            service_date = state.get("temp_service_date")
            if not service_date: return Command(goto="ask_servicedate")

            retry_count = self.get_retries(state, "confirm_servicedate")
            msg = f"Your service date is <break time='300ms'/> <say-as interpret-as='date' format='mdy'>{service_date}</say-as>. Correct?" if retry_count == 0 else f"Sorry, I didn't quite catch that. <break time='300ms'/> Could you please confirm your service date is <break time='300ms'/> <say-as interpret-as='date' format='mdy'>{service_date}</say-as> correct?"

            confirmation = interrupt({"message_to_play": msg, "input_type": "single"})
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            if parsed.confirmation == "1":
                # Success -> Goto Ask Patient or Provider
                return Command(
                    goto="ask_patient_provider",
                    update={"last_user_input": "1", "service_date": service_date, "retries": {}}
                )
            
            elif parsed.confirmation == "2":
                return Command(goto="ask_servicedate", update={"temp_service_date": "", "retries": {}})

            # Retry Logic
            new_retries = self.increment_retries(state, "confirm_servicedate")
            if new_retries["retries"]["confirm_servicedate"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="confirm_servicedate", update=new_retries)

        async def ask_patient_provider(state: IVRState):
            retry_count = self.get_retries(state, "patient_provider")
            msg = "Are you a patient or a provider?" if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say are you a patient or a provider?"

            # Ask Role
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})

            chain = PROMPT['ROLE_CLASSIFIER_LOGIC_PROMPT'] | self.LLM.with_structured_output(RoleClassifierOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            role = parsed.user_role # "1" (Patient) or "2" (Provider)
            
            if role == "1": # Patient -> Skip NPI, go to Fetch
                return Command(
                    goto="fetch_claim_details", 
                    update={"user_role": "patient", "npi": None, "retries": {}}
                )
            elif role == "2": # Provider -> Ask NPI or Tax ID
                return Command(
                    goto="npi_or_tax_id", 
                    update={"user_role": "provider", "retries": {}}
                )
            
            # Unclear -> Retry this question (Simple Loop)
            new_retries = self.increment_retries(state, "patient_provider")
            if new_retries["retries"]["patient_provider"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="ask_patient_provider", update=new_retries)

        async def npi_or_tax_id(state: IVRState):
            retry_count = self.get_retries(state, "npi_or_tax")
            msg = "Which want you like to give either npi id or tax id?"

            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})

            chain = PROMPT['NPI_OR_TAX_ID_SELECTION_PROMPT'] | self.LLM.with_structured_output(NPIOrTaxIDOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            if not parsed.selection or parsed.selection == "0":
                new_retries = self.increment_retries(state, "npi_or_tax")
                if new_retries["retries"]["npi_or_tax"] > self.MAX_RETRIES: return Command(goto="set_status_transfer")
                return Command(goto="npi_or_tax_id", update=new_retries)

            if parsed.selection == "1": return Command(goto="ask_npi", update={"retries": {}})
            if parsed.selection == "2": return Command(goto="ask_tax_id", update={"retries": {}})


        async def ask_npi(state: IVRState):
            retry_count = self.get_retries(state, "npi")
            msg = "Please give me your NPI Number." if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say your NPI number one more time?"
            
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
            
            chain = PROMPT['NPI_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(NPIExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            # Retry Logic
            if not parsed.extracted_npi:
                new_retries = self.increment_retries(state, "npi")
                if new_retries["retries"]["npi"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_npi", update=new_retries)

            # Success -> Goto Confirm NPI
            return Command(
                goto="confirm_npi", 
                update={"last_user_input": parsed.extracted_npi, "temp_npi": parsed.extracted_npi, "retries": {}}
            )

        async def confirm_npi(state: IVRState):
            npi = state.get("temp_npi")
            if not npi: return Command(goto="ask_npi")

            spoken_chars = []

            for ch in npi:
                if ch in number_mappings:
                    spoken_chars.append(number_mappings[ch])
                else:
                    spoken_chars.append(ch)

            rephrase_npi = " <break time='50ms'/> ".join(spoken_chars)

            retry_count = self.get_retries(state, "confirm_npi")
            msg = f"your NPI number is {rephrase_npi}. Correct?" if retry_count == 0 else f"Sorry, I didn't quite catch that. <break time='300ms'/> Could you please confirm your NPI number is {rephrase_npi} correct?"

            confirmation = interrupt({"message_to_play": msg, "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            if parsed.confirmation == "1": 
                # Success -> Goto Fetch
                return Command(
                    goto="fetch_claim_details", 
                    update={"last_user_input": "1", "npi": npi, "retries": {}}
                )

            elif parsed.confirmation == "2":
                return Command(goto="ask_npi", update={"temp_npi": "", "retries": {}})

            # Retry Logic (Go back to Ask NPI)
            new_retries = self.increment_retries(state, "confirm_npi")
            if new_retries["retries"]["confirm_npi"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="confirm_npi", update=new_retries)

        async def ask_tax_id(state: IVRState):
            retry_count = self.get_retries(state, "tax_id")
            msg = "Please give me your Tax ID Number." if retry_count == 0 else "Sorry, I didn't quite catch that. <break time='300ms'/> Could you please say your Tax ID number one more time?"
            
            user_input = interrupt({"message_to_play": msg, "input_type": "varied"})
            
            chain = PROMPT['TAX_ID_EXTRACTOR_SCHEMA_PROMPT'] | self.LLM.with_structured_output(TaxIDExtractorOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": user_input})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            # Retry Logic
            if not parsed.extracted_tax_id:
                new_retries = self.increment_retries(state, "tax_id")
                if new_retries["retries"]["tax_id"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="ask_tax_id", update=new_retries)

            # Success -> Goto Confirm TAX ID
            return Command(
                goto="confirm_tax_id", 
                update={"last_user_input": parsed.extracted_tax_id, "temp_tax_id": parsed.extracted_tax_id, "retries": {}}
            )

        async def confirm_tax_id(state: IVRState):
            tax_id = state.get("temp_tax_id")
            if not tax_id: return Command(goto="ask_tax_id")

            spoken_chars = []

            for ch in tax_id:
                if ch in number_mappings:
                    spoken_chars.append(number_mappings[ch])
                else:
                    spoken_chars.append(ch)

            rephrase_tax_id = " <break time='50ms'/> ".join(spoken_chars)

            retry_count = self.get_retries(state, "confirm_tax_id")
            msg = f"your Tax ID number is {rephrase_tax_id}. Correct?" if retry_count == 0 else f"Sorry, I didn't quite catch that. <break time='300ms'/> Could you please confirm your Tax ID number is {rephrase_tax_id} correct?"

            confirmation = interrupt({"message_to_play": msg, "input_type": "single"})
            
            chain = PROMPT['CONFIRMATION_VALIDATOR_SYSTEM_PROMPT'] | self.LLM.with_structured_output(BinaryConfirmationOutputSchema, include_raw=True)
            before_parsed = await chain.ainvoke({"user_input": confirmation})
            parsed = before_parsed['parsed']

            if parsed.command == "TRANSFER": return Command(goto="set_status_transfer")
            if parsed.command == "MAIN_MENU": return Command(goto="set_status_main_menu")
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            if parsed.confirmation == "1": 
                # Success -> Goto Fetch
                return Command(
                    goto="fetch_claim_details", 
                    update={"last_user_input": "1", "tax_id": tax_id, "retries": {}}
                )

            elif parsed.confirmation == "2":
                return Command(goto="ask_tax_id", update={"temp_tax_id": "", "retries": {}})

            # Retry Logic (Go back to Ask NPI)
            new_retries = self.increment_retries(state, "confirm_tax_id")
            if new_retries["retries"]["confirm_tax_id"] > self.MAX_RETRIES: 
                return Command(goto="set_status_transfer")
            return Command(goto="confirm_tax_id", update=new_retries)

        async def fetch_claim_details(state: IVRState):
            member_id = state.get("member_id")
            dob = state.get("dob")
            user_role = state.get("user_role")
            npi = state.get("npi")
            service_date = state.get("service_date")
            tax_id = state.get("tax_id")

            # 1. Fetch and validate claims details via method
            claims = self._get_validated_claims(member_id, dob, user_role, npi, service_date, tax_id)

            # 2. If validation failed (returned None), immediately transfer the call to customer service
            if claims is None:
                return Command(goto="set_status_transfer")

            # 3. SUCCESS STAGE - Build the Speech
            if not claims:
                msg = "I'm sorry, I could not find any claims for this account. <break time='1s'/> Now, what would you like to do? You can say 'repeat', 'check another', 'transfer', or 'main menu'."
                return Command(goto="handle_final_input", update={"message_to_play": msg})
            
            # 4. Build the Speech
            msg_parts = [f"I found {len(claims)} claim{'s' if len(claims) > 1 else ''} on file."]
            for idx, claim in enumerate(claims):
                spoken_date = f"<say-as interpret-as='date' format='mdy'>{claim['date_of_service']}</say-as>"

                claim_text = (
                    f"Claim number {claim['claim_id']}. Date of service is {spoken_date}. "
                    f"The status is {claim['status']}. "
                    f"The billed amount was {claim['billed_amount']} dollars."
                )
                if claim['status'].lower() == "denied" and "denial_reason" in claim:
                    claim_text += f"The reason for denial is: {claim['denial_reason']}."
                elif claim['status'].lower() == "approved":
                    claim_text += f"The paid amount was {claim['paid_amount']} dollars."

                msg_parts.append(claim_text.strip())

            final_speech = " <break time='1500ms'/> ".join(msg_parts)

            msg = f"{final_speech}. <break time='1s'/> Now, what would you like to do? You can say 'repeat', 'check another', 'transfer', or 'main menu'."

            # Directly transition to input handler with the message
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
            if parsed.command == "EXIT": return Command(goto="set_status_exit")

            # Local Menu Logic
            selection = parsed.selection # e.g. "9", "8", "4", "*"
            
            if selection == "9": # Repeat
                new_retries = self.increment_retries(state, "fetch_details")
                if new_retries["retries"]["fetch_details"] > self.MAX_RETRIES: 
                    return Command(goto="set_status_transfer")
                return Command(goto="fetch_claim_details", update=new_retries)
            elif selection == "8": # Another Claim (Restart Flow)
                # Clear member_id/dob but keep flow active
                return Command(goto="ask_member_id", update={
                    "member_id": None, "dob": None, "npi": None, "user_role": None,
                    "temp_member_id": None, "temp_dob": None, "temp_npi": None,
                    "retries": {}
                })
            elif selection == "4": # Transfer
                return Command(goto="set_status_transfer")
            else: # Default/Exit
                return Command(goto="set_status_main_menu")

        # --- Cleanup Nodes ---

        async def set_status_transfer(state: IVRState):
            return get_reset_state_update("transfer")

        async def set_status_main_menu(state: IVRState):
            return get_reset_state_update("main_menu")

        async def set_status_exit(state: IVRState):
            return get_reset_state_update("exit")


        # Add all nodes
        wf.add_node("ask_member_id", ask_member_id)
        wf.add_node("confirm_member_id", confirm_member_id)
        wf.add_node("ask_dob", ask_dob)
        wf.add_node("confirm_dob", confirm_dob)
        wf.add_node("ask_servicedate", ask_servicedate)
        wf.add_node("confirm_servicedate", confirm_servicedate)
        wf.add_node("ask_patient_provider", ask_patient_provider)
        wf.add_node("npi_or_tax_id", npi_or_tax_id)
        wf.add_node("ask_npi", ask_npi)     
        wf.add_node("confirm_npi", confirm_npi)
        wf.add_node("ask_tax_id", ask_tax_id)
        wf.add_node("confirm_tax_id", confirm_tax_id)
        wf.add_node("fetch_claim_details", fetch_claim_details)
        wf.add_node("handle_final_input", handle_final_input)
        
        wf.add_node("set_status_transfer", set_status_transfer)
        wf.add_node("set_status_main_menu", set_status_main_menu)
        wf.add_node("set_status_exit", set_status_exit)

        wf.set_entry_point("ask_member_id")

        # Since we are using Command(goto="...") for ALL transitions,
        # we do not need to define conditional edges here.
        # The Nodes themselves dictate the flow.
        
        # Just define the end points for the cleanup nodes
        wf.add_edge("set_status_transfer", END)
        wf.add_edge("set_status_main_menu", END)
        wf.add_edge("set_status_exit", END)

        return wf.compile()      ## If anything issue occurred the graph state not changed then give the 'checkpointer=True' in the compile method like this graph.compile(checkpointer=True).
