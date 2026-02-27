from pydantic import BaseModel, Field
from typing import Optional, Literal

class IntentClassifierOutputSchema(BaseModel):
    """Schema for classifying the user's intent into Claim, Auth, or Eligibility."""
    intent: Literal["CLAIM", "ELIGIBILITY", "AUTH", "TRANSFER", "EXIT", "UNKNOWN"] = Field(
        ..., 
        description="Return 'CLAIM' for Claims, 'ELIGIBILITY' for Eligibility, 'AUTH' for Authorization, 'TRANSFER' for Transfer, 'EXIT' for Exit, 'UNKNOWN' for unknown."
    )

class BinaryConfirmationOutputSchema(BaseModel):
    """Schema for validating Yes/No confirmation responses."""
    confirmation: Literal["1", "2"] = Field(
        ..., 
        description="Return '1' if the user confirms (Yes, Correct, Right, Yeah, Yep, Sure, OK, That matches). Return '2' if the user denies (No, Wrong, Incorrect, Wait, Stop, I don't think so)."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class NPIOrTaxIDOutputSchema(BaseModel):
    """Schema for classifying the user's role into Patient or Provider."""
    selection: Literal["1", "2", "0"] = Field(
        ...,
        description="Return '1' for NPI, '2' for Tax ID, or '0' for Fallback."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class TaxIDExtractorOutputSchema(BaseModel):
    """Schema for extracting and formatting a Tax ID."""
    extracted_tax_id: str = Field(
        ...,
        description="The tax ID formatted strictly as a continuous string of digits. If the tax ID is invalid or missing, return an empty string."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class DATEExtractorOutputSchema(BaseModel):
    """Schema for extracting and formatting a Date from user input."""
    extracted_date: str = Field(
        ..., 
        description="The date formatted strictly as MM/DD/YYYY (e.g., 01/23/2004). If the date is invalid or missing, return an empty string."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class MultiClassValidatorOutputSchema(BaseModel):
    """Schema for handling final menu selections (Repeat, Exit, etc)."""
    selection: Literal["9", "*", "8", "4"] = Field(
        ..., 
        description="Return '9' for Repeat, '*' for Exit/End, or '8' to go back to the start. Return '4' for Customer Service or not information user want to find."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class RoleClassifierOutputSchema(BaseModel):
    """Schema for classifying the user's role into Patient or Provider."""
    user_role: Literal["1", "2", "0"] = Field(
        ...,
        description="Return '1' for Patient, '2' for Provider, or '0' for Fallback."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class MemberIDExtractorOutputSchema(BaseModel):
    """Schema for extracting and formatting a Member ID."""
    extracted_member_id: str = Field(
        ...,
        description="The member ID formatted strictly as a continuous string of digits. If the ID is invalid or missing, return an empty string."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class NPIExtractorOutputSchema(BaseModel):
    """Schema for extracting and formatting a NPI."""
    extracted_npi: str = Field(
        ...,
        description="The NPI formatted strictly as a continuous string of digits. If the NPI is invalid or missing, return an empty string."
    )

    command: Literal["TRANSFER", "MAIN_MENU", "EXIT", "None"] | str = Field(
        ...,
        description="Return 'TRANSFER' if the user wants to transfer to a representative. Return 'MAIN_MENU' if the user wants to go back to the main menu. Return 'EXIT' if the user wants to exit. Return 'None' if the user provided the required information."
    )

class CustomerServiceOutputSchema(BaseModel):
    """Schema for handling the Customer Service Agent"""
    is_exit: Literal["1", "0"] = Field(
        ...,
        description="Return '1' if the user wants to exit, '0' otherwise."
    )
    response: str = Field(
        ...,
        description="Return your response."
    )

