import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

class PromptLoader:
    def __init__(self, relative_path: str = "vocode/ivr_agent/policies"):
        """
        Initializes the loader.
        
        Args:
            relative_path: Path to the prompt directory relative to the project root.
        """
        # Resolve the path dynamically. 
        # Path(os.getcwd()) is safe if you always run from project root.
        # Alternatively, use Path(__file__).parent if files are in the same folder as this script.
        self.prompt_dir = Path(os.getcwd()) / relative_path
        self.prompts = {}

    def load_prompts(self) -> dict[str, ChatPromptTemplate]:
        """
        Reads all .txt files in the directory and converts them to LangChain templates.
        """
        if not self.prompt_dir.exists():
            logger.error(f"Prompt directory not found at: {self.prompt_dir}")
            return {}

        logger.info(f"Loading prompts from: {self.prompt_dir}")

        # Iterate over all .txt files using pathlib
        for file_path in self.prompt_dir.glob("*.txt"):
            try:
                # 1. Generate Key: "my_prompt.txt" -> "MY_PROMPT"
                key = file_path.stem.upper()
                
                # 2. Read Content
                content = file_path.read_text(encoding="utf-8")

                # 3. Create Template
                # This structure aligns with your flows: 
                # chain.invoke({"user_input": user_input})
                template = ChatPromptTemplate.from_messages([
                    ("system", content),
                    ("human", "{user_input}"),
                ])

                self.prompts[key] = template
                logger.debug(f"Loaded prompt template: {key}")

            except Exception as e:
                logger.error(f"Failed to load prompt {file_path.name}: {e}")
                raise e

        return self.prompts

# ==========================================
# EXPORTED VARIABLE
# ==========================================
# This variable 'PROMPT' is what you will import in your flows.
# Usage: from prompt_loader import PROMPT 
# OR in setting.py: from vocode.ivr_agent.policies.prompt_loader import PROMPT

loader = PromptLoader()
PROMPT = loader.load_prompts()