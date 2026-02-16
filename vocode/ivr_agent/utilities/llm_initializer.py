from langchain_google_genai import ChatGoogleGenerativeAI
import os


def load_env_if_available(*paths: str) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        for p in paths:
            if p:
                load_dotenv(p, override=False)
    except Exception:
        # silently continue if python-dotenv not installed
        pass


class Gemini():
    def __init__(self, params: dict = None):
        self.params = {
            "model": "gemini-2.0-flash",
            "temperature": 0.0,
            "max_retries": 2,
            "streaming": False
        }
        if params:
            self.params.update(params)
        # FIX: Do NOT initialize the model here. Just set a placeholder.
        load_env_if_available("talkbot.env", ".env", "/home/hellfire/vocode-core/talkbot.env")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        self._llm_instance = None

    def initialize_model(self) -> ChatGoogleGenerativeAI:
        # This method is now safe because it is only called when needed
        llm = ChatGoogleGenerativeAI(
            model=self.params['model'], 
            temperature=self.params['temperature'], 
            max_retries=self.params['max_retries'], 
            streaming=self.params['streaming']
        )
        return llm
    
    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Lazy Loader: Creates the LLM instance the first time it is requested.
        This ensures it happens inside the active Event Loop.
        """
        if self._llm_instance is None:
            self._llm_instance = self.initialize_model()
        return self._llm_instance
    
    # --- Proxy Methods (Optional but helpful) ---
    # These let you use the Gemini object directly like a LangChain model
    
    async def ainvoke(self, input, config=None, **kwargs):
        return await self.get_llm().ainvoke(input, config=config, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        return self.get_llm().invoke(input, config=config, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return self.get_llm().with_structured_output(schema, **kwargs)

    def model_name(self) -> str:
        return self.params['model']
    
    def model_type(self) -> str:
        return "Gemini"


LLM = Gemini().get_llm()