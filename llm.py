import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

GEMINI_MODEL = "gemini-2.5-flash"
LLAMA_MODEL = "llama-3.1-8b-instant"

AVAILABLE_MODELS = [GEMINI_MODEL, LLAMA_MODEL]


def get_llm(model_name: str, streaming: bool = True):
    """Return a configured LangChain chat model.

    Args:
        model_name: One of AVAILABLE_MODELS.
        streaming: Enable streaming (both models support it).

    Returns:
        A LangChain BaseChatModel instance.
    """
    if model_name == GEMINI_MODEL:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=streaming,
            temperature=0.7,
        )
    elif model_name == LLAMA_MODEL:
        return ChatGroq(
            model=LLAMA_MODEL,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            streaming=streaming,
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {AVAILABLE_MODELS}")
