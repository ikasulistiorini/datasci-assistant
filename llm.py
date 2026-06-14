import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

GEMINI_MODEL = "gemini-2.0-flash"
LLAMA_MODEL = "llama-3.1-8b-instant"

AVAILABLE_MODELS = [
    "gemini-2.0-flash",       # free tier: 1500 req/day ← default
    "gemini-1.5-flash",       # free tier: 1500 req/day (fallback)
    "gemini-2.5-flash",       # free tier: 20 req/day (gunakan hemat)
    LLAMA_MODEL,              # open-source via Groq
]


def is_gemini_model(model_name: str) -> bool:
    return model_name.startswith("gemini")


def get_llm(model_name: str, streaming: bool = True):
    if is_gemini_model(model_name):
        return ChatGoogleGenerativeAI(
            model=model_name,
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
