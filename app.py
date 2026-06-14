import io
import os

import streamlit as st
from dotenv import load_dotenv


def _extract_text(content) -> str:
    """Normalize LangChain chunk.content to str.

    Gemini streaming can return content as a list of dicts like
    [{'type': 'text', 'text': '...'}] instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content) if content else ""

load_dotenv()

from llm import AVAILABLE_MODELS, GEMINI_MODEL, get_llm, is_gemini_model
from rag import (
    build_rag_chain,
    build_vectorstore,
    has_vectorstore,
    load_and_chunk_pdfs,
    load_vectorstore,
)
from tools import TOOLS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSci Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DEFAULT_SYSTEM_PROMPT = (
    "You are DataSci Assistant, an expert AI productivity tool for Data Scientists. "
    "Help with data analysis, machine learning concepts, Python code, statistics, "
    "and interpreting charts and results. Be concise, practical, and technically precise."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 DataSci Assistant")
    st.caption("Gemini + Llama | RAG | Vision | Function Calling")
    st.divider()

    selected_model = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=0,
        help="Gemini: supports vision + function calling.\nLlama: open-source via Groq.",
    )

    system_prompt = st.text_area(
        "System Prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=110,
        help="Defines the assistant persona and behavior.",
    )

    st.divider()
    st.subheader("📄 RAG — Document Q&A")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Index documents to enable document Q&A mode.",
    )

    index_btn = st.button(
        "Index Documents",
        type="primary",
        disabled=not uploaded_pdfs,
        use_container_width=True,
    )

    if index_btn and uploaded_pdfs:
        chunk_count = 0
        with st.spinner("Chunking and embedding PDFs... (may take ~1 min for large files)"):
            chunks = load_and_chunk_pdfs(uploaded_pdfs)
            chunk_count = len(chunks)
            build_vectorstore(chunks)
            st.session_state["rag_ready"] = True
        st.success(f"Indexed {chunk_count} chunks from {len(uploaded_pdfs)} file(s).")

    if has_vectorstore():
        st.info("Vectorstore active — questions will use RAG.")

    if st.button("Clear Vectorstore", use_container_width=True, disabled=not has_vectorstore()):
        import shutil
        shutil.rmtree("./vectorstore", ignore_errors=True)
        st.session_state["rag_ready"] = False
        st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_ready" not in st.session_state:
    st.session_state["rag_ready"] = False

# ── API key validation ────────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing. Add it to your .env file and restart the app.")
    st.stop()

if selected_model != GEMINI_MODEL and not GROQ_API_KEY:
    st.warning("GROQ_API_KEY missing. Add it to your .env file or switch to Gemini.")
    st.stop()

# ── Main area header ──────────────────────────────────────────────────────────
st.title("DataSci Assistant")

mode_tags = []
if has_vectorstore():
    mode_tags.append("📄 RAG active")
if is_gemini_model(selected_model):
    mode_tags.append("👁️ vision")
    mode_tags.append("🔧 function calling")
st.caption(
    f"Model: `{selected_model}`"
    + (f" | {' | '.join(mode_tags)}" if mode_tags else " | General chat mode")
)

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Image uploader (Gemini only) ──────────────────────────────────────────────
uploaded_image = None
if is_gemini_model(selected_model):
    uploaded_image = st.file_uploader(
        "Upload image for analysis (optional — Gemini only)",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        key="image_upload",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Image ready for analysis", width=320)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask me anything about data science...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        try:
            # ── Mode 1: Multimodal vision ─────────────────────────────────────
            if uploaded_image and is_gemini_model(selected_model):
                from google import genai as google_genai
                from google.genai import types
                import PIL.Image

                client = google_genai.Client(api_key=GOOGLE_API_KEY)
                img = PIL.Image.open(io.BytesIO(uploaded_image.getvalue()))
                vision_prompt = prompt or "Analyze this image in detail for a Data Scientist."
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=[vision_prompt, img],
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                ):
                    if chunk.text:
                        full_response += chunk.text
                        response_box.markdown(full_response + "▌")
                response_box.markdown(full_response)

            # ── Mode 2: RAG — vectorstore exists ─────────────────────────────
            elif has_vectorstore():
                vectorstore = load_vectorstore()
                llm = get_llm(selected_model)
                rag_chain = build_rag_chain(vectorstore, llm)
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_box.markdown(full_response + "▌")
                response_box.markdown(full_response)

            # ── Mode 3: Direct chat (+ function calling for Gemini) ───────────
            else:
                from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

                llm = get_llm(selected_model)
                llm_with_tools = llm.bind_tools(TOOLS) if is_gemini_model(selected_model) else llm

                # Build message history
                history = [SystemMessage(content=system_prompt)]
                for m in st.session_state["messages"][:-1]:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    else:
                        history.append(AIMessage(content=m["content"]))
                history.append(HumanMessage(content=prompt))

                ai_msg = llm_with_tools.invoke(history)

                # Handle tool calls if any
                if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                    tool_results = []
                    for tc in ai_msg.tool_calls:
                        matched = next((t for t in TOOLS if t.name == tc["name"]), None)
                        if matched:
                            result = matched.invoke(tc.get("args", {}))
                            tool_results.append(
                                ToolMessage(content=str(result), tool_call_id=tc["id"])
                            )
                    final_msgs = history + [ai_msg] + tool_results
                    final = llm.invoke(final_msgs)
                    full_response = _extract_text(final.content)
                    response_box.markdown(full_response)
                else:
                    # Stream directly
                    for chunk in llm_with_tools.stream(history):
                        text = _extract_text(getattr(chunk, "content", ""))
                        if text:
                            full_response += text
                            response_box.markdown(full_response + "▌")
                    response_box.markdown(full_response)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                full_response = (
                    "**Rate limit hit.** The selected model has reached its free-tier daily quota.\n\n"
                    "**Options:**\n"
                    "- Switch to **gemini-2.0-flash** (1500 req/day free) in the sidebar\n"
                    "- Switch to **llama-3.1-8b-instant** (Groq, very generous free tier)\n"
                    "- Wait until tomorrow for the quota to reset\n"
                    "- Add billing at [aistudio.google.com](https://aistudio.google.com) for unlimited usage"
                )
                response_box.warning(full_response)
            else:
                full_response = f"**Error:** {e}"
                response_box.error(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
