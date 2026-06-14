import html
import io
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Normalize LangChain chunk.content to str (handles Gemini list-of-dict format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content) if content else ""


def _make_title(first_msg: str) -> str:
    title = first_msg.strip().replace("\n", " ")
    return (title[:40] + "…") if len(title) > 40 else title


def _create_new_session() -> dict:
    session = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    st.session_state["sessions"].append(session)
    st.session_state["active_session_id"] = session["id"]
    return session


def _get_active_session() -> dict:
    active_id = st.session_state.get("active_session_id")
    for s in st.session_state["sessions"]:
        if s["id"] == active_id:
            return s
    return _create_new_session()


def _render_error(response_box, e: Exception) -> str:
    err_str = str(e)
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
        msg = (
            "**Rate limit hit.** The selected model reached its free-tier daily quota.\n\n"
            "**Options:**\n"
            "- Switch to **gemini-2.0-flash** (1 500 req/day free) in Settings\n"
            "- Switch to **llama-3.1-8b-instant** (Groq — very generous free tier)\n"
            "- Wait until tomorrow for the quota to reset\n"
            "- Add billing at aistudio.google.com for unlimited usage"
        )
        response_box.warning(msg)
    else:
        msg = f"**Error:** {e}"
        response_box.error(msg)
    return msg


# ── Bootstrap ─────────────────────────────────────────────────────────────────

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

# ── Session state init ────────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state["sessions"] = []
if "active_session_id" not in st.session_state:
    st.session_state["active_session_id"] = None
if "rag_ready" not in st.session_state:
    st.session_state["rag_ready"] = False
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []

if not st.session_state["sessions"]:
    _create_new_session()
elif st.session_state["active_session_id"] is None:
    st.session_state["active_session_id"] = st.session_state["sessions"][0]["id"]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* User message: right-aligned bubble */
.user-bubble-wrap {
    display: flex;
    justify-content: flex-end;
    margin: 8px 0 8px 60px;
}
.user-bubble {
    background: #DDE6FF;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    max-width: 78%;
    font-size: 0.95rem;
    line-height: 1.55;
    color: #1E2048;
    white-space: pre-wrap;
    word-break: break-word;
    box-shadow: 0 1px 4px rgba(91,138,248,0.12);
}

/* Sidebar session buttons — compact + left-aligned text */
section[data-testid="stSidebar"] .stButton button {
    text-align: left;
    justify-content: flex-start;
    font-size: 0.85rem;
    padding: 6px 12px;
}

/* Active session button — blue-tinted highlight */
section[data-testid="stSidebar"] .stButton [data-testid="baseButton-primary"] {
    background-color: #DDE6FF !important;
    color: #1E2048 !important;
    border-left: 3px solid #5B8AF8 !important;
    font-weight: 600;
}

/* Caption / secondary text — muted blue-gray */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #6B7AB8 !important;
}

/* Tab strip — active tab blue */
button[data-baseweb="tab"] {
    font-weight: 500;
    color: #6B7AB8;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #5B8AF8 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DataSci Assistant")
    st.caption("Gemini + Llama · RAG · Vision · Function Calling")

    if st.button("＋ New Chat", type="primary", use_container_width=True):
        _create_new_session()
        st.rerun()

    st.divider()

    with st.expander("⚙️ Settings", expanded=False):
        selected_model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=0,
            help="Gemini models support vision + function calling. Llama runs via Groq.",
        )
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=100,
            help="Defines the assistant persona and behavior.",
        )

    st.divider()
    st.markdown("**💬 Recent Chats**")

    active_id = st.session_state["active_session_id"]
    for sess in reversed(st.session_state["sessions"]):
        is_active = sess["id"] == active_id
        if st.button(
            sess["title"],
            key=f"sess_{sess['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["active_session_id"] = sess["id"]
            st.rerun()

# ── API key validation ────────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing. Add it to your .env file and restart the app.")
    st.stop()

if not is_gemini_model(selected_model) and not GROQ_API_KEY:
    st.warning("GROQ_API_KEY missing. Add it to .env or switch to a Gemini model.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_rag = st.tabs(["💬 Chat Assistant", "📄 Document Q&A"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    active_session = _get_active_session()

    mode_tags = []
    if is_gemini_model(selected_model):
        mode_tags += ["👁️ vision", "🔧 function calling"]
    st.caption(
        f"Model: `{selected_model}`"
        + (f"  ·  {' · '.join(mode_tags)}" if mode_tags else "")
    )

    # Render chat history
    for msg in active_session["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble-wrap">'
                f'<div class="user-bubble">{html.escape(msg["content"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Image uploader (Gemini only)
    uploaded_image = None
    if is_gemini_model(selected_model):
        uploaded_image = st.file_uploader(
            "Attach image for visual analysis (Gemini only)",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            key="image_upload",
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Image ready for analysis", width=260)

    # Chat input
    prompt = st.chat_input("Ask me anything about data science...")

    if prompt:
        active_session["messages"].append({"role": "user", "content": prompt})

        # Auto-generate session title from first user message
        if active_session["title"] == "New Chat":
            active_session["title"] = _make_title(prompt)

        # Show user bubble immediately
        st.markdown(
            f'<div class="user-bubble-wrap">'
            f'<div class="user-bubble">{html.escape(prompt)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            try:
                # ── Mode 1: Multimodal vision ──────────────────────────────────
                if uploaded_image and is_gemini_model(selected_model):
                    from google import genai as google_genai
                    from google.genai import types
                    import PIL.Image

                    client = google_genai.Client(api_key=GOOGLE_API_KEY)
                    img = PIL.Image.open(io.BytesIO(uploaded_image.getvalue()))
                    for chunk in client.models.generate_content_stream(
                        model=selected_model,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(system_instruction=system_prompt),
                    ):
                        if chunk.text:
                            full_response += chunk.text
                            response_box.markdown(full_response + "▌")
                    response_box.markdown(full_response)

                # ── Mode 2: Direct chat + function calling ─────────────────────
                else:
                    from langchain_core.messages import (
                        AIMessage, HumanMessage, SystemMessage, ToolMessage,
                    )

                    llm = get_llm(selected_model)
                    llm_with_tools = (
                        llm.bind_tools(TOOLS) if is_gemini_model(selected_model) else llm
                    )

                    history = [SystemMessage(content=system_prompt)]
                    for m in active_session["messages"][:-1]:
                        cls = HumanMessage if m["role"] == "user" else AIMessage
                        history.append(cls(content=m["content"]))
                    history.append(HumanMessage(content=prompt))

                    ai_msg = llm_with_tools.invoke(history)

                    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                        tool_results = []
                        for tc in ai_msg.tool_calls:
                            matched = next((t for t in TOOLS if t.name == tc["name"]), None)
                            if matched:
                                result = matched.invoke(tc.get("args", {}))
                                tool_results.append(
                                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                                )
                        final = llm.invoke(history + [ai_msg] + tool_results)
                        full_response = _extract_text(final.content)
                        response_box.markdown(full_response)
                    else:
                        for chunk in llm_with_tools.stream(history):
                            text = _extract_text(getattr(chunk, "content", ""))
                            if text:
                                full_response += text
                                response_box.markdown(full_response + "▌")
                        response_box.markdown(full_response)

            except Exception as e:
                full_response = _render_error(response_box, e)

        active_session["messages"].append({"role": "assistant", "content": full_response})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT Q&A (RAG)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.subheader("📄 Document Q&A")
    st.caption("Upload PDFs, index them, then ask questions. Answers include source citations.")

    col_upload, col_status = st.columns([2, 1])

    with col_upload:
        uploaded_pdfs = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button(
            "Index Documents",
            type="primary",
            disabled=not uploaded_pdfs,
        ):
            with st.spinner("Chunking and embedding PDFs… (may take ~1 min for large files)"):
                chunks = load_and_chunk_pdfs(uploaded_pdfs)
                build_vectorstore(chunks)
                st.session_state["rag_ready"] = True
                st.session_state["rag_messages"] = []
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_pdfs)} file(s).")

    with col_status:
        if has_vectorstore():
            st.success("Vectorstore active")
            if st.button("🗑️ Clear Vectorstore", use_container_width=True):
                import shutil
                shutil.rmtree("./vectorstore", ignore_errors=True)
                st.session_state["rag_ready"] = False
                st.session_state["rag_messages"] = []
                st.rerun()
        else:
            st.info("No documents indexed yet")

    st.divider()

    # RAG chat history
    for msg in st.session_state["rag_messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble-wrap">'
                f'<div class="user-bubble">{html.escape(msg["content"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    if not has_vectorstore():
        st.info("Upload and index a PDF above to start asking questions.")

    rag_prompt = st.chat_input(
        "Ask about your documents…",
        key="rag_input",
        disabled=not has_vectorstore(),
    )

    if rag_prompt and has_vectorstore():
        st.session_state["rag_messages"].append({"role": "user", "content": rag_prompt})

        st.markdown(
            f'<div class="user-bubble-wrap">'
            f'<div class="user-bubble">{html.escape(rag_prompt)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            try:
                vectorstore = load_vectorstore()
                llm = get_llm(selected_model)
                rag_chain = build_rag_chain(vectorstore, llm)
                for chunk in rag_chain.stream(rag_prompt):
                    full_response += chunk
                    response_box.markdown(full_response + "▌")
                response_box.markdown(full_response)
            except Exception as e:
                full_response = _render_error(response_box, e)

        st.session_state["rag_messages"].append({"role": "assistant", "content": full_response})
