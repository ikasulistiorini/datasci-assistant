import os
import tempfile
import time
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_DIR = "./vectorstore"
EMBEDDING_MODEL = "gemini-embedding-2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
EMBED_BATCH_SIZE = 80   # gemini-embedding-2 free tier: 100 req/min

RAG_SYSTEM_PROMPT = (
    "You are DataSci Assistant, a helpful AI productivity tool for Data Scientists. "
    "Answer the user's question using ONLY the context retrieved from the uploaded documents. "
    "If the answer is not found in the context, say: "
    "'I couldn\\'t find that in the uploaded documents.' "
    "Be concise and cite the source filename and page number when possible."
)


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        output_dimensionality=768,
    )


def load_and_chunk_pdfs(uploaded_files: List) -> List[Document]:
    """Load Streamlit UploadedFile objects, extract text, and split into chunks.

    Args:
        uploaded_files: List of st.UploadedFile from st.file_uploader.

    Returns:
        List of LangChain Document chunks with source_filename metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    all_chunks: List[Document] = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source_filename"] = uploaded_file.name
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
        finally:
            os.unlink(tmp_path)

    return all_chunks


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist them to ChromaDB.

    Processes in batches of EMBED_BATCH_SIZE to respect the
    gemini-embedding-2 free-tier rate limit (100 req/min).

    Args:
        chunks: Document chunks to embed and store.

    Returns:
        The persisted Chroma vectorstore instance.
    """
    embeddings = _get_embedding_model()
    vectorstore = None

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=VECTORSTORE_DIR,
            )
        else:
            time.sleep(65)  # wait for rate-limit window to reset
            vectorstore.add_documents(batch)

    return vectorstore


def load_vectorstore() -> Chroma | None:
    """Load an existing ChromaDB vectorstore from disk.

    Returns:
        Chroma instance if vectorstore exists, else None.
    """
    if not has_vectorstore():
        return None
    embeddings = _get_embedding_model()
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )


def has_vectorstore() -> bool:
    """Return True if a vectorstore has been persisted to disk."""
    return os.path.exists(VECTORSTORE_DIR) and bool(os.listdir(VECTORSTORE_DIR))


def build_rag_chain(vectorstore: Chroma, llm):
    """Build a LCEL RAG chain: retrieve → format → prompt → LLM → parse.

    Args:
        vectorstore: A loaded Chroma vectorstore.
        llm: A LangChain BaseChatModel (Gemini or Groq).

    Returns:
        A runnable LCEL chain that accepts a question string and streams a string answer.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source_filename", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{source}, p.{page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Context from uploaded documents:\n{context}\n\nQuestion: {question}"),
    ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
