# DataSci Assistant

Chatbot berbasis LLM untuk Data Scientist — dibangun sebagai proyek akhir training **Hacktiv8: LLM-Based Tools and Gemini API Integration for Data Scientists**.

## Fitur

| Fitur | Deskripsi |
|---|---|
| Multi-LLM | Gemini 2.5 Flash (default) atau Llama 3.1 8B via Groq — bisa dipilih di sidebar |
| RAG | Upload PDF → index ke ChromaDB → tanya jawab dokumen dengan citation |
| Vision | Upload gambar/chart untuk analisis via Gemini vision |
| Function Calling | Tool `get_current_datetime` dan `explain_ml_metric` otomatis dipanggil Gemini |
| Streaming | Respons muncul real-time token-by-token |
| Chat history | Percakapan tersimpan di session, ada tombol Clear |

## Tech Stack

| Komponen | Library |
|---|---|
| UI | Streamlit |
| Primary LLM + Vision | google-genai + langchain-google-genai |
| Secondary LLM | langchain-groq (Llama 3.1 8B) |
| Embeddings | gemini-embedding-2 (768 dims) |
| Vector DB | ChromaDB via langchain-community |
| PDF Loading | PyPDFLoader (pypdf) |
| Chunking | RecursiveCharacterTextSplitter |
| RAG Chain | LCEL (LangChain Expression Language) |
| Function Calling | @tool + bind_tools() |
| Env | python-dotenv |

## Setup Lokal

**1. Clone repo**
```bash
git clone https://github.com/<username>/datasci-assistant.git
cd datasci-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Buat file `.env`**
```bash
cp .env.example .env
```
Isi dengan API key kamu:
```
GOOGLE_API_KEY=...   # dari https://aistudio.google.com
GROQ_API_KEY=...     # dari https://console.groq.com
```

**4. Jalankan**
```bash
streamlit run app.py
```
Buka `http://localhost:8501` di browser.

## Cara Pakai

**Direct chat** — langsung ketik pertanyaan tentang data science, ML, Python, statistik, dll.

**Function calling** — coba tanya:
- *"What is today's date?"* → Gemini memanggil tool `get_current_datetime`
- *"Explain AUC-ROC"* → Gemini memanggil tool `explain_ml_metric`

**RAG** — upload PDF di sidebar → klik "Index Documents" → tanya isi dokumen.
> Catatan: indexing PDF besar (~100+ chunk) butuh ~1 menit karena rate limit free tier gemini-embedding-2.

**Vision** — pilih model Gemini → upload gambar/chart → ketik pertanyaan.

## Deployment ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Buka [https://share.streamlit.io](https://share.streamlit.io) → login dengan GitHub.
3. Klik **New app** → pilih repo → Main file: `app.py`.
4. Klik **Advanced settings** → **Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "your_key"
   GROQ_API_KEY = "your_key"
   ```
5. Klik **Deploy**. App live di `https://<username>-datasci-assistant-<hash>.streamlit.app`.

> `vectorstore/` tidak perlu di-push ke GitHub (sudah di `.gitignore`).
> Di Streamlit Cloud, user perlu re-index PDF setiap kali app restart (free tier tidak punya persistent storage).

## Struktur Project

```
datasci-assistant/
├── app.py          # Streamlit entry point — UI + orchestration
├── llm.py          # LLM factory: Gemini + Groq/Llama
├── rag.py          # RAG pipeline: PDF → chunk → ChromaDB → LCEL chain
├── tools.py        # Function calling tools
├── requirements.txt
├── .env.example    # Template env vars (di-commit)
├── .env            # Actual keys (TIDAK di-commit)
├── .gitignore
└── README.md
```
