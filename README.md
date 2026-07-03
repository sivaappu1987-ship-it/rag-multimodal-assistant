# Multimodal RAG Assistant & Agentic Troubleshooting Engine

A production-ready **Multimodal Retrieval-Augmented Generation (RAG) Assistant** and **State-Guided Agentic Engine** designed to ingest technical manuals, perform context-aware hierarchical search, handle voice-based communications, execute multi-turn diagnostic troubleshooting workflows, and enforce security policies.

---

## 🚀 Key Features

* **📄 Multimodal Document Ingestion**: Converts uploads (`PDF`, `DOCX`, `PPTX`, `XLSX`, `TXT`) into Markdown using **Microsoft MarkItDown**, performing zero-shot product classification.
* **🔍 Hierarchical Hybrid Search**: Searches Qdrant using a 3-level priority hierarchy (Exact Match -> Family Match -> Global Match) combining dense (MiniLM-L6) and sparse (BM25) candidates using **Reciprocal Rank Fusion (RRF)**.
* **🎙️ Hybrid Voice Layer**: Captures audio input and routes transcription based on language hint (local Whisper for English; remote Sarvam AI Saaras v3 API for Indic languages). Synthesizes speech outputs using **edge-tts** with Microsoft Neural voices.
* **🤖 Agentic Troubleshooting Engine**: Guides users through diagnostic trees tracking session parameters, history, and RAG context blocks across turns.
* **🦜 Unified Agentic Ingestion & Retrieval**: Uses LangGraph `StateGraph` to scrap URLs (BeautifulSoup), cache versions via SQLite file hashes, resolve fuzzy product IDs, classify queries (`qa` vs. `troubleshoot`), and generate troubleshooting steps through a single `POST /agent/run` API.
* **🛡️ Security & Hardening**:
  * **Rate Limiting**: Integrated `slowapi` rate limits on all major endpoints.
  * **File Upload Guard**: Restricts upload sizes to `<= 25MB` and validates file MIME types/extensions.
  * **Prompt Injection Protection**: Employs regex guards (`prompt_guard.py`) to block jailbreak/override instructions.
  * **Isolated Prompts**: Isolates RAG system prompts, instructing the LLM to treat manual instructions strictly as data, not commands.
  * **Secrets & CI**: Never tracks `.env` keys. Includes GitHub Actions workflows for quality checks (`ruff`, `black`, `bandit`) and secret scanning (`detect-secrets`).
* **🧪 24-Test Suite**: Includes comprehensive mock-heavy unit, integration, and security checks under `backend/tests/` running instantly in any environment.

---

## 📁 Project Structure

```text
rag-multimodal-assistant/
├── docker-compose.yml         # Container configuration for Backend + Frontend
├── contributing.md            # Onboarding & Local Setup guide
├── architecture.md            # System architectures & Orchestration flowcharts
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt       # Hardened requirements (slowapi, pytest, bandit, etc.)
│   ├── .env.example           # Environment template file
│   └── app/
│       ├── main.py            # FastAPI Application routes & Rate limiter setup
│       ├── config.py          # Unified Settings manager using pathlib.Path
│       └── services/
│           ├── parser.py      # MarkItDown document parse & metadata extractor
│           ├── chunker.py     # Smart overlapping parser chunker
│           ├── embedder.py    # Local SentenceTransformer embeddings
│           ├── vector_store.py# Qdrant interface (filters, counts, scrolls)
│           ├── hybrid_search.py# In-memory BM25 sparse search and RRF
│           ├── retriever.py   # 3-level prioritized retriever
│           ├── audio.py       # Speech Transcriber (Whisper/Sarvam) & TTS (edge-tts)
│           ├── prompt_guard.py# Prompt injection filter
│           ├── agent_flow.py  # LangGraph unified ingestion & RAG graph
│           └── workflow_manager.py# Troubleshooting state machine
└── frontend/                  # Next.js UI app codebase
```

---

## 🏗️ Architecture & Orchestration

For in-depth explanations, mermaid data flows, and state transition flowcharts, see:
* **[architecture.md](architecture.md)**

---

## ⚙️ Configuration & Run

### 1. Setup Environment
Create a `backend/.env` file from the example template:
```bash
cp backend/.env.example backend/.env
```
Fill in the API keys:
```env
GROQ_API_KEY=your_groq_key
SAMBANOVA_API_KEY=your_sambanova_key
SARVAM_API_KEY=your_sarvam_key
```

### 2. Launch with Docker Compose
To build and run the entire ecosystem (FastAPI Backend + Next.js Frontend + pre-cached embedding weights) locally:
```bash
docker compose up --build
```

### 3. Verification & Testing
To run the 24-test suite locally inside the backend directory:
```bash
cd backend
.\venv\Scripts\pytest -vv
```

---

## 🚀 Access Points
* **Frontend UI**: http://localhost:3000
* **Admin Upload Panel**: http://localhost:3000/admin
* **Backend Docs / API**: http://localhost:8000/docs
* **Unified Agent Flow**: `POST http://localhost:8000/agent/run`
* **Health endpoint**: http://localhost:8000/health
