"""
main.py — FastAPI application for the Multimodal RAG Assistant.
Phase 3: Full RAG pipeline — embed → retrieve → prompt → LLM → grounded answer.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from app.services.parser import ParserService
from app.services.retriever import retrieve_context
from app.services.prompt_builder import build_prompt
from app.config import LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY, SAMBANOVA_API_KEY
from app.services.prompt_guard import is_prompt_injection

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Multimodal RAG Assistant API", version="3.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eagerly initialize services (loads embedding model at startup)
parser_service = ParserService()

# ─── Pydantic Models ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    source_file: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    vectors_stored: int

class UploadResponse(BaseModel):
    filename: str
    markdown_file: str
    chunks_ingested: int
    status: str

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    language: Optional[str] = "en"

class TroubleshootRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)

class AgentRequest(BaseModel):
    query: str
    source_input: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    steps: list[str] = []
    sources: list[dict] = []
    product_id: Optional[str] = None
    clarification_needed: bool = False
    version_info: Optional[str] = None

# ─── LLM Helper ─────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call the configured LLM provider and return the response text."""

    if LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    elif LLM_PROVIDER == "sambanova":
        from openai import OpenAI
        client = OpenAI(
            api_key=SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1",
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    else:
        raise RuntimeError(
            "No LLM API key configured. Add GROQ_API_KEY or SAMBANOVA_API_KEY to backend/.env"
        )

# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    from app.services.vector_store import VectorStoreService
    vs = VectorStoreService()
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "vectors_stored": vs.count(),
    }


@app.get("/files")
def get_files():
    """Retrieve all unique source files loaded in the vector store."""
    from app.services.vector_store import VectorStoreService
    vs = VectorStoreService()
    try:
        files = vs.get_unique_sources()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.get("/products")
def get_products():
    """Retrieve all unique product names loaded in the vector store."""
    from app.services.vector_store import VectorStoreService
    vs = VectorStoreService()
    try:
        products = vs.get_unique_products()
        return {"products": products}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list products: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(payload: ChatRequest, request: Request):
    """
    RAG chat endpoint:
      1. Protect against prompt injection
      2. Retrieve relevant chunks from Qdrant
      3. If no chunks found → return safe fallback
      4. Build grounded prompt
      5. Call LLM
      6. Return answer + unique source files
    """
    if is_prompt_injection(payload.message):
        raise HTTPException(status_code=400, detail="Potential prompt injection detected.")

    fallback = "I could not find that information in the uploaded manuals."

    # Step 1: Retrieve context
    chunks = retrieve_context(payload.message, source_file=payload.source_file)

    # Step 2: Fallback if nothing retrieved
    if not chunks:
        return ChatResponse(answer=fallback, sources=[])

    # Step 3: Build prompt
    prompt = build_prompt(chunks, payload.message)

    # Step 4: Call LLM
    try:
        answer = call_llm(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    # Step 5: Collect unique sources
    normalized_answer = answer.strip().rstrip(".").lower()
    normalized_fallback = fallback.strip().rstrip(".").lower()
    if normalized_answer == normalized_fallback:
        sources = []
    else:
        sources = list(dict.fromkeys(c["source"] for c in chunks))

    return ChatResponse(answer=answer, sources=sources)


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a document, convert it, chunk it, embed it, and store in Qdrant."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Validate file size (25MB limit)
    MAX_FILE_SIZE = 25 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Max size allowed is 25MB."
        )

    # Validate file extension
    allowed_extensions = {".pdf", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt"}
    import os
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {supported}",
        )

    # Validate MIME type
    allowed_mime_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.ms-excel",
        "text/plain"
    }
    if file.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"MIME type '{file.content_type}' is not allowed.",
        )

    try:
        result = parser_service.parse_file(file.filename, content)
        return UploadResponse(
            filename=file.filename,
            markdown_file=result["markdown_file"],
            chunks_ingested=result["chunks_ingested"],
            status="processed",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    hint_lang: str = Form("auto")
):
    """Transcribe uploaded audio file using hybrid local/remote engines."""
    import tempfile
    import os

    # Write to a temporary file
    suffix = os.path.splitext(audio.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        from app.services.audio import transcribe_audio
        result = await transcribe_audio(tmp_path, hint_lang)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/speak")
@limiter.limit("20/minute")
async def speak(payload: SpeakRequest, request: Request):
    """Generate text-to-speech MP3 stream using edge-tts."""
    try:
        from app.services.audio import speak_text
        audio_bytes = await speak_text(payload.text, payload.language)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/troubleshoot")
@limiter.limit("20/minute")
async def troubleshoot(payload: TroubleshootRequest, request: Request):
    """
    Agentic Troubleshooting endpoint.
    Performs multi-turn state-guided diagnosis and recommendation.
    """
    if is_prompt_injection(payload.message):
        raise HTTPException(status_code=400, detail="Potential prompt injection detected.")

    from app.services.workflow_manager import process_troubleshoot_turn
    try:
        result = await process_troubleshoot_turn(payload.session_id, payload.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/run", response_model=AgentResponse)
@limiter.limit("10/minute")
async def agent_run(payload: AgentRequest, request: Request):
    """
    Unified Agentic Ingestion + Retrieval endpoint.
    Scrapes web page or ingests file if provided, and performs grounded retrieval.
    """
    if is_prompt_injection(payload.query):
        raise HTTPException(status_code=400, detail="Potential prompt injection detected.")
        
    inputs = {
        "query": payload.query,
        "source_input": payload.source_input,
        "source_content": None,
        "product_id": None,
        "clarification_needed": False,
        "retrieved_chunks": [],
        "sources": [],
        "mode": "qa",
        "answer": "",
        "steps": [],
        "content_changed": False,
        "version_info": None,
        "clarification_options": []
    }
    
    from app.services.agent_flow import agent_graph
    try:
        # Run graph synchronously
        result = agent_graph.invoke(inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow execution failed: {str(e)}")