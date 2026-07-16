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
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

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
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    steps: list[str] = []
    sources: list[dict] = []
    product_id: Optional[str] = None
    clarification_needed: bool = False
    version_info: Optional[str] = None
    clarification_question: Optional[str] = None
    status: Optional[str] = None

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


def build_clarification_from_ambiguity(raw: str) -> str:
    """
    Validates an LLM-produced ambiguity string before serving it to the user.
    Falls back to a constructed generic-but-topic-aware question if the
    raw string doesn't look like a real question.
    """
    if not raw or not isinstance(raw, str):
        return "Could you clarify what you're referring to?"

    cleaned = raw.strip()

    # Minimum sanity checks — not full NLP validation, just guard against
    # empty strings, junk tokens, or non-question fragments.
    if len(cleaned) < 10:
        return "Could you clarify what you're referring to?"
    if not cleaned.endswith("?"):
        # Doesn't look like a question — still usable as context, but
        # wrap it rather than serve it raw.
        return f"Could you clarify: {cleaned}"

    return cleaned


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(payload: ChatRequest, request: Request):
    """
    Phase 1 & 2 RAG chat endpoint with Query Understanding, Relevance Guard, and Stateful Clarification.
    """
    if is_prompt_injection(payload.message):
        raise HTTPException(status_code=400, detail="Potential prompt injection detected.")

    from app.config import MAX_CLARIFICATION_ATTEMPTS
    session_id = payload.session_id
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    from app.services.session_store import SessionStore
    session_store = SessionStore()
    session = session_store.get(session_id)

    user_message = payload.message

    # --- Phase 2: Context Reconstruction ---
    if session.get("pending_clarification"):
        session["clarification_attempts"] = session.get("clarification_attempts", 0) + 1
        if session["clarification_attempts"] > MAX_CLARIFICATION_ATTEMPTS:
            session["pending_clarification"] = False
            session["clarification_attempts"] = 0
            session_store.save(session_id, session)
            return ChatResponse(
                answer="I'm still having trouble understanding. Could you please state the product model and the issue you're facing in one complete sentence?",
                sources=[],
                needs_clarification=False
            )
        
        from app.services.context_reconstruction import reconstruct_query
        resolved_query, res_conf = reconstruct_query(
            original_query=session.get("last_valid_user_query", ""),
            clarification_question=session.get("clarification_question", ""),
            user_followup=user_message,
            product_hint=session.get("product")
        )

        if is_prompt_injection(resolved_query):
            raise HTTPException(status_code=400, detail="Potential prompt injection detected in reconstructed query.")

        if res_conf == "LOW":
            # If the follow-up makes no sense, trigger another clarification immediately
            session_store.save(session_id, session)
            return ChatResponse(
                answer=f"I didn't quite catch that. {session.get('clarification_question', 'Could you clarify?')}",
                sources=[],
                needs_clarification=True,
                clarification_question=session.get("clarification_question")
            )

        # Replace user_message with the semantically resolved query
        user_message = resolved_query
        session["pending_clarification"] = False
        session["clarification_attempts"] = 0

    fallback = "I could not find that information in the uploaded manuals."

    from app.services.query_understanding import understand_query
    understood = understand_query(user_message)
    input_confidence = understood.get("input_confidence", "LOW")

    # Step 1 & 2: Route on input_confidence
    if input_confidence == "LOW":
        product_hint = understood.get("product_hint")
        issue_hint = understood.get("issue_hint")
        ambiguities = understood.get("ambiguities", [])
        
        if product_hint and issue_hint:
            clarification_q = f"I see this is about the {product_hint} and a {issue_hint} issue — can you tell me a bit more about what's happening?"
        elif product_hint and not issue_hint:
            clarification_q = f"I see this is about the {product_hint} — what's happening with it?"
        elif issue_hint and not product_hint:
            clarification_q = f"Which product is having this {issue_hint} issue?"
        elif ambiguities and isinstance(ambiguities, list) and len(ambiguities) > 0:
            clarification_q = build_clarification_from_ambiguity(ambiguities[0])
        else:
            clarification_q = "Could you tell me more about what you need help with?"

        session["last_valid_user_query"] = user_message
        session["pending_clarification"] = True
        session["clarification_question"] = clarification_q
        session_store.save(session_id, session)

        return ChatResponse(
            answer=clarification_q,
            sources=[],
            needs_clarification=True,
            clarification_question=clarification_q
        )
    elif input_confidence == "MEDIUM":
        ambiguities = understood.get("ambiguities", [])
        if ambiguities and isinstance(ambiguities, list) and len(ambiguities) > 0:
            clarification_q = build_clarification_from_ambiguity(ambiguities[0])
            session["last_valid_user_query"] = user_message
            session["pending_clarification"] = True
            session["clarification_question"] = clarification_q
            session_store.save(session_id, session)
            return ChatResponse(
                answer=clarification_q,
                sources=[],
                needs_clarification=True,
                clarification_question=clarification_q
            )
        # Else proceed to retrieval
        
    # Step 3: Retrieval
    chunks, retrieval_confidence = retrieve_context(
        understood["normalized_query"],
        source_file=payload.source_file,
        query_entities=understood["entities"]
    )

    # Step 4: Route on retrieval_confidence
    if retrieval_confidence == "LOW":
        # Extend fallback for LOW relevance or zero results
        low_fallback = fallback + " The query might be too vague or unrelated to the manuals."
        return ChatResponse(answer=low_fallback, sources=[])
        
    elif retrieval_confidence == "MEDIUM":
        # Check product match if hint provided
        product_hint = understood.get("product_hint")
        if product_hint and chunks:
            top_chunk_product = chunks[0].get("product", "")
            if top_chunk_product and product_hint.lower() not in top_chunk_product.lower():
                # Ask clarifying question if product mismatch
                clarification_q = f"I found some information for {top_chunk_product}, but you asked about {product_hint}. Should I proceed with the details for {top_chunk_product}?"
                return ChatResponse(
                    answer=clarification_q,
                    sources=[],
                    needs_clarification=True,
                    clarification_question=clarification_q
                )
        
        # Build prompt instructing to hedge
        base_prompt = build_prompt(chunks, payload.message)
        prompt = base_prompt + "\n\nNote: The context provided may only partially cover the question. Please hedge your answer and note any uncertainty."
    else:
        # HIGH confidence
        prompt = build_prompt(chunks, payload.message)

    # Step 5: Call LLM
    try:
        answer = call_llm(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    # Collect unique sources
    normalized_answer = answer.strip().rstrip(".").lower()
    normalized_fallback = fallback.strip().rstrip(".").lower()
    if normalized_answer == normalized_fallback or normalized_answer.startswith("i could not find"):
        sources = []
    else:
        sources = list(dict.fromkeys(c["source"] for c in chunks))

    # Successful turn, save product state if extracted
    if understood.get("product_hint"):
        session["product"] = understood.get("product_hint")
    session_store.save(session_id, session)

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
        
    session_id = payload.session_id
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        
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
        "clarification_options": [],
        "session_id": session_id,
        "input_confidence": "LOW",
        "retrieval_confidence": "LOW",
        "clarification_question": None,
        "clarification_attempts": 0,
        "resolved_query": None,
        "retrieval_retries": 0,
        "understood_data": {}
    }
    
    from app.services.agent_flow import agent_graph
    try:
        # Run graph synchronously
        result = agent_graph.invoke(inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow execution failed: {str(e)}")