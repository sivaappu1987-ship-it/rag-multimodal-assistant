"""
agent_flow.py — Unified Agentic Ingestion + Retrieval Flow using LangGraph.
Implements routing, scraping, version caching, fuzzy product ID matching, hybrid RAG,
mode classification (Q&A/Troubleshooting), step generation, and response formatting.
"""
import os
import re
import json
import sqlite3
import hashlib
import requests
from typing import TypedDict, Optional, List, Dict, Any, Literal
from bs4 import BeautifulSoup
from fastapi import HTTPException

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.config import settings, BASE_DIR
from app.services.product_identifier import identify_product as identify_product_service
from app.services.chunker import chunk_markdown
from app.services.embedder import EmbedderService
from app.services.vector_store import VectorStoreService
from app.services.retriever import retrieve_context as retrieve_context_service
from langgraph.graph import StateGraph, END

# --- SQLite Version Cache Registry ---
REGISTRY_DB_PATH = str(settings.INPUT_DIR.parent / "registry.db")

def init_registry_db():
    os.makedirs(os.path.dirname(REGISTRY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(REGISTRY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registry (
            filepath TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_registry_db()

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    source_input: Optional[str]        # URL or local filename
    source_content: Optional[bytes]    # Uploaded raw bytes (for files)
    product_id: Optional[str]
    clarification_needed: bool
    retrieved_chunks: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    mode: Literal["qa", "troubleshoot"]
    answer: str
    steps: Optional[List[str]]
    content_changed: bool
    version_info: Optional[str]
    clarification_options: Optional[List[str]]

# --- Graph Nodes ---

def url_ingest(state: AgentState) -> Dict[str, Any]:
    url = state["source_input"]
    print(f"[AgentFlow] Scraping URL: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL {url}: {str(e)}")
        
    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.title.string.strip() if soup.title else "Scraped Webpage"
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
        
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text_content = "\n".join(chunk for chunk in chunks if chunk)
    
    markdown_content = f"# {title}\n\nSource URL: {url}\n\n{text_content}"
    
    # Generate clean filename
    safe_slug = re.sub(r'[^a-zA-Z0-9]', '_', url.replace("https://", "").replace("http://", ""))[:50]
    filename = f"url_{safe_slug}.txt"
    
    input_dir = str(settings.INPUT_DIR)
    raw_path = os.path.join(input_dir, filename)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    return {
        "source_input": filename,
        "source_content": markdown_content.encode("utf-8")
    }

def file_ingest(state: AgentState) -> Dict[str, Any]:
    filename = os.path.basename(state["source_input"])
    input_dir = str(settings.INPUT_DIR)
    raw_path = os.path.join(input_dir, filename)
    
    if state.get("source_content"):
        with open(raw_path, "wb") as f:
            f.write(state["source_content"])
    elif not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found at: {raw_path}")
        
    # Convert with MarkItDown
    print(f"[AgentFlow] Converting file to markdown: {filename}")
    from markitdown import MarkItDown
    markitdown = MarkItDown()
    result = markitdown.convert(raw_path)
    md_content = result.text_content
    
    output_dir = str(settings.OUTPUT_DIR)
    base_name, _ = os.path.splitext(filename)
    md_filename = f"{base_name}.md"
    md_path = os.path.join(output_dir, md_filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    return {
        "source_content": md_content.encode("utf-8")
    }

def version_check(state: AgentState) -> Dict[str, Any]:
    filename = state["source_input"]
    md_content = state["source_content"].decode("utf-8")
    
    md5_hash = hashlib.md5(md_content.encode("utf-8")).hexdigest()
    
    conn = sqlite3.connect(REGISTRY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT hash, version FROM registry WHERE filepath = ?", (filename,))
    row = cursor.fetchone()
    
    content_changed = True
    version = 1
    
    if row:
        db_hash, db_version = row
        if db_hash == md5_hash:
            content_changed = False
            version = db_version
        else:
            version = db_version + 1
            cursor.execute("UPDATE registry SET hash = ?, version = ?, last_updated = CURRENT_TIMESTAMP WHERE filepath = ?", 
                           (md5_hash, version, filename))
            conn.commit()
    else:
        cursor.execute("INSERT INTO registry (filepath, hash, version) VALUES (?, ?, ?)", 
                       (filename, md5_hash, version))
        conn.commit()
        
    conn.close()
    
    version_status = "unchanged" if not content_changed else ("updated" if version > 1 else "created")
    version_info = f"{filename} (v{version}) - {version_status}"
    print(f"[AgentFlow] Version check: {version_info}")
    
    return {
        "content_changed": content_changed,
        "version_info": version_info
    }

def embed_and_store(state: AgentState) -> Dict[str, Any]:
    filename = state["source_input"]
    md_content = state["source_content"].decode("utf-8")
    
    sample_text = md_content[:1500]
    metadata = identify_product_service(f"File: {filename}\n{sample_text}")
    
    chunks = chunk_markdown(md_content, source_file=filename, metadata=metadata)
    
    embedder = EmbedderService()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        
    vs = VectorStoreService()
    vs.delete_by_filename(filename)
    vs.ingest_chunks(chunks)
    
    print(f"[AgentFlow] Re-embedded and stored {len(chunks)} chunks for {filename}.")
    return {}

def identify_product(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    vs = VectorStoreService()
    existing_products = vs.get_unique_products()
    
    if not existing_products:
        return {"product_id": None, "clarification_needed": False}
        
    extracted = identify_product_service(query)
    extracted_product = extracted.get("product")
    
    if extracted_product:
        matches = [p for p in existing_products if extracted_product.upper() in p.upper() or p.upper() in extracted_product.upper()]
        if len(matches) == 1:
            print(f"[AgentFlow] Identified product: {matches[0]}")
            return {"product_id": matches[0], "clarification_needed": False}
        elif len(matches) > 1:
            print(f"[AgentFlow] Product identification ambiguous between options: {matches}")
            return {
                "product_id": None, 
                "clarification_needed": True, 
                "clarification_options": matches
            }
            
    # Fuzzy match product names directly in the query text
    matches = [p for p in existing_products if p.lower() in query.lower()]
    if len(matches) == 1:
        print(f"[AgentFlow] Identified product (fuzzy match): {matches[0]}")
        return {"product_id": matches[0], "clarification_needed": False}
    elif len(matches) > 1:
        print(f"[AgentFlow] Fuzzy matches ambiguous: {matches}")
        return {
            "product_id": None, 
            "clarification_needed": True, 
            "clarification_options": matches
        }
        
    if len(existing_products) == 1:
        print(f"[AgentFlow] Defaulting to single existing product: {existing_products[0]}")
        return {"product_id": existing_products[0], "clarification_needed": False}
        
    return {"product_id": None, "clarification_needed": False}

def classify_mode(state: AgentState) -> Dict[str, Any]:
    query = state["query"].lower()
    
    trouble_keywords = ["error", "fail", "broken", "troubleshoot", "won't", "diagnose", "fix", "issue", "problem", "fault"]
    has_error_code = bool(re.search(r"\b(e\d{3})\b", query))
    
    if has_error_code or any(k in query for k in trouble_keywords):
        print("[AgentFlow] Keyword classifier: troubleshoot mode.")
        return {"mode": "troubleshoot"}
        
    from app.config import settings
    from app.main import call_llm
    
    if settings.LLM_PROVIDER != "none":
        prompt = f"""Classify the user's technical support query.
Query: "{state["query"]}"
Respond with either 'troubleshoot' (if reporting a problem, error, or failure) or 'qa' (if asking a general information question). Do not include any other text or explanation. Only respond with 'troubleshoot' or 'qa'."""
        try:
            llm_response = call_llm(prompt).strip().lower()
            mode = "troubleshoot" if "troubleshoot" in llm_response else "qa"
            print(f"[AgentFlow] LLM classifier: {mode} mode.")
            return {"mode": mode}
        except Exception as e:
            print(f"[AgentFlow] LLM classifier failed: {str(e)}. Defaulting to qa.")
            
    return {"mode": "qa"}

def retrieve(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    product_id = state["product_id"]
    
    query_entities = {}
    if product_id:
        query_entities = {"product": product_id, "model": product_id}
        
    chunks = retrieve_context_service(query, query_entities=query_entities)
    
    sources = []
    for c in chunks:
        sources.append({
            "source": c["source"],
            "page": c.get("page"),
            "product": c.get("product")
        })
        
    unique_sources = []
    seen = set()
    for s in sources:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)
            
    return {
        "retrieved_chunks": chunks,
        "sources": unique_sources
    }

def generate(state: AgentState) -> Dict[str, Any]:
    chunks = state["retrieved_chunks"]
    query = state["query"]
    mode = state["mode"]
    
    if not chunks:
        return {"answer": "I could not find that information in the uploaded manuals."}
        
    context_str = "\n\n".join([f"--- Source: {c['source']} (Page {c.get('page')}) ---\n{c['content']}" for c in chunks])
    from app.main import call_llm
    
    if mode == "qa":
        prompt = f"""You are a technical support assistant. Answer the user's question using only the provided context. If the answer cannot be found in the context, say "I could not find that information in the uploaded manuals."
Context:
{context_str}

User Question: {query}
Answer:"""
        answer = call_llm(prompt)
        return {"answer": answer}
    else:
        prompt = f"""You are a technical support diagnostic assistant. Analyze the context and provide a step-by-step diagnostic guide for the user's troubleshooting issue.
Generate your response strictly as a JSON object with two fields:
1. "answer": A brief explanation of the problem based on the context.
2. "steps": A JSON list of string steps representing the diagnostic sequence.

Example format:
{{
  "answer": "This is an ink cartridge failure.",
  "steps": ["Step 1: Turn off the printer", "Step 2: Check carriage..."]
}}

Return only valid JSON. Do not write any markdown, backticks, or other text outside the JSON.
Context:
{context_str}

User Query: {query}"""
        response_text = call_llm(prompt)
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(cleaned_text)
            return {
                "answer": data.get("answer", ""),
                "steps": data.get("steps", [])
            }
        except Exception as e:
            print(f"[AgentFlow] Failed to parse troubleshooting JSON: {str(e)}. Raw response: {response_text}")
            return {"answer": response_text, "steps": []}

def format_response(state: AgentState) -> Dict[str, Any]:
    if state.get("clarification_needed"):
        options = state.get("clarification_options", [])
        options_str = ", ".join(options)
        return {
            "answer": f"I detected multiple products matching your request: {options_str}. Please specify which product model you are asking about.",
            "steps": [],
            "sources": [],
            "clarification_needed": True
        }
        
    return {
        "answer": state["answer"],
        "steps": state.get("steps") or [],
        "sources": state.get("sources") or [],
        "product_id": state.get("product_id"),
        "clarification_needed": False,
        "version_info": state.get("version_info")
    }

# --- Graph Assembly ---

def ingest_router(state: AgentState) -> str:
    if state.get("source_input"):
        url = state["source_input"]
        if url.startswith("http://") or url.startswith("https://"):
            return "url_ingest"
        else:
            return "file_ingest"
    else:
        return "identify_product"

def version_router(state: AgentState) -> str:
    if state["content_changed"]:
        return "embed_and_store"
    else:
        return "identify_product"

def product_router(state: AgentState) -> str:
    if state["clarification_needed"]:
        return "format_response"
    else:
        return "classify_mode"

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("url_ingest", url_ingest)
    workflow.add_node("file_ingest", file_ingest)
    workflow.add_node("version_check", version_check)
    workflow.add_node("embed_and_store", embed_and_store)
    workflow.add_node("identify_product", identify_product)
    workflow.add_node("classify_mode", classify_mode)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("format_response", format_response)
    
    workflow.set_conditional_entry_point(
        ingest_router,
        {
            "url_ingest": "url_ingest",
            "file_ingest": "file_ingest",
            "identify_product": "identify_product"
        }
    )
    
    workflow.add_edge("url_ingest", "version_check")
    workflow.add_edge("file_ingest", "version_check")
    
    workflow.add_conditional_edges(
        "version_check",
        version_router,
        {
            "embed_and_store": "embed_and_store",
            "identify_product": "identify_product"
        }
    )
    
    workflow.add_edge("embed_and_store", "identify_product")
    
    workflow.add_conditional_edges(
        "identify_product",
        product_router,
        {
            "format_response": "format_response",
            "classify_mode": "classify_mode"
        }
    )
    
    workflow.add_edge("classify_mode", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Singleton graph instance
agent_graph = build_agent_graph()
