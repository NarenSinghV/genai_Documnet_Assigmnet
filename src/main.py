from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import json
import re
import logging
import os
import requests
from typing import List, Optional

from sentence_transformers import SentenceTransformer, util

from .ingest import ingest_file_to_vectordb, format_provenance
from .vector_store import load_vectordb

# -------------------------
# App
# -------------------------
app = FastAPI(title="GenAI Document Q&A – Local RAG")

# -------------------------
# Storage
# -------------------------
STORAGE_DIR = Path("./uploads")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

STATUS_DIR = STORAGE_DIR / "status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Static frontend (optional)
# -------------------------
WEB_DIR = Path(__file__).parent.parent / "static"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# configure logger
logger = logging.getLogger("genai")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/text-bison-001")


def _call_gemini(prompt: str, max_output_tokens: int = 512, model: str = None) -> str:
    """Call Google Generative Language (Gemini) REST API using an API key.
    Expects GEMINI_API_KEY in environment. Model should be a valid model name
    such as 'models/text-bison-001' or a Gemini model if available.
    This function tries to parse common response shapes and falls back to
    raising a clear exception.
    """
    mdl = model or GEMINI_MODEL
    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    url = f"https://generativelanguage.googleapis.com/v1beta2/{mdl}:generateText?key={api_key}"
    payload = {
        "prompt": {"text": prompt},
        "maxOutputTokens": max_output_tokens
    }
    headers = {"Content-Type": "application/json"}
    logger.info("Calling Gemini model %s (tokens=%s)", mdl, max_output_tokens)
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        logger.error("Gemini API error %s: %s", resp.status_code, resp.text)
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # try common shapes
    text = None
    if isinstance(data, dict):
        # v1beta2: candidates -> [ { "output": "..." } ] or { "candidates": [ {"content": ...} ] }
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            cand = data["candidates"][0]
            text = cand.get("content") or cand.get("output") or cand.get("text")
        if not text and "output" in data:
            # some responses put text directly
            out = data.get("output")
            if isinstance(out, dict):
                text = out.get("text") or out.get("content")
            elif isinstance(out, str):
                text = out
    if not text:
        # last resort: stringify
        text = data.get("candidates", [{}])[0].get("content") if isinstance(data, dict) else None
    if not text:
        raise RuntimeError("Could not parse Gemini response: %s" % (data,))
    return text.strip()

# -------------------------
# Root
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return index.read_text()

    return JSONResponse({
        "status": "ok",
        "message": "GenAI Document Q&A – Local RAG (No OpenAI)",
        "endpoints": [
            "POST /upload",
            "POST /query_rag",
            "GET /status/{file_id}"
        ]
    })

# -------------------------
# Upload & ingest
# -------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    logger.info("Upload request received: %s", getattr(file, "filename", "<no-name>"))
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    path = STORAGE_DIR / filename

    with open(path, "wb") as f:
        f.write(await file.read())

    status_file = STATUS_DIR / f"{file_id}.json"
    status_file.write_text(json.dumps({
        "id": file_id,
        "filename": filename,
        "status": "processing"
    }))

    try:
        vectordb = ingest_file_to_vectordb(str(path))
        status_file.write_text(json.dumps({
            "id": file_id,
            "filename": filename,
            "status": "done",
            "vectordb": bool(vectordb)
        }))
        logger.info("Ingest completed for %s, vectordb=%s", filename, bool(vectordb))
    except Exception as e:
        logger.exception("Ingest failed for %s", filename)
        status_file.write_text(json.dumps({
            "id": file_id,
            "filename": filename,
            "status": "error",
            "error": str(e)
        }))
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "File uploaded and indexed successfully",
        "file_id": file_id,
        "filename": filename
    }

# -------------------------
# Status
# -------------------------
@app.get("/status/{file_id}")
async def status(file_id: str):
    status_file = STATUS_DIR / f"{file_id}.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Status not found")
    return JSONResponse(json.loads(status_file.read_text()))

# -------------------------
# Helpers
# -------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()

# -------------------------
# Embedding model (reload-safe)
# -------------------------
_embedder = None

@app.on_event("startup")
def load_embedder():
    global _embedder
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

def _answer_from_context(context: str, question: str, n: int = 4) -> str:
    sentences = re.split(r'(?<=[\.\?\!])\s+', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if not sentences:
        return context[:1200]

    q_emb = _embedder.encode(question, convert_to_tensor=True)
    s_emb = _embedder.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, s_emb)[0]
    top_idx = scores.argsort(descending=True)[:n]

    selected = [sentences[i] for i in top_idx]
    return " ".join(selected)

# -------------------------
# Request model + dependency
# -------------------------
class QueryRequest(BaseModel):
    q: str
    file_id: Optional[str] = None

async def get_query_payload(request: Request) -> QueryRequest:
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        return QueryRequest(**body)

    form = await request.form()
    return QueryRequest(
        q=form.get("q"),
        file_id=form.get("file_id") or None
    )

# -------------------------
# Query RAG
# -------------------------
@app.post("/query_rag")
async def query_rag(payload: QueryRequest = Depends(get_query_payload)):

    q = payload.q.strip()
    file_id = payload.file_id

    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    # file_id is optional; when provided we restrict search to that document
    source_name = None
    if file_id:
        status_file = STATUS_DIR / f"{file_id}.json"
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Invalid file_id")
        meta = json.loads(status_file.read_text())
        source_name = meta.get("filename")

    vectordb = load_vectordb()
    if not vectordb:
        raise HTTPException(status_code=400, detail="Vector DB not found")

    logger.info("Query received. q=%s, file_id=%s, source_name=%s", q, file_id, source_name)

    try:
        if source_name:
            results = vectordb.similarity_search(q, k=12, filter={"source": source_name})
        else:
            results = vectordb.similarity_search(q, k=12)
    except TypeError:
        results = vectordb.similarity_search(q, k=12)

    if not results:
        logger.info("No results for query: %s", q)
        return {
            "query": q,
            "answer": "No relevant content found in this document." if source_name else "No relevant content found.",
            "sources": []
        }

    docs = [r.page_content for r in results]

    # Rerank using sentence-transformers
    try:
        q_emb = _embedder.encode(q, convert_to_tensor=True)
        d_emb = _embedder.encode(docs, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, d_emb)[0]
        top_k = min(6, len(docs))
        top_indices = scores.argsort(descending=True)[:top_k]
    except Exception:
        logger.exception("Rerank failed, falling back to original ordering")
        top_indices = list(range(min(4, len(docs))))

    selected_chunks = [docs[i] for i in top_indices]
    context = "\n\n".join(_clean_text(chunk) for chunk in selected_chunks)

    # Build a prompt for Gemini
    prompt = (
        "You are an assistant that answers user questions using the provided document context. "
        "Use only the context to answer and cite sources by filename and chunk index when relevant.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer in concise English (no bullet points):"
    )

    # Call Gemini; if not configured or call fails, fall back to local extractor
    try:
        gemini_resp = _call_gemini(prompt)
        answer = gemini_resp
        logger.info("Gemini responded successfully for q=%s", q)
    except Exception as e:
        logger.exception("Gemini call failed, falling back to local extractor: %s", e)
        answer = _answer_from_context(context, q)

    sources = format_provenance([results[i] for i in top_indices])

    return {
        "query": q,
        "answer": answer,
        "sources": sources
    }