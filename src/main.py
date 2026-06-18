from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import json
import re
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
    except Exception as e:
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

    # -----------------------------
    # Load vector DB
    # -----------------------------
    vectordb = load_vectordb()
    if not vectordb:
        raise HTTPException(status_code=400, detail="Vector DB not found")

    # -----------------------------
    # STEP 1: Perform search (filtered by document if source_name set)
    # -----------------------------
    try:
        if source_name:
            results = vectordb.similarity_search(
                q,
                k=12,
                filter={"source": source_name}
            )
        else:
            results = vectordb.similarity_search(q, k=12)
    except TypeError:
        # some vectorstores may not accept filter kwarg
        results = vectordb.similarity_search(q, k=12)

    if not results:
        return {
            "query": q,
            "answer": "No relevant content found in this document." if source_name else "No relevant content found.",
            "sources": []
        }

    # -----------------------------
    # STEP 2: Rerank results (IMPORTANT FIX)
    # -----------------------------
    docs = [r.page_content for r in results]

    q_emb = _embedder.encode(q, convert_to_tensor=True)
    d_emb = _embedder.encode(docs, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, d_emb)[0]

    top_k = min(4, len(docs))
    top_indices = scores.argsort(descending=True)[:top_k]

    selected_chunks = [docs[i] for i in top_indices]

    # -----------------------------
    # STEP 3: Clean + build context
    # -----------------------------
    context = "\n\n".join(
        _clean_text(chunk) for chunk in selected_chunks
    )

    # -----------------------------
    # STEP 4: Generate answer
    # -----------------------------
    answer = _answer_from_context(context, q)

    # -----------------------------
    # STEP 5: return structured response
    # -----------------------------
    return {
        "query": q,
        "answer": answer,
        "sources": format_provenance([results[i] for i in top_indices])
    }