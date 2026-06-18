from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import os
from dotenv import load_dotenv
import json

load_dotenv()

from .ingest import ingest_file_to_vectordb, format_provenance
from .vector_store import load_vectordb, query_vector_store
from .agents import Agent

app = FastAPI(title="GenAI Document Q&A")

STORAGE_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR = STORAGE_DIR / "status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

# Simple API key check for write endpoints
API_KEY = os.getenv("API_KEY")

def require_api_key(x_api_key: str = Header(None)):
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")
    return True

# Serve a simple static frontend
WEB_DIR = Path(__file__).parent.parent / "static"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return index.read_text()
    return {"status": "ok", "message": "GenAI Document Q&A API"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), ok: bool = Depends(require_api_key)):
    # basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    # save file to disk
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    path = STORAGE_DIR / filename
    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)
    # write initial status
    status_file = STATUS_DIR / f"{file_id}.json"
    with open(status_file, "w", encoding="utf-8") as sf:
        json.dump({"id": file_id, "filename": file.filename, "status": "processing"}, sf)
    # ingest into persistent vectordb
    try:
        vectordb = ingest_file_to_vectordb(str(path))
        # update status
        with open(status_file, "w", encoding="utf-8") as sf:
            json.dump({"id": file_id, "filename": file.filename, "status": "done", "vectordb": bool(vectordb)}, sf)
    except Exception as e:
        with open(status_file, "w", encoding="utf-8") as sf:
            json.dump({"id": file_id, "filename": file.filename, "status": "error", "error": str(e)}, sf)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"filename": file.filename, "id": file_id, "vectordb": bool(vectordb)})


@app.get("/status/{file_id}")
async def status(file_id: str):
    status_file = STATUS_DIR / f"{file_id}.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Status not found")
    with open(status_file, "r", encoding="utf-8") as sf:
        return JSONResponse(json.load(sf))


@app.post("/query")
async def query(q: str = Form(...)):
    # basic query that uses existing vectordb if any
    if not q or len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    vectordb = load_vectordb()
    if not vectordb:
        raise HTTPException(status_code=400, detail="Vector DB not found. Ingest documents first.")
    results = query_vector_store(vectordb, q, k=4)
    # return only text snippets
    snippets = []
    for r in results:
        text = getattr(r, "page_content", None) or str(r)
        snippets.append(text)
    return JSONResponse({"query": q, "snippets": snippets})


@app.post("/query_rag")
async def query_rag(q: str = Form(...)):
    if not q or len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    vectordb = load_vectordb()
    if not vectordb:
        raise HTTPException(status_code=400, detail="Vector DB not found. Ingest documents first.")
    results = query_vector_store(vectordb, q, k=4)
    context = "\n\n".join([getattr(r, "page_content", None) or str(r) for r in results])
    # call OpenAI (simple call using openai package)
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt = f"Use the following context to answer the question. Context:\n{context}\n\nQuestion: {q}\nAnswer:" 
        resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=512)
        answer = resp.choices[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # simple hallucination mitigation: include source snippets
    return JSONResponse({"query": q, "answer": answer, "sources": format_provenance(results)})


@app.post("/query_agent")
async def query_agent(q: str = Form(...)):
    if not q or len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    vectordb = load_vectordb()
    if not vectordb:
        raise HTTPException(status_code=400, detail="Vector DB not found. Ingest documents first.")
    agent = Agent("planner1")
    def retriever(query: str):
        return query_vector_store(vectordb, query, k=4)
    result = agent.run(q, retriever)
    return JSONResponse({"query": q, "result": result})
