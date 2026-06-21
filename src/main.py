import os
import re
import uuid
import json
import logging
from pathlib import Path
from typing import Optional

# --- PRODUCTION LOGGING CONSTRAINTS PLATFORM ---
STORAGE_DIR = Path("./uploads")
STATUS_DIR = STORAGE_DIR / "status"
for folder in [STORAGE_DIR, STATUS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 1. Base log configuration routing setup
logger = logging.getLogger("genai")
logger.setLevel(logging.INFO)

# 2. Prevent creating duplicate handlers if uvicorn reloads the file
if not logger.handlers:
    # Standard messaging presentation format layout
    log_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # StreamHandler pipes log strings to your visible Terminal console output window
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # FileHandler persistently saves ALL logging steps into a local document file
    log_file_path = Path("./app.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8", mode="a")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

logger.info("GenAI multi-agent application logging framework initiated successfully.")


from dotenv import load_dotenv
load_dotenv()  # This explicitly injects your .env keys into os.environ

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from .ingest import ingest_file_to_vectordb
from .vector_store import load_vectordb
from .agents import AgentManager

app = FastAPI(title="GenAI Autonomous Agentic RAG Platform")

STORAGE_DIR = Path("./uploads")
STATUS_DIR = STORAGE_DIR / "status"
for folder in [STORAGE_DIR, STATUS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("genai")
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    q: str
    file_id: Optional[str] = None

async def get_query_payload(request: Request) -> QueryRequest:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        return QueryRequest(**body)
    form = await request.form()
    return QueryRequest(q=form.get("q"), file_id=form.get("file_id") or None)

# ----------------------------------------------------------------------
# Root Endpoint (Serves frontend or API map fallback)
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    # Look for the index.html inside your project's static folder
    index_path = STORAGE_DIR.parent / "static" / "index.html"
    
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        
    # Standard JSON fallback layout if you haven't built index.html yet
    return JSONResponse(
        status_code=200,
        content={
            "status": "online",
            "message": "GenAI Agentic RAG Platform is running successfully.",
            "frontend_status": f"Missing interface file at {index_path.name}. Build your HTML ui layer next.",
            "available_agent_endpoints": [
                "POST /upload  -> Register enterprise docs (PDF, CSV, TXT, Excel)",
                "POST /query_rag -> Route queries through multi-agent validation loop"
            ]
        }
    )

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
    status_file.write_text(json.dumps({"id": file_id, "filename": filename, "status": "processing"}))
    
    try:
        vectordb = ingest_file_to_vectordb(str(path))
        if vectordb is not None:
            status_file.write_text(json.dumps({"id": file_id, "filename": filename, "status": "done"}))
        else:
            status_file.write_text(json.dumps({"id": file_id, "filename": filename, "status": "error", "error": "DB write failure"}))
    except Exception as e:
        status_file.write_text(json.dumps({"id": file_id, "filename": filename, "status": "error", "error": str(e)}))
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"message": "File indexed successfully into agent knowledge base", "file_id": file_id, "filename": filename}

@app.post("/query_rag")
async def query_rag(payload: QueryRequest = Depends(get_query_payload)):
    query_str = payload.q.strip()
    file_id = payload.file_id
    
    logger.info("--- New Inbound Request Received ---")
    logger.info(f"HTTP Payload Details -> Query: '{query_str}', File Reference Session ID: {file_id}")
    
    if len(query_str) < 3:
        logger.warning("Aborting Request: Query validation constraint check failed.")
        raise HTTPException(status_code=400, detail="Query validation error: request too brief.")
        
    source_name = None
    if file_id:
        status_file = STATUS_DIR / f"{file_id}.json"
        if not status_file.exists():
            logger.error("Context Matching Exception: Document reference key tracking ID mapping not found.")
            raise HTTPException(status_code=404, detail="Requested file context handle tracking session not found.")
        meta = json.loads(status_file.read_text())
        source_name = meta.get("filename")
        logger.info(f"Session mapped cleanly. Restricting query window to filename context: {source_name}")

    vectordb = load_vectordb()
    if not vectordb:
        logger.critical("System Engine Crash: Vector store initialization failed.")
        raise HTTPException(status_code=400, detail="System Core Error: Agent Knowledge Vector Store is empty.")

    try:
        manager = AgentManager(vectordb=vectordb)
        agent_dispatch_response = manager.run(query=query_str, source_name=source_name)
        
        # --- SAFE LINE-BY-LINE CONTEXT PARSING ENGINE ---
        raw_answer = agent_dispatch_response.get("answer", "")
        
        # Clean out any stray markdown structure fragments safely
        raw_answer = (
            raw_answer.replace("###", "")
                      .replace("**", "")
                      .replace("*", "")
                      .replace("`", "")
                      .replace("---", "")
        )
        
        # Split purely by raw line changes to preserve the full generated text
        raw_lines = raw_answer.splitlines()
        final_valid_lines = []
        
        for line in raw_lines:
            cleaned_line = line.strip()
            # Retain only actual text rows, leaving numbering intact
            if cleaned_line:
                final_valid_lines.append(cleaned_line)
                
        # Join lines cleanly with a standard single new-line character
        final_clean_text = "\n".join(final_valid_lines)

        # Write execution footprint results to log file
        logger.info("--- Final Sanitized User-Facing Chatbot Answer ---")
        logger.info(f"\n{final_clean_text}")
        logger.info("--------------------------------------------------")

        return {
            "query": query_str,
            "answer": final_clean_text
        }
        
    except Exception as e:
        logger.exception("Fatal Exception encountered during main pipeline run process.")
        raise HTTPException(status_code=500, detail=str(e))
