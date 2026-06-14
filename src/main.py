from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="GenAI Document Q&A")

@app.get("/")
async def root():
    return {"status": "ok", "message": "GenAI Document Q&A API"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # placeholder for ingestion
    contents = await file.read()
    size = len(contents)
    return JSONResponse({"filename": file.filename, "size": size})

@app.post("/query")
async def query(q: str):
    # placeholder for query handling
    return JSONResponse({"query": q, "answer": "This is a stub response."})
