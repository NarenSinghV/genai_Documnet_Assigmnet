# genai_Documnet_Assigmnet

Generative AI Document Q&A (Capstone)

## Overview

This repository is a starter scaffold for a Generative AI document question-answering application using a FastAPI backend. It provides endpoints to upload documents, ingest them into a vector store, and query the knowledge base using an LLM + RAG pipeline. Agent scaffolding is included for planning/reasoning components.

## Quick start

1. Create and activate a Python virtual environment (already created in this workspace):
   - PowerShell:
     . .\.venv\Scripts\Activate.ps1
2. Install dependencies:
   python -m pip install -r requirements.txt
3. Create a `.env` file from `.env.example` and set your OpenAI key and any other secrets.
4. Run the API server:
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

## Files

- `src/main.py` — FastAPI app (upload and query endpoints).
- `src/ingest.py` — document ingestion and chunking routines.
- `src/vector_store.py` — vector store wrapper (Chroma) and embedding helpers.
- `src/agents.py` — agent skeleton that uses planner/retriever/LLM to answer queries.
- `.env.example` — environment variable examples.
- `.gitignore` — ignores .venv and .env.

## Notes and next steps

- This scaffold uses LangChain + Chroma + OpenAI embeddings by default. You can swap embedding/LLM providers.
- Add tests, CI, and deployment configs when ready.
- Do not commit secrets — keep `.env` ignored.

## License

Add your license and project details.
