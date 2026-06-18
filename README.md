# genai_Documnet_Assigmnet

Generative AI Document Q&A (Capstone) — RAG using HuggingFace embeddings + ChromaDB

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that runs locally:

- PDF / file upload -> text extraction
- Text chunking
- Local embeddings (HuggingFace) — no external API required by default
- ChromaDB (persistent vector store)
- Semantic search (top-K retrieval)
- Answer generation from retrieved context (simple prompt executor / agent scaffold)

Note: OpenAI integration is present in the repository but commented/outlined so you can run fully offline using HuggingFace embeddings.

## Quick start (Windows / PowerShell)

1. Create and activate a virtual environment (if not already present):
   py -3 -m venv .venv
   . .\.venv\Scripts\Activate.ps1

2. Install Python dependencies:
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

3. Configure environment variables:
   - Copy `.env.example` to `.env` and edit as needed.
   - `OPENAI_API_KEY` is optional if you plan to use OpenAI features; the default project uses HuggingFace embeddings so a key is not required.
   - `CHROMA_PERSIST_DIRECTORY` (default: `./chromadb`) is where Chroma stores persisted vectors.

4. Start the API server (use the venv Python to ensure correct packages):
   .\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000

5. Ingest documents (upload) — examples:
   - Use the provided Python test uploader `upload_test.py` in the repo root (recommended on Windows to avoid shell quoting issues):
     & .\.venv\Scripts\python.exe upload_test.py

   - Or use curl (careful with filenames containing & or spaces):
     curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@uploads/yourfile.pdf"

6. Query the RAG endpoint
   - /query returns raw snippets
   - /query_rag returns an answer generated from retrieved context
   - /query_agent runs the agent orchestration (retrieval -> reasoning -> verification)

## Run locally (Windows / PowerShell)

1. Create and activate virtual environment

   py -3 -m venv .venv
   . .\.venv\Scripts\Activate.ps1

2. Upgrade pip and install project dependencies

   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   # ensure langchain text-splitters and chromadb are available
   .\.venv\Scripts\python.exe -m pip install "langchain[text-splitters]" chromadb langchain-openai

3. Start the FastAPI server (use the venv Python)

   .\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000

4. Upload a file (recommended: use the provided Python uploader to avoid shell quoting issues)

   & .\.venv\Scripts\python.exe upload_test.py

   Or with curl (careful with filenames that contain spaces or &):

   curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@uploads/yourfile.pdf"

5. Query the RAG endpoint (example using curl)

   curl.exe -X POST "http://127.0.0.1:8000/query_rag" -F "q=What is the deadline for submission?"

6. Run tests

   .\.venv\Scripts\python.exe -m pytest -q

## Troubleshooting

- If you see errors about missing langchain.text_splitter, run:

  .\.venv\Scripts\python.exe -m pip install "langchain[text-splitters]"


- If uploads complete but `vectordb` is false in the response, inspect the corresponding status JSON in `uploads/status/<id>.json` for errors (common: Chroma migration warning).
- If Chroma reports a deprecated configuration and you do not need existing data, remove the persisted directory and re-ingest:
  Remove-Item -Recurse -Force .\chromadb

- If you need to preserve an existing Chroma DB, follow the migration instructions and tool: `pip install chroma-migrate` and run `chroma-migrate` as described in the Chroma docs.

## Development notes

- Embeddings: the code is configured to prefer a local HuggingFace embeddings provider. If you uncomment or enable OpenAI usage you must set `OPENAI_API_KEY` and accept external API usage.
- Vector store: uses Chroma (chromadb). Ensure `chromadb` is installed in the same Python environment used to run the server.

## Testing

- Unit tests for ingestion exist in `tests/` and can be run with pytest from the venv:
  .\.venv\Scripts\python.exe -m pytest -q

## Security

- Never commit secrets. `.env` is ignored by the recommended `.gitignore` in this repo.

## Files of interest

- `src/main.py` — FastAPI app and endpoints
- `src/ingest.py` — document loaders, chunking, ingest pipeline
- `src/vector_store.py` — embedding provider and Chroma wrapper
- `src/agents.py` — agent orchestration (retrieval, reasoner, verifier)
- `static/` — simple frontend for manual testing
