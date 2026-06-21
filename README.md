# GenAI Document Agent Platform

Generative AI Autonomous Document Q&A (Capstone) — Multi-Agent RAG Platform powered by HuggingFace embeddings, ChromaDB, and the official Google GenAI SDK.

## Overview

This project implements an enterprise-grade Autonomous Retrieval-Augmented Generation (RAG) platform structured into a synchronized multi-agent workflow. The application features isolated document context matching, automatic validation metrics, and structured streaming logging.

### System Components
- **Inbound Ingestion Layer (`ingest.py`)**: PDF, CSV, Excel, and text processing using `PyPDF2` and `Pandas`. Extracted text is segmented natively via `RecursiveCharacterTextSplitter`.
- **Persistent Vector Array (`vector_store.py`)**: Stores vector matrices locally using **ChromaDB**. High-performance local vector conversions are handled via a custom `all-MiniLM-L6-v2` SentenceTransformer adapter interface.
- **Multi-Agent Orchestration Engine (`agents.py`)**: 
  - `RetrievalAgent`: Pulls high-relevancy context blocks (\(k=5\)) isolated strictly by the file session handle.
  - `ReasoningAgent`: Generates high-density summaries using the modern stable **Google Gemini 2.5 Flash** runtime engine.
  - `VerifierAgent`: A hallucination guardrail agent that performs real-time token proximity checks to establish string correctness scores.
- **Enterprise Router (`main.py`)**: A production FastAPI server handling session state tracking, text sanitization formatting, and logging.

---

## Technical Architecture Diagram

```text
                               ┌───────────────────────────────┐
                               │     User UI Interface Layer   │
                               └───────────────┬───────────────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼ (POST /upload)                                      ▼ (POST /query_rag)
┌──────────────────────────────────────┐               ┌──────────────────────────────────────┐
│ Inbound Ingestion Pipeline (ingest)   │               │ Multi-Agent Framework Core (agents)  │
├──────────────────────────────────────┤               ├──────────────────────────────────────┤
│ • PyPDF2 / Pandas Content Extractor  │               │ • Request Payload Session Mapper     │
│ • Recursive Character Token Splitter │               │ • Dynamic Token Context Evaluator    │
│ • Key-Isolated Metadata Tagger       │               │ • Real-time UTF-8 File Logger        │
└──────────────────┬───────────────────┘               └──────────────────┬───────────────────┘
                   │                                                      │
                   │ (Write Vector Collections)                           │ (Orchestrated Pipeline Execution)
                   ▼                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Enterprise Vector Database Storage Array (ChromaDB)                     │
│                             Embedding Base Transformation Engine: all-MiniLM-L6-v2                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                          │
       ┌──────────────────────────────────────────────────────────────────┴────────────────────────────────┐
       ▼ (1. Semantic Lookup)                                             ▼ (2. Prompt Synthesis)          ▼ (3. Guardrail Match)
┌──────────────────────────────┐                                   ┌──────────────────────────────┐ ┌──────────────────────────────┐
│       RetrievalAgent         │                                   │        ReasoningAgent        │ │        VerifierAgent         │
├──────────────────────────────┤                                   ├──────────────────────────────┤ ├──────────────────────────────┤
│ Fetches optimal context maps │                                   │ Generates high-density summaries│ Calculates context proximity │
│ utilizing source file-level  │──────────────────────────────────►│ using Google Gemini 2.5      │►│ tokens to establish          │
│ metadata filters (k=5).      │                                   │ Flash core engines.          │ │ factual verification scores. │
└──────────────────────────────┘                                   └──────────────────────────────┘ └──────────────────────────────┘
                                                                                                                   │
                                                                                                                   ▼
                                                                                                    ┌──────────────────────────────┐
                                                                                                    │     Sanitised User Screen    │
                                                                                                    └──────────────────────────────┘
```

---

## Quick Start (Windows / PowerShell)

### 1. Create and Activate a Virtual Environment
```powershell
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install google-genai python-dotenv langchain-community sentence-transformers PyPDF2 pandas openpyxl
```

### 3. Configure Environment Variables
Create a file named `.env` in the root folder of the project. Declare your credentials exactly like this:
```env
GEMINI_API_KEY=AIzaSyYourActualGoogleAIStudioKeyHere
```

### 4. Start the API Server
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Ingest Documents
Use the interactive frontend by navigating to `http://127.0.0.1:8000/` or use the provided Python uploader tool:
```powershell
& .\.venv\Scripts\python.exe upload_test.py
```
Alternatively, upload files using cURL:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@uploads/your_document.pdf"
```

### 6. Query the Multi-Agent RAG Endpoint
Submit a request using cURL to fetch an ultra-clean, text-sanitized, human-readable chatbot response:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/query_rag" -H "Content-Type: application/json" -d "{\"q\":\"Summarise the document in 10 simple points\"}"
```

---

## Production Diagnostics & Logging

The platform includes a real-time event logging system. Every file conversion, semantic retrieval step, LLM token communication event, verification audit score, and final user response block is printed to the terminal console and permanently saved to a file called `app.log` in the root folder.

### Sample In-Line Trace Log Format:
```text
[2026-06-21 21:49:34] INFO [genai.run:45] -> [AgentManager] Starting Multi-Agent Pipeline Execution for query: 'Summarise the document in 10 simple points'
[2026-06-21 21:49:34] INFO [genai.run:47] -> [RetrievalAgent] Fetching semantic contexts matching query. Target file filter: ad9e2ff4_Test_upload.pdf
[2026-06-21 21:49:34] INFO [genai.run:52] -> [RetrievalAgent] Successfully gathered 5 relevant context blocks.
[2026-06-21 21:49:34] INFO [genai.run:56] -> [ReasoningAgent] Dispatching synchronized structured content payload to Gemini core engine...
[2026-06-21 21:49:41] INFO [genai.run:58] -> [ReasoningAgent] Received raw execution text stream response (Length: 1140 characters).
[2026-06-21 21:49:41] INFO [genai.verify:14] -> [VerifierAgent] Beginning hallucination verification loop...
[2026-06-21 21:49:41] INFO [genai.verify:24] -> [VerifierAgent] Completed verification. Score: 0.76
[2026-06-21 21:49:41] INFO [genai.query_rag:95] -> --- Final Sanitized User-Facing Chatbot Answer ---
1. HTML lists are used to group items together to improve readability.
2. There are three types of lists: Ordered List, Unordered List, and Description List.
...
```

---

## Troubleshooting

- **Configuration Error Message**: If your interface shows a `Configuration Error` screen, double-check that your `.env` file is in the root directory and that the `GEMINI_API_KEY` name is spelled correctly without spaces.
- **Chroma DB Migration Issues**: If your application crashes after a framework dependency update, safely delete old local collection data using PowerShell to force clean serialization:
  ```powershell
  Remove-Item -Recurse -Force .\chromadb
  ```
- **Text Truncation Bugs**: If points cut off mid-sentence, it means your `k` context lookups are pulling too much text or too little. The codebase is pre-configured to `k=5`, which balances context size and model token limits perfectly.

---

## Repository Architecture Map

- `src/main.py` — FastAPI gateway router, real-time logging, and text sanitization engines.
- `src/agents.py` — Multi-agent setup (`AgentManager`, `RetrievalAgent`, `ReasoningAgent`, `VerifierAgent`) using the `google-genai` SDK.
- `src/ingest.py` — Multi-format file loaders (`PyPDF2`, `Pandas`), recursive data slicing, and metadata isolation filters.
- `src/vector_store.py` — Local HuggingFace SentenceTransformer wrapper matching Chroma expectations.
- `static/` — HTML/JS single-page frontend interface layer.
