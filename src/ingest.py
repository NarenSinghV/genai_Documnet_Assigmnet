import os
from typing import List, Dict, Any
import logging

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as e:
            raise ImportError(
                "RecursiveCharacterTextSplitter not found. Install 'langchain' with the text-splitters module or 'langchain_text_splitters' package."
            ) from e

from .vector_store import create_or_update_vector_store

logger = logging.getLogger("genai")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

logger.info("ingest module loaded")


def load_text_from_file(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            texts = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(texts)
        except Exception:
            return ""
    elif ext in [".csv"]:
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_csv(index=False)
        except Exception:
            return ""
    elif ext in [".xls", ".xlsx"]:
        try:
            import pandas as pd
            df = pd.read_excel(path)
            return df.to_csv(index=False)
        except Exception:
            return ""
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def ingest_file_to_vectordb(path: str, metadata: Dict[str, Any] = None, persist_directory: str = "./chromadb"):
    """Load file, chunk text with metadata, and add to persistent vector DB."""
    metadata = metadata or {}
    logger.info("Ingesting file: %s", path)
    text = load_text_from_file(path)
    if not text:
        logger.warning("No text extracted from file: %s", path)
        return None
    chunks = chunk_text(text)
    logger.info("Created %s chunks for %s", len(chunks), path)
    
    metadatas = []
    for i, c in enumerate(chunks):
        meta = dict(metadata)
        # Ensure the filename key aligns perfectly with downstream filters
        meta.update({
            "source": str(os.path.basename(path)),
            "chunk_index": int(i),
        })
        metadatas.append(meta)
    try:
        vectordb = create_or_update_vector_store(chunks, metadatas=metadatas, persist_directory=persist_directory)
        logger.info("Vector DB updated for %s", path)
        return vectordb
    except Exception as e:
        logger.exception("vectordb creation failed for %s: %s", path, e)
        return None


def format_provenance(documents: List[Any]):
    """Format a list of retrieved documents into short provenance entries."""
    out = []
    for d in documents:
        meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
        src = meta.get("source") or meta.get("id") or "unknown"
        chunk_idx = meta.get("chunk_index")
        snippet = (getattr(d, "page_content", None) or str(d))[:300]
        out.append({"source": src, "chunk_index": chunk_idx, "snippet": snippet})
    return out