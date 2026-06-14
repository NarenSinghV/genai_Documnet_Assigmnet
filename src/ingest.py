import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_text_from_file(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".pdf":
        # simple helper: if you need more robust PDF parsing, add PyPDF2 or pdfplumber
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            texts = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(texts)
        except Exception:
            return ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
