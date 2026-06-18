import os
from typing import List, Optional, Sequence, Dict, Any

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def _get_embeddings():
    return OpenAIEmbeddings()


def load_vectordb(persist_directory: str = "./chromadb") -> Optional[Chroma]:
    """Load an existing Chroma vector DB if present, else return None."""
    embeddings = _get_embeddings()
    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        # Chroma constructor with persist_directory loads existing DB
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return None


def create_or_update_vector_store(
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    persist_directory: str = "./chromadb",
) -> Chroma:
    """Create or update a Chroma vector store with texts and optional metadatas.

    metadatas should be a sequence with same length as texts.
    """
    embeddings = _get_embeddings()
    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # add_texts supports metadatas
        vectordb.add_texts(list(texts), metadatas=list(metadatas) if metadatas else None)
    else:
        vectordb = Chroma.from_texts(list(texts), embeddings, metadatas=list(metadatas) if metadatas else None, persist_directory=persist_directory)
    return vectordb


def query_vector_store(vectordb: Chroma, query: str, k: int = 4):
    """Return top-k documents for the query from the provided vector store (with metadata)."""
    return vectordb.similarity_search(query, k=k)
