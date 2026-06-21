import os
import logging
from typing import Sequence, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("genai")

class HFEmbeddingsWrapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return list(map(float, vec))

def _get_embeddings():
    return HFEmbeddingsWrapper()

def load_vectordb(persist_directory: str = "./chromadb") -> Optional[Chroma]:
    if not os.path.exists(persist_directory) or not any(os.scandir(persist_directory)):
        return None
    return Chroma(persist_directory=persist_directory, embedding_function=_get_embeddings())

def create_or_update_vector_store(
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    persist_directory: str = "./chromadb",
) -> Chroma:
    embeddings = _get_embeddings()
    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectordb.add_texts(texts=list(texts), metadatas=list(metadatas) if metadatas else None)
    else:
        vectordb = Chroma.from_texts(
            texts=list(texts),
            embedding=embeddings,
            metadatas=list(metadatas) if metadatas else None,
            persist_directory=persist_directory
        )
    return vectordb
