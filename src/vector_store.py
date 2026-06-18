import os
from typing import Sequence, Dict, Any, Optional

# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# -------------------------
# Embeddings
# -------------------------
# def _get_embeddings():
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY is not set in environment")

    # return OpenAIEmbeddings(
    #     api_key=api_key,
    #     model="text-embedding-3-small"
    # )

from langchain_community.embeddings import HuggingFaceEmbeddings

def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------------------------
# Load existing DB
# -------------------------
def load_vectordb(persist_directory: str = "./chromadb") -> Optional[Chroma]:
    if not os.path.exists(persist_directory):
        return None

    if not any(os.scandir(persist_directory)):
        return None

    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


# -------------------------
# Create / Update DB
# -------------------------
def create_or_update_vector_store(
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    persist_directory: str = "./chromadb",
) -> Chroma:
    embeddings = _get_embeddings()

    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vectordb.add_texts(
            texts=list(texts),
            metadatas=list(metadatas) if metadatas else None
        )
    else:
        vectordb = Chroma.from_texts(
            texts=list(texts),
            embedding=embeddings,
            metadatas=list(metadatas) if metadatas else None,
            persist_directory=persist_directory
        )

    vectordb.persist()
    return vectordb


# -------------------------
# Query
# -------------------------
def query_vector_store(vectordb: Chroma, query: str, k: int = 4):
    return vectordb.similarity_search(query, k=k)