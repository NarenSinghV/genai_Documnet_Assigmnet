import logging
import os

logger = logging.getLogger("genai")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/text-bison-001")
logger.info("vector_store loaded; GEMINI_MODEL=%s", GEMINI_MODEL)

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
    # Prefer a local HuggingFace SentenceTransformer for offline embeddings.
    try:
        logger.info("Using local sentence-transformers for embeddings")
        from sentence_transformers import SentenceTransformer

        class HFEmbeddingsWrapper:
            def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts):
                # returns list[list[float]]
                vecs = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
                return [list(map(float, v)) for v in vecs]

            def embed_query(self, text: str):
                vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
                return list(map(float, vec))

        return HFEmbeddingsWrapper()
    except Exception as ex:
        logger.warning("Local sentence-transformers not available: %s", ex)
        # Fall back to OpenAI embeddings (if available)
        try:
            logger.info("Falling back to OpenAI embeddings")
            return OpenAIEmbeddings()
        except Exception as e:
            logger.exception("No embedding provider available")
            raise ImportError("No embedding provider available. Install 'sentence-transformers' or configure OpenAI embeddings.")

# -------------------------
# Load existing DB
# -------------------------
def load_vectordb(persist_directory: str = "./chromadb") -> Optional[Chroma]:
    logger.info("Loading vectordb from %s", persist_directory)
    if not os.path.exists(persist_directory):
        logger.info("Persist directory does not exist: %s", persist_directory)
        return None

    if not any(os.scandir(persist_directory)):
        logger.info("Persist directory empty: %s", persist_directory)
        return None

    embeddings = _get_embeddings()
    logger.info("Opening Chroma vectordb (persist=%s)", persist_directory)
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
    logger.info("Creating/updating vector store. persist_directory=%s; documents=%s", persist_directory, len(texts))
    embeddings = _get_embeddings()

    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        logger.info("Updating existing vectordb at %s", persist_directory)
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vectordb.add_texts(
            texts=list(texts),
            metadatas=list(metadatas) if metadatas else None
        )
    else:
        logger.info("Creating new vectordb at %s", persist_directory)
        vectordb = Chroma.from_texts(
            texts=list(texts),
            embedding=embeddings,
            metadatas=list(metadatas) if metadatas else None,
            persist_directory=persist_directory
        )

    vectordb.persist()
    logger.info("Persisted vectordb at %s", persist_directory)
    return vectordb


# -------------------------
# Query
# -------------------------
def query_vector_store(vectordb: Chroma, query: str, k: int = 4):
    logger.info("Querying vectordb (k=%s): %s", k, query[:120])
    try:
        return vectordb.similarity_search(query, k=k)
    finally:
        logger.info("Query complete")