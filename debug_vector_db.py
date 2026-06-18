import os, traceback
from src.vector_store import load_vectordb, _get_embeddings, query_vector_store

print("CWD:", os.getcwd())
print("CHROMA_DIR exists:", os.path.exists("./chromadb"))

try:
    db = load_vectordb()
    print("vectordb loaded:", bool(db))
except Exception as e:
    print("vectordb load error:", repr(e))
    db = None

try:
    emb = _get_embeddings()
    print("embeddings provider:", type(emb).__name__)
    try:
        qvec = emb.embed_query("disaster response")
        print("embed dim:", len(qvec))
    except Exception as e:
        print("embed error:", repr(e))
except Exception as e:
    print("_get_embeddings error:", repr(e))

if db:
    try:
        docs = query_vector_store(db, "disaster response", k=4)
        print("retrieved count:", len(docs))
        for i, d in enumerate(docs):
            meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            print("---", i, meta.get("source"), meta.get("chunk_index"))
            content = getattr(d, "page_content", "")
            print(content[:600].replace("\n", " "))
    except Exception:
        traceback.print_exc()
else:
    print("No vectordb to query")
