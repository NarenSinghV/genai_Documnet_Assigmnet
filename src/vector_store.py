from typing import List

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def create_vector_store(docs: List[str], persist_directory: str = "./chromadb"):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(docs, embeddings, persist_directory=persist_directory)
    return vectordb


def query_vector_store(vectordb, query: str, k: int = 4):
    results = vectordb.similarity_search(query, k=k)
    return results
